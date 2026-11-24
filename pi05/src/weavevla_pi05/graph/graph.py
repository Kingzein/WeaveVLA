"""Four-dimensional geometry; post-pre-GNN edge GRU; BPTT=1.

Metadata controls validity/update policy but is never concatenated into edge
features. The only physical edge encoder input is [dx,dy,dz,center].
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .base_graph import SemanticGeometricGraph, EdgeAwareLayer, mlp
from .geometry import default_recipe


class SparseLayer(EdgeAwareLayer):
    def forward(self, nodes, edges, src, dst):
        n, dim = nodes.shape
        q, k, v = self.qkv(nodes).reshape(n, 3, self.heads, self.width).unbind(1)
        scores = torch.einsum('ihd,jhd->hij', q, k) / self.width**.5
        bias = nodes.new_zeros(n, n, self.heads)
        bias[src, dst] = self.bias(edges)
        allowed = torch.eye(n, device=nodes.device, dtype=torch.bool)
        allowed[src, dst] = True
        weights = (scores+bias.permute(2, 0, 1)).masked_fill(~allowed[None], -torch.inf).softmax(-1)
        edge_values = nodes.new_zeros(n, n, self.heads, self.width)
        edge_values[src, dst] = self.edge_value(edges).reshape(-1, self.heads, self.width)
        message = torch.einsum('hij,ijhd->ihd', weights, v.unsqueeze(0)+edge_values)
        updated = self.node_norm(nodes+self.out(message.reshape(n, dim)))
        updated = self.ffn_norm(updated+self.ffn(updated))
        edges = self.edge_norm(edges+self.edge_update(torch.cat([nodes[src], nodes[dst], edges], -1)))
        return updated, edges


@torch.no_grad()
def fps(points, count=128):
    if len(points) <= count:
        return points.detach()
    indices, distance = [], torch.full((len(points),), torch.inf, device=points.device)
    nxt = (points-points.mean(0)).square().sum(-1).argmax()
    for _ in range(count):
        indices.append(nxt)
        distance = torch.minimum(distance, (points-points[nxt]).square().sum(-1))
        nxt = distance.argmax()
    return points[torch.stack(indices)].detach()


def build_pairs(ids, prompt_ids, gripper, distances, observed, previous=(), target=2):
    """Prompt clique + gripper links + prompt-only nearest K object union.

Rank all observed other objects, including prompt objects, before deduplication.
Existing task edges consume nearest-K slots; never refill with farther objects.
Non-prompt objects are passive endpoints. Preserve eligible occluded edges.
"""
    if not isinstance(target, int) or target < 0:
        raise ValueError('prompt_local_k must be a nonnegative integer')
    ids = set(ids)
    observed = set(observed)
    prompt = set(prompt_ids) & ids
    objects = ids - ({gripper} if gripper else set())
    prompt &= objects
    base = {(a, b) for a in prompt for b in prompt if a != b}
    if gripper in ids:
        base |= {(a, gripper) for a in prompt} | {(gripper, a) for a in prompt}
    base |= {(a, b) for a, b in previous if a in ids and b in ids and a != b
             and a in objects and b in objects and (a in prompt or b in prompt)
             and (a not in observed or b not in observed)}
    base |= {(b, a) for a, b in tuple(base)}
    added = set()
    for a in sorted(prompt & observed):
        candidates = (objects & observed) - {a}
        ranked = sorted(candidates, key=lambda b: (distances[a, b], b))
        for b in ranked[:target]:
            added.update(((a, b), (b, a)))
    return sorted(base | added)


class EdgeMemoryGraph(SemanticGeometricGraph):
    def __init__(self, vl_dim, cfg=None, memory=True):
        cfg = dict(default_recipe()['graph'], **(dict(cfg) if cfg is not None else {}))
        if cfg.get('bptt', 1) != 1 or cfg.get('geometry_input_dim', 4) != 4 or cfg.get('edge_status_input', False):
            raise ValueError('This version requires BPTT=1 and geometry-only 4D edges')
        if cfg.get('geometry_loss_dim', 4) != 4:
            raise ValueError('Surface loss extension requires a separate recipe')
        super().__init__(vl_dim, cfg)
        dim = int(cfg['hidden_dim'])
        self.cfg = cfg
        self.edge_encoder = mlp(4, dim, dim)
        self.edge_gru = nn.GRUCell(dim, dim)
        self.memory_fusion = mlp(2*dim, dim, dim)
        for layer in list(self.pre)+list(self.task):
            layer.__class__ = SparseLayer
        self.memory_enabled = memory
        self.states, self.contexts, self.last_trace = {}, [], []
        self.record_trace = True

    def reset(self, key=None):
        if key is None:
            self.states.clear()
        else:
            self.states.pop(str(key), None)

    def detach_memory(self):
        for state in self.states.values():
            for obj in state['registry'].values():
                obj['memory'] = obj['memory'].detach()
            for edge in state['edge_memory'].values():
                edge['hidden'] = edge['hidden'].detach()

    def temporal_state_dict(self):
        self.detach_memory()
        return copy.deepcopy(self.states)

    def forward(self, objects_batch, semantic_batch, vl_hidden, attention_mask=None,
                compute_geometry=True, visual_batch=None):
        if not (len(self.contexts) == len(objects_batch) == len(semantic_batch) == len(vl_hidden)):
            raise ValueError('Exactly one context per batch slot is required')
        if len({str(c['episode_key']) for c in self.contexts}) != len(self.contexts):
            raise ValueError('Concurrent batch slots cannot share an episode memory')
        if self.use_visual_patches and visual_batch is None:
            raise ValueError('Visual node features required')
        outputs, losses, traces = [], [], []
        device, dtype = vl_hidden.device, self.queries.dtype
        for b, (objects, semantics, ctx) in enumerate(zip(objects_batch, semantic_batch, self.contexts)):
            key, step = str(ctx['episode_key']), int(ctx['step'])
            state = self.states.setdefault(key, dict(registry={}, edge_memory={}, pairs=[], step=-1, generation=None))
            if step <= state['step']:
                raise ValueError('Nonsequential graph update: reset or restore matching cursor')
            if len(objects) != len(semantics):
                raise ValueError('Semantic vectors and instance rows differ')
            ids = [str(o['id']) for o in objects]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate instance ID')
            generations = {int(o['geometry_generation']) for o in objects}
            if len(generations) > 1:
                raise ValueError('Mixed coordinate generations in one graph')
            generation = next(iter(generations), ctx.get('geometry_generation', state['generation']))
            if generation != state['generation']:
                state['edge_memory'].clear()
                state['pairs'] = []
                # Content may survive, but old coordinates cannot enter a new frame.
                for entry in state['registry'].values():
                    entry['points'] = None
            state['generation'], state['step'] = generation, step
            reg, current_content, sources = state['registry'], {}, {}
            for i, obj in enumerate(objects):
                oid, label = str(obj['id']), str(obj['label'])
                if label == 'robot':
                    raise ValueError('Only the gripper robot node is permitted')
                source = obj['geometry_source']
                if source not in ('fresh', 'static_supported', 'occluded_hold', 'missing', 'tcp_only'):
                    raise ValueError('Unknown geometry source')
                points = torch.as_tensor(obj['points'], dtype=torch.float32, device=device)
                valid = points.ndim == 2 and points.shape[1] == 3 and len(points) >= 16 and bool(torch.isfinite(points).all())
                prev = reg.get(oid)
                if prev and label != prev['label']:
                    raise ValueError('Instance identity changed label')
                if source in ('fresh', 'static_supported', 'tcp_only'):
                    if source != 'tcp_only' and (not valid or not obj['visible_now']):
                        raise ValueError('Observed geometry requires current valid evidence')
                    if label == 'robot gripper' and obj.get('position_source') != 'tcp':
                        raise ValueError('Gripper requires live TCP, not depth centroid')
                    center = torch.as_tensor(obj['geometry_center'],device=device,dtype=torch.float32)
                    if center.shape != (3,) or not bool(torch.isfinite(center).all()):raise ValueError('Invalid geometry center')
                    shape_center = points.mean(0) if len(points) else center
                    indices = torch.linspace(0, len(points)-1, min(len(points), self.max_points), device=device).long()
                    shape = self.point_encoder((points[indices]-shape_center).to(dtype)).amax(0) if len(points) else torch.zeros(int(self.cfg['hidden_dim']),device=device,dtype=dtype)
                    parts = [self.semantic(semantics[i].to(device=device, dtype=dtype)), shape]
                    if self.use_visual_patches:
                        parts.append(self.visual_projection(visual_batch[b][i].to(device=device, dtype=dtype)))
                    content = prev['memory'].to(device=device,dtype=dtype) if source=='tcp_only' and prev else self.node_encoder(torch.cat(parts))
                    current_content[oid] = content
                    # Registry points is only a non-None availability marker in the
                    # center-distance graph. Shape encoding above still uses all input
                    # points; no surface-distance consumer requires FPS here.
                    reg[oid] = dict(memory=content.detach(), label=label, center=center.detach(),
                                    points=points[:1].detach().clone(), seen=step,
                                    uv=obj.get('pixel_uv_normalized', [.5, .5]))
                elif prev:
                    if source == 'occluded_hold' and valid and label != 'robot gripper':
                        prev['points'] = points[:1].detach().clone()
                        prev['center'] = points.mean(0).detach()
                    else:
                        prev['points'] = None
                sources[oid] = source
            # The geometry processor must return all known IDs, including missing.
            for oid in reg.keys()-sources.keys():
                reg[oid]['points'] = None
                sources[oid] = 'missing'
            grippers = [oid for oid, r in reg.items() if r['label'] == 'robot gripper']
            if len(grippers) > 1:
                raise ValueError('Multiple grippers')
            gripper = grippers[0] if grippers else None
            usable = sorted(oid for oid, r in reg.items() if r['points'] is not None)
            observed = {oid for oid in usable if sources[oid] in ('fresh', 'static_supported', 'tcp_only')}
            distances = {}
            for i, a in enumerate(usable):
                for c in usable[i+1:]:
                    d = (reg[c]['center']-reg[a]['center']).norm().detach()
                    distances[a, c] = distances[c, a] = d
            # Keep the exact GPU norm arithmetic; transfer once instead of
            # synchronizing for every scalar comparison in Python sorting.
            if distances:
                distance_keys = list(distances)
                distance_values = torch.stack([distances[k] for k in distance_keys]).cpu().tolist()
                distances = dict(zip(distance_keys, distance_values))
            pairs = build_pairs(usable, ctx['prompt_ids'], gripper, distances, observed,
                                state['pairs'], self.cfg.get('prompt_local_k', 2))
            state['pairs'] = pairs
            # Known occluded prompt nodes survive even if all their edges expire.
            active = sorted(set(usable) | (set(map(str, ctx['prompt_ids'])) & reg.keys()) | set(grippers))
            if len(active) > self.max_objects:
                raise ValueError('Active graph exceeds max_objects; never truncate silently')
            ttl = self.cfg['edge_memory_ttl_steps']
            state['edge_memory'] = {p: e for p, e in state['edge_memory'].items() if step-e['updated'] <= ttl}
            edge_trace = []
            if active:
                nodes = torch.stack([current_content[a] if a in current_content else reg[a]['memory'].to(device=device, dtype=dtype) for a in active])
                index = {a: i for i, a in enumerate(active)}
                src = torch.tensor([index[a] for a, _ in pairs], device=device, dtype=torch.long)
                dst = torch.tensor([index[c] for _, c in pairs], device=device, dtype=torch.long)
                geometry = []
                for a, c in pairs:
                    delta = reg[c]['center']-reg[a]['center']
                    geometry.append(torch.cat([delta, delta.norm().reshape(1)]))
                geometry = torch.stack(geometry) if geometry else nodes.new_empty((0, 4))
                edges = self.edge_encoder(geometry.to(dtype))
                for layer in self.pre:
                    nodes, edges = layer(nodes, edges, src, dst)
                observed_flags = [a in observed and c in observed for a, c in pairs]
                if len(edges):
                    allowed = torch.tensor(observed_flags, dtype=torch.bool, device=device)
                    recurrent_dtype = self.edge_gru.weight_ih.dtype
                    previous = torch.stack([state['edge_memory'][p]['hidden'].detach().to(device=device, dtype=recurrent_dtype)
                                            if p in state['edge_memory'] and self.memory_enabled else torch.zeros_like(edges[i], dtype=recurrent_dtype)
                                            for i, p in enumerate(pairs)])
                    hidden = previous.clone()
                    if self.memory_enabled and any(observed_flags):
                        # Keep recurrent arithmetic/storage in the module dtype
                        # (FP32 by default), independent of outer BF16 autocast.
                        with torch.autocast(device.type, enabled=False):
                            hidden[allowed] = self.edge_gru(edges[allowed].to(recurrent_dtype), previous[allowed])
                    for i, pair in enumerate(pairs):
                        if self.memory_enabled and observed_flags[i]:
                            state['edge_memory'][pair] = dict(hidden=hidden[i].detach().clone(), updated=step)
                        if self.record_trace:
                            edge_trace.append(dict(source=pair[0], target=pair[1], geometry=geometry[i].detach().float().cpu().tolist(),
                                                   gru_updated=observed_flags[i] and self.memory_enabled,
                                                   last_gru_update_step=state['edge_memory'].get(pair, {}).get('updated')))
                    edges = self.memory_fusion(torch.cat([edges, hidden], -1))
                    mask = None if attention_mask is None else ~attention_mask[b:b+1].bool()
                    if mask is not None and bool(mask.all()):
                        raise ValueError('No valid VLM tokens')
                    context = self.relation_attention(edges[None], vl_hidden[b:b+1].to(dtype), vl_hidden[b:b+1].to(dtype),
                                                      key_padding_mask=mask, need_weights=False)[0][0]
                    edges = self.fusion_norm(edges+context)
                    edges = self.fusion_ffn_norm(edges+self.fusion_ffn(edges))
                for layer in self.task:
                    nodes, edges = layer(nodes, edges, src, dst)
                bank = torch.cat([nodes, edges])
                valid = torch.tensor([a in observed and c in observed for a, c in pairs], device=device, dtype=torch.bool)
                if compute_geometry and valid.any():
                    losses.append(F.smooth_l1_loss(self.geometry_head(edges[valid]).float(), geometry[valid, :4].detach().float()))
            else:
                bank = self.empty_scene
            queries = self.queries
            for layer in self.qformer:
                queries = layer(queries, bank[None])
            outputs.append(self.output_projection(queries)[0])
            if self.record_trace:
                traces.append(dict(episode_key=key, step=step, nodes=[dict(id=a, label=reg[a]['label'],
                    source=sources[a], prompt=a in set(map(str, ctx['prompt_ids'])), uv=reg[a]['uv']) for a in active], edges=edge_trace))
        self.last_trace = traces
        loss = torch.stack(losses).mean() if losses else vl_hidden.sum()*0
        if compute_geometry:
            loss = loss+sum(p.reshape(-1)[0]*0 for p in self.parameters())
        return torch.stack(outputs).to(vl_hidden.dtype), loss
