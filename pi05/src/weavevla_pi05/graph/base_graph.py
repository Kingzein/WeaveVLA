"""Object-centric relation reasoning; no recurrent graph state.

Objects: {label: str, points: float[P,3], id: optional scalar}.
All point clouds must share one metric coordinate frame. Full directed graph,
without self edges; self attention remains available for isolated nodes.
"""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, out))


class EdgeAwareLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError('graph hidden_dim must be divisible by num_heads')
        self.heads, self.width = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.bias = nn.Linear(dim, heads)
        self.edge_value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.edge_update = mlp(3 * dim, 2 * dim, dim)
        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.ffn = mlp(dim, 4 * dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, nodes, edges, src, dst):
        n, dim = nodes.shape
        q, k, v = self.qkv(nodes).reshape(n, 3, self.heads, self.width).unbind(1)
        score = torch.einsum('ihd,jhd->hij', q, k) / self.width ** .5
        bias = nodes.new_zeros(n, n, self.heads)
        bias[src, dst] = self.bias(edges)
        weights = (score + bias.permute(2, 0, 1)).softmax(-1)
        edge_values = nodes.new_zeros(n, n, self.heads, self.width)
        edge_values[src, dst] = self.edge_value(edges).reshape(-1, self.heads, self.width)
        message = torch.einsum('hij,ijhd->ihd', weights, v.unsqueeze(0) + edge_values)
        updated = self.node_norm(nodes + self.out(message.reshape(n, dim)))
        updated = self.ffn_norm(updated + self.ffn(updated))
        edges = self.edge_norm(edges + self.edge_update(torch.cat([nodes[src], nodes[dst], edges], -1)))
        return updated, edges


class QueryLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = mlp(dim, 4 * dim, dim)

    def forward(self, queries, bank):
        queries = self.norms[0](queries + self.self_attn(queries, queries, queries, need_weights=False)[0])
        queries = self.norms[1](queries + self.cross_attn(queries, bank, bank, need_weights=False)[0])
        return self.norms[2](queries + self.ffn(queries))


class SemanticGeometricGraph(nn.Module):
    def __init__(self, vl_dim, cfg):
        super().__init__()
        dim, heads = int(cfg.get('hidden_dim', 256)), int(cfg.get('num_heads', 8))
        count = int(cfg.get('num_queries', 16))
        self.max_objects = int(cfg.get('max_objects', 32))
        self.max_points = int(cfg.get('max_points', 512))
        if min(count, self.max_objects, self.max_points) < 1:
            raise ValueError('num_queries, max_objects and max_points must be positive')
        self.semantic = nn.Linear(vl_dim, dim)
        self.use_visual_patches = bool(cfg.get('use_visual_patches', False))
        if self.use_visual_patches:
            self.visual_projection = nn.Linear(vl_dim, dim)
        self.point_encoder = mlp(3, dim, dim)  # PointNet-style shared MLP + max pool
        self.node_encoder = mlp((3 if self.use_visual_patches else 2) * dim, dim, dim)
        self.edge_encoder = mlp(4, dim, dim)
        self.pre = nn.ModuleList([EdgeAwareLayer(dim, heads) for _ in range(int(cfg.get('pre_layers', 2)))])
        self.relation_attention = nn.MultiheadAttention(dim, heads, kdim=vl_dim, vdim=vl_dim, batch_first=True)
        self.fusion_norm = nn.LayerNorm(dim)
        self.fusion_ffn = mlp(dim, 4 * dim, dim)
        self.fusion_ffn_norm = nn.LayerNorm(dim)
        self.task = nn.ModuleList([EdgeAwareLayer(dim, heads) for _ in range(int(cfg.get('task_layers', 2)))])
        self.geometry_head = mlp(dim, dim, 4)
        self.queries = nn.Parameter(torch.randn(1, count, dim) * .02)
        self.empty_scene = nn.Parameter(torch.zeros(1, dim))
        self.qformer = nn.ModuleList([QueryLayer(dim, heads) for _ in range(int(cfg.get('qformer_layers', 2)))])
        self.output_projection = nn.Linear(dim, vl_dim)

    def forward(self, objects_batch, semantic_batch, vl_hidden, attention_mask=None, compute_geometry=True, visual_batch=None):
        if len(objects_batch) != vl_hidden.shape[0] or len(semantic_batch) != len(objects_batch):
            raise ValueError('Graph and VLM batch sizes must match')
        if self.use_visual_patches and (visual_batch is None or len(visual_batch) != len(objects_batch)):
            raise ValueError('Visual patch features required for every graph scene')
        outputs, geo_losses = [], []
        dtype, device = self.queries.dtype, vl_hidden.device
        for b, (objects, semantics) in enumerate(zip(objects_batch, semantic_batch)):
            if len(objects) > self.max_objects:
                raise ValueError(f'Object count exceeds max_objects={self.max_objects}; filter perception explicitly')
            if len(semantics) != len(objects):
                raise ValueError('One semantic embedding is required per object')
            if objects:
                pcs, centers = [], []
                for obj in objects:
                    points = torch.as_tensor(obj['points'], device=device, dtype=torch.float32)
                    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0 or not torch.isfinite(points).all():
                        raise ValueError('Each object requires a nonempty finite [P,3] point cloud')
                    center = points.mean(0)
                    centers.append(center)
                    # Local shape feature; centroid is explicitly used only in edges.
                    indices = torch.linspace(0, len(points) - 1, min(len(points), self.max_points), device=device).long()
                    pcs.append(self.point_encoder((points[indices] - center).to(dtype)).amax(0))
                parts = [self.semantic(semantics.to(device=device, dtype=dtype)), torch.stack(pcs)]
                if self.use_visual_patches:
                    visual = visual_batch[b]
                    if visual.shape != semantics.shape or not torch.isfinite(visual).all():
                        raise ValueError('One finite VLM patch vector required per object')
                    parts.append(self.visual_projection(visual.to(device=device, dtype=dtype)))
                nodes = self.node_encoder(torch.cat(parts, -1))
                centers = torch.stack(centers)
                src, dst = (~torch.eye(len(objects), dtype=torch.bool, device=device)).nonzero(as_tuple=True)
                delta = centers[dst] - centers[src]
                geometry = torch.cat([delta, delta.norm(dim=-1, keepdim=True)], -1)
                edges = self.edge_encoder(geometry.to(dtype))
                for layer in self.pre:
                    nodes, edges = layer(nodes, edges, src, dst)
                if len(edges):
                    mask = None if attention_mask is None else ~attention_mask[b:b+1].bool()
                    if mask is not None and mask.all():
                        raise ValueError('VLM attention mask must contain at least one valid token')
                    context = self.relation_attention(edges.unsqueeze(0), vl_hidden[b:b+1].to(dtype), vl_hidden[b:b+1].to(dtype), key_padding_mask=mask, need_weights=False)[0][0]
                    edges = self.fusion_norm(edges + context)
                    edges = self.fusion_ffn_norm(edges + self.fusion_ffn(edges))
                for layer in self.task:
                    nodes, edges = layer(nodes, edges, src, dst)
                bank = torch.cat([nodes, edges], 0)
                if compute_geometry and len(edges):
                    geo_losses.append(F.smooth_l1_loss(self.geometry_head(edges).float(), geometry.detach(), reduction='none'))
            else:
                bank = self.empty_scene
            queries = self.queries
            for layer in self.qformer:
                queries = layer(queries, bank.unsqueeze(0))
            outputs.append(self.output_projection(queries)[0])
        loss = torch.cat(geo_losses).mean() if geo_losses else vl_hidden.sum() * 0
        # Sparse/empty scenes must remain compatible with DDP without find_unused_parameters.
        # Also marks the training-only head used on edge-free batches.
        if compute_geometry:
            loss = loss + sum(p.reshape(-1)[0] * 0 for p in self.parameters())
        return torch.stack(outputs).to(vl_hidden.dtype), loss
