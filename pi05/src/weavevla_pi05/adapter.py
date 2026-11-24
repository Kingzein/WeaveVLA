"""Isolated action-expert adapter for pinned OpenPI PI0Pytorch(pi05=True).

This is a model-level integration, not a training launcher or online perception
server. It requires validated, causally prepared geometry at the configured control-step cadence.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS

from .alignment import preprocess_aligned
from .graph.graph import EdgeMemoryGraph
from .graph.visual_patch import mask_cell_weights


def image_digest(image):
    """Hash the already oriented/resized/normalized observation, before augmentation."""
    image = image.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(image.shape), str(image.dtype))).encode())
    h.update(image.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass
class GraphScene:
    objects: list[dict]
    label_ids: torch.Tensor  # [N,L], PaliGemma tokenizer IDs, without prompt wrappers
    label_mask: torch.Tensor  # same shape; excludes special tokens and padding
    context: dict  # episode_key, control step, prompt_ids, geometry_generation
    image_digests: dict[str, str]
    geometry_stride: int  # must match the configured temporal geometry recipe


def tokenize_object_labels(objects, sentencepiece_processor, max_length=64):
    """Use the SAME PaliGemma SentencePiece vocabulary as the backbone.

    No task wrappers, BOS/EOS or newline tokens are included in the label mean.
    Pass the processor behind OpenPI PaligemmaTokenizer, not a Qwen tokenizer.
    """
    rows = [sentencepiece_processor.encode(o['label'].strip().replace('_', ' ').replace('\n', ' '),
            add_bos=False, add_eos=False)[:max_length] for o in objects]
    if any(not row for row in rows):
        raise ValueError("Empty object label encoding")
    width = max((len(row) for row in rows), default=1)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
        valid[i, :len(row)] = True
    return ids, valid


def pool_patches(cells, masks, objects, rows, columns):
    """Mask area weights on SigLIP spatial cells; keep gradient to current patches."""
    if cells.shape[0] != rows * columns or len(masks) != len(objects):
        raise ValueError("SigLIP grid/object layout mismatch")
    result = []
    for obj, mask in zip(objects, masks, strict=True):
        # Cropping can remove a currently observed object's visual evidence.
        # Its raw geometric visibility is unchanged; never pool background.
        if not obj['visible_now'] or not bool(mask.any()):
            result.append(cells.new_zeros(cells.shape[-1]))
            continue
        weights = mask_cell_weights(mask, rows, columns).to(cells.device)
        with torch.autocast(cells.device.type, enabled=False):
            result.append((cells.float() * weights[:, None]).sum(0).to(cells.dtype))
    return torch.stack(result) if result else cells.new_empty((0, cells.shape[-1]))


class GraphPI05(PI0Pytorch):
    def __init__(self, config, *, graph_cfg=None, graph_enabled=True, geometry_stride=8):
        if not config.pi05:
            raise ValueError("This adapter targets pi0.5")
        if config.pytorch_compile_mode is not None:
            raise ValueError("Disable whole-sampler torch.compile for mutable graph memory")
        super().__init__(config)
        width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.graph_model = EdgeMemoryGraph(width, graph_cfg)
        if type(geometry_stride) is not int or geometry_stride < 1 or geometry_stride > config.action_horizon:
            raise ValueError("geometry_stride must fit inside the action horizon")
        self.geometry_stride = geometry_stride
        # Read out directly into the action expert's embedding space.
        latent_width = self.graph_model.output_projection.in_features
        self.graph_model.output_projection = torch.nn.Linear(latent_width, self.action_in_proj.out_features)
        self._expert_graph_tokens = None
        self.graph_enabled = graph_enabled
        self.geometry_loss = None
        self.last_spatial_trace = None
        self._scene_batch = None
        self._transformed_masks = None
        self._prefix_calls = 0

    def load_base_weights(self, state_dict):
        """Strict base-only load. Missing keys may belong ONLY to the new graph.

        Do this before constructing an optimizer/DDP wrapper. An already adapted
        checkpoint should instead be loaded with normal strict load_state_dict.
        """
        own = self.state_dict()
        expected = {k for k in own if not k.startswith('graph_model.')}
        if set(state_dict) != expected:
            raise ValueError('Base checkpoint keys do not exactly match the pi0.5 backbone')
        if any(state_dict[k].shape != own[k].shape for k in expected):
            raise ValueError('Base checkpoint tensor shape mismatch')
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected or any(not k.startswith('graph_model.') for k in missing):
            raise RuntimeError('Unexpected base-checkpoint load result')

    def reset_graph(self, episode_key=None):
        if self._scene_batch is not None:
            raise RuntimeError("Cannot reset during a policy call")
        self.graph_model.reset(episode_key)

    def _validate(self, observation, scenes):
        if scenes is None or len(scenes) != len(observation.state):
            raise ValueError("One graph scene is required per observation")
        keys = [str(s.context['episode_key']) for s in scenes]
        if len(keys) != len(set(keys)):
            raise ValueError("Batch slots must have distinct episode memories")
        for b, scene in enumerate(scenes):
            if scene.geometry_stride != self.geometry_stride:
                raise ValueError(f"Expected causal geometry with stride={self.geometry_stride}")
            last = self.graph_model.states.get(keys[b])
            if last is not None and int(scene.context['step']) - last['step'] != self.geometry_stride:
                raise ValueError(f"Expected next observation after {self.geometry_stride} control steps; reset at episode boundary")
            if scene.label_ids.ndim != 2 or len(scene.label_ids) != len(scene.objects) or scene.label_mask.shape != scene.label_ids.shape:
                raise ValueError("Object labels and instance rows do not align")
            if scene.label_ids.dtype != torch.long or scene.label_mask.dtype != torch.bool:
                raise ValueError("Label IDs must be int64 and valid-token mask bool")
            if len(scene.objects) and not bool(scene.label_mask.any(-1).all()):
                raise ValueError("Each object needs at least one label token")
            used_keys = set()
            for obj in scene.objects:
                view = obj['image_index']
                if view not in (0, 1):
                    raise ValueError("This graph uses only the two real cameras")
                key = IMAGE_KEYS[view]
                used_keys.add(key)
                if not bool(observation.image_masks[key][b]):
                    raise ValueError("Object cannot refer to a padded camera")
                mask = torch.as_tensor(obj['visual_mask'])
                if mask.dtype != torch.bool or mask.shape != (224, 224):
                    raise ValueError("Node visual mask must match the 224x224 model input")
            for key in used_keys:
                if scene.image_digests.get(key) != image_digest(observation.images[key][b]):
                    raise ValueError("RGB/mask observation fingerprint mismatch")

    @contextmanager
    def _request(self, observation, scenes):
        if self._scene_batch is not None:
            raise RuntimeError("Graph model calls must be serialized")
        self._validate(observation, scenes)
        old = {str(s.context['episode_key']): copy.deepcopy(self.graph_model.states.get(str(s.context['episode_key']))) for s in scenes}
        self._scene_batch = scenes
        self._prefix_calls = 0
        self.geometry_loss = None
        try:
            yield
            if self._prefix_calls != 1:
                raise RuntimeError("A graph observation must be consumed exactly once")
        except BaseException:
            for key, state in old.items():
                if state is None:
                    self.graph_model.states.pop(key, None)
                else:
                    self.graph_model.states[key] = state
            self.geometry_loss = None
            raise
        finally:
            self._scene_batch = None
            self._expert_graph_tokens = None
            self._transformed_masks = None
            self.graph_model.contexts = []

    def _preprocess_observation(self, observation, *, train=True):
        if self._scene_batch is None:
            return super()._preprocess_observation(observation, train=train)
        pixel_masks = {key: [] for key in IMAGE_KEYS}
        for view, key in enumerate(IMAGE_KEYS):
            for scene in self._scene_batch:
                masks = [torch.as_tensor(o['visual_mask']) for o in scene.objects if o['image_index'] == view]
                pixel_masks[key].append(torch.stack(masks) if masks else torch.empty(0, 224, 224, dtype=torch.bool))
        obs, self._transformed_masks, self.last_spatial_trace = preprocess_aligned(observation, pixel_masks, train=train)
        return list(obs.images.values()), list(obs.image_masks.values()), obs.tokenized_prompt, obs.tokenized_prompt_mask, obs.state

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        prefix, padding, attention = super().embed_prefix(images, img_masks, lang_tokens, lang_masks)
        if self._scene_batch is None:
            return prefix, padding, attention
        self._prefix_calls += 1
        if self._prefix_calls != 1:
            raise RuntimeError("Graph/GRU repeated within one observation")
        # Reuse image embeddings from the original prefix; no second SigLIP pass.
        patch = self.paligemma_with_expert.paligemma.config.vision_config.patch_size
        maps, offset = [], 0
        for image in images:
            h, w = image.shape[-2:] if image.shape[1] == 3 else image.shape[1:3]
            if h % patch or w % patch:
                raise ValueError("Image dimensions must be divisible by SigLIP patch size")
            rows, columns = h // patch, w // patch
            maps.append((prefix[:, offset:offset + rows * columns], rows, columns))
            offset += rows * columns
        if offset + lang_tokens.shape[1] != prefix.shape[1]:
            raise ValueError("Unexpected prefix packing")
        semantics, visuals = [], []
        for b, scene in enumerate(self._scene_batch):
            ids = scene.label_ids.to(prefix.device)
            valid = scene.label_mask.to(prefix.device)[..., None]
            embedded = self.paligemma_with_expert.embed_language_tokens(ids)
            semantics.append((embedded * valid).sum(1) / valid.sum(1).clamp_min(1))
            per_object = [None] * len(scene.objects)
            for view, (cells, rows, columns) in enumerate(maps):
                indices = [i for i, o in enumerate(scene.objects) if o['image_index'] == view]
                values = pool_patches(cells[b], self._transformed_masks[IMAGE_KEYS[view]][b], [scene.objects[i] for i in indices], rows, columns)
                for i, value in zip(indices, values, strict=True):
                    per_object[i] = value
            visuals.append(torch.stack(per_object) if per_object else prefix.new_empty((0, prefix.shape[-1])))
        self.graph_model.contexts = [s.context for s in self._scene_batch]
        with torch.autocast(prefix.device.type, enabled=False):
            tokens, self.geometry_loss = self.graph_model([s.objects for s in self._scene_batch], semantics,
                prefix, padding, compute_geometry=self.training, visual_batch=visuals)
        self._expert_graph_tokens = tokens
        # VLM sees exactly its original prefix, never graph tokens.
        return prefix, padding, attention

    def embed_suffix(self, state, noisy_actions, timestep):
        suffix, padding, attention, cond = super().embed_suffix(state, noisy_actions, timestep)
        if self._scene_batch is None:
            return suffix, padding, attention, cond
        tokens = self._expert_graph_tokens
        if tokens is None:
            raise RuntimeError("Graph must be computed once before expert denoising")
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=suffix.device)
        boundary = torch.zeros_like(valid)
        boundary[:, 0] = True
        # Separate attention block: graph reads prefix + graph, actions read both;
        # original prefix cannot attend to graph or action suffix.
        return (torch.cat([tokens.to(suffix.dtype), suffix], 1),
                torch.cat([valid, padding], 1), torch.cat([boundary, attention], 1), cond)

    def forward(self, observation, actions, noise=None, time=None, *, graph_batch=None):
        if not self.graph_enabled:
            self.geometry_loss = None
            return super().forward(observation, actions, noise=noise, time=time)
        with self._request(observation, graph_batch):
            return super().forward(observation, actions, noise=noise, time=time)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, *, graph_batch=None):
        if not self.graph_enabled:
            self.geometry_loss = None
            return super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
        if self.training:
            raise ValueError("Use eval() for policy inference")
        with self._request(observation, graph_batch):
            return super().sample_actions(device, observation, noise=noise, num_steps=num_steps)
