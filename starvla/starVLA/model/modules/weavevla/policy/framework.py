import torch
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T as BaseQwen

class Qwen_GR00T(BaseQwen):
    """training and inference share mask-coverage-weighted conditioning."""
    def _graph_conditioning(self, examples, hidden, mask, compute_geometry, visual_tokens=None, qwen_inputs=None):
        if not self.graph_enabled:
            return hidden, mask, hidden.new_zeros(())
        with torch.autocast(hidden.device.type, enabled=False):
            from starVLA.model.modules.graph.camera_metadata import attach_graph_camera
            graph_cfg = self.config.framework.graph
            calibrated = []
            for example in examples:
                example = dict(example)
                if "graph_camera_intrinsics" not in example and graph_cfg.get("camera_intrinsics") is None:
                    cameras = self.config.datasets.vla_data.get("graph_cameras")
                    if cameras is not None:
                        key = example.get("graph_camera_key", graph_cfg.get("evaluation_camera"))
                        if key is None and "default" not in cameras:
                            raise ValueError("Select graph.evaluation_camera or example graph_camera_key for multi-camera evaluation")
                        example = attach_graph_camera(example, self.config.datasets.vla_data, key or "default")
                calibrated.append(example)
            objects = self.graph_perception(calibrated, track=not compute_geometry)
            semantics = []
            embedding = self.qwen_vl_interface.model.get_input_embeddings()
            tokenizer = self.qwen_vl_interface.processor.tokenizer
            for scene in objects:
                if scene:
                    encoded = tokenizer([obj["label"] for obj in scene], padding=True,
                                        truncation=True, max_length=64,
                                        add_special_tokens=False, return_tensors="pt")
                    ids = encoded["input_ids"].to(hidden.device)
                    valid = encoded["attention_mask"].to(hidden.device).unsqueeze(-1)
                    word_embeddings = embedding(ids)
                    semantics.append((word_embeddings * valid).sum(1) / valid.sum(1).clamp_min(1))
                else:
                    semantics.append(hidden.new_empty((0, hidden.shape[-1])))
            visual_batch = None
            if graph_cfg.get("use_visual_patches", True):
                from .visual_patch import gather_object_patches
                visual_batch = gather_object_patches(
                    visual_tokens, qwen_inputs.get("image_grid_thw"), objects,
                    [len(e["image"]) if isinstance(e["image"], (list, tuple)) else 1 for e in examples],
                    int(self.qwen_vl_interface.model.config.vision_config.spatial_merge_size),
                )
            graph_tokens, geometry_loss = self.graph_model(
                objects, semantics, hidden, mask, compute_geometry=compute_geometry, visual_batch=visual_batch
            )
        # Preserve every original VLM token and its original mask. Only append
        # projected graph tokens to the existing GR00T cross-attention context.
        if mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
        mask = torch.cat([mask.to(device=hidden.device, dtype=torch.bool),
                          torch.ones(graph_tokens.shape[:2], device=hidden.device, dtype=torch.bool)], dim=1)
        return torch.cat([hidden, graph_tokens], dim=1), mask, geometry_loss


