"""Frozen monocular RGB perception. Never reads sensor depth/RGB-D fields."""
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image


def rgb_image(value):
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError('Workspace RGB must be HWC or CHW')
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = value.transpose(1, 2, 0)
    if value.dtype != np.uint8:
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Float workspace RGB must be in [0,1]; use uint8 for [0,255]')
        value = (value * 255).round().astype(np.uint8)
    return Image.fromarray(value).convert('RGB')


def camera_matrix(example, cfg, size):
    """Scale calibration from its reference (height,width) to current RGB size."""
    value = example.get('graph_camera_intrinsics', cfg.get('camera_intrinsics'))
    resolution = example.get('graph_intrinsics_size', cfg.get('intrinsics_size'))
    if value is None or resolution is None:
        raise ValueError('Monocular graph requires camera_intrinsics and intrinsics_size=[height,width] (config or graph_* example fields)')
    matrix = np.asarray(value, dtype=np.float32).copy()
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError('camera_intrinsics must be a finite 3x3 pinhole matrix with positive focal lengths')
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError('intrinsics_size must be [height,width] with positive values')
    height, width = size
    matrix[0] *= width / resolution[1]
    matrix[1] *= height / resolution[0]
    return matrix


def unproject(depth, mask, intrinsics):
    """Backproject predicted metric depth to camera XYZ; filters invalid pixels."""
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError('Predicted depth and instance mask must have the same HxW shape')
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[ys, xs]
    # General calibrated pinhole inverse also supports skew.
    rays = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(intrinsics).T
    return (rays * z[:, None]).astype(np.float32)


class IdentityTracker:
    """Category-constrained nearest-centroid association only; no learned graph memory."""
    def __init__(self, threshold=.2):
        self.threshold = threshold
        self.streams = OrderedDict()

    def assign(self, objects, stream, reset=False):
        if reset:
            self.streams.pop(stream, None)
        previous, next_id = self.streams.pop(stream, ([], 0))
        centers = [np.asarray(o['points']).mean(0) for o in objects]
        candidates = sorted((float(np.linalg.norm(c - old_c)), i, j)
                            for i, (o, c) in enumerate(zip(objects, centers))
                            for j, (label, old_c, _) in enumerate(previous) if o['label'] == label)
        used, matched = set(), set()
        for distance, i, j in candidates:
            if distance <= self.threshold and i not in matched and j not in used:
                objects[i]['id'] = previous[j][2]
                used.add(j)
                matched.add(i)
        for i, obj in enumerate(objects):
            if i not in matched:
                obj['id'] = next_id
                next_id += 1
        self.streams[stream] = ([(o['label'], c, o['id']) for o, c in zip(objects, centers)], next_id)
        if len(self.streams) > 128:
            self.streams.popitem(last=False)
        return objects


class MonocularObjectPerception(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        self.cfg = dict(cfg)
        model_id = cfg.get('depth_model', 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
        self.depth_processor = AutoImageProcessor.from_pretrained(model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
        if getattr(self.depth_model.config, 'depth_estimation_type', None) != 'metric':
            raise ValueError('Graph geometry requires a metric monocular depth checkpoint, not relative/inverse depth')
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        checkpoint = cfg.get('detector_checkpoint')
        if checkpoint:
            self.detector = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            self.detector.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        else:
            self.detector = maskrcnn_resnet50_fpn(weights=weights)
        self.categories = weights.meta['categories']
        self.requires_grad_(False)
        self.train(False)
        self.tracker = IdentityTracker(float(cfg.get('tracking_distance', .2)))

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # torchvision ROI/NMS kernels do not uniformly support bfloat16.
        # Keep the frozen RGB teachers in FP32 when a policy/server casts the
        # trainable model to BF16; device moves still propagate normally.
        self.depth_model.float()
        self.detector.float()
        return result

    def train(self, mode=True):
        # Frozen teachers stay deterministic even when the policy enters train().
        return super().train(False)

    @torch.no_grad()
    def forward(self, examples, track=False):
        device = next(self.depth_model.parameters()).device
        batches = []
        for example in examples:
            # No consumption of depth, RGB-D or externally supplied 3D points.
            images = example['image']
            source = example.get('graph_rgb')
            if source is not None and self.cfg.get('use_visual_patches', False) and not example.get('graph_rgb_matches_workspace', False):
                raise ValueError('graph_rgb must be a full-frame resize of the selected VLM camera; set graph_rgb_matches_workspace only after verifying alignment')
            if source is None:
                source = images[int(example.get('graph_workspace_image_index', self.cfg.get('workspace_image_index', 0)))] if isinstance(images, (list, tuple)) else images
            image = rgb_image(source)
            width, height = image.size
            intrinsics = camera_matrix(example, self.cfg, (height, width))
            with torch.autocast(device.type, enabled=False):
                inputs = self.depth_processor(images=image, return_tensors='pt').to(device)
                # Match frozen module dtype (e.g. checkpoints cast by an inference server).
                inputs = {k: v.to(dtype=next(self.depth_model.parameters()).dtype) if v.is_floating_point() else v for k, v in inputs.items()}
                predicted = self.depth_model(**inputs).predicted_depth
                depth = F.interpolate(predicted.unsqueeze(1).float(), size=(height, width), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
                if 'graph_instances' in example:
                    # Optional externally detected 2D masks; depth is STILL predicted here.
                    instances = example['graph_instances']
                else:
                    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(device=device, dtype=next(self.detector.parameters()).dtype) / 255
                    detection = self.detector([tensor])[0]
                    instances = [{'label': self.categories[int(label)], 'mask': mask[0].float().cpu().numpy() > .5}
                                 for label, mask, score in zip(detection['labels'], detection['masks'], detection['scores'])
                                 if float(score) >= float(self.cfg.get('detection_threshold', .5))]
            objects = []
            for instance in instances[:int(self.cfg.get('max_objects', 32))]:
                points = unproject(depth, instance['mask'], intrinsics)
                if len(points) >= int(self.cfg.get('min_points', 16)):
                    # Choose a valid object pixel nearest the mask centroid; for
                    # concave masks the arithmetic centroid may be background.
                    ys, xs = np.nonzero(np.asarray(instance['mask'], dtype=bool) & np.isfinite(depth) & (depth > 0))
                    nearest = np.argmin((xs-xs.mean())**2 + (ys-ys.mean())**2)
                    objects.append({'label': str(instance['label']), 'points': points, 'id': len(objects),
                                    'image_index': int(example.get('graph_workspace_image_index', self.cfg.get('workspace_image_index', 0))),
                                    'pixel_uv_normalized': [(float(xs[nearest])+.5)/width, (float(ys[nearest])+.5)/height]})
            # Shuffled training samples never update trackers. Inference requires an
            # explicit stream ID, so unrelated requests cannot silently share identity.
            if track and 'graph_stream_id' in example:
                objects = self.tracker.assign(objects, str(example['graph_stream_id']), bool(example.get('graph_reset', False)))
            batches.append(objects)
        return batches
