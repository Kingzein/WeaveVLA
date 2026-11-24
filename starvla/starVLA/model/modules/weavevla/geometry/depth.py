"""Pinned DA3 metric adapter: BF16 network, FP32 metric conversion/interpolation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class DA3Metric:
    def __init__(self, weights, device='cuda', process_res=518):
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from depth_anything_3.cfg import create_object
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from depth_anything_3.utils.alignment import apply_metric_scaling
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This recipe requires CUDA BF16 support')
        root = Path(weights)
        config = json.loads((root/'config.json').read_text())
        if config['model_name'] != 'da3metric-large':
            raise ValueError('Wrong depth model: expected DA3METRIC-LARGE')
        self.model = create_object(OmegaConf.create(config['config']))
        weights_dict = load_file(str(root/'model.safetensors'))
        if not all(k.startswith('model.') for k in weights_dict):
            raise ValueError('Unexpected DA3 checkpoint key layout')
        self.model.load_state_dict({k[6:]: v for k, v in weights_dict.items()}, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.processor, self.metric_scaling = InputProcessor(), apply_metric_scaling
        self.process_res = int(process_res)
        import inspect
        self.provenance = dict(model='depth-anything/DA3METRIC-LARGE', weights_sha256=file_sha256(root/'model.safetensors'),
                               config_sha256=file_sha256(root/'config.json'),
                               input_processor_sha256=file_sha256(inspect.getfile(InputProcessor)),
                               metric_conversion_sha256=file_sha256(inspect.getfile(apply_metric_scaling)),
                               process_res=self.process_res, process_res_method='upper_bound_resize',
                               network_dtype='bfloat16', output_dtype='float32', depth_definition='camera_axis_z_metres')

    @torch.inference_mode()
    def __call__(self, rgb, K):
        image = rgb.convert('RGB') if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb, np.uint8))
        K = np.asarray(K, np.float32)
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError('K must correspond to the supplied RGB grid')
        x, _, intrinsics = self.processor([image], intrinsics=K[None].copy(),
                                          process_res=self.process_res, process_res_method='upper_bound_resize', sequential=True)
        # One image avoids batch shape unification crops. Resize-only preprocessing
        # allows direct interpolation to the original grid; K is transformed by DA3.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            canonical = self.model(x[None].to(self.device))['depth']
        with torch.autocast('cuda', enabled=False):
            metric = self.metric_scaling(canonical.float(), intrinsics[None].to(self.device).float())
            depth = torch.nn.functional.interpolate(metric, size=(image.height, image.width), mode='bicubic', align_corners=False)[0, 0]
        depth = depth.float().cpu().numpy()
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        meta = dict(self.provenance, image_size=[image.height, image.width], actual_input_size=list(x.shape[-2:]),
                    K=K.tolist(), K_processed=intrinsics[0].tolist(), camera_frame_convention='x_right_y_down_z_forward')
        return depth, meta
