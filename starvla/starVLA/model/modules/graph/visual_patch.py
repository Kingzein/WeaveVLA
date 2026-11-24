"""Align Qwen3-VL post-merger visual encoder tokens with object pixels.

These are encoder features before the language decoder, not final LLM states.
Qwen3 spatial merger emits row-major merged cells for each input image.
"""
import torch


def capture_visual_forward(interface, enabled, **inputs):
    if not enabled:
        return interface(**inputs), None
    model = interface.model
    if getattr(model.config, 'model_type', None) != 'qwen3_vl':
        raise ValueError('Node visual patches currently require Qwen3-VL')
    if inputs.get('pixel_values_videos') is not None:
        raise ValueError('Node visual patches require still images, not packed videos')
    captured = []
    def capture(module, args, output):
        tokens = output[0] if isinstance(output, tuple) else output
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError('Unexpected Qwen3 visual encoder output')
        captured.append(tokens)
    handle = model.model.visual.register_forward_hook(capture)
    try:
        outputs = interface(**inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Expected exactly one packed image encoder call')
    return outputs, captured[0]


def gather_object_patches(tokens, grid_thw, objects_batch, image_counts, merge_size):
    if tokens is None or tokens.ndim != 2:
        raise ValueError('Expected packed visual encoder tokens')
    if merge_size < 1:
        raise ValueError('Invalid spatial merge size')
    if grid_thw is None or len(grid_thw) != sum(image_counts) or len(objects_batch) != len(image_counts):
        raise ValueError('Visual grids must match batch image counts')
    grids = grid_thw.detach().cpu().tolist()
    maps, offset = [], 0
    for t, h, w in grids:
        if t != 1 or h < 1 or w < 1 or h % merge_size or w % merge_size:
            raise ValueError('Expected divisible 2D image grids')
        h, w = h // merge_size, w // merge_size
        count = h * w
        if offset+count > len(tokens):
            raise ValueError('Visual tokens shorter than grid layout')
        maps.append(tokens[offset:offset+count].reshape(h,w,-1))
        offset += count
    if offset != len(tokens):
        raise ValueError('Visual token count does not match merged grids')
    result=[]; image_offset=0
    for objects, count in zip(objects_batch, image_counts):
        features=[]
        for obj in objects:
            view=int(obj['image_index'])
            if not 0 <= view < count:
                raise ValueError('Object workspace camera is not in VLM images')
            uv=torch.as_tensor(obj['pixel_uv_normalized'], dtype=torch.float64)
            if uv.shape != (2,) or not torch.isfinite(uv).all() or (uv < 0).any() or (uv >= 1).any():
                raise ValueError('Object image coordinates must be finite normalized [u,v] in [0,1)')
            feature_map=maps[image_offset+view]; h,w=feature_map.shape[:2]
            x,y=int(float(uv[0])*w),int(float(uv[1])*h)
            features.append(feature_map[y,x])
        result.append(torch.stack(features) if features else tokens.new_empty((0,tokens.shape[-1])))
        image_offset+=count
    return result
