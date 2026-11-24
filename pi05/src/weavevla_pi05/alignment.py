"""OpenPI spatial augmentations with the same transforms applied to node masks.

RGB arithmetic/RNG order follows pinned OpenPI preprocessing_pytorch.py.
Geometry remains in the original camera chart; only visual masks are warped.
No detector, depth estimator, or temporal geometry state consumes augmented RGB.
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS


def resize_masks(masks, size):
    if len(masks) == 0:
        return masks.new_empty((0, *size))
    return F.interpolate(masks[:, None].float(), size=size, mode="nearest")[:, 0].bool()


def resize_image_and_masks(image, masks, size=(224, 224)):
    """Explicit aligned resize/pad before normalization, CHW image and NHW masks.

    Uses the same integer resize size and asymmetric remainder padding as OpenPI.
    The caller must also apply any camera orientation transform to BOTH inputs.
    """
    if image.ndim != 3 or image.shape[0] != 3 or masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("Image and masks must share the input grid")
    h, w = image.shape[-2:]
    ratio = max(w / size[1], h / size[0])
    rh, rw = int(h / ratio), int(w / ratio)
    resized = F.interpolate(image[None].float(), size=(rh, rw), mode="bilinear", align_corners=False)[0]
    masks = resize_masks(masks, (rh, rw))
    top, rem_h = divmod(size[0] - rh, 2)
    left, rem_w = divmod(size[1] - rw, 2)
    pads = (left, left + rem_w, top, top + rem_h)
    if image.dtype == torch.uint8:
        resized = resized.round().clamp(0, 255).to(torch.uint8)
        fill = 0
    else:
        resized = resized.clamp(-1, 1).to(image.dtype)
        fill = -1
    return F.pad(resized, pads, value=fill), F.pad(masks, pads, value=False)


def preprocess_aligned(observation, pixel_masks, *, train):
    """Return processed observation, transformed masks and spatial trace.

    pixel_masks[key][batch] is [number_of_objects_in_that_view,H,W]. The minimal
    adapter accepts already aligned 224x224 observations, just like the standard
    policy transforms. It rejects silently resized/misaligned external masks.
    """
    if not set(IMAGE_KEYS).issubset(observation.images):
        raise ValueError("Missing OpenPI camera keys")
    out_images, out_masks, out_pixel_masks, traces = {}, {}, {}, {}
    for key in IMAGE_KEYS:
        image = observation.images[key]
        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        if image.shape[1:3] != (224, 224):
            raise ValueError("Align RGB and masks to 224x224 before graph observation construction")
        masks = [m.to(image.device).clone() for m in pixel_masks[key]]
        if len(masks) != len(image) or any(m.dtype != torch.bool or m.ndim != 3 or m.shape[-2:] != (224, 224) for m in masks):
            raise ValueError("Mask batch/grid does not match images")
        trace = {"crop": None, "angle": 0.0}
        if train:
            image = image / 2.0 + 0.5
            if "wrist" not in key:
                height, width = image.shape[1:3]
                ch, cw = int(height * .95), int(width * .95)
                max_h, max_w = height - ch, width - cw
                if max_h > 0 and max_w > 0:
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h:start_h + ch, start_w:start_w + cw, :]
                    masks = [m[:, start_h:start_h + ch, start_w:start_w + cw] for m in masks]
                    trace["crop"] = [int(start_h), int(start_w), ch, cw]
                image = F.interpolate(image.permute(0, 3, 1, 2), size=(height, width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                masks = [resize_masks(m, (height, width)) for m in masks]
                angle = torch.rand(1, device=image.device) * 10 - 5
                trace["angle"] = float(angle)
                if torch.abs(angle) > .1:
                    angle_rad = angle * torch.pi / 180.
                    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
                    gx, gy = torch.linspace(-1, 1, width, device=image.device), torch.linspace(-1, 1, height, device=image.device)
                    gy, gx = torch.meshgrid(gy, gx, indexing="ij")
                    gx, gy = gx.unsqueeze(0).expand(len(image), -1, -1), gy.unsqueeze(0).expand(len(image), -1, -1)
                    grid = torch.stack([gx * cos_a - gy * sin_a, gx * sin_a + gy * cos_a], dim=-1)
                    image = F.grid_sample(image.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=False).permute(0, 2, 3, 1)
                    masks = [F.grid_sample(m[:, None].float(), grid[b:b+1].expand(len(m), -1, -1, -1), mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].bool() if len(m) else m for b, m in enumerate(masks)]
            brightness_factor = .7 + torch.rand(1, device=image.device) * .6
            image = image * brightness_factor
            contrast_factor = .6 + torch.rand(1, device=image.device) * .8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean
            saturation_factor = .5 + torch.rand(1, device=image.device) * 1.
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor
            image = image.clamp(0, 1) * 2. - 1.
        out_images[key] = image.permute(0, 3, 1, 2) if channels_first else image
        out_masks[key] = observation.image_masks.get(key, torch.ones(observation.state.shape[:-1], dtype=torch.bool, device=image.device))
        out_pixel_masks[key], traces[key] = masks, trace
    processed = SimpleNamespace(images=out_images, image_masks=out_masks,
                                state=observation.state,
                                tokenized_prompt=observation.tokenized_prompt,
                                tokenized_prompt_mask=observation.tokenized_prompt_mask)
    return processed, out_pixel_masks, traces
