"""current-instance mask area weights over post-merger visual cells.

No learned parameters, centroid fallback, or additional visual tokens. Masks
must use the same uncropped image coordinates as the corresponding token grid.
"""
import torch

MODE = 'mask_coverage_area_weighted_mean'


def mask_cell_weights(mask, grid_h, grid_w):
    """Return normalized mask area per cell, in FP32 on the mask device.

    For divisible image/grid sizes this is simply a count of foreground pixels.
    Otherwise split a pixel's area across the cells its footprint intersects;
    do not count an entire boundary pixel multiple times. An empty mask returns
    zeros; the caller decides whether the object's state permits an empty mask.
    """
    mask = torch.as_tensor(mask)
    if mask.ndim != 2 or mask.dtype != torch.bool or min(mask.shape) < 1:
        raise ValueError('Expected aligned 2D boolean current mask')
    if not isinstance(grid_h, int) or not isinstance(grid_w, int) or min(grid_h, grid_w) < 1:
        raise ValueError('Expected positive integer cell grid')
    mh, mw = mask.shape
    if mh % grid_h == 0 and mw % grid_w == 0:
        areas = mask.reshape(grid_h, mh // grid_h, grid_w, mw // grid_w).sum((1, 3), dtype=torch.float32)
    else:
        # Integer coordinates avoid ambiguous rounding at shared cell borders.
        # Overlap lengths are divided by grid size to obtain input-pixel units.
        y = torch.arange(mh, device=mask.device)[None, :]
        row = torch.arange(grid_h, device=mask.device)[:, None]
        oy = (torch.minimum((y + 1) * grid_h, (row + 1) * mh)
              - torch.maximum(y * grid_h, row * mh)).clamp_min(0).float() / grid_h
        x = torch.arange(mw, device=mask.device)[None, :]
        col = torch.arange(grid_w, device=mask.device)[:, None]
        ox = (torch.minimum((x + 1) * grid_w, (col + 1) * mw)
              - torch.maximum(x * grid_w, col * mw)).clamp_min(0).float() / grid_w
        # Mask metadata has no gradient. Explicitly disable outer BF16 autocast
        # for the area calculation, including non-divisible grids.
        with torch.autocast(mask.device.type, enabled=False):
            areas = oy @ mask.float() @ ox.T
    total = areas.sum()
    if not bool(total > 0):
        return areas.reshape(-1)
    return (areas / total).reshape(-1)


def gather_object_patches(tokens, grid_thw, objects_batch, image_counts, merge_size):
    if tokens is None or tokens.ndim != 2 or not tokens.is_floating_point() or not isinstance(merge_size, int) or merge_size < 1:
        raise ValueError('Invalid visual tokens or merge size')
    if any(not isinstance(c, int) or c < 0 for c in image_counts):
        raise ValueError('Invalid image counts')
    if grid_thw is None or len(grid_thw) != sum(image_counts) or len(objects_batch) != len(image_counts):
        raise ValueError('Image layout mismatch')
    maps, offset = [], 0
    for t, h, w in grid_thw.detach().cpu().tolist():
        if any(not isinstance(v, int) for v in (t, h, w)) or t != 1 or min(h, w) < 1 or h % merge_size or w % merge_size:
            raise ValueError('Expected divisible still-image grids')
        h, w = h // merge_size, w // merge_size
        maps.append((tokens[offset:offset + h * w], h, w))
        offset += h * w
    if offset != len(tokens):
        raise ValueError('Packed token count mismatch')
    result, image_offset = [], 0
    for objects, count in zip(objects_batch, image_counts):
        features = []
        for obj in objects:
            view = int(obj['image_index'])
            if not 0 <= view < count:
                raise ValueError('Invalid image index')
            cells, gh, gw = maps[image_offset + view]
            mask = torch.as_tensor(obj['visual_mask'])
            if mask.ndim != 2 or mask.dtype != torch.bool or min(mask.shape) < 1:
                raise ValueError('Expected aligned 2D boolean current mask')
            # Same V6 semantics: never pool background for held/occluded nodes.
            # The graph's existing history handling remains responsible for them.
            if not bool(obj['visible_now']) or not bool(mask.any()):
                if obj['visible_now'] and obj['geometry_source'] != 'tcp_only':
                    raise ValueError('Visible object has empty visual mask')
                features.append(tokens.new_zeros(tokens.shape[-1]))
                continue
            weights = mask_cell_weights(mask, gh, gw)
            selected = (weights > 0).nonzero(as_tuple=True)[0]
            if not len(selected):
                raise ValueError('Nonempty mask selected no patches')
            # Normally masks are CPU metadata. Select there before the transfer
            # so CUDA nonzero does not introduce a per-node synchronization.
            active_weights = weights.index_select(0, selected).to(device=tokens.device)
            selected = selected.to(device=tokens.device)
            # FP32 accumulation; retain gradients to every contributing token.
            value = (cells.index_select(0, selected).float()
                     * active_weights.unsqueeze(-1)).sum(0)
            features.append(value.to(tokens.dtype))
        result.append(torch.stack(features) if features else tokens.new_empty((0, tokens.shape[-1])))
        image_offset += count
    return result
