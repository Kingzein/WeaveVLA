import numpy as np

def attach_masks(objects, masks, ids):
    masks=np.asarray(masks)
    keys=list(map(str,ids))
    if masks.dtype!=bool or masks.ndim!=3 or len(keys)!=len(masks) or len(set(keys))!=len(keys):
        raise ValueError('Invalid instance mask mapping')
    by_id=dict(zip(keys,masks))
    for obj in objects:
        key=str(obj['id'])
        if obj['visible_now'] and key not in by_id:raise ValueError('Visible instance missing current mask')
        obj['visual_mask']=by_id[key].copy() if key in by_id else np.zeros(masks.shape[1:],bool)
    return objects
