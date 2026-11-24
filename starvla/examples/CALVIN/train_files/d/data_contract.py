"""The official CALVIN/Libero action and image packing contract."""
import numpy as np


def action_chunk(actions, step, horizon=8):
    actions = np.asarray(actions)
    assert actions.ndim == 2 and actions.shape[1] == 7 and len(actions)
    assert 0 <= step < len(actions) and horizon == 8
    # Official modality.json omits absolute, whose schema default is True.
    # retrieve_data_and_pad therefore repeats the entire final action row.
    indices = np.minimum(np.arange(step, step+horizon), len(actions)-1)
    return actions[indices].astype(np.float64)


def normalize_actions(chunk, stats):
    x = np.asarray(chunk, dtype=np.float64).copy()
    assert x.shape == (8, 7) and np.isfinite(x).all()
    # Official Normalizer constructs float32 statistic tensors then casts them
    # to the float64 padded data tensor. Preserve that rounding before packing.
    lo = np.asarray(stats['min'], dtype=np.float32).astype(np.float64)[:6]
    hi = np.asarray(stats['max'], dtype=np.float32).astype(np.float64)[:6]
    assert lo.shape == hi.shape == (6,) and np.isfinite(lo).all() and np.isfinite(hi).all()
    varying = hi != lo
    x[:, :6] = np.divide(x[:, :6]-lo, hi-lo, out=np.zeros_like(x[:, :6]), where=varying)*2-1
    x[:, np.where(~varying)[0]] = 0
    # No quantile normalization, clipping, or binary remapping of the gripper.
    return x.astype(np.float16)


def denormalize_actions(actions, stats):
    x = np.asarray(actions, dtype=np.float32).copy()
    assert x.shape[-1] == 7 and np.isfinite(x).all()
    lo, hi = np.asarray(stats['min'], np.float32), np.asarray(stats['max'], np.float32)
    x[..., :6] = (x[..., :6]+1)*.5*(hi[:6]-lo[:6])+lo[:6]
    return x
