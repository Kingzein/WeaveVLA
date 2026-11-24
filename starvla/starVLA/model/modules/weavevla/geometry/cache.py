"""Atomic episode cache; separate causal processor for each sampling phase.

Raw inputs never contain simulator object poses/depth. Legacy masks can only
produce explicitly diagnostic caches, which the training loader rejects.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .geometry import TemporalGeometry, compatible_recipe_digests, default_recipe, recipe_digest

from .contracts import ROBOT_MASK_VERSION, validate_masks


def rgb_instruction_digest(rgb, instruction):
    # Same source contract as the existing starVLA cache (verified in adapter).
    h = hashlib.sha256()
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    h.update(instruction.encode())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def save_frame(path, objects, depth, masks, rgb, wrist, K, metadata, action=None, state=None, robot_mask=None, gripper_mask=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = dict(depth_raw_da3_fp32=np.asarray(depth, np.float32), masks=np.asarray(masks, bool),
                  rgb=np.asarray(rgb, np.uint8), wrist=np.asarray(wrist, np.uint8), K=np.asarray(K, np.float32))
    meta = dict(metadata, objects=[{k:v for k,v in o.items() if k != 'points'} for o in objects])
    for i, obj in enumerate(objects): fields[f'points_{i}'] = np.asarray(obj['points'], np.float32)
    if action is not None: fields['action'] = np.asarray(action)
    if state is not None: fields['state'] = np.asarray(state)
    if robot_mask is not None: fields['robot_mask'] = np.asarray(robot_mask)
    if gripper_mask is not None: fields['gripper_mask'] = np.asarray(gripper_mask)
    fields['metadata'] = json.dumps(meta, ensure_ascii=False, allow_nan=False)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **fields)
    # Read the committed representation before publishing it.
    with np.load(tmp, allow_pickle=False) as z:
        assert z['depth_raw_da3_fp32'].dtype == np.float32
        assert len(json.loads(str(z['metadata']))['objects']) == len(objects)
    tmp.replace(path)


def read_frame(path, recipe=None, allow_diagnostic=False, allow_provisional=False):
    recipe = recipe or default_recipe()
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        if meta['recipe_sha256'] not in compatible_recipe_digests(recipe):
            raise ValueError('Cache recipe mismatch')
        provisional = allow_provisional and meta.get('training_quality') == 'provisional_legacy_masks'
        if not allow_diagnostic and not meta['trainable'] and not provisional:
            raise ValueError('Diagnostic/legacy-mask cache cannot enter training')
        rgb = z['rgb'].copy()
        if meta['rgb_instruction_sha256'] != rgb_instruction_digest(rgb, meta['instruction']):
            raise ValueError('RGB/instruction identity mismatch')
        objects = meta['objects']
        for i, obj in enumerate(objects): obj['points'] = z[f'points_{i}'].copy()
        arrays = {k:z[k].copy() for k in ['rgb','wrist','depth_raw_da3_fp32','masks','K','action','state','robot_mask','gripper_mask'] if k in z}
    from .tcp import position
    tcp = position(meta['geometry']['tcp'])
    grippers = [o for o in objects if o['label']=='robot gripper']
    if len(grippers)!=1:raise ValueError('Expected exactly one TCP gripper')
    np.testing.assert_allclose(grippers[0]['geometry_center'],tcp,rtol=0,atol=1e-7)
    if grippers[0].get('position_source')!='tcp':raise ValueError('Wrong gripper coordinate source')
    if meta['trainable']:
        instances=[dict(id=i,label=n) for i,n in zip(meta['mask_instance_ids'],meta['mask_instance_labels'])]
        validate_masks(rgb,arrays['masks'],instances,meta['mask_provenance'],arrays.get('robot_mask'),arrays.get('gripper_mask'),meta['control_step'],meta['rgb_instruction_sha256'])
    return objects, meta, arrays


class EpisodeWriter:
    def __init__(self, root, episode_id, instruction, recipe=None, diagnostic=False, provisional=False):
        import fcntl
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/'writer.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.recipe = recipe or default_recipe()
        self.episode, self.instruction, self.diagnostic = str(episode_id), instruction, diagnostic
        if diagnostic and provisional: raise ValueError('Diagnostic and provisional modes are distinct')
        self.provisional = provisional
        self.phases, self.frames, self.last_step = {}, [], -1
        self.depth_provenance = None
        if (self.root/'complete.json').exists():
            raise FileExistsError('Episode already committed; validate and reuse, do not overwrite')
        # Restarting an incomplete episode must replay from its beginning; files
        # are atomically regenerated only under the exclusive episode lock.

    def append(self, step, rgb, wrist, depth_raw, masks, instances, K, depth_meta,
               mask_provenance, prompt_ids, action=None, state=None, fixed_camera=True,
               camera_key='main', robot_mask=None, gripper_mask=None, tcp=None):
        if step != self.last_step+1:
            raise ValueError('Dense causal source frames must start at 0 without gaps')
        if depth_meta.get('model') != 'depth-anything/DA3METRIC-LARGE' or depth_meta.get('network_dtype') != 'bfloat16':
            raise ValueError('Not the selected DA3 BF16 output')
        depth_signature = {k:depth_meta[k] for k in ['weights_sha256','config_sha256','input_processor_sha256','metric_conversion_sha256']}
        if self.depth_provenance is not None and self.depth_provenance != depth_signature:
            raise ValueError('Depth implementation changed within episode')
        self.depth_provenance = depth_signature
        verified_mask = False
        if not self.diagnostic and not self.provisional:
            validate_masks(rgb,masks,instances,mask_provenance,robot_mask,gripper_mask,step,rgb_instruction_digest(rgb,self.instruction))
            verified_mask = True
        if not verified_mask and not self.diagnostic and not self.provisional:
            raise ValueError('Rebuild/verify visible-scene robot and segmentation masks before training cache')
        stride = self.recipe['geometry']['stride']
        phase = step % stride
        processor = self.phases.setdefault(phase, TemporalGeometry(self.recipe['geometry']))
        objects, trace = processor.update(self.episode, step, rgb, depth_raw, masks, instances, K, fixed_camera, camera_key, tcp=tcp)
        path = self.root/f'{step:06d}.npz'
        metadata = dict(episode_id=self.episode, instruction=self.instruction, control_step=step, phase=phase,
                        recipe_sha256=recipe_digest(self.recipe), trainable=bool(verified_mask and not self.diagnostic),
                        training_quality='provisional_legacy_masks' if self.provisional else ('diagnostic' if self.diagnostic else 'verified_visible_scene'),
                        rgb_instruction_sha256=rgb_instruction_digest(rgb, self.instruction),
                        mask_instance_ids=[str(o['id']) for o in instances], mask_instance_labels=[o['label'] for o in instances],
                        prompt_ids=list(map(str,prompt_ids)),
                        mask_provenance=mask_provenance, depth_provenance=depth_meta, geometry=trace)
        save_frame(path, objects, depth_raw, masks, rgb, wrist, K, metadata, action, state, robot_mask, gripper_mask)
        self.frames.append(dict(step=step, path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), trainable=metadata['trainable']))
        self.last_step = step
        return objects, trace

    def finish(self, expected_frames):
        if len(self.frames) != expected_frames:
            raise ValueError('Incomplete episode')
        # No learned 256D features or GRU hidden states are stored in this cache.
        row = dict(episode_id=self.episode, instruction=self.instruction, recipe=self.recipe,
                   recipe_sha256=recipe_digest(self.recipe), frames=self.frames, depth_provenance=self.depth_provenance,
                   trainable=all(f['trainable'] for f in self.frames), complete=True,
                   training_quality='provisional_legacy_masks' if self.provisional else ('diagnostic' if self.diagnostic else 'verified_visible_scene'),
                   phases=sorted(self.phases), phase_policy='independent_causal_state_per_offset')
        atomic_json(self.root/'complete.json', row)
        self.lock.close()
        return row
