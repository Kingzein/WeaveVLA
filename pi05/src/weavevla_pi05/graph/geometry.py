"""Causal, GT-free geometry. Shared by offline cache and live inference.

All ages are control steps. Inputs are RGB uint8, axial metric depth and K on
the same grid. Instance IDs must be stable; changing labels is a hard error.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def default_recipe():
    return json.loads(Path(__file__).with_name('recipe.json').read_text())


def recipe_digest(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fps(points, count=128):
    points = np.asarray(points, np.float32)
    if len(points) <= count:
        return points.copy()
    indices, distance = [], np.full(len(points), np.inf)
    next_index = np.square(points-points.mean(0)).sum(-1).argmax()
    for _ in range(count):
        indices.append(next_index)
        distance = np.minimum(distance, np.square(points-points[next_index]).sum(-1))
        next_index = distance.argmax()
    return points[indices].copy()


def unproject(depth, mask, K):
    valid = mask & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    rays = np.stack([x, y, np.ones(len(x))], -1) @ np.linalg.inv(K).T
    return (rays * depth[y, x, None]).astype(np.float32)


def erode(mask, pixels):
    out = mask.copy()
    for _ in range(pixels):
        p = np.pad(out, 1)
        h, w = out.shape
        out = np.logical_and.reduce([p[i:i+h, j:j+w] for i in range(3) for j in range(3)])
    return out


class TemporalGeometry:
    def __init__(self, config=None):
        self.cfg = dict(default_recipe()['geometry'], **(config or {}))
        self.reset()

    def reset(self):
        self.episode = None
        self.step = None
        self.camera_key = None
        self.registry = {}
        self.last_scale = 1.
        self.last_scale_step = None
        self.generation = 0
        self.trace = {}

    def state_dict(self):
        return copy.deepcopy({k: v for k, v in vars(self).items() if k != 'trace'})

    def load_state_dict(self, state):
        if state['cfg'] != self.cfg:
            raise ValueError('Temporal geometry recipe changed during resume')
        self.__dict__.update(copy.deepcopy(state))

    def _stable(self, mask, rgb, ref):
        if ref is None or ref['mask'].shape != mask.shape:
            return False, np.zeros_like(mask)
        old = ref['mask']
        h, w = mask.shape
        core = erode(mask & old, max(1, round(self.cfg['erosion_px_224'] * min(h, w)/224)))
        minimum = max(1, round(self.cfg['core_pixels_224'] * h*w/(224*224)))
        if core.sum() < minimum or not mask.any() or not old.any():
            return False, core
        # Normalize centroid differences independently along both image axes.
        shift = (np.argwhere(mask).mean(0)-np.argwhere(old).mean(0)) * [224/h, 224/w]
        ok = ((mask & old).sum()/max(1, (mask | old).sum()) >= self.cfg['mask_iou']
              and np.linalg.norm(shift) <= self.cfg['centroid_px_224']
              and abs(mask.sum()/old.sum()-1) <= self.cfg['area_change']
              and np.abs(rgb[core].astype(np.float32)-ref['rgb'][core].astype(np.float32)).mean() <= self.cfg['rgb_mad'])
        return bool(ok), core

    def update(self, episode, step, rgb, depth_raw, masks, instances, K,
               fixed_camera=True, camera_key='main', tcp=None):
        from .tcp import position
        tcp_position = position(tcp)
        rgb = np.asarray(rgb)
        depth_raw = np.asarray(depth_raw, np.float32)
        K = np.asarray(K, np.float32)
        masks = np.asarray(masks, bool)
        h, w = depth_raw.shape
        if rgb.shape != (h, w, 3) or masks.shape != (len(instances), h, w):
            raise ValueError('RGB/depth/mask grid mismatch')
        if rgb.dtype != np.uint8 or K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError('Expected uint8 RGB and valid grid-aligned K')
        ids = [str(o['id']) for o in instances]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate stable instance IDs')
        if sum(o['label'] == 'robot gripper' for o in instances) > 1:
            raise ValueError('At most one gripper')
        if self.episode != str(episode):
            self.reset()
            self.episode = str(episode)
        if self.step is not None and step-self.step != self.cfg['stride']:
            raise ValueError('Geometry must advance one configured graph update; replay missing observations')
        signature = (str(camera_key), h, w, K.tobytes())
        reset_reason = None
        if self.camera_key is not None and signature != self.camera_key:
            reset_reason = 'camera_or_calibration_changed'
        if not fixed_camera:
            reset_reason = 'moving_camera_hold_disabled'
        if reset_reason:
            self.registry.clear()
            self.last_scale, self.last_scale_step = 1., None
            self.generation += 1
        self.camera_key = signature
        self.step = int(step)
        observations, logs, anchor_ids = {}, [], []
        for oid, obj, mask in zip(ids, instances, masks):
            if obj['label'] == 'robot':
                continue
            prev = self.registry.get(oid)
            if prev and prev['label'] != obj['label']:
                raise ValueError('Identity label changed: explicit episode/ID reset required')
            valid = mask & np.isfinite(depth_raw) & (depth_raw > 0)
            seen = int(valid.sum()) >= self.cfg['min_points']
            stable_prev, _ = self._stable(mask, rgb, prev.get('previous') if prev else None)
            stable_ref, core = self._stable(mask, rgb, prev.get('anchor') if prev else None)
            stable = bool(fixed_camera and seen and obj['label'] != 'robot gripper' and stable_prev and stable_ref)
            count = prev['stable_count']+1 if stable else 0
            observations[oid] = dict(obj=obj, mask=mask, seen=seen, stable=stable, count=count)
            if stable and count >= self.cfg['confirmations']:
                reference = prev['anchor']['depth_corrected']
                core &= np.isfinite(reference) & (reference > 0) & valid
                minimum = max(1, round(self.cfg['core_pixels_224']*h*w/(224*224)))
                if core.sum() >= minimum:
                    logs.append(float(np.median(np.log(reference[core])-np.log(depth_raw[core]))))
                    anchor_ids.append(oid)
        reliable = len(logs) >= self.cfg['min_anchors'] and fixed_camera
        if reliable:
            scale = float(np.clip(np.exp(np.median(logs)), self.cfg['scale_min'], self.cfg['scale_max']))
            self.last_scale, self.last_scale_step = scale, step
            scale_reason = 'anchors'
        elif (self.cfg['scale_fallback'] == 'retain_short_then_reset' and self.last_scale_step is not None
              and step-self.last_scale_step <= self.cfg['scale_ttl_steps']):
            scale, scale_reason = self.last_scale, 'short_retention_no_holding'
        else:
            scale, scale_reason = 1., 'unit_fallback_no_holding'
            if self.last_scale_step is not None:
                # A new coordinate baseline cannot share old references/edge memory.
                self.last_scale_step, self.last_scale = None, 1.
                self.generation += 1
                for r in self.registry.values():
                    r['anchor'] = r['previous'] = None
                    r['stable_count'] = 0
                for obs in observations.values():
                    obs['stable'], obs['count'] = False, 0
                reset_reason = 'scale_reference_expired'
        corrected = depth_raw * np.float32(scale)
        current = {oid: unproject(corrected, obs['mask'], K) for oid, obs in observations.items() if obs['seen']}
        centers = {oid: p.mean(0) for oid, p in current.items()}
        radii = {oid: float(np.linalg.norm(p-centers[oid], axis=1).max()) for oid,p in current.items()}
        gripper = next((oid for oid, o in observations.items() if o['obj']['label'] == 'robot gripper'), None)
        if gripper is not None:
            centers[gripper] = tcp_position
            # Radius uses the physical TCP anchor but does not rescale TCP.
            radii[gripper] = float(np.linalg.norm(current[gripper]-tcp_position, axis=1).max()) if gripper in current else 0.
        result = []
        # Include known IDs missing from the detector/tracker output as occluded.
        for oid in sorted(set(observations) | set(self.registry)):
            obs, prev = observations.get(oid), self.registry.get(oid)
            label = obs['obj']['label'] if obs else prev['label']
            seen = bool(obs and obs['seen'])
            gap, hold = None, False
            if seen:
                if oid != gripper and gripper in centers:
                    gap = float(np.linalg.norm(centers[oid]-centers[gripper]))
                hold = bool(prev and reliable and obs['stable'] and obs['count'] >= self.cfg['confirmations']
                            and gripper in current and gap is not None and gap > self.cfg['near_gripper_m'] + radii[oid] + radii[gripper]
                            and prev['generation'] == self.generation)
                points = prev['points'].copy() if hold else current[oid]
                source = 'static_supported' if hold else 'fresh'
                measured = prev['measured'] if hold else step
                y, x = np.nonzero(obs['mask'])
                uv = [(float(x.mean())+.5)/w, (float(y.mean())+.5)/h]
                frame = dict(mask=obs['mask'].copy(), rgb=rgb.copy(), depth_corrected=corrected.copy())
                anchor = prev['anchor'] if prev and obs['stable'] and prev['anchor'] is not None else frame
                entry = dict(label=label, points=points.copy(), measured=int(measured), seen=int(step),
                             evidence_step=int(step), previous=frame, anchor=anchor,
                             stable_count=obs['count'], uv=uv, generation=self.generation)
                self.registry[oid] = entry
            elif label == 'robot gripper':
                points = np.empty((0,3),np.float32)
                source = 'tcp_only'
                measured, uv = step, (prev['uv'] if prev else [.5,.5])
                entry = dict(label=label,points=points,measured=step,seen=step,evidence_step=step,previous=None,anchor=None,stable_count=0,uv=uv,generation=self.generation)
                self.registry[oid] = entry
            elif prev:
                prev['previous'] = None
                prev['stable_count'] = 0
                # Never reuse gripper coordinates as if its position were current.
                usable = (label != 'robot gripper' and fixed_camera and prev['generation'] == self.generation
                          and step-prev['seen'] <= self.cfg['occlusion_ttl_steps'])
                points = prev['points'].copy() if usable else np.empty((0, 3), np.float32)
                source = 'occluded_hold' if usable else 'missing'
                measured, uv, entry = prev['measured'], prev['uv'], prev
            else:
                continue  # never-seen objects have no placeholder
            center = tcp_position if label == 'robot gripper' else (points.mean(0) if len(points) else None)
            mask = obs['mask'] if obs else np.zeros((h, w), bool)
            result.append(dict(id=oid, label=label, points=points, geometry_center=None if center is None else center.tolist(), position_source='tcp' if label=='robot gripper' else 'pointcloud_centroid', geometry_source=source,
                               visible_now=seen, geometry_age=int(step-measured),
                               last_geometry_update_step=int(measured), last_seen_step=entry['seen'],
                               static_evidence_step=entry['evidence_step'], stable_count=entry['stable_count'],
                               geometry_generation=self.generation, image_index=0,
                               pixel_uv_normalized=uv, mask_area_fraction=float(mask.mean()),
                               valid_depth_ratio=float((mask & np.isfinite(depth_raw) & (depth_raw > 0)).sum()/max(1, mask.sum())),
                               gap_to_gripper_m=gap, hold_decision=hold))
        self.trace = dict(episode_id=self.episode, control_step=step, scale_factor=scale,
                          scale_reason=scale_reason, scale_reliable=reliable, anchor_ids=anchor_ids,
                          generation=self.generation, reset_reason=reset_reason, tcp=tcp,
                          nodes=[{k: v for k, v in o.items() if k != 'points'} for o in result])
        return result, copy.deepcopy(self.trace)
