"""Validate committed, trainable caches before constructing a formal run."""
import argparse
import hashlib
import json
from pathlib import Path

from .cache import read_frame, atomic_json
from .geometry import compatible_recipe_digests, default_recipe


def validate_manifest(path, full=True, allow_provisional=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    recipe = default_recipe()
    expected = manifest.get('recipe_sha256')
    if not manifest.get('complete') or expected not in compatible_recipe_digests(recipe):
        raise ValueError('Incomplete manifest or different recipe')
    episodes = manifest['episodes']
    if len({r['key'] for r in episodes}) != len(episodes):
        raise ValueError('Duplicate episodes')
    total, suites, depth_signatures = 0, set(), set()
    for record in episodes:
        marker = (path.parent/record['completion']).resolve()
        complete = json.loads(marker.read_text())
        quality = complete.get('training_quality', 'verified_visible_scene' if complete['trainable'] else 'diagnostic')
        if not complete['complete'] or (not complete['trainable'] and not (allow_provisional and quality == 'provisional_legacy_masks')) or complete['recipe_sha256'] != expected:
            raise ValueError(f'Unverified episode: {record["key"]}')
        if complete['episode_id'] != record['key'] or complete['instruction'] != record['instruction']:
            raise ValueError('Episode/instruction mismatch')
        depth_signatures.add(json.dumps(complete['depth_provenance'],sort_keys=True))
        phases = {str(i): [] for i in range(recipe['geometry']['stride'])}
        for index, row in enumerate(complete['frames']):
            if row['step'] != index:
                raise ValueError('Dense source frame gap')
            frame = marker.parent/row['path']
            phases[str(index % 8)].append(str(frame.resolve()))
            if full or index in (0, len(complete['frames'])//2, len(complete['frames'])-1):
                if hashlib.sha256(frame.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError('Committed frame checksum mismatch')
                objects, meta, arrays = read_frame(frame, recipe, allow_provisional=allow_provisional)
                if meta['recipe_sha256'] != expected:
                    raise ValueError('Frame recipe does not match manifest')
                if meta.get('training_quality',quality) != quality: raise ValueError('Mixed training quality')
                if meta['episode_id'] != record['key'] or meta['control_step'] != index or 'action' not in arrays:
                    raise ValueError('Frame identity/action mismatch')
                import numpy as np
                if arrays['action'].shape!=(8,7) or not np.isfinite(arrays['action']).all():
                    raise ValueError('Expected finite normalized 8x7 action chunk')
                if meta['instruction'] != record['instruction'] or meta['phase'] != index % recipe['geometry']['stride']:
                    raise ValueError('Frame instruction/phase mismatch')
                ids = meta['mask_instance_ids']
                if len(ids) != len(set(ids)) or len(ids) != len(meta['mask_instance_labels']):
                    raise ValueError('Duplicate/misaligned mask instance IDs')
                if not set(meta['prompt_ids']).issubset(set(ids)):
                    raise ValueError('Prompt binding references a different instance registry')
                if arrays['depth_raw_da3_fp32'].dtype != np.float32 or not np.isfinite(arrays['depth_raw_da3_fp32']).all():
                    raise ValueError('Invalid DA3 depth storage')
                for obj in objects:
                    points = obj['points']
                    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                        raise ValueError('Invalid model-input point cloud')
                    if obj['geometry_age'] != index - obj['last_geometry_update_step'] or obj['geometry_age'] < 0:
                        raise ValueError('Noncausal geometry timestamp')
        if any(not frames for frames in phases.values()):
            raise ValueError('Every sampling offset needs real observations')
        if {k: [str((path.parent/f).resolve()) for f in v] for k,v in record['frames_by_offset'].items()} != phases:
            raise ValueError('Sampling phase manifest mismatch')
        total += len(complete['frames'])
        suites.add(record['dataset'])
    report = dict(passed=True, full=full, recipe_sha256=expected, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  episodes=len(episodes), control_frames=total, suites=sorted(suites), allow_provisional=allow_provisional)
    if len(depth_signatures)!=1: raise ValueError('Mixed DA3 depth model/preprocessing versions')
    if manifest.get('scope') == 'full_libero' and (len(episodes) != 1693 or total != 273465 or len(suites) != 4):
        raise ValueError('Full LIBERO data coverage differs from agreed dataset')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-provisional', action='store_true')
    args = parser.parse_args()
    report = validate_manifest(args.manifest, allow_provisional=args.allow_provisional)
    atomic_json(args.output, report)
    print(json.dumps(report))
