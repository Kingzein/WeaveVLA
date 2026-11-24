"""Rebuild DA3 geometry from exported RGB/actions and audited segmentation.

Formal mode requires visible-scene mask provenance. --diagnostic-legacy-masks
is explicitly non-trainable and exists for implementation/performance checks.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import EpisodeWriter, atomic_json
from .depth import DA3Metric, file_sha256
from .geometry import default_recipe, recipe_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--legacy-export-root', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--raw-depth-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--mask-root')
    parser.add_argument('--tcp-root', required=True)
    parser.add_argument('--diagnostic-legacy-masks', action='store_true')
    parser.add_argument('--provisional-legacy-masks', action='store_true')
    parser.add_argument('--one-per-suite', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--parts', type=int, default=1)
    args = parser.parse_args()
    if args.diagnostic_legacy_masks and args.provisional_legacy_masks: raise ValueError('Choose one quality mode')
    if args.max_frames and not args.diagnostic_legacy_masks:
        raise ValueError('Truncated episodes are diagnostic only')
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    import fcntl
    part_lock=(root/f'part_{args.part}.lock').open('a')
    fcntl.flock(part_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recipe = default_recipe()
    source = json.loads(Path(args.source_manifest).read_text())
    records = source['episodes']
    if args.one_per_suite:
        records = [next(r for r in records if r['dataset']==name) for name in sorted({r['dataset'] for r in records})]
    records = records[args.part::args.parts]
    bindings = json.loads(Path(args.bindings).read_text())['episodes']
    bindings_sha256 = file_sha256(args.bindings)
    # The existing bindings must retain the original task instruction. They use
    # the selected structured parser/MiniLM method; source file hash is recorded.
    predictor = DA3Metric(args.weights) if not args.raw_depth_root else None
    raw_root = Path(args.raw_depth_root).resolve() if args.raw_depth_root else None
    if raw_root:
        depth_complete = json.loads((raw_root/'complete.json').read_text())
        if not depth_complete['complete']: raise ValueError('Raw DA3 stage not complete')
    depth_provenance = None
    rows = []; started = time.monotonic(); depth_seconds = []
    for record in records:
        dense = []
        for offset, frames in record['frames_by_offset'].items():
            dense += [(int(offset)+8*i, Path(f)) for i, f in enumerate(frames)]
        dense.sort()
        if args.max_frames: dense = dense[:args.max_frames]
        if [s for s, _ in dense] != list(range(len(dense))): raise ValueError('Source phase frames not dense')
        destination = root/'episodes'/record['key']
        marker = destination/'complete.json'
        raw_rows = None
        if raw_root:
            raw_marker = json.loads((raw_root/'episodes'/record['key']/'complete.json').read_text())
            if raw_marker['recipe_sha256'] != depth_complete['recipe_sha256'] or raw_marker['episode_id'] != record['key']:
                raise ValueError('Raw depth episode recipe mismatch')
            raw_rows = raw_marker['frames']
            if len(raw_rows) != len(dense): raise ValueError('Raw depth count mismatch')
        if marker.exists():
            prior = json.loads(marker.read_text())
            if prior['recipe_sha256'] != recipe_digest(recipe) or len(prior['frames']) != len(dense):
                raise ValueError('Existing cache has a different recipe/frame count')
            expected_quality='provisional_legacy_masks' if args.provisional_legacy_masks else ('diagnostic' if args.diagnostic_legacy_masks else 'verified_visible_scene')
            if prior.get('training_quality') != expected_quality: raise ValueError('Cache quality mode changed')
            depth_provenance=prior['depth_provenance']
        else:
            writer = EpisodeWriter(destination, record['key'], record['instruction'], recipe,
                                   args.diagnostic_legacy_masks, args.provisional_legacy_masks)
            binding = bindings[record['key']]
            if binding['parsed']['instruction'] != record['instruction']: raise ValueError('Binding instruction changed')
            for step, exported in dense:
                if not exported.is_absolute(): exported = Path(args.legacy_export_root)/exported
                with np.load(exported, allow_pickle=False) as z:
                    meta = json.loads(str(z['metadata']))
                    if meta['step'] != step or meta['lang'] != record['instruction'] or meta['episode_key'] != record['key']:
                        raise ValueError('Source export identity mismatch')
                    rgb = np.array(Image.open(exported.parent/meta['images'][0]).convert('RGB'))
                    wrist = np.array(Image.open(exported.parent/meta['images'][1]).convert('RGB'))
                    action = z['action'].copy(); state = z['state'].copy() if 'state' in z else None
                    stored_K = z['K'].copy() if 'K' in z else None
                source_cache = Path(args.mask_root)/record['key']/f'{step:06d}.npz' if args.mask_root else Path(meta['source_cache'])
                with np.load(source_cache, allow_pickle=False) as z:
                    sm = json.loads(str(z['metadata']))
                    if sm['step'] != step or sm['dataset'] != record['dataset'] or sm['episode'] != record['episode']:
                        raise ValueError('Source segmentation identity mismatch')
                    # Match the original hash implementation, without importing
                    # simulator/perception models into the depth worker.
                    from starVLA.model.modules.graph.cache import digest
                    if digest(Image.fromarray(rgb), record['instruction']) != sm['rgb_instruction_sha256']:
                        raise ValueError('Source RGB/mask digest mismatch')
                    masks = z['masks'].copy()
                    robot_mask = z['robot_mask'].copy()
                    gripper_mask = z['gripper_mask'].copy()
                    instances = [dict(id=str(i), label=str(n)) for i,n in zip(z['object_ids'],z['labels'])]
                K = np.array([[309.01933598375615,0,127],[0,309.01933598375615,127],[0,0,1]],np.float32)
                K[0] *= rgb.shape[1]/256; K[1] *= rgb.shape[0]/256
                if stored_K is not None: K = np.asarray(stored_K, np.float32)
                if raw_root:
                    from .cache import rgb_instruction_digest
                    row = raw_rows[step]
                    raw_path = raw_root/'episodes'/record['key']/row['file']
                    if row['step'] != step or file_sha256(raw_path) != row['sha256']: raise ValueError('Raw depth checksum mismatch')
                    with np.load(raw_path, allow_pickle=False) as raw:
                        rm = json.loads(str(raw['metadata']))
                        if (rm['episode_id'],rm['control_step'],rm['instruction']) != (record['key'],step,record['instruction']):
                            raise ValueError('Raw depth identity mismatch')
                        if rm['rgb_instruction_sha256'] != rgb_instruction_digest(rgb,record['instruction']): raise ValueError('Raw depth RGB mismatch')
                        np.testing.assert_array_equal(raw['K'],K)
                        depth=raw['depth_raw_da3_fp32'].copy();dm=rm['depth']
                else:
                    import torch
                    torch.cuda.synchronize(); tic = time.monotonic()
                    depth, dm = predictor(rgb, K)
                    torch.cuda.synchronize(); depth_seconds.append(time.monotonic()-tic)
                depth_provenance = {k:dm[k] for k in ('weights_sha256','config_sha256','input_processor_sha256','metric_conversion_sha256')}
                provenance = dict(sm.get('mask_provenance', {}))
                provenance.update(dict(robot_mask_version=sm.get('robot_mask',{}).get('version'),
                    verified_rgb_instances=sm.get('mask_provenance',{}).get('verified_rgb_instances') is True,
                    source=str(source_cache), source_recipe=sm['recipe'],
                    source_mask_status=sm.get('robot_mask_status'), source_external_occlusion=sm.get('geometry_external_occlusion'),
                    detector_sha256=sm.get('detector_sha256'), bindings_sha256=bindings_sha256))
                tcp = json.loads((Path(args.tcp_root)/record['key']/f'{step:06d}.json').read_text())
                if tcp['control_step'] != step or tcp['episode_id'] != record['key'] or tcp['rgb_file_sha256'] != file_sha256(exported.parent/meta['images'][0]):
                    raise ValueError('TCP frame identity mismatch')
                writer.append(step,rgb,wrist,depth,masks,instances,K,dm,provenance,binding['prompt_ids'],action,state,robot_mask=robot_mask,gripper_mask=gripper_mask,tcp=tcp)
            writer.finish(len(dense))
        phases = {str(o):[str((destination/f'{step:06d}.npz').resolve()) for step,_ in dense if step%8==o] for o in range(8)}
        rows.append(dict(key=record['key'],dataset=record['dataset'],episode=record['episode'],instruction=record['instruction'],
                         split=record.get('split','train'),frames=phases['0'],frames_by_offset=phases,
                         completion=str(marker.resolve())))
        atomic_json(root/f'progress_{args.part}.json',dict(episodes=len(rows),target=len(records),last=record['key'],seconds=time.monotonic()-started))
        print('CACHED',len(rows),len(records),record['key'],len(dense),flush=True)
    manifest = dict(complete=True,scope='diagnostic' if args.diagnostic_legacy_masks or args.one_per_suite else ('full_libero_part' if args.parts>1 else 'full_libero'),
                    trainable=not (args.diagnostic_legacy_masks or args.provisional_legacy_masks),
                    training_quality='provisional_legacy_masks' if args.provisional_legacy_masks else ('diagnostic' if args.diagnostic_legacy_masks else 'verified_visible_scene'),
                    recipe_sha256=recipe_digest(recipe),episodes=rows,
                    control_frames=sum(sum(len(fs) for fs in r['frames_by_offset'].values()) for r in rows),
                    depth_provenance=depth_provenance,elapsed_seconds=time.monotonic()-started,
                    measured_depth_seconds_mean=float(np.mean(depth_seconds[2:])) if len(depth_seconds)>2 else None)
    atomic_json(root/f'manifest_{args.part}.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('episodes','depth_provenance')}),flush=True)


if __name__ == '__main__': main()
