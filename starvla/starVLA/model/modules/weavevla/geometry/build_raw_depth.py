"""Independent DA3 raw depth stage; no masks, GT depth or temporal decisions.

Episodes are disjoint across workers. Each FP32 frame is committed atomically;
an interrupted episode is checked and resumed without altering prior outputs.
"""
import argparse
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import atomic_json, rgb_instruction_digest
from .depth import DA3Metric, file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-manifest', required=True)
    ap.add_argument('--export-root', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--diagnostic-subset', action='store_true')
    args = ap.parse_args()
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root/f'worker_{args.worker}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads(Path(args.source_manifest).read_text())
    assert source['complete']
    if not args.diagnostic_subset:
        assert len(source['episodes']) == 1693 and source['control_frames'] == 273465
    assert 0 <= args.worker < args.workers
    # Greedy, frame-balanced static assignment, saved for reproducible resumption.
    bins = [[] for _ in range(args.workers)]; sizes = [0]*args.workers
    for rec in sorted(source['episodes'], key=lambda r: (-r['control_frames'], r['key'])):
        i = min(range(args.workers), key=lambda j: (sizes[j], j))
        bins[i].append(rec); sizes[i] += rec['control_frames']
    records = bins[args.worker]
    assignment = dict(source_sha256=file_sha256(args.source_manifest), workers=args.workers,
                      worker=args.worker, episodes=[r['key'] for r in records], frames=sizes[args.worker])
    plan = root/f'plan_{args.worker}.json'
    if plan.exists(): assert json.loads(plan.read_text()) == assignment, 'Assignment/source changed'
    else: atomic_json(plan, assignment)
    atomic_json(root/f'status_{args.worker}.json', dict(state='loading_da3', pid=os.getpid(), **assignment))
    model = DA3Metric(args.weights)
    recipe = dict(depth=model.provenance, camera='agentview',
                  source_sha256=assignment['source_sha256'], stage='raw_depth_only', diagnostic_subset=args.diagnostic_subset)
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    atomic_json(root/f'recipe_{args.worker}.json', dict(sha256=recipe_hash, **recipe))
    started = time.monotonic(); frames = 0; episodes = 0; inferred = 0; depth_seconds = 0.
    pool = ThreadPoolExecutor(max_workers=2)

    def load(rec, step, path):
        path = Path(path)
        if not path.is_absolute(): path = Path(args.export_root)/path
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            stored_K = z['K'].copy() if 'K' in z else None
        assert (meta['episode_key'], meta['lang'], meta['step']) == (rec['key'], rec['instruction'], step)
        rgb_path = path.parent/meta['images'][0]
        with Image.open(rgb_path) as im: rgb = np.array(im.convert('RGB'))
        # Standard fixed LIBERO agentview calibration, scaled to the exported RGB.
        K = np.array([[309.01933598375615, 0, 127], [0, 309.01933598375615, 127], [0, 0, 1]], np.float32)
        K[0] *= rgb.shape[1]/256; K[1] *= rgb.shape[0]/256
        if stored_K is not None: K = np.asarray(stored_K, np.float32)
        return rgb, K, str(path), str(rgb_path)

    def progress(state, **extra):
        elapsed = time.monotonic()-started
        atomic_json(root/f'status_{args.worker}.json', dict(state=state, pid=os.getpid(),
            worker=args.worker, frames=frames,
            target_frames=sizes[args.worker], episodes=episodes, target_episodes=len(records),
            elapsed_seconds=elapsed, inferred_frames=inferred,
            depth_seconds_mean=depth_seconds/max(inferred, 1), updated_at=time.time(), **extra))

    try:
        for rec in records:
            out = root/'episodes'/rec['key']; out.mkdir(parents=True, exist_ok=True)
            marker = out/'complete.json'
            dense = sorted((int(o)+8*i, p) for o, paths in rec['frames_by_offset'].items() for i, p in enumerate(paths))
            assert [s for s, _ in dense] == list(range(rec['control_frames']))
            if marker.exists():
                done = json.loads(marker.read_text())
                assert done['recipe_sha256'] == recipe_hash and done['episode_id'] == rec['key']
                assert len(done['frames']) == len(dense)
                for row in done['frames']:
                    assert file_sha256(out/row['file']) == row['sha256'], 'Depth output damaged'
                frames += len(dense); episodes += 1; progress('running', episode=rec['key']); continue
            rows = []; pending = pool.submit(load, rec, *dense[0])
            for i, (step, path) in enumerate(dense):
                rgb, K, exported, rgb_path = pending.result()
                if i+1 < len(dense): pending = pool.submit(load, rec, *dense[i+1])
                target = out/f'{step:06d}.npz'
                identity = dict(recipe_sha256=recipe_hash, episode_id=rec['key'], control_step=step,
                    rgb_instruction_sha256=rgb_instruction_digest(rgb, rec['instruction']),
                    source_export=exported, source_rgb=rgb_path, instruction=rec['instruction'])
                if target.exists():
                    with np.load(target, allow_pickle=False) as z:
                        prior = json.loads(str(z['metadata']))
                        assert all(prior[k] == v for k,v in identity.items())
                        assert z['depth_raw_da3_fp32'].dtype == np.float32 and z['depth_raw_da3_fp32'].shape == rgb.shape[:2]
                else:
                    tic = time.monotonic(); depth, meta = model(rgb, K)
                    depth_seconds += time.monotonic()-tic; inferred += 1
                    if not np.isfinite(depth).any(): raise ValueError('DA3 produced no valid depth')
                    tmp = out/f'{step:06d}.{os.getpid()}.tmp.npz'
                    np.savez_compressed(tmp, depth_raw_da3_fp32=depth, K=K,
                        metadata=json.dumps(dict(identity, depth=meta), allow_nan=False))
                    with np.load(tmp, allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['depth_raw_da3_fp32'], depth)
                    tmp.replace(target)
                rows.append(dict(step=step, file=target.name, sha256=file_sha256(target)))
                frames += 1
                if frames % 25 == 0: progress('running', episode=rec['key'], control_step=step)
            atomic_json(marker, dict(complete=True, episode_id=rec['key'], recipe_sha256=recipe_hash,
                instruction=rec['instruction'], frames=rows, stage='raw_depth_only', geometry_cache_complete=False))
            episodes += 1; progress('running', episode=rec['key'])
            print('DEPTH_EPISODE', args.worker, episodes, len(records), frames, sizes[args.worker], rec['key'], flush=True)
        progress('complete')
        atomic_json(root/f'complete_{args.worker}.json', dict(complete=True, recipe_sha256=recipe_hash,
            frames=frames, episodes=episodes, stage='raw_depth_only', geometry_cache_complete=False))
    except BaseException as exc:
        progress('failed', error=repr(exc)); raise
    finally:
        pool.shutdown(wait=True)


if __name__ == '__main__': main()
