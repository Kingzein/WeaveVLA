"""Run official LIBERO tasks against the bundled policy service."""
import gc
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from multiprocessing.connection import Client

ROOT = Path(os.environ['WEAVEVLA_RUNTIME_ROOT'])
SDK = Path(os.environ['EVAL_STANDARD_SOURCE'])
OUT = Path(os.environ['EVAL_OUT'])
sys.path.insert(0, str(SDK))
os.environ['MUJOCO_GL'] = 'egl'
import imageio
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from sync_robot_mask import LiveRobotMask
from render_context import install as bind_render_context
# Explicitly load this run's protocol, independent of similarly named Plus code.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import SUITES, TRIALS, SETTLING_STEPS, atomic, validate_episode
bind_render_context()
assert Path(benchmark.__file__).resolve().is_relative_to(SDK.resolve())
np.random.seed(7)
SUITE = os.environ['EVAL_SUITE']
MAX_STEPS = SUITES[SUITE]
suite = benchmark.get_benchmark_dict()[SUITE]()
assert suite.n_tasks == 10
pairs = [tuple(p) for p in json.loads(os.environ['EVAL_PAIRS'])]
expected = set(pairs)
assert len(pairs) == len(expected) and all(0 <= t < 10 and 0 <= r < 50 for t, r in pairs)
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / 'results.json'
previous = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
if previous:
    assert previous['checkpoint'] == os.environ['EVAL_CKPT']
    assert previous['benchmark'] == SUITE and previous['expected_pairs'] == [list(p) for p in pairs]
results = list(previous.get('episodes', []))
completed = {validate_episode(OUT, row, expected) for row in results}
assert len(completed) == len(results)
for task, trial in pairs:
    f = OUT / f'task_{task:02d}' / f'trial_{trial:02d}' / 'result.json'
    if (task, trial) not in completed and f.exists():
        row = json.loads(f.read_text())
        completed.add(validate_episode(OUT, row, expected))
        results.append(row)

def quat2axisangle(q):
    q = np.asarray(q)
    w = np.clip(q[3], -1., 1.)
    d = np.sqrt(1 - w * w)
    return np.zeros(3) if np.isclose(d, 0.) else q[:3] * (2 * np.arccos(w) / d)

start = time.time()
elapsed_before = previous.get('elapsed_s', 0)
def status(stage, **extra):
    n = len(results)
    wins = sum(bool(r['success']) for r in results)
    atomic(RESULTS, dict(status=stage, benchmark=SUITE, benchmark_kind='standard_LIBERO',
        checkpoint=os.environ['EVAL_CKPT'], 
        tasks=10, trials_per_task=TRIALS, expected_pairs=[list(p) for p in pairs],
        planned_episodes=len(pairs), completed_episodes=n, successes=wins,
        success_rate=wins / n if n else None, max_control_steps=MAX_STEPS,
        settling_steps=SETTLING_STEPS, execute_steps=8, predicted_horizon=int(os.environ['EVAL_HORIZON']),
        elapsed_s=elapsed_before + time.time() - start,
        episodes=sorted(results, key=lambda r: (r['task'], r['trial'])), **extra))

status('running')
conn = Client(os.environ['EVAL_SOCK'], family='AF_UNIX', authkey=b'weavevla-libero-goal-eval')
metadata = json.loads(conn.recv_bytes())
assert metadata['action_chunk_size'] == 8
try:
    for task_id in sorted({t for t, r in pairs}):
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        assert len(states) >= TRIALS
        instruction = str(task.language)
        assert instruction.strip()
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256, render_gpu_device_id=int(os.environ['EVAL_GPU']))
        env.seed(7)
        env.env.hard_reset = False
        geometry = None
        try:
            for trial in sorted(r for t, r in pairs if t == task_id):
                if (task_id, trial) in completed:
                    continue
                if geometry is not None:
                    geometry.close()
                    geometry = None
                env.reset()
                obs = env.set_init_state(states[trial])
                for _ in range(SETTLING_STEPS):
                    obs, _, _, _ = env.step([0.] * 6 + [-1.])
                geometry = LiveRobotMask(env)
                folder = OUT / f'task_{task_id:02d}' / f'trial_{trial:02d}'
                live = OUT / 'live' / f'task_{task_id:02d}' / f'trial_{trial:02d}'
                # Keep interrupted artifacts; only uncommitted attempts can be superseded.
                if folder.exists():
                    assert not (folder / 'result.json').exists()
                    folder.rename(folder.with_name(folder.name + '_partial_' + str(time.time_ns())))
                if live.exists():
                    live.rename(live.with_name(live.name + '_partial_' + str(time.time_ns())))
                folder.mkdir(parents=True)
                prompt_meta = dict(raw_sdk_instruction=instruction,
                    instruction_source='official_standard_task_language', prompt_version='standard_sdk_direct',
                    original_task=task.name)
                status('running', active_task=task_id, active_trial=trial, instruction=instruction)
                frames, wrists, actions = [], [], []
                success = False
                episode_start = time.time()
                for step in range(MAX_STEPS):
                    request_start = time.perf_counter()
                    rgb = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                    wrist_rgb = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
                    frames.append(rgb)
                    wrists.append(wrist_rgb)
                    primary = Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR)
                    wrist = Image.fromarray(wrist_rgb).resize((224, 224), Image.Resampling.BILINEAR)
                    geometry_start = time.perf_counter()
                    robot, grip, K = geometry.render(np.asarray(primary))
                    geometry_ms = (time.perf_counter() - geometry_start) * 1000
                    request = io.BytesIO()
                    np.savez(request, primary=np.asarray(primary), wrist=np.asarray(wrist),
                        robot=robot, grip=grip, K=K,
                        state=np.r_[obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']],
                        metadata=json.dumps(dict(robot_mask=geometry.last_metadata, task=task_id,
                            trial=trial, step=step, instruction=instruction, **prompt_meta)))
                    rpc_start = time.perf_counter()
                    conn.send_bytes(request.getvalue())
                    if not conn.poll(900):
                        raise TimeoutError(f'Policy timeout: {SUITE}/{task_id}/{trial}/{step}')
                    reply = json.loads(conn.recv_bytes())
                    if 'error' in reply:
                        raise RuntimeError(reply['error'])
                    timing = dict(reply['timing'], geometry_ms=geometry_ms,
                        rpc_ms=(time.perf_counter() - rpc_start) * 1000,
                        observation_to_action_ms=(time.perf_counter() - request_start) * 1000)
                    with (folder / 'inference_timing.jsonl').open('a') as stream:
                        stream.write(json.dumps(timing) + '\n')
                    action = np.asarray(reply['action'], dtype=np.float32)
                    assert action.shape == (7,) and np.isfinite(action).all()
                    action = action.copy()
                    action[6] = 1. - 2. * float(action[6] > .5)
                    actions.append(action.copy())
                    obs, _, success, _ = env.step(action.tolist())
                    if success:
                        break
                video = folder / ('success.mp4' if success else 'failure.mp4')
                imageio.mimwrite(str(video), frames, fps=25)
                imageio.mimwrite(str(folder / 'rgb_dual.mp4'), [np.concatenate([a, b], axis=1) for a, b in zip(frames, wrists)], fps=25)
                np.save(folder / 'actions.npy', np.stack(actions))
                row = dict(robot_mask_version=geometry.version, task=task_id, trial=trial,
                    task_name=task.name, instruction=instruction, **prompt_meta,
                    success=bool(success), steps=len(actions), seconds=time.time() - episode_start,
                    video=str(video.relative_to(OUT)), rgb_video=str((folder / 'rgb_dual.mp4').relative_to(OUT)))
                atomic(folder / 'result.json', row)
                results.append(row)
                completed.add((task_id, trial))
                status('running')
                print('ROLLOUT_RESULT', json.dumps(row), 'TOTAL', len(results), flush=True)
        finally:
            if geometry is not None:
                geometry.close()
            env.close()
            gc.collect()
    assert completed == expected
    status('complete')
    conn.send_bytes(b'STOP')
    print('STANDARD_SHARD_COMPLETE', SUITE, len(results), flush=True)
except Exception:
    status('error', error=traceback.format_exc())
    raise
finally:
    conn.close()
