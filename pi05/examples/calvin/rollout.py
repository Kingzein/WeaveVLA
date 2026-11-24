"""Run CALVIN D five-task chains with the official oracle."""
import argparse
import json
import os
import random
import subprocess
import time
from multiprocessing import Pipe
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def request(connection, **message):
    if message:
        connection.send(message)
    if not connection.poll(900):
        raise TimeoutError(f"CALVIN policy did not respond: {message.get('op', 'startup')}")
    result = connection.recv()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result


def run_chain(env, oracle, annotations, policy, initial, tasks, ident):
    import numpy as np
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    random.seed(ident)
    np.random.seed(ident)
    robot, scene = get_env_state_for_initial_condition(initial)
    env.reset(robot_obs=robot, scene_obs=scene)
    request(policy, op="start", id=ident)
    attempts = []
    for task in tasks:
        obs = env.get_obs()
        language = annotations[task][0].split("\n")[0].replace("\u2019", "'")
        request(policy, op="reset")
        start_info = env.get_info()
        for step in range(360):
            action = np.asarray(request(policy, op="step", language=language,
                                        robot_obs=obs["robot_obs"],
                                        rgb_static=obs["rgb_obs"]["rgb_static"],
                                        rgb_gripper=obs["rgb_obs"]["rgb_gripper"]), np.float32)
            if action.shape != (7,) or not np.isfinite(action).all() or action[6] not in (-1, 1):
                raise ValueError("Invalid CALVIN action")
            obs, _, _, current = env.step(action)
            success = bool(oracle.get_task_info_for_set(start_info, current, {task}))
            if success:
                break
        attempts.append(dict(task=task, success=success, steps=step + 1))
        if not success:
            break
    return dict(id=ident, length=sum(item["success"] for item in attempts), attempts=attempts)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequences", type=Path, default=here.parents[1] / "assets/calvin/sequences.json")
    parser.add_argument("--env-config", required=True, type=Path)
    parser.add_argument("--calvin-conf", required=True, type=Path)
    parser.add_argument("--env-root", required=True, type=Path)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    sequences = json.loads(args.sequences.read_text())
    if len(sequences) != 1000 or any(len(tasks) != 5 for _, tasks in sequences):
        parser.error("Use 1,000 five-task sequences")
    if not 0 <= args.lane < args.lanes or args.lanes > len(sequences):
        parser.error("Invalid shard")
    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive")

    import hydra
    from omegaconf import OmegaConf

    annotations = OmegaConf.load(args.calvin_conf / "annotations/new_playtable_validation.yaml")
    oracle = hydra.utils.instantiate(OmegaConf.load(
        args.calvin_conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    ))
    cfg = OmegaConf.load(args.env_config)
    cfg.data_path = str(args.env_root / "data")
    for name in list(cfg.cameras):
        if "tactile" in name:
            del cfg.cameras[name]
    cfg.env.use_egl = True
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    read_end, write_end = Pipe()
    child_env = dict(os.environ, PYTHONPATH=os.environ["EVAL_POLICY_PATH"])
    child = subprocess.Popen(
        [os.environ["EVAL_POLICY_PYTHON"], str(here.parents[1] / "scripts/serve_policy.py"),
         "calvin", "--pipe-fd", str(write_end.fileno()), "--checkpoint", str(args.checkpoint)], pass_fds=[write_end.fileno()], env=child_env,
    )
    write_end.close()
    try:
        receipt = request(read_end)
        if not receipt.get("ready"):
            raise RuntimeError(receipt)
        rows = []
        ids = list(range(args.lane, len(sequences), args.lanes))[:args.limit]
        result = dict(checkpoint=str(args.checkpoint), planned=len(ids), frontend=receipt, rows=rows)
        for ident in ids:
            started = time.monotonic()
            initial, tasks = sequences[ident]
            row = run_chain(env, oracle, annotations, read_end, initial, tasks, ident)
            row["seconds"] = time.monotonic() - started
            rows.append(row)
            result["completed"] = len(rows)
            result["success_rates"] = {
                length: sum(r["length"] >= length for r in rows) / len(rows)
                for length in range(1, 6)
            }
            result["mean_chain_length"] = sum(r["length"] for r in rows) / len(rows)
            write_json(args.output / "results.json", result)
            print("D_SEQUENCE", ident, row["length"], flush=True)
    finally:
        if child.poll() is None:
            try:
                read_end.send(dict(op="close"))
                child.wait(timeout=40)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        read_end.close()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
