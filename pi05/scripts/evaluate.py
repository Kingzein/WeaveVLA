"""Launch a LIBERO or CALVIN rollout shard in isolated policy and simulation environments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="benchmark", required=True)
    libero = sub.add_parser("libero")
    calvin = sub.add_parser("calvin")
    for command in (libero, calvin):
        command.add_argument("--checkpoint", type=Path, required=True, help="Torch inference checkpoint")
        command.add_argument("--output", type=Path, required=True, help="New results directory for this shard")
        command.add_argument("--runtime-root", type=Path, required=True, help="Original graph perception and asset root")
        command.add_argument("--policy-python", default=sys.executable)
        command.add_argument("--sim-python", default=sys.executable)
        command.add_argument("--prompt-root", type=Path, help="Defaults to the runtime root")
        command.add_argument("--da3-source", type=Path, help="Optional DA3 Python source directory")
        command.add_argument("--sam-source", type=Path, help="Optional SAM3 Python source directory")
        command.add_argument("--tokenizer", required=True, type=Path)
    libero.add_argument("--libero-source", required=True, type=Path)
    libero.add_argument("--suite", required=True, choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"))
    libero.add_argument("--stats", type=Path, required=True, help="Training-time action normalization metadata")
    libero.add_argument("--pairs", type=Path, help="JSON list of [task, trial]; default is all 500")
    libero.add_argument("--render-gpu", type=int, default=0)
    libero.add_argument("--depth-weights", type=Path, required=True)
    libero.add_argument("--cutie-source", type=Path, required=True)
    libero.add_argument("--cutie-weights", type=Path, required=True)
    libero.add_argument("--sam-weights", type=Path, required=True)
    libero.add_argument("--yolo-weights", type=Path, required=True)
    libero.add_argument("--sam-python", required=True)
    libero.add_argument("--yolo-python", required=True)
    calvin.add_argument("--frontend-root", type=Path, required=True, help="Frozen CALVIN frontend code and detector assets")
    calvin.add_argument("--frontend-python", required=True)
    calvin.add_argument("--detector-sha256")
    calvin.add_argument("--env-root", type=Path, required=True, help="CALVIN calvin_env source root with its data and conf")
    calvin.add_argument("--env-config", type=Path, required=True, help="D validation merged_config.yaml")
    calvin.add_argument("--calvin-conf", type=Path, required=True, help="CALVIN calvin_models/conf")
    calvin.add_argument("--sequences", type=Path, default=ASSETS / "calvin/sequences.json")
    calvin.add_argument("--calibration", type=Path, default=ASSETS / "calvin/calibration.json")
    calvin.add_argument("--lanes", type=int, default=1)
    calvin.add_argument("--lane", type=int, default=0)
    calvin.add_argument("--limit", type=int)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            value = value.expanduser().resolve()
            if name != "output" and not value.exists():
                parser.error(f"Missing --{name.replace('_', '-')}: {value}")
            setattr(args, name, value)
    if args.output.exists():
        parser.error(f"Choose a new output directory: {args.output}")
    return args


def env_for(args):
    runtime = args.runtime_root
    paths = [ROOT / "src"]
    paths.extend(source for source in (runtime, args.da3_source, args.sam_source) if source is not None)
    prefix = os.pathsep.join(map(str, paths))
    extra = os.environ.get("PYTHONPATH")
    if extra:
        prefix += ":" + extra
    env = dict(os.environ, PYTHONPATH=prefix, WEAVEVLA_RUNTIME_ROOT=str(runtime),
               EVAL_OUT=str(args.output), EVAL_CKPT=str(args.checkpoint),
               WEAVE_PROMPT_ROOT=str(args.prompt_root or runtime),
               EVAL_POLICY_PYTHON=args.policy_python, EVAL_POLICY_PATH=prefix,
               PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", JAX_PLATFORMS="cpu")
    for attribute, key in (("stats", "EVAL_STATS"), ("tokenizer", "PI05_TOKENIZER")):
        value = getattr(args, attribute, None)
        if value is not None:
            env[key] = str(value)
        else:
            env.pop(key, None)
    return env


def libero(args, env):
    pairs = json.loads(args.pairs.read_text()) if args.pairs else [
        [task, trial] for task in range(10) for trial in range(50)
    ]
    keys = [tuple(pair) for pair in pairs]
    if (not pairs or len(keys) != len(set(keys)) or any(
            len(key) != 2 or type(key[0]) is not int or type(key[1]) is not int
            or not (0 <= key[0] < 10 and 0 <= key[1] < 50) for key in keys)):
        raise ValueError("Pass unique [task 0–9, trial 0–49] pairs")
    env.update(EVAL_SUITE=args.suite, EVAL_DATASET=args.suite,
               EVAL_PAIRS=json.dumps(pairs), EVAL_GPU=str(args.render_gpu),
               EVAL_HORIZON="10", EVAL_STANDARD_SOURCE=str(args.libero_source),
               CUTIE_SOURCE=str(args.cutie_source), CUTIE_WEIGHTS=str(args.cutie_weights),
               LIBERO_SAM_WEIGHTS=str(args.sam_weights), LIBERO_YOLO_WEIGHTS=str(args.yolo_weights),
               DA3_WEIGHTS=str(args.depth_weights), SAM3_PYTHON=args.sam_python,
               YOLO_PYTHON=args.yolo_python)
    with tempfile.TemporaryDirectory(prefix="weave-eval-", dir="/tmp") as folder:
        env["EVAL_SOCK"] = str(Path(folder) / "policy.sock")
        with (args.output / "policy.log").open("w") as log:
            policy = subprocess.Popen([args.policy_python, "-u", str(ROOT / "scripts/serve_policy.py"), "libero"],
                                      env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 1200
                ready = args.output / "server_ready.json"
                while not ready.is_file():
                    if policy.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"LIBERO policy startup failed. See {log.name}")
                    time.sleep(0.25)
                subprocess.run([args.sim_python, "-u", str(ROOT / "examples/libero/client.py")], env=env, check=True)
            finally:
                if policy.poll() is None:
                    try:
                        policy.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        policy.terminate()
                        try:
                            policy.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            policy.kill()
                            policy.wait()


def calvin(args, env):
    env.update(CALVIN_FRONTEND_ROOT=str(args.frontend_root),
               CALVIN_FRONTEND_PYTHON=args.frontend_python,
               CALVIN_ENV_ROOT=str(args.env_root), CALVIN_CALIBRATION=str(args.calibration))
    if args.detector_sha256:
        env["CALVIN_DETECTOR_SHA256"] = args.detector_sha256
    command = [args.sim_python, "-u", str(ROOT / "examples/calvin/rollout.py"),
               "--checkpoint", str(args.checkpoint), "--output", str(args.output)]
    for name in ("env_config", "env_root", "calvin_conf", "sequences", "lane", "lanes", "limit"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    subprocess.run(command, env=env, check=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = libero if args.benchmark == "libero" else calvin
    command(args, env_for(args))


if __name__ == "__main__":
    main()
