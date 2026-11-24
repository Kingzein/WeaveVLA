"""Train a new π0.5 + Graph policy from OpenPI base weights."""
import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import asdict

from .config import CONFIGS, config_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=config_name, choices=CONFIGS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True, help="OpenPI pi05_base/params directory")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path)
    parser.add_argument("--fsdp-devices", type=int, help="Devices per model shard group; inferred by default")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--peak-lr", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--geo-weight", type=float, help="Geometry loss coefficient")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = CONFIGS[args.config]
    for name in ("steps", "save_interval", "peak_lr", "warmup", "geo_weight", "seed"):
        if getattr(args, name) is None:
            key = "geometry_weight" if name == "geo_weight" else name
            setattr(args, name, getattr(config, key))
    for path in (args.manifest, args.tokenizer):
        if not path.is_file():
            parser.error(f"Input file missing: {path}")
    if not args.base_params.is_dir():
        parser.error(f"OpenPI base params directory missing: {args.base_params}")
    if args.output.exists():
        parser.error(f"Use a new output directory: {args.output}")
    if args.steps < 1 or args.save_interval < 1 or args.warmup < 0 or args.peak_lr <= 0 or args.geo_weight < 0:
        parser.error("Invalid steps, save interval, or loss/learning rate settings")
    if config.dataset != "libero":
        if args.prompt_root is None or not args.prompt_root.is_dir():
            parser.error("--prompt-root must point to the directory containing the CALVIN language assets")
        os.environ["WEAVE_PROMPT_ROOT"] = str(args.prompt_root.resolve())
    return args


def training_settings(config, args, fsdp):
    return asdict(config) | {
        "config": args.config,
        "fsdp_devices": fsdp,
        "physical_batch": config.physical_batch,
        "steps": args.steps,
        "save_interval": args.save_interval,
        "peak_lr": args.peak_lr,
        "warmup": args.warmup,
        "geometry_weight": args.geo_weight,
        "seed": args.seed,
    }


def main():
    args = parse_args()

    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    import torch
    from flax.traverse_util import flatten_dict
    from openpi.models import gemma, model
    from openpi.shared import array_typing as at

    from weavevla_pi05.graph.graph import EdgeMemoryGraph
    from .host_data import Host, Prefetch

    config = CONFIGS[args.config]
    libero = config.dataset == "libero"
    if libero:
        from .libero.core import Core, state_from
        from .libero.source_stream import Streams
    else:
        from .calvin.core import Core, state_from
        from .calvin.d_to_d.source_stream import Streams

    physical_batch = config.physical_batch
    devices = jax.devices()
    device_count = len(devices)
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("Training requires CUDA devices")
    fsdp = args.fsdp_devices if args.fsdp_devices is not None else (2 if device_count % 2 == 0 else 1)
    if fsdp < 1 or device_count % fsdp or physical_batch % (device_count // fsdp):
        raise ValueError("Device mesh must divide both the visible devices and the training batch")

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    graph = EdgeMemoryGraph(gemma.get_config("gemma_2b").width)
    graph.output_projection = torch.nn.Linear(
        graph.output_projection.in_features, gemma.get_config("gemma_300m").width
    )
    graph_params = {name: tensor.detach().numpy() for name, tensor in graph.state_dict().items()}
    del graph
    with at.disable_typechecking():
        core = Core(
            graph_params, fsdp=fsdp, peak_lr=args.peak_lr, warmup=args.warmup,
            action_horizon=config.action_horizon, geo_weight=args.geo_weight,
        )
        base = model.restore_params(args.base_params, restore_type=np.ndarray, dtype=np.float32)
        if "graph_params" in base:
            raise ValueError("Use the original π0.5 base weights")
        base["graph_params"] = graph_params
        actual = flatten_dict(base)
        expected = flatten_dict(core.p_shape.to_pure_dict())
        if set(actual) != set(expected):
            raise ValueError(f"Base weight structure differs: {sorted(set(actual) ^ set(expected))[:8]}")
        for key, leaf in expected.items():
            if actual[key].shape != leaf.shape:
                raise ValueError(f"Parameter shape differs at {key}: {actual[key].shape} != {leaf.shape}")
        params = jax.device_put(state_from(core.p_shape, base), core.ps)
        ema = jax.device_put(state_from(core.p_shape, base), core.ps)
        del base, actual, expected, graph_params
        optimizer = jax.jit(core.tx.init, in_shardings=(core.ps,), out_shardings=core.os)(params)
        memory = jax.device_put({
            "nodes": np.zeros((physical_batch, 32, 256), np.float32),
            "edges": np.zeros((physical_batch, 32, 32, 256), np.float32),
        }, core.ds)
        rng = jax.device_put(jax.random.PRNGKey(args.seed), core.rep)
        tree = (params, optimizer, ema, memory, rng)

        run = args.output.resolve()
        run.mkdir(parents=True, exist_ok=False)
        (run / "config.json").write_text(json.dumps(training_settings(config, args, fsdp), indent=2) + "\n")
        streams = [Streams(rank, args.manifest.resolve(), args.tokenizer.resolve()) for rank in range(config.stream_count)]
        prefetch = Prefetch(Host(streams, capacity=config.stream_capacity))
        try:
            batch = prefetch.take()
            update = core.compile(batch, memory)
            with (run / "metrics.jsonl").open("w", buffering=1) as metrics:
                for step in range(1, args.steps + 1):
                    params, optimizer, ema, memory, rng, info = update(
                        *tree[:4], *core.inputs(batch), tree[4]
                    )
                    tree = (params, optimizer, ema, memory, rng)
                    row = {key: float(value) for key, value in info.items()}
                    if not all(math.isfinite(value) for value in row.values()):
                        raise FloatingPointError(f"Nonfinite training metric at step {step}")
                    row.update(step=step, lr=float(core.schedule(step - 1)))
                    metrics.write(json.dumps(row) + "\n")
                    if step % args.save_interval == 0 or step == args.steps:
                        with ocp.PyTreeCheckpointer() as checkpointer:
                            checkpointer.save(run / f"step_{step:06d}", {"params": params.to_pure_dict()})
                    if step < args.steps:
                        batch = prefetch.take()
        finally:
            prefetch.close()


if __name__ == "__main__":
    main()
