import json
from pathlib import Path

from accelerate.logging import get_logger
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets"):
    if dataset_py != "lerobot_datasets":
        raise ValueError(f"Unsupported dataset_py: {dataset_py!r}")

    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    vla_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(
        data_cfg=vla_cfg,
        balance_dataset_weights=vla_cfg.get("balance_dataset_weights", False),
        balance_trajectory_weights=vla_cfg.get("balance_trajectory_weights", False),
    )
    num_workers = int(vla_cfg.get("num_workers", 4))
    loader_options = {
        "batch_size": vla_cfg.per_device_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(vla_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(vla_cfg.get("persistent_workers", True))
        loader_options["prefetch_factor"] = int(vla_cfg.get("prefetch_factor", 2))

    loader = DataLoader(dataset, **loader_options)
    if not dist.is_initialized() or dist.get_rank() == 0:
        dataset.save_dataset_statistics(Path(cfg.output_dir) / "dataset_statistics.json")
    return loader
