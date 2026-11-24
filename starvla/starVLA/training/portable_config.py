"""Serialize model and training settings without machine-local paths."""

import ntpath

from omegaconf import OmegaConf


_SAVED_FIELDS = (
    "run_id",
    "run_root_dir",
    "output_dir",
    "seed",
    "is_debug",
    "version_id",
    "framework",
    "datasets",
    "trainer",
    "memory_training",
)

_LOCAL_FIELDS = {
    "base_vlm",
    "data_root_dir",
    "detector_checkpoint",
    "graph_cache_root",
    "observation_manifest",
    "output_dir",
    "pretrained_checkpoint",
    "run_root_dir",
}


def _without_local_paths(value):
    if isinstance(value, dict):
        return {
            key: None if key in _LOCAL_FIELDS else _without_local_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_local_paths(item) for item in value]
    if isinstance(value, str) and (value.startswith(("/", "\\", "~", "file:")) or ntpath.isabs(value)):
        return None
    return value


def save_portable_config(config, path):
    settings = OmegaConf.to_container(config, resolve=True)
    saved = {key: settings[key] for key in _SAVED_FIELDS if key in settings}
    OmegaConf.save(OmegaConf.create(_without_local_paths(saved)), path)
