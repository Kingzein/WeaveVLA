"""Named training configurations for the supported benchmark splits."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    action_horizon: int
    batch_size: int
    stream_count: int
    stream_capacity: int
    geometry_weight: float
    geometry_stride: int = 8
    steps: int = 30_000
    save_interval: int = 3_000
    peak_lr: float = 5e-5
    warmup: int = 10_000
    seed: int = 433

    @property
    def physical_batch(self) -> int:
        return self.stream_count * self.stream_capacity


CONFIGS = {
    "pi05_libero": TrainConfig("libero", 10, 256, 6, 44, 0.1),
    "pi05_calvin_d": TrainConfig("calvin_d", 8, 48, 6, 8, 0.1),
}
_ALIASES = {"libero": "pi05_libero", "calvin-d": "pi05_calvin_d"}


def config_name(value: str) -> str:
    return _ALIASES.get(value, value)
