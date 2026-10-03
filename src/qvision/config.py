"""Typed experiment configuration loaded from YAML.

Every knob that affects a result lives here, so a run directory's ``config.yaml`` is enough to
reproduce it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid config: {message}")


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/fashion-mnist"
    val_size: int = 5000
    # Use only the first N (shuffled) training examples; None means the full training split.
    train_subset: int | None = None

    def __post_init__(self) -> None:
        _require(self.val_size > 0, f"data.val_size must be > 0, got {self.val_size}")
        _require(
            self.train_subset is None or self.train_subset > 0,
            f"data.train_subset must be null or > 0, got {self.train_subset}",
        )


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, int] = (16, 32)
    hidden: int = 128
    dropout: float = 0.25
    num_classes: int = 10

    def __post_init__(self) -> None:
        _require(
            len(self.channels) == 2 and all(c > 0 for c in self.channels),
            f"model.channels must be two positive ints, got {list(self.channels)}",
        )
        _require(self.hidden > 0, f"model.hidden must be > 0, got {self.hidden}")
        _require(0.0 <= self.dropout < 1.0, f"model.dropout must be in [0, 1), got {self.dropout}")
        _require(self.num_classes >= 2, f"model.num_classes must be >= 2, got {self.num_classes}")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 3
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        _require(self.epochs > 0, f"train.epochs must be > 0, got {self.epochs}")
        _require(self.batch_size > 0, f"train.batch_size must be > 0, got {self.batch_size}")
        _require(self.lr > 0, f"train.lr must be > 0, got {self.lr}")
        _require(
            self.weight_decay >= 0, f"train.weight_decay must be >= 0, got {self.weight_decay}"
        )
        _require(self.patience >= 1, f"train.patience must be >= 1, got {self.patience}")
        _require(self.min_delta >= 0, f"train.min_delta must be >= 0, got {self.min_delta}")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    run_name: str = "fmnist-cnn"
    runs_dir: str = "runs"
    threads: int = 2
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        _require(self.threads >= 1, f"threads must be >= 1, got {self.threads}")
        _require(bool(self.run_name), "run_name must not be empty")
        subset = self.data.train_subset
        _require(
            subset is None or subset >= self.train.batch_size,
            f"data.train_subset ({subset}) must be >= train.batch_size ({self.train.batch_size})",
        )

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"]["channels"] = list(self.model.channels)
        return d


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw)
    model_raw = dict(raw.pop("model", {}) or {})
    if "channels" in model_raw:
        model_raw["channels"] = tuple(model_raw["channels"])
    sections = {
        "data": _build(DataConfig, raw.pop("data", None)),
        "model": _build(ModelConfig, model_raw),
        "train": _build(TrainConfig, raw.pop("train", None)),
    }
    return replace(_build(Config, raw), **sections)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh) or {})
