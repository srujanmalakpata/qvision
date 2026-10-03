"""A tiny MLflow-style run tracker: one directory per run with config, metrics and summary.

runs/<run_name>/
    config.yaml     exact configuration used
    env.json        library versions, thread count, platform
    metrics.csv     one row per epoch
    summary.json    final numbers (best epoch, test metrics, ...)
    model.pt        best checkpoint (git-ignored)
"""

from __future__ import annotations

import csv
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        # Load average at measurement time: a high value means latency numbers are contended.
        "load_average_1m": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
    }


class RunTracker:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self._fieldnames: list[str] | None = None
        # A fresh run overwrites the previous metrics file rather than appending to it.
        self.metrics_path.unlink(missing_ok=True)

    def log_config(self, config: dict[str, Any]) -> None:
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        write_json(self.run_dir / "env.json", environment_info())

    def log_metrics(self, row: dict[str, float | int]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row)
        new_file = not self.metrics_path.exists()
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def log_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.run_dir / "summary.json", summary)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
