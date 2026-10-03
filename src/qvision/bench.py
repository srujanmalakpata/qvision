"""Small, honest latency measurement helper."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable


def summarize_ms(samples: list[float]) -> dict[str, float]:
    samples = sorted(samples)
    p90_index = min(len(samples) - 1, round(0.9 * (len(samples) - 1)))
    return {
        "median_ms": round(statistics.median(samples), 3),
        "p90_ms": round(samples[p90_index], 3),
        "mean_ms": round(statistics.fmean(samples), 3),
        "repeats": len(samples),
    }


def measure_interleaved_ms(
    fns: dict[str, Callable[[], object]], warmup: int = 5, repeats: int = 30
) -> dict[str, dict[str, float]]:
    """Time several functions in round-robin order.

    On a shared machine background load drifts over time; interleaving makes every candidate
    see (roughly) the same load, so their *relative* latencies stay comparable.
    """
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    samples: dict[str, list[float]] = {name: [] for name in fns}
    for _ in range(repeats):
        for name, fn in fns.items():
            t0 = time.perf_counter()
            fn()
            samples[name].append((time.perf_counter() - t0) * 1000.0)
    return {name: summarize_ms(s) for name, s in samples.items()}
