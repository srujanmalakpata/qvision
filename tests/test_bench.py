import time

import pytest

from qvision.bench import measure_interleaved_ms, summarize_ms


def test_summarize_ms_reports_ordered_statistics():
    stats = summarize_ms([5.0, 1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    assert stats["repeats"] == 11
    assert stats["median_ms"] == 6.0
    assert stats["p90_ms"] == 10.0
    assert stats["mean_ms"] == pytest.approx(155 / 11, abs=1e-3)


def test_interleaved_measurement_times_every_candidate():
    calls = {"fast": 0, "slow": 0}

    def make(name, seconds):
        def fn():
            calls[name] += 1
            time.sleep(seconds)

        return fn

    stats = measure_interleaved_ms(
        {"fast": make("fast", 0.0), "slow": make("slow", 0.003)}, warmup=2, repeats=4
    )
    assert calls == {"fast": 6, "slow": 6}
    assert stats["slow"]["median_ms"] > stats["fast"]["median_ms"]
    assert stats["fast"]["repeats"] == 4
