"""Dynamic INT8 backend selection, including builds whose default engine is none."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from qvision.study import select_quantized_engine


@pytest.mark.parametrize(
    "supported, expected",
    [
        (["none", "qnnpack", "fbgemm", "x86"], "x86"),
        (["qnnpack", "fbgemm"], "fbgemm"),
        (["qnnpack"], "qnnpack"),
    ],
)
def test_select_quantized_engine_prefers_supported_cpu_backend(monkeypatch, supported, expected):
    backend = SimpleNamespace(supported_engines=supported, engine="none")
    monkeypatch.setattr(torch.backends, "quantized", backend)
    assert select_quantized_engine() == expected
    assert backend.engine == expected


@pytest.mark.parametrize("supported", [[], ["none"], ["onednn", "none"]])
def test_select_quantized_engine_fails_clearly_without_supported_backend(monkeypatch, supported):
    backend = SimpleNamespace(supported_engines=supported, engine="none")
    monkeypatch.setattr(torch.backends, "quantized", backend)
    with pytest.raises(RuntimeError, match="No supported dynamic INT8 quantization engine"):
        select_quantized_engine()
    assert backend.engine == "none"


def test_dynamic_int8_runs_with_default_engine_in_fresh_process(tmp_path):
    # A fresh process uses this build's real default (none on macOS arm64), independent of
    # engine selection in other tests. Some builds reject assigning none after initialization.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from qvision.config import ModelConfig
from qvision.model import SmallCNN
from qvision.study import torch_dynamic_int8

torch.set_num_threads(2)
torch.manual_seed(0)
model = SmallCNN(ModelConfig(channels=(4, 8), hidden=16, dropout=0.0)).eval()
quantized = torch_dynamic_int8(model)
engine = torch.backends.quantized.engine
assert engine in torch.backends.quantized.supported_engines
assert engine in {"x86", "fbgemm", "qnnpack"}
assert any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in quantized.modules())
images = torch.randn(3, 1, 28, 28)
with torch.inference_mode():
    logits = quantized(images)
    reference = model(images)
assert logits.shape == (3, 10)
assert torch.isfinite(logits).all()
torch.testing.assert_close(logits, reference, atol=0.01, rtol=0.05)
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
