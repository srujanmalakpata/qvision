"""Export the fp32 model to ONNX and check ONNX Runtime agrees with PyTorch."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from qvision.bench import measure_interleaved_ms

INPUT_NAME = "image"
OUTPUT_NAME = "logits"


@contextlib.contextmanager
def _quiet_logger(name: str, level: int = logging.ERROR) -> Iterator[None]:
    """Raise one logger's threshold for the ``with`` block, then restore it.

    The exporter's registration logger prints one warning per optional torchvision op it cannot
    find (torchvision is not a dependency); those lines are noise here. Python warnings are left
    alone, so real exporter fallbacks or numerical warnings still surface.
    """
    logger = logging.getLogger(name)
    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)


def export_onnx(model: nn.Module, path: str | Path, opset: int = 18) -> Path:
    """Export with the ``torch.export``-based exporter and a dynamic batch dimension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    example = torch.zeros(2, 1, 28, 28)
    batch = torch.export.Dim("batch", min=1, max=4096)
    with _quiet_logger("torch.onnx._internal.exporter._registration"):
        program = torch.onnx.export(
            model,
            (example,),
            dynamo=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_shapes={"x": {0: batch}},
            opset_version=opset,
            external_data=False,
            verbose=False,
        )
    program.save(str(path), external_data=False)
    return path


def ort_session(path: str | Path, threads: int | None = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def check_parity(
    model: nn.Module,
    onnx_path: str | Path,
    inputs: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Compare logits from PyTorch and ONNX Runtime on ``inputs``."""
    session = ort_session(onnx_path)
    max_abs, agree = 0.0, 0
    within = True
    model = model.eval()
    for start in range(0, len(inputs), batch_size):
        x = inputs[start : start + batch_size]
        with torch.inference_mode():
            ref = model(x).numpy()
        got = session.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0]
        max_abs = max(max_abs, float(np.abs(ref - got).max()))
        agree += int((ref.argmax(1) == got.argmax(1)).sum())
        within &= bool(np.allclose(ref, got, atol=atol, rtol=rtol))
    return {
        "n": int(len(inputs)),
        "max_abs_logit_diff": max_abs,
        "top1_agreement": agree / len(inputs),
        "atol": atol,
        "rtol": rtol,
        "allclose": within,
    }


def compare_latency(
    model: nn.Module,
    onnx_path: str | Path,
    batch: torch.Tensor,
    threads: int,
    repeats: int = 50,
) -> dict[str, dict[str, float]]:
    """Interleaved PyTorch-vs-ONNX Runtime latency on the same batch and thread budget."""
    session = ort_session(onnx_path, threads=threads)
    feed = {INPUT_NAME: batch.numpy()}
    model = model.eval()

    @torch.inference_mode()
    def torch_run() -> torch.Tensor:
        return model(batch)

    return measure_interleaved_ms(
        {
            "pytorch_fp32": torch_run,
            "onnxruntime_fp32": lambda: session.run([OUTPUT_NAME], feed),
        },
        repeats=repeats,
    )
