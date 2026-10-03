"""Quantization study: compare fp32 against several INT8/INT4 variants.

Variants
--------
* ``fp32``                 - the trained model.
* ``int8_weight_only``     - this repo's per-channel symmetric INT8 quantizer (all Conv2d + Linear).
* ``int4_weight_only``     - same quantizer at 4 bits, two weights packed per byte.
* ``int8_torch_dynamic``   - ``torch.ao.quantization.quantize_dynamic`` on Linear layers
                             (INT8 weights *and* dynamically quantized activations, using
                             PyTorch's integer GEMM kernels; convolutions stay fp32).

For each variant we report test accuracy and its delta vs fp32, a *paired* comparison against
fp32 on the same test images (discordant counts, exact McNemar p-value and a paired-bootstrap 95%
CI of the accuracy delta), top-1 agreement with fp32, serialized state-dict size, and CPU latency
per batch.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from qvision.bench import measure_interleaved_ms
from qvision.evaluate import labels_of, predict_proba
from qvision.metrics import paired_comparison
from qvision.quantize import quantize_model_weights, serialized_size_bytes, weight_error_stats


def select_quantized_engine() -> str:
    """Select the process-wide CPU backend before packing dynamic INT8 weights."""
    supported = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No supported dynamic INT8 quantization engine; available: {supported}")


def torch_dynamic_int8(model: nn.Module) -> nn.Module:
    """PyTorch's built-in dynamic quantization (Linear layers only)."""
    from torch.ao.quantization import quantize_dynamic

    select_quantized_engine()
    with warnings.catch_warnings():
        # Silence only the two known deprecation notices: torch.ao.quantization is moving to the
        # separate `torchao` package, and torch.ao's quantized-tensor helpers warn that they are
        # deprecated too. Any other warning still surfaces.
        warnings.filterwarnings(
            "ignore", message=r"torch\.ao\.quantization is deprecated", category=DeprecationWarning
        )
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.ao\.")
        return quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8)


def build_variants(model: nn.Module) -> dict[str, nn.Module]:
    model = model.eval()
    return {
        "fp32": model,
        "int8_weight_only": quantize_model_weights(model, bits=8),
        "int4_weight_only": quantize_model_weights(model, bits=4),
        "int8_torch_dynamic": torch_dynamic_int8(model),
    }


def _forward_fn(model: nn.Module, batch: torch.Tensor) -> Callable[[], torch.Tensor]:
    @torch.inference_mode()
    def run() -> torch.Tensor:
        return model(batch)

    return run


def run_study(
    model: nn.Module,
    test_ds: TensorDataset,
    batch_sizes: tuple[int, ...] = (1, 256),
    repeats: int = 50,
    n_boot: int = 2000,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    y_true = labels_of(test_ds)
    images = test_ds.tensors[0]
    variants = build_variants(model)
    fp32_pred: np.ndarray | None = None
    fp32_acc = fp32_size = None
    results: dict[str, Any] = {}
    for name, variant in variants.items():
        pred = predict_proba(variant, test_ds).argmax(axis=1)
        acc = float((pred == y_true).mean())
        size = serialized_size_bytes(variant)
        if fp32_pred is None:
            fp32_pred, fp32_acc, fp32_size = pred, acc, size
        paired = paired_comparison(fp32_pred == y_true, pred == y_true, n_boot=n_boot, seed=seed)
        entry: dict[str, Any] = {
            "accuracy": acc,
            "accuracy_delta_pp": round((acc - fp32_acc) * 100, 3),
            "paired_vs_fp32": {
                "only_fp32_correct": paired.only_a_correct,
                "only_variant_correct": paired.only_b_correct,
                "delta_pp_ci95": [round(v * 100, 3) for v in paired.delta_ci95],
                "mcnemar_exact_p": paired.mcnemar_p,
            },
            "top1_agreement_with_fp32": float((pred == fp32_pred).mean()),
            "size_bytes": size,
            "compression_vs_fp32": round(fp32_size / size, 2),
            "latency": {},
        }
        results[name] = entry
        log(
            f"{name:20s} acc {acc:.4f} ({entry['accuracy_delta_pp']:+.2f} pp, "
            f"McNemar p {paired.mcnemar_p:.3g})  {size} B"
        )

    for bs in batch_sizes:
        fns = {name: _forward_fn(v, images[:bs]) for name, v in variants.items()}
        for name, stats in measure_interleaved_ms(fns, warmup=5, repeats=repeats).items():
            results[name]["latency"][f"batch_{bs}"] = stats
        medians = ", ".join(
            f"{n}: {results[n]['latency'][f'batch_{bs}']['median_ms']}" for n in fns
        )
        log(f"latency batch {bs} (median ms, interleaved): {medians}")

    return {
        "variants": results,
        "weight_error": {
            "int8": weight_error_stats(model, 8),
            "int4": weight_error_stats(model, 4),
        },
        "settings": {
            "n_test": len(y_true),
            "paired_bootstrap_resamples": n_boot,
            "batch_sizes": list(batch_sizes),
            "latency_repeats": repeats,
            "latency_method": "round-robin interleaved across variants, after 5 warmup calls",
            "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "quantized_engine": torch.backends.quantized.engine,
        },
    }


def study_to_markdown(study: dict[str, Any]) -> str:
    batch_keys = [f"batch_{bs}" for bs in study["settings"]["batch_sizes"]]
    header = (
        "| Variant | Accuracy | Δ vs fp32 (pp) | Paired 95% CI of Δ (pp) | Discordant b/c | "
        "McNemar p | Agreement | Size (KB) | Compression | "
        + " | ".join(f"Latency {k.replace('_', ' ')} (ms, median)" for k in batch_keys)
        + " |"
    )
    lines = [header, "|---|" + "---:|" * (8 + len(batch_keys))]
    for name, v in study["variants"].items():
        lat = " | ".join(f"{v['latency'][k]['median_ms']:.2f}" for k in batch_keys)
        p = v["paired_vs_fp32"]
        lo, hi = p["delta_pp_ci95"]
        lines.append(
            f"| {name} | {v['accuracy']:.4f} | {v['accuracy_delta_pp']:+.2f} | "
            f"{lo:+.2f} to {hi:+.2f} | {p['only_fp32_correct']}/{p['only_variant_correct']} | "
            f"{p['mcnemar_exact_p']:.3g} | "
            f"{v['top1_agreement_with_fp32']:.4f} | {v['size_bytes'] / 1024:.1f} | "
            f"{v['compression_vs_fp32']:.2f}x | {lat} |"
        )
    lines += [
        "",
        "b = test images only fp32 classifies correctly, c = only the variant does. McNemar's "
        f"exact test and the paired bootstrap use the same {study['settings']['n_test']} test "
        "images for both models, accounting for correlated errors an unpaired accuracy CI omits.",
    ]
    return "\n".join(lines)
