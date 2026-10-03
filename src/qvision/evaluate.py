"""Run a model over a dataset and assemble a full evaluation report."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qvision.data import CLASS_NAMES
from qvision.metrics import (
    bootstrap_ci,
    confusion_matrix,
    expected_calibration_error,
    per_class_prf,
)


@torch.inference_mode()
def predict_proba(model: nn.Module, dataset: TensorDataset, batch_size: int = 1000) -> np.ndarray:
    """Softmax probabilities, shape ``[N, num_classes]``."""
    model.eval()
    out = [torch.softmax(model(x), dim=1) for x, _ in DataLoader(dataset, batch_size=batch_size)]
    return torch.cat(out).numpy()


def labels_of(dataset: TensorDataset) -> np.ndarray:
    return dataset.tensors[1].numpy()


def evaluation_report(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_names: tuple[str, ...] = CLASS_NAMES,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Accuracy (+ bootstrap 95% CI), macro-F1, per-class P/R/F1, confusion matrix, ECE and the
    reliability table (per confidence bin: count, mean confidence, accuracy)."""
    y_pred = probs.argmax(axis=1)
    correct = y_pred == y_true
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    prf = per_class_prf(cm)
    calib = expected_calibration_error(probs, y_true)
    lo, hi = bootstrap_ci(correct, n_boot=n_boot, seed=seed)
    return {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": [lo, hi],
        "bootstrap_resamples": n_boot,
        "macro_f1": float(prf.f1.mean()),
        "ece_15_bins": calib.ece,
        "reliability": [
            {
                "bin": f"({calib.bin_edges[i]:.3f}, {calib.bin_edges[i + 1]:.3f}]",
                "count": int(calib.bin_count[i]),
                "mean_confidence": float(calib.bin_confidence[i]),
                "accuracy": float(calib.bin_accuracy[i]),
            }
            for i in range(len(calib.bin_count))
            if calib.bin_count[i] > 0
        ],
        "per_class": {
            name: {
                "precision": float(prf.precision[i]),
                "recall": float(prf.recall[i]),
                "f1": float(prf.f1[i]),
                "support": int(prf.support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def evaluate_model(model: nn.Module, dataset: TensorDataset, **kwargs: Any) -> dict[str, Any]:
    return evaluation_report(predict_proba(model, dataset), labels_of(dataset), **kwargs)


def report_to_markdown(report: dict[str, Any]) -> str:
    """Render the per-class table and confusion matrix as Markdown."""
    lo, hi = report["accuracy_ci95"]
    lines = [
        f"Accuracy: **{report['accuracy']:.4f}** (95% bootstrap CI {lo:.4f}-{hi:.4f}, "
        f"n={report['n']}), macro-F1 {report['macro_f1']:.4f}, ECE {report['ece_15_bins']:.4f}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in report["per_class"].items():
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['support']} |"
        )
    names = report["class_names"]
    short = [n.split("/")[0][:7] for n in names]
    lines += ["", "Confusion matrix (rows = true, columns = predicted):", ""]
    lines.append("| true \\ pred | " + " | ".join(short) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for name, row in zip(short, report["confusion_matrix"], strict=True):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        "Reliability (15 equal-width confidence bins, empty bins omitted):",
        "",
        "| Confidence bin | Count | Mean confidence | Accuracy | Gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in report["reliability"]:
        gap = b["accuracy"] - b["mean_confidence"]
        lines.append(
            f"| {b['bin']} | {b['count']} | {b['mean_confidence']:.3f} | {b['accuracy']:.3f} "
            f"| {gap:+.3f} |"
        )
    return "\n".join(lines)
