"""Classification metrics implemented from first principles with NumPy.

Everything here is a pure function of arrays, which keeps it easy to unit-test against
scikit-learn and to reason about from their inputs and outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of examples with true class ``i`` predicted as ``j``."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    flat = y_true * num_classes + y_pred
    return np.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


@dataclass(frozen=True)
class PerClass:
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def per_class_prf(cm: np.ndarray) -> PerClass:
    """Precision, recall and F1 per class from a confusion matrix (0 where undefined)."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, actual)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return PerClass(precision, recall, f1, actual)


@dataclass(frozen=True)
class Calibration:
    ece: float
    bin_edges: np.ndarray
    bin_confidence: np.ndarray  # mean confidence per bin (NaN if empty)
    bin_accuracy: np.ndarray  # accuracy per bin (NaN if empty)
    bin_count: np.ndarray


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> Calibration:
    """Top-label ECE: the sample-weighted mean |accuracy - confidence| over confidence bins.

    Bins are right-closed, ``(lo, hi]``, except the first which also includes 0.
    """
    probs = np.asarray(probs, dtype=np.float64)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    count = np.bincount(bin_ids, minlength=n_bins)
    conf_sum = np.bincount(bin_ids, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(bin_ids, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_conf = conf_sum / count
        bin_acc = acc_sum / count
    ece = float(np.abs(acc_sum - conf_sum).sum() / len(confidence))
    return Calibration(ece, edges, bin_conf, bin_acc, count)


def bootstrap_ci(
    correct: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 250,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of a 0/1 ``correct`` vector.

    Resamples the test set with replacement ``n_boot`` times; the CI is the
    ``alpha/2`` and ``1 - alpha/2`` quantiles of the resampled accuracies.
    """
    correct = np.asarray(correct, dtype=np.float64)
    n = len(correct)
    if n == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        stats[start : start + size] = correct[idx].mean(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass(frozen=True)
class PairedComparison:
    """Two classifiers scored on the *same* examples (e.g. fp32 vs a quantized variant)."""

    n: int
    only_a_correct: int  # b: A right, B wrong
    only_b_correct: int  # c: A wrong, B right
    accuracy_delta: float  # acc(B) - acc(A)
    delta_ci95: tuple[float, float]  # paired bootstrap percentile CI of acc(B) - acc(A)
    mcnemar_p: float  # exact two-sided McNemar p-value


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant counts ``b`` and ``c``.

    Under H0 (both classifiers equally accurate) each discordant example is equally likely to
    favour either one, so ``min(b, c)`` follows Binomial(b + c, 0.5). Integer arithmetic keeps
    it exact for any test-set size.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_comparison(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 250,
) -> PairedComparison:
    """Compare two 0/1 correctness vectors computed on the same examples.

    The bootstrap uses the same resampled example indices for both correctness vectors,
    accounting for their correlated errors. Separate unpaired accuracy CIs would be too wide
    for the accuracy delta.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_vec = np.asarray(correct_b, dtype=bool)
    if a.shape != b_vec.shape or a.ndim != 1:
        raise ValueError("correct_a and correct_b must be 1-D and the same length")
    n = len(a)
    if n == 0:
        raise ValueError("cannot compare empty arrays")
    only_a = int((a & ~b_vec).sum())
    only_b = int((~a & b_vec).sum())
    diff = b_vec.astype(np.float64) - a.astype(np.float64)  # per-example delta in {-1, 0, 1}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        stats[start : start + size] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return PairedComparison(
        n=n,
        only_a_correct=only_a,
        only_b_correct=only_b,
        accuracy_delta=float(diff.mean()),
        delta_ci95=(float(lo), float(hi)),
        mcnemar_p=mcnemar_exact_p(only_a, only_b),
    )
