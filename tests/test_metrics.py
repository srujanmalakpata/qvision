import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sk_confusion
from sklearn.metrics import precision_recall_fscore_support

from qvision.evaluate import evaluation_report, report_to_markdown
from qvision.metrics import (
    bootstrap_ci,
    confusion_matrix,
    expected_calibration_error,
    mcnemar_exact_p,
    paired_comparison,
    per_class_prf,
)


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 400)
    y_pred = np.where(rng.random(400) < 0.7, y_true, rng.integers(0, 5, 400))
    return y_true, y_pred


def test_confusion_matrix_matches_sklearn(predictions):
    y_true, y_pred = predictions
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred, 5), sk_confusion(y_true, y_pred, labels=range(5))
    )


def test_per_class_prf_matches_sklearn(predictions):
    y_true, y_pred = predictions
    ours = per_class_prf(confusion_matrix(y_true, y_pred, 5))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(5), zero_division=0)
    np.testing.assert_allclose(ours.precision, p)
    np.testing.assert_allclose(ours.recall, r)
    np.testing.assert_allclose(ours.f1, f)
    np.testing.assert_array_equal(ours.support, s)


def test_prf_handles_never_predicted_class():
    cm = confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 1]), 3)
    prf = per_class_prf(cm)
    assert prf.precision[2] == 0 and prf.recall[2] == 0 and prf.f1[2] == 0


def test_ece_is_zero_for_perfectly_calibrated_bins():
    # 10 samples at confidence 0.8, 8 of them correct -> accuracy == confidence in that bin.
    probs = np.tile([0.8, 0.2], (10, 1))
    y = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probs, y).ece == pytest.approx(0.0, abs=1e-12)


def test_ece_of_overconfident_model():
    probs = np.tile([0.9, 0.1], (10, 1))
    y = np.array([0] * 5 + [1] * 5)  # 50% accurate at 90% confidence
    calib = expected_calibration_error(probs, y, n_bins=10)
    assert calib.ece == pytest.approx(0.4)
    assert calib.bin_count.sum() == 10


def test_ece_bins_include_confidence_one_and_low_values():
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    calib = expected_calibration_error(probs, np.array([0, 0]), n_bins=10)
    assert calib.bin_count[-1] == 1 and calib.bin_count.sum() == 2


def test_bootstrap_ci_brackets_accuracy_and_shrinks_with_n():
    rng = np.random.default_rng(1)
    small = rng.random(200) < 0.9
    large = rng.random(20000) < 0.9
    lo_s, hi_s = bootstrap_ci(small, n_boot=500, seed=0)
    lo_l, hi_l = bootstrap_ci(large, n_boot=500, seed=0)
    assert lo_s <= small.mean() <= hi_s
    assert lo_l <= large.mean() <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)
    # Normal approximation: half-width ~ 1.96 * sqrt(p(1-p)/n).
    expected = 1.96 * np.sqrt(0.9 * 0.1 / 20000)
    assert (hi_l - lo_l) / 2 == pytest.approx(expected, rel=0.2)


def test_bootstrap_ci_is_deterministic_for_a_seed():
    x = np.random.default_rng(2).random(300) < 0.5
    assert bootstrap_ci(x, n_boot=300, seed=5) == bootstrap_ci(x, n_boot=300, seed=5)


def test_evaluation_report_structure():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(10), size=100)
    y = rng.integers(0, 10, 100)
    report = evaluation_report(probs, y, n_boot=200)
    assert report["n"] == 100
    assert np.array(report["confusion_matrix"]).sum() == 100
    assert set(report["per_class"]) == set(report["class_names"])
    lo, hi = report["accuracy_ci95"]
    assert lo <= report["accuracy"] <= hi
    md = report_to_markdown(report)
    assert "| Ankle boot |" in md
    # Reliability table: one row per non-empty confidence bin, counts add up to n.
    bins = report["reliability"]
    assert sum(b["count"] for b in bins) == 100
    assert all(b["count"] > 0 and 0 <= b["accuracy"] <= 1 for b in bins)
    assert "Reliability" in md


@pytest.mark.parametrize("b, c", [(0, 0), (3, 0), (10, 4), (29, 0), (120, 91), (180, 151)])
def test_mcnemar_exact_matches_scipy_binomtest(b, c):
    from scipy.stats import binomtest

    expected = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    assert mcnemar_exact_p(b, c) == pytest.approx(expected, rel=1e-9)
    assert mcnemar_exact_p(b, c) == mcnemar_exact_p(c, b)


def test_paired_comparison_counts_discordant_pairs_and_brackets_delta():
    rng = np.random.default_rng(4)
    a = rng.random(5000) < 0.9
    b = a.copy()
    flip = rng.choice(5000, size=150, replace=False)
    b[flip] = ~b[flip]
    cmp = paired_comparison(a, b, n_boot=500, seed=0)
    assert cmp.only_a_correct + cmp.only_b_correct == 150
    assert cmp.accuracy_delta == pytest.approx(b.mean() - a.mean())
    lo, hi = cmp.delta_ci95
    assert lo <= cmp.accuracy_delta <= hi
    # Pairing makes the CI of the *difference* much narrower than an unpaired accuracy CI.
    lo_a, hi_a = bootstrap_ci(a, n_boot=500, seed=0)
    assert (hi - lo) < (hi_a - lo_a)


def test_paired_comparison_identical_models():
    a = np.random.default_rng(5).random(1000) < 0.8
    cmp = paired_comparison(a, a, n_boot=100)
    assert cmp.only_a_correct == cmp.only_b_correct == 0
    assert cmp.delta_ci95 == (0.0, 0.0) and cmp.mcnemar_p == 1.0
