"""PR-AUC, calibration metrics, Platt scaling and the extended binary report."""

import numpy as np
import pytest

from deepfake_detector.evaluation.calibration import fit_platt
from deepfake_detector.evaluation.metrics import (
    average_precision,
    binary_report,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


def test_average_precision_matches_bruteforce():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 2)
    # brute force: mean precision at each positive's rank, ties grouped (sklearn's step-wise definition)
    ap = 0.0
    for t in np.unique(s)[::-1]:
        sel = s >= t
        tp = (y[sel] == 1).sum()
        new_pos = ((s == t) & (y == 1)).sum()
        ap += new_pos / y.sum() * tp / sel.sum()
    assert average_precision(y, s) == pytest.approx(ap)
    assert average_precision([0, 1, 1], [0.1, 0.8, 0.9]) == 1.0


def test_brier_ece_and_bins():
    y = np.array([0, 0, 1, 1])
    assert brier_score(y, [0, 0, 1, 1]) == 0.0
    assert brier_score(y, [1, 1, 0, 0]) == 1.0
    assert expected_calibration_error(y, [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0)
    bins = reliability_bins(y, [0.05, 0.15, 0.85, 0.95], n_bins=10)
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 4
    assert expected_calibration_error([1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_platt_recovers_scale_and_improves_calibration():
    rng = np.random.default_rng(1)
    z_true = rng.normal(0, 2, 20000)
    y = (rng.random(z_true.size) < 1 / (1 + np.exp(-z_true))).astype(int)
    overconfident = 3.0 * z_true + 1.0  # a model whose logits are 3x too large and shifted
    platt = fit_platt(overconfident, y)
    assert platt.a == pytest.approx(1 / 3, rel=0.05) and platt.b == pytest.approx(-1 / 3, abs=0.1)
    raw = 1 / (1 + np.exp(-overconfident))
    assert expected_calibration_error(y, platt.transform(overconfident)) < expected_calibration_error(y, raw)
    with pytest.raises(ValueError):
        fit_platt([0.1, 0.2], [1, 1])


def test_binary_report_rates():
    r = binary_report([0, 0, 0, 1, 1], [0.9, 0.1, 0.2, 0.8, 0.3]).to_dict()
    assert r["fpr"] == pytest.approx(1 / 3) and r["fnr"] == pytest.approx(1 / 2)
    assert r["n_real"] == 3 and r["n_fake"] == 2 and 0 < r["pr_auc"] <= 1
