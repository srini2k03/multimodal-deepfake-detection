import math

import numpy as np
import pytest

from deepfake_detector.evaluation.metrics import binary_report, confusion_matrix, equal_error_rate, roc_auc, roc_curve
from deepfake_detector.inference.fusion import aggregate_frames, late_fusion


def test_auc_perfect_random_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert math.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_auc_matches_bruteforce():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 2)  # include ties
    pos, neg = s[y == 1], s[y == 0]
    brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
    assert roc_auc(y, s) == pytest.approx(brute)
    fpr, tpr, _ = roc_curve(y, s)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    assert trapezoid(tpr, fpr) == pytest.approx(brute)


def test_eer():
    eer, _ = equal_error_rate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert eer == pytest.approx(0.0)
    eer, _ = equal_error_rate([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])
    assert 0.0 < eer < 1.0


def test_confusion_and_report():
    assert confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1]).tolist() == [[1, 1], [0, 2]]
    r = binary_report([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert r.accuracy == 0.75 and r.recall == 1.0
    assert r.precision == pytest.approx(2 / 3) and r.f1 == pytest.approx(0.8)
    assert r.to_dict()["confusion_matrix"] == [[1, 1], [0, 2]]
    with pytest.raises(ValueError):
        binary_report([0, 2], [0.1, 0.2])


def test_frame_aggregation():
    s = [0.1, 0.2, 0.9, 0.95]
    assert aggregate_frames(s, "mean") == pytest.approx(0.5375)
    assert aggregate_frames(s, "max") == 0.95
    assert aggregate_frames(s, "median") == pytest.approx(0.55)
    assert aggregate_frames(s, "topk_mean", 0.5) == pytest.approx(0.925)
    assert 0.5 < aggregate_frames(s, "confidence_weighted") < 0.95
    assert math.isnan(aggregate_frames([], "mean"))
    with pytest.raises(ValueError):
        aggregate_frames(s, "mode")


def test_late_fusion_weighted_logit_and_missing():
    w = {"visual": 0.7, "audio": 0.3}
    assert late_fusion({"visual": 0.8, "audio": 0.2}, w) == pytest.approx(0.62)
    assert late_fusion({"visual": 0.8, "audio": None}, w) == pytest.approx(0.8)
    assert late_fusion({"visual": 0.9, "audio": 0.9}, w, "logit") == pytest.approx(0.9)
    with pytest.raises(ValueError):
        late_fusion({"visual": None, "audio": float("nan")}, w)
