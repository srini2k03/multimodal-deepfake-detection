"""Binary classification metrics implemented in NumPy (no scikit-learn dependency).

Convention: label 1 = fake (positive class), scores are P(fake).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be binary (0/1)")
    return y, s


def roc_curve(y_true, y_score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr, thresholds)`` with thresholds in decreasing order."""
    y, s = _as_arrays(y_true, y_score)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    tps, fps = np.r_[0, tps], np.r_[0, fps]
    thresholds = np.r_[np.inf, s[idx]]
    pos, neg = y.sum(), y.size - y.sum()
    tpr = tps / pos if pos else np.zeros_like(tps, dtype=float)
    fpr = fps / neg if neg else np.zeros_like(fps, dtype=float)
    return fpr.astype(float), tpr.astype(float), thresholds


def roc_auc(y_true, y_score) -> float:
    """Area under ROC via the Mann-Whitney U statistic (ties get half credit)."""
    y, s = _as_arrays(y_true, y_score)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    combined = np.r_[pos, neg][order]
    # average ranks for ties
    i = 0
    while i < combined.size:
        j = i
        while j + 1 < combined.size and combined[j + 1] == combined[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def equal_error_rate(y_true, y_score) -> tuple[float, float]:
    """Return ``(eer, threshold)`` where FPR ~= FNR, linearly interpolated on the ROC."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    idx = int(np.where(diff >= 0)[0][0]) if (diff >= 0).any() else len(diff) - 1
    if idx == 0:
        return float(fpr[0]), float(thr[1] if len(thr) > 1 else 0.5)
    # interpolate between idx-1 and idx
    d0, d1 = diff[idx - 1], diff[idx]
    t = d0 / (d0 - d1) if d0 != d1 else 0.0
    eer = fpr[idx - 1] + t * (fpr[idx] - fpr[idx - 1])
    threshold = thr[idx] if np.isfinite(thr[idx]) else thr[min(idx + 1, len(thr) - 1)]
    return float(eer), float(threshold)


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """2x2 matrix ``[[TN, FP], [FN, TP]]``."""
    y, p = _as_arrays(y_true, y_pred)
    p = p.astype(np.int64)
    return np.array(
        [
            [int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())],
            [int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())],
        ]
    )


@dataclass
class BinaryReport:
    n: int
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float
    eer_threshold: float
    threshold: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def binary_report(y_true, y_score, threshold: float = 0.5) -> BinaryReport:
    y, s = _as_arrays(y_true, y_score)
    pred = (s >= threshold).astype(np.int64)
    cm = confusion_matrix(y, pred)
    _tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    has_both = 0 < y.sum() < y.size
    eer, eer_thr = equal_error_rate(y, s) if has_both else (float("nan"), float("nan"))
    return BinaryReport(
        n=int(y.size),
        auc=roc_auc(y, s),
        accuracy=float((pred == y).mean()) if y.size else float("nan"),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        eer=eer,
        eer_threshold=eer_thr,
        threshold=float(threshold),
        confusion_matrix=cm.tolist(),
    )
