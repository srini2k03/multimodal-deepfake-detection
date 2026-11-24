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


def average_precision(y_true, y_score) -> float:
    """Area under the precision-recall curve (step-wise average precision, positives = fake)."""
    y, s = _as_arrays(y_true, y_score)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    idx = np.r_[np.where(np.diff(s))[0], y.size - 1]  # last index of each distinct score
    tps = np.cumsum(y)[idx]
    precision = tps / (idx + 1)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier_score(y_true, y_prob) -> float:
    y, p = _as_arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """Equal-width probability bins with mean confidence, observed fake rate and count."""
    y, p = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = which == b
        n = int(mask.sum())
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "mean_prob": float(p[mask].mean()) if n else None,
                "fake_rate": float(y[mask].mean()) if n else None,
            }
        )
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) * |fake_rate_b - mean_prob_b| over equal-width bins."""
    y, _ = _as_arrays(y_true, y_prob)
    if y.size == 0:
        return float("nan")
    bins = reliability_bins(y_true, y_prob, n_bins)
    return float(sum(b["n"] / y.size * abs(b["fake_rate"] - b["mean_prob"]) for b in bins if b["n"]))


# --------------------------------------------------------------------------- ASVspoof conventions
# ASVspoof scores countermeasure (CM) outputs as "higher = more bona fide". The helpers below
# follow the official ASVspoof 2019 evaluation package (eval_metrics.py) so numbers are comparable.
ASVSPOOF2019_COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,
    "Pnon": (1 - 0.05) * 0.01,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}


def det_curve(target_scores, nontarget_scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(frr, far, thresholds)`` exactly as ASVspoof's ``compute_det_curve``."""
    tar = np.asarray(target_scores, dtype=np.float64).ravel()
    non = np.asarray(nontarget_scores, dtype=np.float64).ravel()
    all_scores = np.concatenate((tar, non))
    labels = np.concatenate((np.ones(tar.size), np.zeros(non.size)))
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_sums = np.cumsum(labels)
    non_sums = non.size - (np.arange(1, all_scores.size + 1) - tar_sums)
    frr = np.concatenate(([0.0], tar_sums / tar.size))
    far = np.concatenate(([1.0], non_sums / non.size))
    thresholds = np.concatenate(([all_scores[indices[0]] - 0.001], all_scores[indices]))
    return frr, far, thresholds


def eer_from_scores(target_scores, nontarget_scores) -> tuple[float, float]:
    """``(eer, threshold)`` with targets scoring higher (ASVspoof ``compute_eer``)."""
    frr, far, thr = det_curve(target_scores, nontarget_scores)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2.0), float(thr[i])


def min_tdcf(bonafide_cm, spoof_cm, asv_target, asv_nontarget, asv_spoof, cost_model: dict | None = None) -> dict:
    """Minimum normalised tandem detection cost (ASVspoof 2019 t-DCF, Kinnunen et al. 2018/2020).

    ``*_cm`` are countermeasure scores (higher = bona fide); ``asv_*`` are the organisers'
    ASV scores for target, non-target and spoofed trials. The ASV threshold is its EER point.
    """
    cm = cost_model or ASVSPOOF2019_COST_MODEL
    tar, non, spf = (np.asarray(a, dtype=np.float64) for a in (asv_target, asv_nontarget, asv_spoof))
    _, asv_thr = eer_from_scores(tar, non)
    pfa_asv = float(np.mean(non >= asv_thr))
    pmiss_asv = float(np.mean(tar < asv_thr))
    pmiss_spoof_asv = float(np.mean(spf < asv_thr))
    pmiss_cm, pfa_cm, thr = det_curve(bonafide_cm, spoof_cm)
    c1 = cm["Ptar"] * (cm["Cmiss_cm"] - cm["Cmiss_asv"] * pmiss_asv) - cm["Pnon"] * cm["Cfa_asv"] * pfa_asv
    c2 = cm["Cfa_cm"] * cm["Pspoof"] * (1 - pmiss_spoof_asv)
    if c1 < 0 or c2 < 0:
        raise ValueError("t-DCF weights are negative; the ASV scores look wrong")
    tdcf = (c1 * pmiss_cm + c2 * pfa_cm) / min(c1, c2)
    i = int(np.argmin(tdcf))
    return {
        "min_tdcf": float(tdcf[i]),
        "threshold": float(thr[i]),
        "asv": {"eer_threshold": asv_thr, "pfa": pfa_asv, "pmiss": pmiss_asv, "pmiss_spoof": pmiss_spoof_asv},
    }


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
    pr_auc: float = float("nan")
    fpr: float = float("nan")  # real media flagged as fake / all real
    fnr: float = float("nan")  # fakes missed / all fakes
    n_real: int = 0
    n_fake: int = 0

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
    tn = int(_tn)
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
        pr_auc=average_precision(y, s) if has_both else float("nan"),
        fpr=float(fp / (fp + tn)) if fp + tn else float("nan"),
        fnr=float(fn / (fn + tp)) if fn + tp else float("nan"),
        n_real=int(tn + fp),
        n_fake=int(fn + tp),
    )
