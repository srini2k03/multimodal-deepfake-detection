"""Pure-NumPy score aggregation (frames -> video) and late fusion (modalities -> final)."""

from __future__ import annotations

import math

import numpy as np

AGGREGATIONS = ("mean", "median", "max", "topk_mean", "confidence_weighted")


def _clip(p: np.ndarray | float, eps: float = 1e-6):
    return np.clip(p, eps, 1.0 - eps)


def logit(p):
    p = _clip(np.asarray(p, dtype=np.float64))
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def aggregate_frames(scores: list[float] | np.ndarray, method: str = "mean", k_fraction: float = 0.25) -> float:
    """Collapse per-frame P(fake) into a single video-level probability.

    * ``mean`` / ``median`` / ``max`` - as named.
    * ``topk_mean`` - mean of the top ``k_fraction`` most suspicious frames; robust when
      only part of the clip is manipulated.
    * ``confidence_weighted`` - frames far from 0.5 get more weight (|p-0.5|).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return float("nan")
    if method == "mean":
        return float(s.mean())
    if method == "median":
        return float(np.median(s))
    if method == "max":
        return float(s.max())
    if method == "topk_mean":
        k = max(1, math.ceil(s.size * k_fraction))
        return float(np.sort(s)[-k:].mean())
    if method == "confidence_weighted":
        w = np.abs(s - 0.5) + 1e-3
        return float((w * s).sum() / w.sum())
    raise ValueError(f"Unknown aggregation '{method}'. Options: {AGGREGATIONS}")


def late_fusion(
    modality_probs: dict[str, float | None],
    weights: dict[str, float],
    method: str = "weighted",
) -> float:
    """Fuse available modality probabilities; missing modalities (``None``/NaN) are skipped
    and the remaining weights are renormalised.

    ``weighted`` averages probabilities; ``logit`` averages log-odds (sharper when the
    modalities agree, and equivalent to a product-of-experts under independence).
    """
    avail = {
        k: float(v)
        for k, v in modality_probs.items()
        if v is not None and not math.isnan(float(v)) and weights.get(k, 0.0) > 0
    }
    if not avail:
        raise ValueError("No modality probabilities available for fusion")
    total = sum(weights[k] for k in avail)
    if method == "weighted":
        return float(sum(weights[k] * p for k, p in avail.items()) / total)
    if method == "logit":
        z = sum(weights[k] * float(logit(p)) for k, p in avail.items()) / total
        return float(sigmoid(z))
    raise ValueError("method must be 'weighted' or 'logit'")
