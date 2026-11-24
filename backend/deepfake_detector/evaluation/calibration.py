"""Post-hoc probability calibration (Platt scaling) fitted on a held-out split.

A detector's raw sigmoid output is rarely a calibrated probability, and the shift from the
validation to the test distribution (e.g. unseen attacks) makes it worse. Platt scaling fits
``p = sigmoid(a * z + b)`` on held-out logits ``z`` by Newton's method on the log-loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PlattScaler:
    a: float = 1.0
    b: float = 0.0

    def transform(self, logits) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))

    def to_dict(self) -> dict:
        return asdict(self)


def fit_platt(logits, labels, *, l2: float = 1e-6, iters: int = 100) -> PlattScaler:
    """Fit Platt scaling with Platt's (1999) smoothed targets to avoid overconfident fits."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    n_pos, n_neg = y.sum(), y.size - y.sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Platt scaling needs both classes")
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    x = np.stack([z, np.ones_like(z)], axis=1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (p - t) + l2 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return PlattScaler(a=float(w[0]), b=float(w[1]))
