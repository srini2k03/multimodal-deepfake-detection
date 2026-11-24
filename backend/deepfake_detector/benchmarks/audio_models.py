"""Audio countermeasure systems benchmarked on ASVspoof 2019 LA.

* ``silence_lr``: logistic regression on duration and leading/trailing silence only. It never
  looks at the speech, so whatever it achieves measures a dataset shortcut, not detection.
* ``logmel_lr``: logistic regression on per-band mean and std of the log-mel spectrogram.
* ``cnn`` / ``cnn_trim``: the repository's log-mel CNN, trained on untrimmed or
  silence-trimmed audio.

All systems output a logit for "spoof" (label 1). ASVspoof metrics use ``-logit`` as the
bona fide score.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import AudioConfig, AudioTrainConfig
from ..evaluation.metrics import eer_from_scores
from .asvspoof import FeatureSplit, iter_batches

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- linear baselines
@dataclass
class LogisticRegression:
    """L2-regularised logistic regression fitted with Newton's method on standardised inputs."""

    l2: float = 1e-3
    balanced: bool = True
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, iters: int = 50) -> LogisticRegression:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_, self.scale_ = x.mean(axis=0), x.std(axis=0) + 1e-8
        z = np.c_[(x - self.mean_) / self.scale_, np.ones(len(x))]
        sw = np.ones(len(y))
        if self.balanced:
            pos = y.mean()
            sw = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
        w = np.zeros(z.shape[1])
        reg = self.l2 * len(y) * np.r_[np.ones(z.shape[1] - 1), 0.0]
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
            grad = z.T @ (sw * (p - y)) + reg * w
            hess = (z * (sw * p * (1 - p))[:, None]).T @ z + np.diag(reg + 1e-9)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef_ = w
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        z = (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_
        return z @ self.coef_[:-1] + self.coef_[-1]


def silence_features(fs: FeatureSplit) -> np.ndarray:
    """Duration and silence only: no spectral information at all."""
    cols = [fs.column(k) for k in ("duration_s", "lead_silence_s", "trail_silence_s")]
    return np.stack([*cols, *(np.log1p(c) for c in cols)], axis=1)


def logmel_stats_features(features: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Per-band mean and standard deviation over time: ``(N, 2 * n_mels)``."""
    out = [np.c_[b.mean(axis=2), b.std(axis=2)] for b in iter_batches(features, batch)]
    return np.concatenate(out) if out else np.zeros((0, 0))


# --------------------------------------------------------------------------- CNN
@dataclass
class TrainResult:
    model: object
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_eer: float = float("nan")
    seconds: float = 0.0


def logit_model(model):
    import keras

    return keras.Model(model.input, model.get_layer("logit").output, name=f"{model.name}_logit")


def predict_logits(model, features: np.ndarray, batch: int = 512) -> np.ndarray:
    """Spoof logits for ``(N, n_mels, frames)`` features (memmap-friendly, float16 ok)."""
    lm = logit_model(model)
    out = [np.asarray(lm(b[..., None], training=False)).reshape(-1) for b in iter_batches(features, batch)]
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def spec_augment(x: np.ndarray, rng: np.random.Generator, max_f: int = 8, max_t: int = 40) -> np.ndarray:
    """One random frequency mask and one random time mask per example (in place, zero fill)."""
    n, n_mels, n_frames = x.shape
    fw = rng.integers(0, max_f + 1, n)
    tw = rng.integers(0, max_t + 1, n)
    f0 = rng.integers(0, np.maximum(1, n_mels - fw))
    t0 = rng.integers(0, np.maximum(1, n_frames - tw))
    for i in range(n):
        x[i, f0[i] : f0[i] + fw[i], :] = 0.0
        x[i, :, t0[i] : t0[i] + tw[i]] = 0.0
    return x


def val_subset(labels: np.ndarray, max_spoof: int, seed: int) -> np.ndarray:
    """All bona fide plus a seeded sample of at most ``max_spoof`` spoofs (model selection only)."""
    rng = np.random.default_rng(seed)
    bona = np.where(labels == 0)[0]
    spoof = np.where(labels == 1)[0]
    if spoof.size > max_spoof:
        spoof = rng.choice(spoof, max_spoof, replace=False)
    return np.sort(np.r_[bona, spoof])


def train_cnn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    audio_cfg: AudioConfig,
    tcfg: AudioTrainConfig,
    run_dir: Path | None = None,
) -> TrainResult:
    """Train the log-mel CNN with class-balanced epochs; select the epoch with the lowest dev EER."""
    import keras
    import tensorflow as tf

    from ..models.audio import build_audio_cnn

    keras.utils.set_random_seed(tcfg.seed)
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(tcfg.seed)
    model = build_audio_cnn(audio_cfg.n_mels, train_x.shape[2], filters=tuple(audio_cfg.cnn_filters))
    lr = tcfg.learning_rate
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy")

    bona = np.where(train_y == 0)[0]
    spoof = np.where(train_y == 1)[0]
    n_spoof = min(spoof.size, round(tcfg.spoof_per_bonafide * bona.size))
    vidx = val_subset(val_y, tcfg.val_max_spoof, tcfg.seed)
    vx, vy = np.asarray(val_x[vidx], dtype=np.float16), val_y[vidx]
    result = TrainResult(model=model)
    best_weights, best_eer, since_best, since_decay = None, np.inf, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.perf_counter()
        idx = np.sort(np.r_[bona, rng.choice(spoof, n_spoof, replace=False)])
        x = np.asarray(train_x[idx], dtype=np.float32)
        y = train_y[idx].astype(np.float32)
        if tcfg.spec_augment:
            spec_augment(x, rng)
        order = rng.permutation(len(idx))
        hist = model.fit(x[order, ..., None], y[order], batch_size=tcfg.batch_size, epochs=1, verbose=0)
        del x
        logits = predict_logits(model, vx, batch=512)
        eer, _ = eer_from_scores(-logits[vy == 0], -logits[vy == 1])
        row = {
            "epoch": epoch,
            "loss": float(hist.history["loss"][-1]),
            "val_eer": eer,
            "lr": lr,
            "seconds": round(time.perf_counter() - t0, 1),
            "n_train": len(idx),
        }
        result.history.append(row)
        logger.info("epoch", extra=row)
        if eer < best_eer - 1e-5:
            best_eer, best_weights, since_best, since_decay = eer, model.get_weights(), 0, 0
            result.best_epoch = epoch
        else:
            since_best += 1
            since_decay += 1
            if since_best >= tcfg.early_stopping_patience:
                logger.info("early stopping", extra={"epoch": epoch, "best_epoch": result.best_epoch})
                break
            if since_decay >= tcfg.lr_decay_patience:
                lr *= tcfg.lr_decay_factor
                model.optimizer.learning_rate.assign(lr)
                since_decay = 0
    if best_weights is not None:
        model.set_weights(best_weights)
    result.best_val_eer = float(best_eer)
    result.seconds = round(time.perf_counter() - t_start, 1)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        model.save(run_dir / "model.keras")
    return result
