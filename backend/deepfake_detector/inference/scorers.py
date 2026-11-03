"""Model wrappers with a uniform interface for trained, random-weight and placeholder scorers."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameScorer(Protocol):
    name: str
    kind: str  # "trained", "random_weights" or "placeholder"
    input_size: int

    def predict(self, batch: np.ndarray) -> np.ndarray: ...

    def heatmap(self, image: np.ndarray, image_bgr: np.ndarray) -> np.ndarray | None: ...


class AudioScorer(Protocol):
    kind: str

    def predict(self, clips: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- Keras-backed
class KerasFrameScorer:
    def __init__(self, name: str, model, kind: str) -> None:
        self.name = name
        self.model = model
        self.kind = kind
        self.input_size = int(model.input_shape[1])

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if len(batch) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self.model.predict(batch, batch_size=16, verbose=0), dtype=np.float32).reshape(-1)

    def heatmap(self, image: np.ndarray, image_bgr: np.ndarray) -> np.ndarray | None:
        from ..explain.gradcam import compute_gradcam

        try:
            return compute_gradcam(self.model, image)
        except Exception:
            logger.exception("grad-cam failed", extra={"model": self.name})
            return None


class KerasTemporalScorer:
    def __init__(self, model, kind: str = "trained") -> None:
        self.model = model
        self.kind = kind
        self.seq_len = int(model.input_shape[1])
        self.input_size = int(model.input_shape[2])

    def predict_sequence(self, frames: np.ndarray) -> float:
        if len(frames) == 0:
            return float("nan")
        idx = np.linspace(0, len(frames) - 1, self.seq_len).astype(int)
        seq = frames[idx][None, ...]
        return float(np.asarray(self.model.predict(seq, verbose=0)).reshape(-1)[0])


class KerasAudioScorer:
    def __init__(self, model, kind: str) -> None:
        self.model = model
        self.kind = kind
        self.n_frames = int(model.input_shape[2])

    def predict(self, clips: np.ndarray) -> np.ndarray:
        if len(clips) == 0:
            return np.zeros((0,), dtype=np.float32)
        t = clips.shape[2]
        if t != self.n_frames:  # pad / crop the time axis to the trained length
            clips = (
                clips[:, :, : self.n_frames]
                if t > self.n_frames
                else np.pad(clips, ((0, 0), (0, 0), (0, self.n_frames - t), (0, 0)))
            )
        return np.asarray(self.model.predict(clips, verbose=0), dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- placeholders
def _stable_unit(data: bytes) -> float:
    return int(hashlib.sha1(data).hexdigest()[:8], 16) / float(0xFFFFFFFF)


class PlaceholderFrameScorer:
    """Deterministic, model-free stand-in used when TensorFlow is unavailable.

    Produces stable pseudo-scores from a hash of the down-sampled frame so the UI can be
    exercised end-to-end. These numbers carry NO forensic meaning.
    """

    kind = "placeholder"

    def __init__(self, name: str, input_size: int = 299) -> None:
        self.name = name
        self.input_size = input_size

    def predict(self, batch: np.ndarray) -> np.ndarray:
        out = []
        for img in batch:
            small = cv2.resize(((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8), (8, 8))
            out.append(0.2 + 0.6 * _stable_unit(small.tobytes() + self.name.encode()))
        return np.asarray(out, dtype=np.float32)

    def heatmap(self, image: np.ndarray, image_bgr: np.ndarray) -> np.ndarray | None:
        from ..explain.gradcam import pseudo_saliency

        return pseudo_saliency(image_bgr)


class PlaceholderAudioScorer:
    kind = "placeholder"

    def predict(self, clips: np.ndarray) -> np.ndarray:
        return np.asarray(
            [0.2 + 0.6 * _stable_unit(np.round(c, 1).astype(np.float16).tobytes()) for c in clips], dtype=np.float32
        )


# --------------------------------------------------------------------------- loading
def load_keras_model(path: Path):
    import keras

    from ..models import ensemble, temporal  # noqa: F401 - registers custom layers

    return keras.models.load_model(path, compile=False)
