"""Face detection backends built on OpenCV.

Two detectors are provided behind a common interface:

* :class:`DnnFaceDetector` - OpenCV's ResNet-10 SSD (``res10_300x300``), accurate and
  reasonably fast on CPU. Model files are fetched by ``scripts/download_face_models.sh``.
* :class:`HaarFaceDetector` - classic Viola-Jones cascade bundled with OpenCV. No
  downloads required, used as the automatic fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import FaceConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned face box in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


class FaceDetector:
    """Base class: ``detect`` returns faces sorted by descending area."""

    name = "base"

    def __init__(self, min_face_size: int = 40) -> None:
        self.min_face_size = min_face_size

    def _detect(self, image_bgr: np.ndarray) -> list[FaceBox]:  # pragma: no cover - abstract
        raise NotImplementedError

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        faces = [f for f in self._detect(image_bgr) if f.w >= self.min_face_size and f.h >= self.min_face_size]
        return sorted(faces, key=lambda f: (f.area, f.score), reverse=True)

    def largest(self, image_bgr: np.ndarray) -> FaceBox | None:
        faces = self.detect(image_bgr)
        return faces[0] if faces else None


class HaarFaceDetector(FaceDetector):
    name = "haar"

    def __init__(self, min_face_size: int = 40, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        super().__init__(min_face_size)
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This OpenCV build has no Haar cascade support (removed in OpenCV 5). "
                "Install opencv-python-headless<5 or use face.detector=dnn."
            )
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():  # pragma: no cover - broken OpenCV install
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def _detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return [FaceBox(int(x), int(y), int(w), int(h), 1.0) for (x, y, w, h) in rects]


class DnnFaceDetector(FaceDetector):
    name = "dnn"

    def __init__(self, prototxt: str, weights: str, confidence: float = 0.6, min_face_size: int = 40) -> None:
        super().__init__(min_face_size)
        if not (Path(prototxt).exists() and Path(weights).exists()):
            raise FileNotFoundError(
                f"DNN face model files not found ({prototxt}, {weights}). "
                "Run scripts/download_face_models.sh or set face.detector=haar."
            )
        self._net = cv2.dnn.readNetFromCaffe(prototxt, weights)
        self.confidence = confidence

    def _detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self._net.setInput(blob)
        detections = self._net.forward()
        faces: list[FaceBox] = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.confidence:
                continue
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append(FaceBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), score))
        return faces


def build_face_detector(cfg: FaceConfig) -> FaceDetector:
    """Instantiate the configured detector, falling back to Haar if DNN files are missing."""
    if cfg.detector == "dnn":
        try:
            return DnnFaceDetector(cfg.dnn_prototxt, cfg.dnn_weights, cfg.confidence, cfg.min_face_size)
        except FileNotFoundError as exc:
            logger.warning("Falling back to Haar face detector: %s", exc)
    return HaarFaceDetector(min_face_size=cfg.min_face_size)
