"""Landmark-free face alignment and cropping.

Alignment strategy (no facial-landmark model required):

1. Expand the detector box by a relative ``margin`` and make it square so the crop keeps
   the forehead/chin/cheeks where blending artifacts typically live.
2. If ``align`` is enabled, run OpenCV's eye cascade inside the upper half of the face.
   When two plausible eyes are found, rotate the frame about the face centre so the
   inter-ocular line is horizontal (roll correction).
3. Crop and resize to a fixed ``crop_size`` (299 for Xception/InceptionV3).

If eye detection fails we gracefully fall back to bounding-box normalisation only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .face_detector import FaceBox

_EYE_CASCADE = None


def _eye_cascade():
    global _EYE_CASCADE
    if _EYE_CASCADE is None and hasattr(cv2, "CascadeClassifier"):
        _EYE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
    return _EYE_CASCADE


@dataclass(frozen=True)
class AlignedFace:
    crop_bgr: np.ndarray
    box: FaceBox
    angle_deg: float
    aligned: bool


def square_box(box: FaceBox, margin: float, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Return an expanded square box ``(x1, y1, x2, y2)`` clipped to the image."""
    h_img, w_img = image_shape[:2]
    cx, cy = box.center
    side = max(box.w, box.h) * (1.0 + 2.0 * margin)
    side = min(side, max(h_img, w_img))
    half = side / 2.0
    x1, y1 = round(cx - half), round(cy - half)
    x2, y2 = round(cx + half), round(cy + half)
    # Shift the box back inside the frame instead of shrinking it when possible.
    if x1 < 0:
        x2, x1 = x2 - x1, 0
    if y1 < 0:
        y2, y1 = y2 - y1, 0
    if x2 > w_img:
        x1, x2 = max(0, x1 - (x2 - w_img)), w_img
    if y2 > h_img:
        y1, y2 = max(0, y1 - (y2 - h_img)), h_img
    return x1, y1, x2, y2


def estimate_roll(image_bgr: np.ndarray, box: FaceBox) -> float | None:
    """Estimate in-plane rotation (degrees) from the two most prominent eyes, if found."""
    x, y, w, h = box.x, box.y, box.w, box.h
    upper = image_bgr[max(0, y) : max(0, y) + h // 2 + 1, max(0, x) : max(0, x) + w]
    if upper.size == 0:
        return None
    gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY) if upper.ndim == 3 else upper
    cascade = _eye_cascade()
    if cascade is None:  # no cascade support in this OpenCV build: skip roll correction
        return None
    min_eye = max(8, w // 10)
    eyes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_eye, min_eye))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = sorted(eyes, key=lambda e: e[0])
    c1 = (ex1 + ew1 / 2.0, ey1 + eh1 / 2.0)
    c2 = (ex2 + ew2 / 2.0, ey2 + eh2 / 2.0)
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    if dx < w * 0.15:  # eyes too close together: probably a false positive
        return None
    angle = math.degrees(math.atan2(dy, dx))
    return angle if abs(angle) <= 30.0 else None


def rotate_image(image: np.ndarray, angle_deg: float, center: tuple[float, float]) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def align_and_crop(
    image_bgr: np.ndarray,
    box: FaceBox,
    crop_size: int = 299,
    margin: float = 0.3,
    align: bool = True,
) -> AlignedFace:
    """Rotate (optionally) and crop a square, margin-padded face patch of ``crop_size``."""
    angle = estimate_roll(image_bgr, box) if align else None
    source = image_bgr
    if angle is not None and abs(angle) > 1.0:
        source = rotate_image(image_bgr, angle, box.center)
    x1, y1, x2, y2 = square_box(box, margin, source.shape)
    crop = source[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return AlignedFace(crop_bgr=crop, box=box, angle_deg=float(angle or 0.0), aligned=angle is not None)


def center_crop(image_bgr: np.ndarray, crop_size: int = 299) -> np.ndarray:
    """Fallback when no face is detected: central square crop."""
    h, w = image_bgr.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return cv2.resize(image_bgr[y0 : y0 + side, x0 : x0 + side], (crop_size, crop_size), interpolation=cv2.INTER_AREA)
