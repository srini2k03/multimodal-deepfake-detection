"""Image perturbations shared by training augmentation and the robustness benchmark (uint8 RGB)."""

from __future__ import annotations

import cv2
import numpy as np


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:  # pragma: no cover
        raise ValueError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def rescale(img: np.ndarray, factor: float) -> np.ndarray:
    """Downscale by ``factor`` then back up: removes high-frequency detail like a low-res upload."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def center_crop(img: np.ndarray, fraction: float) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * fraction), int(w * fraction)
    y, x = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y : y + ch, x : x + cw], (w, h), interpolation=cv2.INTER_LINEAR)


ROBUSTNESS_CONDITIONS = [
    "clean",
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "jpeg_q30",
    "downscale_0.5",
    "downscale_0.25",
    "blur_sigma1",
    "blur_sigma2",
    "noise_sigma5",
    "noise_sigma15",
    "brightness_0.7",
    "brightness_1.3",
    "crop_0.8",
]


def apply_condition(img: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "clean":
        return img
    kind, _, value = condition.partition("_")
    if kind == "jpeg":
        return jpeg(img, int(value.removeprefix("q")))
    if kind == "downscale":
        return rescale(img, float(value))
    if kind == "blur":
        return blur(img, float(value.removeprefix("sigma")))
    if kind == "noise":
        return noise(img, float(value.removeprefix("sigma")), rng)
    if kind == "brightness":
        return brightness(img, float(value))
    if kind == "crop":
        return center_crop(img, float(value))
    raise ValueError(f"Unknown condition '{condition}'")


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Training augmentation applied identically to both classes.

    Flips and mild photometric jitter keep blending artefacts intact; random JPEG
    re-compression and down/up-scaling teach invariance to the processing that real
    uploads receive (and blunt the "was this file ever JPEG-compressed" shortcut).
    """
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = jpeg(img, int(rng.integers(60, 96)))
    if rng.random() < 0.25:
        img = rescale(img, float(rng.uniform(0.5, 1.0)))
    if rng.random() < 0.3:
        img = brightness(img, float(rng.uniform(0.85, 1.15)))
    return np.ascontiguousarray(img)


def to_model_input(batch_uint8: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 in [-1, 1] (the repository-wide model input convention)."""
    return batch_uint8.astype(np.float32) / 127.5 - 1.0
