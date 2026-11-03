from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from deepfake_detector.config import Config

HAS_TF = importlib.util.find_spec("tensorflow") is not None


def pytest_collection_modifyitems(config, items):
    if HAS_TF:
        return
    skip = pytest.mark.skip(reason="TensorFlow not installed")
    for item in items:
        if "tf" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.data.image_size = 96
    c.face.crop_size = 96
    c.api.weights_dir = str(tmp_path / "weights")
    c.api.upload_dir = str(tmp_path / "uploads")
    c.api.frames_per_video = 4
    c.api.demo_backend = "numpy"
    c.audio.clip_seconds = 1.0
    return c.validate()


def synthetic_face(size: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 90, np.uint8)
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 4, size // 3), 0, 0, 360, (150, 180, 220), -1)
    return img


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "face.png"
    cv2.imwrite(str(path), synthetic_face())
    return path


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (128, 128))
    for i in range(20):
        frame = synthetic_face(128, seed=i)
        cv2.putText(frame, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    from deepfake_detector.preprocessing.audio import write_wav

    sr = 16000
    t = np.arange(int(2.5 * sr)) / sr
    path = tmp_path / "tone.wav"
    write_wav(path, 0.4 * np.sin(2 * np.pi * 440 * t), sr)
    return path
