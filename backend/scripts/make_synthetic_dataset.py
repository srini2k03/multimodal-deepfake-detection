"""Generate a tiny SYNTHETIC dataset in the ``generic`` layout for pipeline smoke tests.

    python scripts/make_synthetic_dataset.py --out data/synthetic --videos 12

The clips are procedurally drawn shapes, not faces, and the "fake" class simply has a
blocky high-frequency patch. It exists only to exercise extraction -> training ->
evaluation end-to-end; numbers obtained on it say nothing about deepfake detection.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import cv2
import numpy as np


def draw_frame(rng: np.random.Generator, t: int, fake: bool, size: int = 160) -> np.ndarray:
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (40 + t % 30, 80, 120)
    cx = size // 2 + int(10 * math.sin(t / 5))
    cv2.ellipse(img, (cx, size // 2), (45, 60), 0, 0, 360, (150, 180, 220), -1)
    cv2.circle(img, (cx - 18, size // 2 - 15), 6, (30, 30, 30), -1)
    cv2.circle(img, (cx + 18, size // 2 - 15), 6, (30, 30, 30), -1)
    if fake:
        patch = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        patch = cv2.resize(patch, (40, 24), interpolation=cv2.INTER_NEAREST)
        y0, x0 = size // 2 + 15, cx - 20
        img[y0 : y0 + 24, x0 : x0 + 40] = cv2.addWeighted(img[y0 : y0 + 24, x0 : x0 + 40], 0.5, patch, 0.5, 0)
    return img


def write_video(path: Path, fake: bool, frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 160))
    for t in range(frames):
        writer.write(draw_frame(rng, t, fake))
    writer.release()


def write_tone(path: Path, fake: bool, seconds: float = 4.5, sr: int = 16000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    sig = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    if fake:
        sig += 0.1 * np.sign(np.sin(2 * np.pi * 1800 * t))
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--videos", type=int, default=12, help="videos per class")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    for label in ("real", "fake"):
        (args.out / "videos" / label).mkdir(parents=True, exist_ok=True)
        (args.out / "audio" / label).mkdir(parents=True, exist_ok=True)
        for i in range(args.videos):
            seed = i + (1000 if label == "fake" else 0)
            write_video(args.out / "videos" / label / f"clip_{i:03d}.avi", label == "fake", args.frames, seed)
            write_tone(args.out / "audio" / label / f"tone_{i:03d}.wav", label == "fake")
    print(f"Synthetic smoke-test data written to {args.out}")


if __name__ == "__main__":
    main()
