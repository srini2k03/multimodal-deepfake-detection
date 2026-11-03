"""Frame sampling from video files with OpenCV."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class SampledFrame:
    index: int
    timestamp_s: float
    image_bgr: np.ndarray


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if count <= 0:  # some containers do not report frame counts
            count = 0
            while cap.grab():
                count += 1
    finally:
        cap.release()
    return VideoInfo(count, fps if fps > 0 else 25.0, width, height)


def sample_indices(frame_count: int, num_frames: int, strategy: str = "uniform", stride: int = 10) -> list[int]:
    """Choose frame indices to decode.

    ``uniform`` spreads ``num_frames`` evenly over the video (centre of each segment);
    ``stride`` takes every ``stride``-th frame, capped at ``num_frames``.
    """
    if frame_count <= 0 or num_frames <= 0:
        return []
    if strategy == "stride":
        return list(range(0, frame_count, max(1, stride)))[:num_frames]
    if strategy != "uniform":
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if num_frames >= frame_count:
        return list(range(frame_count))
    edges = np.linspace(0, frame_count, num_frames + 1)
    centres = ((edges[:-1] + edges[1:]) / 2.0).astype(int)
    return sorted(set(int(min(c, frame_count - 1)) for c in centres))


def iter_frames(path: str | Path, indices: list[int], fps: float | None = None) -> Iterator[SampledFrame]:
    """Yield the requested frames by sequential grab/retrieve (robust across codecs)."""
    wanted = sorted(set(indices))
    if not wanted:
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = fps or float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
    try:
        pos, target_i = 0, 0
        while target_i < len(wanted):
            if not cap.grab():
                break
            if pos == wanted[target_i]:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield SampledFrame(pos, pos / fps, frame)
                target_i += 1
            pos += 1
    finally:
        cap.release()


def sample_frames(
    path: str | Path, num_frames: int, strategy: str = "uniform", stride: int = 10
) -> tuple[VideoInfo, list[SampledFrame]]:
    info = probe_video(path)
    indices = sample_indices(info.frame_count, num_frames, strategy, stride)
    return info, list(iter_frames(path, indices, info.fps))
