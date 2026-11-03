"""Upload validation: extension whitelist and content sniffing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..inference.predictor import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS
from ..preprocessing.video import probe_video

ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS


class ValidationError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def check_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if not filename or not ext:
        raise ValidationError("File name must have an extension")
    if ext not in ALLOWED_EXTS:
        raise ValidationError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTS))}", status=415)
    return ext


def check_content(path: Path, media_type: str, max_video_seconds: int) -> dict:
    """Make sure the file really decodes as the claimed media type; return basic metadata."""
    if path.stat().st_size == 0:
        raise ValidationError("Uploaded file is empty")
    if media_type == "image":
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValidationError("File is not a decodable image", status=422)
        return {"width": int(img.shape[1]), "height": int(img.shape[0])}
    if media_type == "video":
        try:
            info = probe_video(path)
        except ValueError as exc:
            raise ValidationError("File is not a decodable video", status=422) from exc
        if info.frame_count <= 0:
            raise ValidationError("Video contains no frames", status=422)
        if info.duration_s > max_video_seconds:
            raise ValidationError(
                f"Video is {info.duration_s:.0f}s long; the limit is {max_video_seconds}s", status=413
            )
        return {
            "frames": info.frame_count,
            "fps": round(info.fps, 3),
            "duration": round(info.duration_s, 3),
            "width": info.width,
            "height": info.height,
        }
    return {}
