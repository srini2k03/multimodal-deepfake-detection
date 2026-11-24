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


def probe_audio(path: Path) -> dict:
    """Duration and sample rate without decoding everything (libsndfile, then ffmpeg)."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValidationError("Audio file contains no samples", status=422)
        return {"duration": round(info.frames / info.samplerate, 3), "sample_rate": int(info.samplerate),
                "channels": int(info.channels)}  # fmt: skip
    except (ImportError, RuntimeError, TypeError):
        pass  # formats libsndfile cannot read (mp3/m4a on some builds) are checked by full decode
    from ..preprocessing.audio import AudioUnavailableError, load_audio

    try:
        signal = load_audio(path, 16000)
    except AudioUnavailableError as exc:
        raise ValidationError("File is not decodable audio", status=422) from exc
    if signal.size == 0:
        raise ValidationError("Audio file contains no samples", status=422)
    return {"duration": round(signal.size / 16000, 3), "sample_rate": None, "channels": 1}


def check_content(path: Path, media_type: str, max_video_seconds: int, max_audio_seconds: int = 600) -> dict:
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
    if media_type == "audio":
        meta = probe_audio(path)
        if meta["duration"] > max_audio_seconds:
            raise ValidationError(
                f"Audio is {meta['duration']:.0f}s long; the limit is {max_audio_seconds}s", status=413
            )
        return meta
    return {}
