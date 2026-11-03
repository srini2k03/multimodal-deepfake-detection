"""Helpers for optional heavy dependencies (TensorFlow, ffmpeg)."""

from __future__ import annotations

import functools
import importlib.util
import shutil


@functools.lru_cache(maxsize=1)
def tf_available() -> bool:
    """Return True if TensorFlow can be imported (checked without importing it)."""
    return importlib.util.find_spec("tensorflow") is not None


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def require_tf():
    """Import and return TensorFlow, raising a helpful error if it is missing."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "TensorFlow is required for this operation. Install it with `pip install -r backend/requirements.txt`."
        ) from exc
    return tf
