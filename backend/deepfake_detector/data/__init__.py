"""Dataset discovery (pure Python) and ``tf.data`` pipelines (import ``.datasets`` explicitly)."""

from .layouts import FAKE, REAL, VideoRecord, class_weights, discover, read_manifest, write_manifest

__all__ = ["FAKE", "REAL", "VideoRecord", "class_weights", "discover", "read_manifest", "write_manifest"]
