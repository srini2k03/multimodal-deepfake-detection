"""Inference engine. Heavy imports (TensorFlow) only happen when models are loaded."""

from .fusion import aggregate_frames, late_fusion
from .predictor import DeepfakeDetector, media_type_for

__all__ = ["DeepfakeDetector", "aggregate_frames", "late_fusion", "media_type_for"]
