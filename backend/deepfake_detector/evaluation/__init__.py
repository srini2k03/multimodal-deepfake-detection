"""Evaluation metrics (NumPy) and the evaluation CLI (``python -m deepfake_detector.evaluation.evaluate``)."""

from .metrics import BinaryReport, binary_report, confusion_matrix, equal_error_rate, roc_auc, roc_curve

__all__ = ["BinaryReport", "binary_report", "confusion_matrix", "equal_error_rate", "roc_auc", "roc_curve"]
