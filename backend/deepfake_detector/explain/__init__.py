"""Explainability utilities (Grad-CAM)."""

from .gradcam import compute_gradcam, encode_png_base64, normalize_heatmap, overlay_heatmap

__all__ = ["compute_gradcam", "encode_png_base64", "normalize_heatmap", "overlay_heatmap"]
