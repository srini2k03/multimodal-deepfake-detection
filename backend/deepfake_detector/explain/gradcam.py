"""Grad-CAM heatmaps for the frame classifiers.

Grad-CAM (Selvaraju et al., 2017) weights the last convolutional feature maps by the
spatially averaged gradient of the "fake" logit, highlighting the regions that pushed
the decision. For deepfakes these are typically blending boundaries, mouth and eyes.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np


def compute_gradcam(model, image: np.ndarray, conv_layer: str | None = None) -> np.ndarray:
    """Return a ``[0, 1]`` heatmap of shape ``image.shape[:2]``.

    ``model`` must be a classifier built by :func:`build_frame_classifier`; ``image`` is
    a single preprocessed ``(H, W, 3)`` array in ``[-1, 1]``.
    """
    import keras
    import tensorflow as tf

    from ..models.backbones import BACKBONES, get_backbone

    backbone = get_backbone(model)
    if conv_layer is None:
        key = backbone.name.removesuffix("_backbone")
        conv_layer = BACKBONES[key][1] if key in BACKBONES else _last_conv_layer(backbone)
    grad_model = keras.Model(backbone.inputs, [backbone.get_layer(conv_layer).output, backbone.output])
    head = model.layers[model.layers.index(backbone) + 1 :]

    x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, features = grad_model(x, training=False)
        tape.watch(conv_out)
        y = features
        for layer in head:
            if layer.name == "prob":
                break  # use the pre-sigmoid logit for better-conditioned gradients
            y = layer(y, training=False)
        target = tf.reduce_sum(tf.cast(y, tf.float32))
    grads = tape.gradient(target, conv_out)
    if grads is None:  # pragma: no cover - defensive
        return np.zeros(image.shape[:2], dtype=np.float32)
    conv = tf.cast(conv_out[0], tf.float32)
    weights = tf.reduce_mean(tf.cast(grads[0], tf.float32), axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1)).numpy()
    return normalize_heatmap(cv2.resize(cam, (image.shape[1], image.shape[0])))


def _last_conv_layer(backbone) -> str:
    for layer in reversed(backbone.layers):
        shape = getattr(layer, "output", None)
        if shape is not None and len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No 4D layer found for Grad-CAM")


def normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam.astype(np.float32))
    cam -= cam.min()
    peak = cam.max()
    return cam / peak if peak > 1e-12 else np.zeros_like(cam)


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET-coloured heatmap over a BGR uint8 image."""
    heat = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0)


def encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:  # pragma: no cover
        raise ValueError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def pseudo_saliency(image_bgr: np.ndarray) -> np.ndarray:
    """High-frequency energy map used ONLY in numpy demo mode (not a model explanation)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return normalize_heatmap(cv2.GaussianBlur(lap, (0, 0), sigmaX=max(2.0, gray.shape[0] / 30)))
