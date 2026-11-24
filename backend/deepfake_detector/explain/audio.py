"""Audio explanations: log-mel spectrogram images and Grad-CAM on the audio CNN.

Grad-CAM over a spectrogram highlights the time-frequency regions whose activations pushed the
spoof logit up. It is diagnostic evidence about what the model attends to (for ASVspoof models
this is often silence or high-frequency bands), not proof that a region is synthetic.
"""

from __future__ import annotations

import cv2
import numpy as np

_GRAD_MODELS: dict = {}


def spectrogram_image(mel: np.ndarray, scale: int = 3) -> np.ndarray:
    """``(n_mels, frames)`` log-mel -> BGR uint8 image, low frequencies at the bottom."""
    m = np.asarray(mel, dtype=np.float32)
    lo, hi = np.percentile(m, 1), np.percentile(m, 99)
    norm = np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap(np.uint8(255 * norm[::-1]), cv2.COLORMAP_VIRIDIS)
    return cv2.resize(img, (m.shape[1] * 2, m.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def audio_gradcam(model, mel: np.ndarray, conv_layer: str = "last_conv") -> np.ndarray:
    """Grad-CAM heatmap ``(n_mels, frames)`` in [0, 1] for one standardised log-mel clip."""
    import keras
    import tensorflow as tf

    from .gradcam import normalize_heatmap

    key = (id(model), conv_layer)
    cached = _GRAD_MODELS.get(key)
    if cached is None or cached[0] is not model:
        grad_model = keras.Model(model.input, [model.get_layer(conv_layer).output, model.get_layer("logit").output])
        cached = (model, grad_model)
        _GRAD_MODELS[key] = cached
    grad_model = cached[1]
    x = tf.convert_to_tensor(np.asarray(mel, dtype=np.float32)[None, ..., None])
    with tf.GradientTape() as tape:
        conv, logit = grad_model(x, training=False)
        tape.watch(conv)
        target = tf.reduce_sum(logit)
    grads = tape.gradient(target, conv)
    if grads is None:  # pragma: no cover - defensive
        return np.zeros(mel.shape, dtype=np.float32)
    weights = tf.reduce_mean(grads[0], axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    return normalize_heatmap(cv2.resize(cam, (mel.shape[1], mel.shape[0]), interpolation=cv2.INTER_LINEAR))
