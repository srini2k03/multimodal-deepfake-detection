"""Compact CNN over log-mel spectrograms for synthetic/cloned speech detection."""

from __future__ import annotations

import keras
from keras import layers


def _conv_block(x, filters: int, name: str, dropout: float):
    for i in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)


def build_audio_cnn(
    n_mels: int = 64,
    n_frames: int = 401,
    *,
    filters: tuple[int, ...] = (32, 64, 128),
    dropout: float = 0.2,
) -> keras.Model:
    """Input ``(n_mels, n_frames, 1)`` standardised log-mel -> P(fake) in float32."""
    inputs = keras.Input((n_mels, n_frames, 1), name="log_mel")
    x = inputs
    for i, f in enumerate(filters):
        x = _conv_block(x, f, f"block{i + 1}", dropout)
    x = layers.Conv2D(filters[-1] * 2, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="fc")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    x = layers.Dense(1, name="logit")(x)
    outputs = layers.Activation("sigmoid", dtype="float32", name="prob")(x)
    return keras.Model(inputs, outputs, name="audio_cnn")
