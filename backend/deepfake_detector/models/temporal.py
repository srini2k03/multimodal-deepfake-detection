"""Frame-level -> video-level temporal aggregation heads."""

from __future__ import annotations

import keras
from keras import layers, ops

from .backbones import build_backbone, get_backbone


@keras.saving.register_keras_serializable(package="deepfake_detector")
class AttentionPooling(layers.Layer):
    """Additive (Bahdanau-style) attention pooling over the time axis.

    ``score_t = v^T tanh(W h_t + b)``, ``alpha = softmax(score)``, ``out = sum_t alpha_t h_t``.
    Lets the model focus on the few frames that carry manipulation artifacts instead of
    averaging them away.
    """

    def __init__(self, units: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh", name="proj")
        self.score = layers.Dense(1, use_bias=False, name="score")

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.score.build((*input_shape[:-1], self.units))
        super().build(input_shape)

    def attention_weights(self, inputs):
        scores = ops.squeeze(self.score(self.proj(inputs)), axis=-1)  # (B, T)
        return ops.softmax(ops.cast(scores, "float32"), axis=-1)

    def call(self, inputs):
        alpha = ops.cast(self.attention_weights(inputs), inputs.dtype)
        return ops.sum(inputs * ops.expand_dims(alpha, -1), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg


def build_temporal_model(
    name: str,
    seq_len: int = 8,
    input_size: int = 299,
    *,
    pooling: str = "attention",
    weights: str | None = "imagenet",
    dropout: float = 0.5,
    attention_units: int = 128,
    frame_classifier: keras.Model | None = None,
) -> keras.Model:
    """Shared CNN over each frame -> mean/attention pooling -> P(fake) for the whole clip.

    Passing a trained ``frame_classifier`` reuses its fine-tuned backbone so the temporal
    head can be trained on top of frame-level training (recommended).
    """
    if pooling not in ("mean", "attention"):
        raise ValueError("pooling must be 'mean' or 'attention'")
    backbone = (
        get_backbone(frame_classifier) if frame_classifier is not None else build_backbone(name, input_size, weights)
    )
    inputs = keras.Input((seq_len, input_size, input_size, 3), name="frames")
    feats = layers.TimeDistributed(backbone, name="frame_features")(inputs, training=False)
    if pooling == "mean":
        pooled = layers.GlobalAveragePooling1D(name="temporal_pool")(feats)
    else:
        pooled = AttentionPooling(attention_units, name="temporal_pool")(feats)
    x = layers.Dropout(dropout, name="head_dropout")(pooled)
    x = layers.Dense(1, name="logit")(x)
    outputs = layers.Activation("sigmoid", dtype="float32", name="prob")(x)
    return keras.Model(inputs, outputs, name=f"{name}_temporal_{pooling}")
