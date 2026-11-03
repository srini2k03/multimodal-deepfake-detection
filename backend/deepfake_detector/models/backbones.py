"""Xception / InceptionV3 frame classifiers with two-stage transfer learning support."""

from __future__ import annotations

from collections.abc import Callable

import keras
from keras import layers

# name -> (constructor, last convolutional layer used for Grad-CAM)
BACKBONES: dict[str, tuple[Callable[..., keras.Model], str]] = {
    "xception": (keras.applications.Xception, "block14_sepconv2_act"),
    "inception_v3": (keras.applications.InceptionV3, "mixed10"),
}
MIN_INPUT = {"xception": 71, "inception_v3": 75}


def backbone_layer_name(name: str) -> str:
    return f"{name}_backbone"


def build_backbone(name: str, input_size: int = 299, weights: str | None = "imagenet") -> keras.Model:
    """ImageNet feature extractor with global average pooling (no classification top)."""
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {sorted(BACKBONES)}")
    if input_size < MIN_INPUT[name]:
        raise ValueError(f"{name} requires input_size >= {MIN_INPUT[name]}")
    ctor, _ = BACKBONES[name]
    return ctor(
        include_top=False,
        weights=weights,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
        name=backbone_layer_name(name),
    )


def build_frame_classifier(
    name: str,
    input_size: int = 299,
    *,
    weights: str | None = "imagenet",
    dropout: float = 0.5,
) -> keras.Model:
    """Backbone + dropout + single-logit head. Outputs P(fake) in float32.

    The backbone is called with ``training=False`` so BatchNorm statistics stay frozen
    during fine-tuning - the standard recipe for small-batch transfer learning.
    """
    inputs = keras.Input((input_size, input_size, 3), name="image")
    backbone = build_backbone(name, input_size, weights)
    x = backbone(inputs, training=False)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    x = layers.Dense(1, name="logit")(x)
    outputs = layers.Activation("sigmoid", dtype="float32", name="prob")(x)
    return keras.Model(inputs, outputs, name=f"{name}_classifier")


def get_backbone(model: keras.Model) -> keras.Model:
    """Find the CNN backbone in a frame classifier or (TimeDistributed) temporal model."""
    for layer in model.layers:
        inner = layer.layer if isinstance(layer, layers.TimeDistributed) else layer
        if inner.name.endswith("_backbone") and isinstance(inner, keras.Model):
            return inner
    raise ValueError(f"No backbone sub-model found in {model.name}")


def set_trainable(model: keras.Model, stage: str, unfreeze_fraction: float = 0.35) -> int:
    """Configure trainability for a training stage and return #trainable backbone layers.

    * ``head``: backbone frozen, only the classification head learns.
    * ``finetune``: the top ``unfreeze_fraction`` of backbone layers become trainable;
      BatchNormalization layers always remain frozen.
    """
    backbone = get_backbone(model)
    if stage == "head":
        backbone.trainable = False
        return 0
    if stage != "finetune":
        raise ValueError("stage must be 'head' or 'finetune'")
    if not 0.0 < unfreeze_fraction <= 1.0:
        raise ValueError("unfreeze_fraction must be in (0, 1]")
    backbone.trainable = True
    cutoff = int(len(backbone.layers) * (1.0 - unfreeze_fraction))
    n_trainable = 0
    for i, layer in enumerate(backbone.layers):
        trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
        n_trainable += int(trainable)
    return n_trainable
