"""Weighted ensemble of frame classifiers sharing one input."""

from __future__ import annotations

import keras
from keras import layers


@keras.saving.register_keras_serializable(package="deepfake_detector")
class WeightedAverage(layers.Layer):
    """Convex combination of model outputs with fixed (normalised) weights."""

    def __init__(self, member_weights: list[float], **kwargs) -> None:
        super().__init__(**kwargs)
        total = float(sum(member_weights))
        if total <= 0:
            raise ValueError("member_weights must sum to a positive value")
        self.member_weights = [float(w) / total for w in member_weights]

    def call(self, inputs):
        if len(inputs) != len(self.member_weights):
            raise ValueError("number of inputs must match number of weights")
        out = inputs[0] * self.member_weights[0]
        for x, w in zip(inputs[1:], self.member_weights[1:], strict=True):
            out = out + x * w
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        cfg = super().get_config()
        cfg["member_weights"] = self.member_weights
        return cfg


def build_ensemble(members: dict[str, keras.Model], weights: list[float] | None = None) -> keras.Model:
    """Return a model whose outputs are ``{"ensemble": p, <member>: p_member, ...}``."""
    if not members:
        raise ValueError("at least one member model is required")
    names = list(members)
    weights = weights or [1.0] * len(names)
    first = members[names[0]]
    inputs = keras.Input(first.input_shape[1:], name="image")
    member_outs = {n: members[n](inputs) for n in names}
    combined = WeightedAverage(weights, name="weighted_average")([member_outs[n] for n in names])
    outputs = {"ensemble": layers.Activation("linear", dtype="float32", name="ensemble")(combined)}
    for n in names:
        outputs[n] = layers.Activation("linear", dtype="float32", name=n)(member_outs[n])
    return keras.Model(inputs, outputs, name="frame_ensemble")


def ensemble_probability(per_model: dict[str, float], weights: dict[str, float]) -> float:
    """NumPy-free helper used at inference time when members are run separately."""
    num = sum(per_model[n] * weights.get(n, 0.0) for n in per_model)
    den = sum(weights.get(n, 0.0) for n in per_model)
    return float(num / den) if den > 0 else float(sum(per_model.values()) / max(len(per_model), 1))


__all__ = ["WeightedAverage", "build_ensemble", "ensemble_probability"]
