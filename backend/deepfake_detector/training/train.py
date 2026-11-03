"""Training CLI with two-stage transfer learning, mixed precision and class weighting.

Examples::

    # frame-level classifiers (run once per backbone)
    python -m deepfake_detector.training.train --model xception
    python -m deepfake_detector.training.train --model inception_v3

    # video-level temporal head on top of a trained frame model
    python -m deepfake_detector.training.train --model temporal --backbone xception

    # audio branch
    python -m deepfake_detector.training.train --model audio

Stage 1 trains only the new head with the ImageNet backbone frozen; stage 2 unfreezes
the top ``train.unfreeze_fraction`` of backbone layers (BatchNorm kept frozen) and
fine-tunes with a much lower learning rate. The best checkpoint (by validation AUC) is
exported to ``<api.weights_dir>/<name>.keras`` for the API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from ..utils.logging import configure_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_precision(enabled: bool) -> str:
    import keras
    import tensorflow as tf

    if enabled and tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        return "mixed_float16"
    if enabled:
        logger.warning("mixed precision requested but no GPU found; training in float32")
    keras.mixed_precision.set_global_policy("float32")
    return "float32"


def compile_model(model, lr: float, label_smoothing: float = 0.0) -> None:
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def build_callbacks(run_dir: Path, cfg: Config, best_so_far: float | None = None) -> list:
    """Standard callbacks. ``best_so_far`` stops stage 2 from overwriting a better stage-1 checkpoint."""
    import keras

    t = cfg.train
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(run_dir / "best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=best_so_far,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=t.early_stopping_patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=t.reduce_lr_patience, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
        keras.callbacks.TerminateOnNaN(),
    ]
    if importlib.util.find_spec("tensorboard") is not None:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard")))
    return callbacks


def _merge_history(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for h in histories:
        if h is None:
            continue
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(float(x) for x in v)
    return merged


def two_stage_fit(model, train, val, cfg: Config, run_dir: Path, class_weight: dict | None):
    from ..models.backbones import set_trainable

    t = cfg.train
    set_trainable(model, "head")
    compile_model(model, t.lr_head, t.label_smoothing)
    logger.info("stage 1: training head", extra={"epochs": t.epochs_head})
    h1 = model.fit(
        train.dataset,
        validation_data=val.dataset,
        epochs=t.epochs_head,
        class_weight=class_weight,
        callbacks=build_callbacks(run_dir, cfg),
        verbose=2,
    )

    h2 = None
    if t.epochs_finetune > 0:
        best_stage1 = max(h1.history.get("val_auc", [0.0]))
        n = set_trainable(model, "finetune", t.unfreeze_fraction)
        compile_model(model, t.lr_finetune, t.label_smoothing)
        logger.info("stage 2: fine-tuning", extra={"epochs": t.epochs_finetune, "trainable_layers": n})
        h2 = model.fit(
            train.dataset,
            validation_data=val.dataset,
            epochs=t.epochs_head + t.epochs_finetune,
            initial_epoch=len(h1.epoch),
            class_weight=class_weight,
            callbacks=build_callbacks(run_dir, cfg, best_stage1),
            verbose=2,
        )
    return _merge_history(h1, h2)


def _export(run_dir: Path, cfg: Config, export_name: str, model, meta: dict) -> Path:
    weights_dir = Path(cfg.api.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "best.keras"
    target = weights_dir / f"{export_name}.keras"
    if best.exists():
        shutil.copy2(best, target)
    else:
        model.save(target)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("exported model", extra={"path": str(target)})
    return target


def train(cfg: Config, model_kind: str, backbone: str = "xception", init_from: str | None = None) -> Path:
    import keras

    from ..data.datasets import audio_dataset, compute_class_weights, frame_dataset, label_counts, sequence_dataset
    from ..models import build_audio_cnn, build_frame_classifier, build_temporal_model
    from ..preprocessing.audio import frames_per_clip

    set_seed(cfg.data.seed)
    policy = configure_precision(cfg.train.mixed_precision)
    run_dir = Path(cfg.train.output_dir) / f"{model_kind}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = "imagenet" if cfg.model.pretrained else None

    if model_kind in ("xception", "inception_v3"):
        train_ds, val_ds = frame_dataset(cfg, "train"), frame_dataset(cfg, "val")
        model = build_frame_classifier(model_kind, cfg.data.image_size, weights=weights, dropout=cfg.model.dropout)
        export_name = model_kind
    elif model_kind == "temporal":
        if cfg.model.temporal_pooling == "none":
            cfg.model.temporal_pooling = "attention"
        train_ds, val_ds = sequence_dataset(cfg, "train"), sequence_dataset(cfg, "val")
        frame_model = None
        init_path = Path(init_from) if init_from else Path(cfg.api.weights_dir) / f"{backbone}.keras"
        if init_path.exists():
            from ..inference.scorers import load_keras_model

            frame_model = load_keras_model(init_path)
            logger.info("initialising temporal backbone", extra={"from": str(init_path)})
        model = build_temporal_model(
            backbone,
            cfg.data.seq_len,
            cfg.data.image_size,
            pooling=cfg.model.temporal_pooling,
            weights=weights,
            dropout=cfg.model.dropout,
            attention_units=cfg.model.attention_units,
            frame_classifier=frame_model,
        )
        export_name = "temporal"
    elif model_kind == "audio":
        train_ds, val_ds = audio_dataset(cfg, "train"), audio_dataset(cfg, "val")
        model = build_audio_cnn(cfg.audio.n_mels, frames_per_clip(cfg.audio))
        export_name = "audio_cnn"
    else:
        raise ValueError(f"Unknown model kind '{model_kind}'")

    cw = compute_class_weights(train_ds.labels) if cfg.train.class_weighting else None
    logger.info(
        "data",
        extra={
            "train": label_counts(train_ds.labels),
            "val": label_counts(val_ds.labels),
            "class_weight": cw,
            "policy": policy,
        },
    )

    if model_kind == "audio":
        t = cfg.train
        compile_model(model, t.lr_head, t.label_smoothing)
        hist = model.fit(
            train_ds.dataset,
            validation_data=val_ds.dataset,
            epochs=t.epochs_head + t.epochs_finetune,
            class_weight=cw,
            callbacks=build_callbacks(run_dir, cfg),
            verbose=2,
        )
        history = _merge_history(hist)
    else:
        history = two_stage_fit(model, train_ds, val_ds, cfg, run_dir, cw)

    keras.mixed_precision.set_global_policy("float32")
    meta = {
        "model": model_kind,
        "backbone": backbone,
        "policy": policy,
        "class_weight": cw,
        "history": history,
        "config": cfg.to_dict(),
    }
    return _export(run_dir, cfg, export_name, model, meta)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True, choices=["xception", "inception_v3", "temporal", "audio"])
    parser.add_argument(
        "--backbone", default="xception", choices=["xception", "inception_v3"], help="backbone for --model temporal"
    )
    parser.add_argument("--init-from", default=None, help="frame model .keras to initialise the temporal backbone")
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)

    configure_logging()
    cfg = load_config(args.config)
    if args.epochs_head is not None:
        cfg.train.epochs_head = args.epochs_head
    if args.epochs_finetune is not None:
        cfg.train.epochs_finetune = args.epochs_finetune
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    train(cfg, args.model, args.backbone, args.init_from)


if __name__ == "__main__":
    main()
