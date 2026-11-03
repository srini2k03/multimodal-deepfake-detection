"""``tf.data`` input pipelines for frame-level, sequence-level and audio training.

All image pipelines emit tensors scaled to ``[-1, 1]``, which is the preprocessing used
by both Xception and InceptionV3, so one pipeline can feed every backbone and the
ensemble.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..config import Config
from .layouts import class_weights, read_manifest

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class SplitData:
    dataset: tf.data.Dataset
    labels: list[int]
    steps: int


def _frames_by_split(crops_root: Path, split: str) -> dict[str, tuple[list[str], int]]:
    manifest = crops_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found. Run the face extraction step first.")
    out: dict[str, tuple[list[str], int]] = {}
    for row in read_manifest(manifest):
        if row["split"] != split:
            continue
        frames = sorted(str(p) for p in (crops_root / row["video_id"]).glob("frame_*.png"))
        if frames:
            out[row["video_id"]] = (frames, int(row["label"]))
    return out


def preprocess_image(image: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.image.resize(tf.cast(image, tf.float32), (image_size, image_size))
    return image / 127.5 - 1.0


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Light, artifact-preserving augmentation on ``[-1, 1]`` images.

    Aggressive blurs/resampling would erase the blending traces the model must learn,
    so we restrict ourselves to flips and mild photometric jitter.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation((image + 1.0) / 2.0, 0.85, 1.15) * 2.0 - 1.0
    return tf.clip_by_value(image, -1.0, 1.0)


def _load_png(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image.set_shape([None, None, 3])
    return preprocess_image(image, image_size)


def frame_dataset(cfg: Config, split: str, *, training: bool | None = None,
                  max_frames_per_video: int | None = None) -> SplitData:
    """Frame-level dataset of ``(image, label)``; every crop is an independent sample."""
    training = split == "train" if training is None else training
    d = cfg.data
    videos = _frames_by_split(Path(d.crops_root), split)
    paths, labels = [], []
    rng = np.random.default_rng(d.seed)
    for frames, label in videos.values():
        if max_frames_per_video and len(frames) > max_frames_per_video:
            frames = sorted(rng.choice(frames, max_frames_per_video, replace=False).tolist())
        paths.extend(frames)
        labels.extend([label] * len(frames))
    if not paths:
        raise ValueError(f"No frames found for split '{split}' in {d.crops_root}")

    ds = tf.data.Dataset.from_tensor_slices((paths, np.asarray(labels, np.float32)))
    if training:
        ds = ds.shuffle(min(len(paths), d.shuffle_buffer), seed=d.seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_png(p, d.image_size), y), num_parallel_calls=AUTOTUNE)
    if training and d.augment:
        ds = ds.map(lambda x, y: (augment_image(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(d.batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    steps = int(np.ceil(len(paths) / d.batch_size))
    return SplitData(ds, labels, steps)


def _pick_sequence(frames: list[str], seq_len: int, rng: np.random.Generator, training: bool) -> list[str]:
    if len(frames) >= seq_len:
        if training:
            start = int(rng.integers(0, len(frames) - seq_len + 1))
            return frames[start : start + seq_len]
        idx = np.linspace(0, len(frames) - 1, seq_len).astype(int)
        return [frames[i] for i in idx]
    reps = int(np.ceil(seq_len / len(frames)))
    return (frames * reps)[:seq_len]


def sequence_dataset(cfg: Config, split: str, *, training: bool | None = None) -> SplitData:
    """Video-level dataset of ``(frames[T, H, W, 3], label)`` for temporal heads."""
    training = split == "train" if training is None else training
    d = cfg.data
    videos = _frames_by_split(Path(d.crops_root), split)
    if not videos:
        raise ValueError(f"No videos found for split '{split}' in {d.crops_root}")
    rng = np.random.default_rng(d.seed)
    seqs = [_pick_sequence(frames, d.seq_len, rng, training) for frames, _ in videos.values()]
    labels = [label for _, label in videos.values()]

    ds = tf.data.Dataset.from_tensor_slices((np.asarray(seqs), np.asarray(labels, np.float32)))
    if training:
        ds = ds.shuffle(len(seqs), seed=d.seed, reshuffle_each_iteration=True)

    def load_seq(paths: tf.Tensor, y: tf.Tensor):
        frames = tf.map_fn(lambda p: _load_png(p, d.image_size), paths, fn_output_signature=tf.float32)
        if training and d.augment:
            # Same flip for the whole clip keeps temporal consistency.
            frames = tf.cond(tf.random.uniform([]) > 0.5, lambda: tf.reverse(frames, axis=[2]), lambda: frames)
        return frames, y

    ds = ds.map(load_seq, num_parallel_calls=AUTOTUNE)
    batch = max(1, d.batch_size // d.seq_len)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return SplitData(ds, labels, int(np.ceil(len(seqs) / batch)))


def audio_dataset(cfg: Config, split: str, *, training: bool | None = None) -> SplitData:
    """Clip-level log-mel dataset built from ``extract audio`` output."""
    training = split == "train" if training is None else training
    manifest = Path(f"{cfg.data.audio_root.rstrip('/')}_features") / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found. Run `extract audio` first.")
    xs, ys = [], []
    for row in read_manifest(manifest):
        if row["split"] != split:
            continue
        clips = np.load(row["features"]).astype(np.float32)
        xs.append(clips)
        ys.extend([int(row["label"])] * len(clips))
    if not xs:
        raise ValueError(f"No audio clips for split '{split}'")
    x = np.concatenate(xs)
    ds = tf.data.Dataset.from_tensor_slices((x, np.asarray(ys, np.float32)))
    if training:
        ds = ds.shuffle(len(ys), seed=cfg.data.seed, reshuffle_each_iteration=True)

        def spec_augment(m, y):
            # Random time/frequency masking (SpecAugment-lite).
            f0 = tf.random.uniform([], 0, tf.shape(m)[0] - 8, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, tf.shape(m)[1] - 20, dtype=tf.int32)
            fmask = tf.logical_or(tf.range(tf.shape(m)[0]) < f0, tf.range(tf.shape(m)[0]) >= f0 + 8)
            tmask = tf.logical_or(tf.range(tf.shape(m)[1]) < t0, tf.range(tf.shape(m)[1]) >= t0 + 20)
            mask = tf.cast(fmask[:, None, None] & tmask[None, :, None], m.dtype)
            return m * mask, y

        ds = ds.map(spec_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.data.batch_size).prefetch(AUTOTUNE)
    return SplitData(ds, ys, int(np.ceil(len(ys) / cfg.data.batch_size)))


def video_frame_groups(cfg: Config, split: str) -> dict[str, tuple[list[str], int]]:
    """Expose per-video frame lists (used by video-level evaluation)."""
    return _frames_by_split(Path(cfg.data.crops_root), split)


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    return class_weights(labels)


def label_counts(labels: list[int]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for y in labels:
        counts[int(y)] += 1
    return dict(counts)
