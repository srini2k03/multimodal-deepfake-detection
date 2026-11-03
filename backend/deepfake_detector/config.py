"""Typed configuration loaded from YAML with environment-variable overrides.

Every section is a dataclass so values are discoverable and validated in one place.
Use :func:`load_config` to obtain a :class:`Config`; unknown keys raise ``ValueError``
so typos in YAML files are caught early.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

SUPPORTED_BACKBONES = ("xception", "inception_v3")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass
class FaceConfig:
    detector: str = "haar"  # "dnn" (OpenCV res10 SSD) or "haar"
    dnn_prototxt: str = "models/face/deploy.prototxt"
    dnn_weights: str = "models/face/res10_300x300_ssd_iter_140000.caffemodel"
    confidence: float = 0.6
    margin: float = 0.3  # relative padding around the detected box
    min_face_size: int = 40
    align: bool = True  # rotate so the eye line is horizontal when eyes are found
    crop_size: int = 299


@dataclass
class SamplingConfig:
    frames_per_video: int = 32
    strategy: str = "uniform"  # "uniform" or "stride"
    stride: int = 10
    max_frames: int = 64


@dataclass
class DataConfig:
    dataset: str = "faceforensics"  # "faceforensics", "celebdf" or "generic"
    raw_root: str = "data/raw"
    crops_root: str = "data/crops"
    audio_root: str = "data/audio"
    cache_dir: str = "data/cache"
    image_size: int = 299
    batch_size: int = 32
    seq_len: int = 8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    shuffle_buffer: int = 2048
    augment: bool = True
    seed: int = 42
    compression: str = "c23"  # FaceForensics++ compression level


@dataclass
class ModelConfig:
    backbones: list[str] = field(default_factory=lambda: list(SUPPORTED_BACKBONES))
    ensemble_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    pretrained: bool = True
    dropout: float = 0.5
    temporal_pooling: str = "none"  # "none", "mean" or "attention"
    attention_units: int = 128


@dataclass
class TrainConfig:
    epochs_head: int = 3
    epochs_finetune: int = 12
    lr_head: float = 1e-3
    lr_finetune: float = 2e-5
    unfreeze_fraction: float = 0.35
    mixed_precision: bool = True
    class_weighting: bool = True
    label_smoothing: float = 0.0
    early_stopping_patience: int = 4
    reduce_lr_patience: int = 2
    output_dir: str = "artifacts"


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 8000.0
    clip_seconds: float = 4.0


@dataclass
class FusionConfig:
    method: str = "weighted"  # "weighted" (probability space) or "logit"
    visual_weight: float = 0.7
    audio_weight: float = 0.3
    threshold: float = 0.5


@dataclass
class ApiConfig:
    weights_dir: str = "weights"
    max_upload_mb: int = 100
    max_video_seconds: int = 300
    frames_per_video: int = 16
    frame_aggregation: str = "mean"  # see inference.fusion.AGGREGATIONS
    gradcam: bool = True
    demo_mode: str = "auto"  # "auto" (demo when weights missing), "on" or "off"
    demo_backend: str = "auto"  # "auto", "keras" (random weights) or "numpy"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    async_workers: int = 2
    job_ttl_seconds: int = 3600
    upload_dir: str = "uploads"


@dataclass
class Config:
    face: FaceConfig = field(default_factory=FaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> Config:
        for name in self.model.backbones:
            if name not in SUPPORTED_BACKBONES:
                raise ValueError(f"Unsupported backbone '{name}'. Choose from {SUPPORTED_BACKBONES}.")
        if len(self.model.ensemble_weights) != len(self.model.backbones):
            raise ValueError("model.ensemble_weights must have one entry per backbone")
        if any(w < 0 for w in self.model.ensemble_weights) or sum(self.model.ensemble_weights) <= 0:
            raise ValueError("model.ensemble_weights must be non-negative with a positive sum")
        if self.model.temporal_pooling not in ("none", "mean", "attention"):
            raise ValueError("model.temporal_pooling must be 'none', 'mean' or 'attention'")
        if self.face.detector not in ("dnn", "haar"):
            raise ValueError("face.detector must be 'dnn' or 'haar'")
        if self.fusion.method not in ("weighted", "logit"):
            raise ValueError("fusion.method must be 'weighted' or 'logit'")
        if not 0.0 < self.fusion.threshold < 1.0:
            raise ValueError("fusion.threshold must be in (0, 1)")
        if self.api.demo_mode not in ("auto", "on", "off"):
            raise ValueError("api.demo_mode must be 'auto', 'on' or 'off'")
        if self.api.demo_backend not in ("auto", "keras", "numpy"):
            raise ValueError("api.demo_backend must be 'auto', 'keras' or 'numpy'")
        if self.api.frame_aggregation not in ("mean", "median", "max", "topk_mean", "confidence_weighted"):
            raise ValueError("api.frame_aggregation is not a supported aggregation method")
        if self.data.image_size < 75:
            raise ValueError("data.image_size must be >= 75 (InceptionV3 minimum input)")
        return self


def _build_dataclass(cls: type, values: dict[str, Any] | None, path: str = "") -> Any:
    values = values or {}
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config key(s) at '{path or 'root'}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in values:
            continue
        value = values[f.name]
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint):
            kwargs[f.name] = _build_dataclass(hint, value, f"{path}.{f.name}".strip("."))
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        items = [s.strip() for s in raw.split(",") if s.strip()]
        if current and isinstance(current[0], float):
            return [float(s) for s in items]
        return items
    return raw


def apply_env_overrides(cfg: Config, environ: dict[str, str] | None = None) -> Config:
    """Apply ``DFD_<SECTION>__<KEY>`` environment variables, e.g. ``DFD_API__MAX_UPLOAD_MB=50``."""
    environ = dict(os.environ if environ is None else environ)
    for key, raw in environ.items():
        if not key.startswith("DFD_") or "__" not in key:
            continue
        section_name, _, attr = key[4:].lower().partition("__")
        section = getattr(cfg, section_name, None)
        if section is None or not hasattr(section, attr):
            continue
        setattr(section, attr, _coerce(raw, getattr(section, attr)))
    return cfg


def load_config(path: str | os.PathLike[str] | None = None, *, env: bool = True) -> Config:
    """Load a YAML config (defaults to ``configs/default.yaml``) and validate it."""
    cfg_path = Path(path) if path else Path(os.environ.get("DFD_CONFIG", DEFAULT_CONFIG_PATH))
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = _build_dataclass(Config, data)
    if env:
        apply_env_overrides(cfg)
    return cfg.validate()
