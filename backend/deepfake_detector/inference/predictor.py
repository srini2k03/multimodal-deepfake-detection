"""End-to-end multi-modal inference for images, videos and audio files.

Weights are looked up in ``api.weights_dir``::

    weights/xception.keras        frame classifier (required for trained mode)
    weights/inception_v3.keras    frame classifier (required for trained mode)
    weights/temporal.keras        optional video-level temporal head
    weights/audio_cnn.keras       optional audio branch

When frame-classifier weights are missing and ``api.demo_mode`` is ``auto``/``on``,
the detector runs in *demo mode* with randomly initialised networks (or a deterministic
placeholder if TensorFlow is not installed). Every response says so explicitly.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import __version__
from ..config import Config, load_config
from ..preprocessing.audio import AudioUnavailableError, audio_to_mel_clips, frames_per_clip, load_audio
from ..preprocessing.pipeline import FaceCrop, FaceExtractor
from ..utils.optional import tf_available
from .fusion import aggregate_frames, late_fusion
from .scorers import (
    KerasAudioScorer,
    KerasFrameScorer,
    KerasTemporalScorer,
    PlaceholderAudioScorer,
    PlaceholderFrameScorer,
    load_keras_model,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

DEMO_WARNING = (
    "DEMO MODE: no trained weights were found, so scores come from untrained "
    "(randomly initialised) networks or a deterministic placeholder. They carry NO "
    "forensic meaning. Train the models and place the .keras files in the weights directory."
)


def media_type_for(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    raise ValueError(f"Unsupported file type '{ext}'")


def preprocess_crop(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 127.5 - 1.0


class DeepfakeDetector:
    """Loads all available models once and serves thread-safe predictions."""

    def __init__(self, cfg: Config | None = None, *, extractor: FaceExtractor | None = None) -> None:
        self.cfg = cfg or load_config()
        self.weights_dir = Path(self.cfg.api.weights_dir)
        self.extractor = extractor or FaceExtractor(self.cfg.face, self.cfg.sampling)
        self._lock = threading.Lock()
        self.frame_scorers: dict[str, Any] = {}
        self.temporal: KerasTemporalScorer | None = None
        self.audio: Any | None = None
        self._load()

    # ------------------------------------------------------------------ loading
    def _demo_backend(self) -> str:
        backend = self.cfg.api.demo_backend
        if backend == "auto":
            return "keras" if tf_available() else "numpy"
        if backend == "keras" and not tf_available():
            logger.warning("demo_backend=keras requested but TensorFlow is missing; using placeholder")
            return "numpy"
        return backend

    def _load(self) -> None:
        api, size = self.cfg.api, self.cfg.data.image_size
        force_demo = api.demo_mode == "on"
        missing = []
        for name in self.cfg.model.backbones:
            path = self.weights_dir / f"{name}.keras"
            if path.exists() and not force_demo:
                if not tf_available():
                    raise RuntimeError(f"Found {path} but TensorFlow is not installed")
                self.frame_scorers[name] = KerasFrameScorer(name, load_keras_model(path), "trained")
                logger.info("loaded frame model", extra={"model": name, "path": str(path)})
            else:
                missing.append(name)
        if missing:
            if api.demo_mode == "off":
                raise FileNotFoundError(f"Missing weights for {missing} in {self.weights_dir} and demo_mode is 'off'")
            backend = self._demo_backend()
            for name in missing:
                if backend == "keras":
                    from ..models.backbones import build_frame_classifier

                    model = build_frame_classifier(name, size, weights=None, dropout=self.cfg.model.dropout)
                    self.frame_scorers[name] = KerasFrameScorer(name, model, "random_weights")
                else:
                    self.frame_scorers[name] = PlaceholderFrameScorer(name, size)
            logger.warning("running in demo mode", extra={"models": missing, "backend": backend})

        temporal_path = self.weights_dir / "temporal.keras"
        if temporal_path.exists() and not force_demo and tf_available():
            self.temporal = KerasTemporalScorer(load_keras_model(temporal_path))

        if self.cfg.audio.enabled:
            audio_path = self.weights_dir / "audio_cnn.keras"
            if audio_path.exists() and not force_demo and tf_available():
                self.audio = KerasAudioScorer(load_keras_model(audio_path), "trained")
            elif api.demo_mode != "off":
                if self._demo_backend() == "keras":
                    from ..models.audio import build_audio_cnn

                    model = build_audio_cnn(self.cfg.audio.n_mels, frames_per_clip(self.cfg.audio))
                    self.audio = KerasAudioScorer(model, "random_weights")
                else:
                    self.audio = PlaceholderAudioScorer()

    # ------------------------------------------------------------------ status
    @property
    def mode(self) -> str:
        kinds = {s.kind for s in self.frame_scorers.values()}
        if kinds == {"trained"}:
            return "trained"
        return "partial" if "trained" in kinds else "demo"

    def info(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "mode": self.mode,
            "warning": DEMO_WARNING if self.mode != "trained" else None,
            "tensorflow": tf_available(),
            "weights_dir": str(self.weights_dir),
            "models": {
                **{n: {"modality": "visual", "status": s.kind} for n, s in self.frame_scorers.items()},
                **({"temporal": {"modality": "visual", "status": self.temporal.kind}} if self.temporal else {}),
                **({"audio_cnn": {"modality": "audio", "status": self.audio.kind}} if self.audio else {}),
            },
            "ensemble_weights": self._ensemble_weights(),
            "fusion": {
                "method": self.cfg.fusion.method,
                "weights": {"visual": self.cfg.fusion.visual_weight, "audio": self.cfg.fusion.audio_weight},
            },
            "frame_aggregation": self.cfg.api.frame_aggregation,
            "threshold": self.cfg.fusion.threshold,
        }

    def _ensemble_weights(self) -> dict[str, float]:
        raw = dict(zip(self.cfg.model.backbones, self.cfg.model.ensemble_weights, strict=True))
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------------ visual
    def _score_crops(self, crops: list[FaceCrop]) -> dict[str, Any]:
        weights = self._ensemble_weights()
        size = self.cfg.data.image_size
        batch = np.stack([preprocess_crop(c.crop_bgr, size) for c in crops])
        per_model = {name: scorer.predict(batch) for name, scorer in self.frame_scorers.items()}
        ensemble = sum(per_model[n] * weights[n] for n in per_model) / sum(weights[n] for n in per_model)

        face_mask = np.array([c.face_found for c in crops])
        use = face_mask if face_mask.any() else np.ones_like(face_mask)
        method = self.cfg.api.frame_aggregation
        visual = aggregate_frames(ensemble[use], method)
        per_model_video = {n: aggregate_frames(s[use], method) for n, s in per_model.items()}

        temporal_prob = None
        if self.temporal is not None and use.sum() > 0:
            temporal_prob = self.temporal.predict_sequence(batch[use])
            visual = float((visual + temporal_prob) / 2.0)

        frames = [
            {
                "index": int(c.frame_index),
                "timestamp": round(float(c.timestamp_s), 3),
                "face_found": bool(c.face_found),
                "probability": round(float(ensemble[i]), 5),
                "per_model": {n: round(float(s[i]), 5) for n, s in per_model.items()},
            }
            for i, c in enumerate(crops)
        ]

        gradcam = None
        if self.cfg.api.gradcam:
            gradcam = self._gradcam(crops, batch, ensemble, use)

        return {
            "probability": round(float(visual), 5),
            "per_model": {
                n: {
                    "probability": round(float(p), 5),
                    "weight": round(weights[n], 4),
                    "status": self.frame_scorers[n].kind,
                }
                for n, p in per_model_video.items()
            },
            "temporal_probability": None if temporal_prob is None else round(temporal_prob, 5),
            "aggregation": method,
            "frames": frames,
            "faces_detected": int(face_mask.sum()),
            "frames_analyzed": len(crops),
        }, gradcam

    def _gradcam(self, crops, batch, ensemble, use) -> dict[str, Any] | None:
        from ..explain.gradcam import encode_png_base64, overlay_heatmap

        candidates = np.where(use)[0]
        idx = int(candidates[np.argmax(ensemble[candidates])])
        weights = self._ensemble_weights()
        name = max(self.frame_scorers, key=lambda n: weights.get(n, 0.0))
        heat = self.frame_scorers[name].heatmap(batch[idx], crops[idx].crop_bgr)
        if heat is None:
            return None
        return {
            "frame_index": int(crops[idx].frame_index),
            "model": name,
            "method": "grad-cam" if self.frame_scorers[name].kind != "placeholder" else "pseudo-saliency (demo)",
            "overlay": encode_png_base64(overlay_heatmap(crops[idx].crop_bgr, heat)),
            "face": encode_png_base64(crops[idx].crop_bgr),
        }

    # ------------------------------------------------------------------ audio
    def _score_audio(self, path: Path) -> dict[str, Any]:
        if self.audio is None:
            return {"available": False, "reason": "audio branch disabled"}
        try:
            signal = load_audio(path, self.cfg.audio.sample_rate)
        except AudioUnavailableError as exc:
            return {"available": False, "reason": str(exc)}
        if signal.size < self.cfg.audio.sample_rate * 0.5 or float(np.abs(signal).max(initial=0.0)) < 1e-4:
            return {"available": False, "reason": "audio track is empty or silent"}
        clips = audio_to_mel_clips(signal, self.cfg.audio)
        scores = self.audio.predict(clips)
        clip_s = self.cfg.audio.clip_seconds
        return {
            "available": True,
            "probability": round(float(np.mean(scores)), 5),
            "status": self.audio.kind,
            "duration": round(signal.size / self.cfg.audio.sample_rate, 3),
            "segments": [
                {"start": round(i * clip_s, 3), "end": round((i + 1) * clip_s, 3), "probability": round(float(s), 5)}
                for i, s in enumerate(scores)
            ],
        }

    # ------------------------------------------------------------------ public API
    def predict_file(self, path: str | Path, media_type: str | None = None) -> dict[str, Any]:
        path = Path(path)
        media_type = media_type or media_type_for(path)
        start = time.perf_counter()
        with self._lock:
            visual, gradcam, audio = None, None, None
            if media_type == "image":
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError("Could not decode image")
                visual, gradcam = self._score_crops([self.extractor.crop_image(image)])
            elif media_type == "video":
                result = self.extractor.extract_video(path, self.cfg.api.frames_per_video, use_cache=False)
                if not result.crops:
                    raise ValueError("No frames could be decoded from the video")
                visual, gradcam = self._score_crops(result.crops)
                audio = self._score_audio(path)
            elif media_type == "audio":
                audio = self._score_audio(path)
                if not audio.get("available"):
                    raise ValueError(f"Audio could not be analysed: {audio.get('reason')}")
            else:
                raise ValueError(f"Unsupported media type '{media_type}'")
        return self._assemble(media_type, visual, audio, gradcam, start)

    def _assemble(self, media_type, visual, audio, gradcam, start) -> dict[str, Any]:
        f = self.cfg.fusion
        probs = {
            "visual": visual["probability"] if visual else None,
            "audio": audio["probability"] if audio and audio.get("available") else None,
        }
        probability = late_fusion(probs, {"visual": f.visual_weight, "audio": f.audio_weight}, f.method)
        mode = (
            self.mode if visual is not None else ("trained" if audio and audio.get("status") == "trained" else "demo")
        )
        return {
            "media_type": media_type,
            "mode": mode,
            "warning": DEMO_WARNING if mode != "trained" else None,
            "label": "FAKE" if probability >= f.threshold else "REAL",
            "probability_fake": round(probability, 5),
            "threshold": f.threshold,
            "modalities": {"visual": visual, "audio": audio},
            "fusion": {
                "method": f.method,
                "weights": {"visual": f.visual_weight, "audio": f.audio_weight},
                "used": [k for k, v in probs.items() if v is not None],
            },
            "gradcam": gradcam,
            "processing_ms": round((time.perf_counter() - start) * 1000.0, 1),
            "version": __version__,
        }
