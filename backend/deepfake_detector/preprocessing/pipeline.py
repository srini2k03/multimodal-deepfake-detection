"""High-level face extraction pipeline with on-disk crop caching."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import FaceConfig, SamplingConfig
from .alignment import align_and_crop, center_crop
from .face_detector import FaceDetector, build_face_detector
from .video import sample_frames

logger = logging.getLogger(__name__)


@dataclass
class FaceCrop:
    frame_index: int
    timestamp_s: float
    crop_bgr: np.ndarray
    face_found: bool
    box: tuple[int, int, int, int] | None = None
    angle_deg: float = 0.0


@dataclass
class ExtractionResult:
    source: str
    crops: list[FaceCrop] = field(default_factory=list)
    fps: float = 0.0
    frame_count: int = 0

    @property
    def faces_found(self) -> int:
        return sum(c.face_found for c in self.crops)


class FaceExtractor:
    """Detect, align and crop faces from images or sampled video frames."""

    def __init__(
        self,
        face_cfg: FaceConfig,
        sampling_cfg: SamplingConfig | None = None,
        detector: FaceDetector | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.face_cfg = face_cfg
        self.sampling_cfg = sampling_cfg or SamplingConfig()
        self.detector = detector or build_face_detector(face_cfg)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    # ------------------------------------------------------------------ images
    def crop_image(
        self, image_bgr: np.ndarray, frame_index: int = 0, timestamp_s: float = 0.0, fallback_center: bool = True
    ) -> FaceCrop | None:
        box = self.detector.largest(image_bgr)
        if box is None:
            if not fallback_center:
                return None
            return FaceCrop(frame_index, timestamp_s, center_crop(image_bgr, self.face_cfg.crop_size), False)
        aligned = align_and_crop(image_bgr, box, self.face_cfg.crop_size, self.face_cfg.margin, self.face_cfg.align)
        return FaceCrop(
            frame_index, timestamp_s, aligned.crop_bgr, True, (box.x, box.y, box.w, box.h), aligned.angle_deg
        )

    # ------------------------------------------------------------------ videos
    def _cache_key(self, path: Path, num_frames: int) -> str:
        stat = path.stat()
        payload = json.dumps(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "frames": num_frames,
                "face": asdict(self.face_cfg),
                "sampling": asdict(self.sampling_cfg),
                "detector": self.detector.name,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _load_cache(self, cache_path: Path) -> ExtractionResult | None:
        meta_file = cache_path / "meta.json"
        if not meta_file.exists():
            return None
        meta = json.loads(meta_file.read_text())
        crops = []
        for item in meta["crops"]:
            img = cv2.imread(str(cache_path / item["file"]))
            if img is None:
                return None
            box = tuple(item["box"]) if item["box"] else None
            crops.append(
                FaceCrop(item["frame_index"], item["timestamp_s"], img, item["face_found"], box, item["angle_deg"])
            )
        return ExtractionResult(meta["source"], crops, meta["fps"], meta["frame_count"])

    def _save_cache(self, cache_path: Path, result: ExtractionResult) -> None:
        cache_path.mkdir(parents=True, exist_ok=True)
        items = []
        for crop in result.crops:
            name = f"frame_{crop.frame_index:06d}.png"
            cv2.imwrite(str(cache_path / name), crop.crop_bgr)
            items.append(
                {
                    "file": name,
                    "frame_index": crop.frame_index,
                    "timestamp_s": crop.timestamp_s,
                    "face_found": crop.face_found,
                    "box": list(crop.box) if crop.box else None,
                    "angle_deg": crop.angle_deg,
                }
            )
        meta = {"source": result.source, "fps": result.fps, "frame_count": result.frame_count, "crops": items}
        (cache_path / "meta.json").write_text(json.dumps(meta, indent=2))

    def extract_video(
        self, path: str | Path, num_frames: int | None = None, keep_faceless: bool = True, use_cache: bool = True
    ) -> ExtractionResult:
        path = Path(path)
        num_frames = num_frames or self.sampling_cfg.frames_per_video
        cache_path = None
        if self.cache_dir and use_cache:
            cache_path = self.cache_dir / f"{path.stem}_{self._cache_key(path, num_frames)}"
            cached = self._load_cache(cache_path)
            if cached is not None:
                logger.debug("crop cache hit", extra={"video": str(path)})
                return cached

        info, frames = sample_frames(path, num_frames, self.sampling_cfg.strategy, self.sampling_cfg.stride)
        result = ExtractionResult(str(path), fps=info.fps, frame_count=info.frame_count)
        for frame in frames:
            crop = self.crop_image(frame.image_bgr, frame.index, frame.timestamp_s, fallback_center=keep_faceless)
            if crop is not None:
                result.crops.append(crop)
        if cache_path is not None:
            self._save_cache(cache_path, result)
        return result


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
