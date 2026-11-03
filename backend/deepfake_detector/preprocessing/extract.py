"""Offline preprocessing CLI.

Faces (video datasets)::

    python -m deepfake_detector.preprocessing.extract faces --config configs/default.yaml --workers 4

writes ``<crops_root>/<video_id>/frame_XXXXXX.png`` and ``<crops_root>/manifest.csv``.

Audio (``<audio_root>/{real,fake}/**``)::

    python -m deepfake_detector.preprocessing.extract audio --config configs/default.yaml

writes per-file log-mel clips ``<audio_root>_features/<id>.npy`` and a manifest.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from ..config import Config, load_config
from ..data.layouts import AUDIO_EXTS, VIDEO_EXTS, assign_splits, discover, discover_generic, write_manifest
from ..utils.logging import configure_logging
from .audio import AudioUnavailableError, audio_to_mel_clips, load_audio
from .pipeline import FaceExtractor

logger = logging.getLogger(__name__)

_WORKER_EXTRACTOR: FaceExtractor | None = None


def _init_worker(cfg: Config) -> None:
    global _WORKER_EXTRACTOR
    cv2.setNumThreads(1)
    _WORKER_EXTRACTOR = FaceExtractor(cfg.face, cfg.sampling, cache_dir=None)


def _extract_one(
    video_id: str, path: str, out_dir: str, overwrite: bool, keep_faceless: bool = False
) -> tuple[str, int]:
    assert _WORKER_EXTRACTOR is not None
    target = Path(out_dir) / video_id
    if target.is_dir() and not overwrite:
        existing = len(list(target.glob("frame_*.png")))
        if existing:
            return video_id, existing
    target.mkdir(parents=True, exist_ok=True)
    try:
        result = _WORKER_EXTRACTOR.extract_video(path, keep_faceless=keep_faceless, use_cache=False)
    except ValueError as exc:
        logger.warning("skipping unreadable video", extra={"video": path, "error": str(exc)})
        return video_id, 0
    for crop in result.crops:
        cv2.imwrite(str(target / f"frame_{crop.frame_index:06d}.png"), crop.crop_bgr)
    return video_id, len(result.crops)


def extract_faces(
    cfg: Config, workers: int = 1, overwrite: bool = False, limit: int | None = None, keep_faceless: bool = False
) -> Path:
    d = cfg.data
    records = discover(
        d.dataset,
        d.raw_root,
        compression=d.compression,
        val_fraction=d.val_fraction,
        test_fraction=d.test_fraction,
        seed=d.seed,
    )
    if limit:
        records = records[:limit]
    if not records:
        raise SystemExit(f"No videos found under {d.raw_root} for layout '{d.dataset}'")
    logger.info("discovered videos", extra={"count": len(records), "dataset": d.dataset})
    out_dir = Path(d.crops_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init_worker, initargs=(cfg,)) as pool:
        futures = [
            pool.submit(_extract_one, r.video_id, r.path, str(out_dir), overwrite, keep_faceless) for r in records
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            vid, n = fut.result()
            counts[vid] = {"n_frames": n}
            if i % 50 == 0 or i == len(futures):
                logger.info("progress", extra={"done": i, "total": len(futures)})
    kept = [r for r in records if counts.get(r.video_id, {}).get("n_frames", 0) > 0]
    manifest = out_dir / "manifest.csv"
    write_manifest(kept, manifest, ["n_frames"], counts)
    logger.info("wrote manifest", extra={"path": str(manifest), "videos": len(kept)})
    return manifest


def extract_audio(cfg: Config) -> Path:
    d, a = cfg.data, cfg.audio
    items = discover_generic(Path(d.audio_root), AUDIO_EXTS | VIDEO_EXTS)
    records = assign_splits(items, d.val_fraction, d.test_fraction, d.seed)
    if not records:
        raise SystemExit(f"No audio found under {d.audio_root}/{{real,fake}}")
    feat_dir = Path(f"{d.audio_root.rstrip('/')}_features")
    feat_dir.mkdir(parents=True, exist_ok=True)
    extras: dict[str, dict] = {}
    kept = []
    for rec in records:
        try:
            clips = audio_to_mel_clips(load_audio(rec.path, a.sample_rate), a)
        except AudioUnavailableError as exc:
            logger.warning("skipping audio", extra={"path": rec.path, "error": str(exc)})
            continue
        if len(clips) == 0:
            continue
        npy = feat_dir / f"{rec.video_id}.npy"
        np.save(npy, clips.astype(np.float16))
        extras[rec.video_id] = {"features": str(npy), "n_clips": len(clips)}
        kept.append(rec)
    manifest = feat_dir / "manifest.csv"
    write_manifest(kept, manifest, ["features", "n_clips"], extras)
    logger.info("wrote audio manifest", extra={"path": str(manifest), "files": len(kept)})
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    faces = sub.add_parser("faces", help="extract aligned face crops from a video dataset")
    faces.add_argument("--config", default=None)
    faces.add_argument("--workers", type=int, default=4)
    faces.add_argument("--overwrite", action="store_true")
    faces.add_argument("--limit", type=int, default=None, help="process only the first N videos (smoke test)")
    faces.add_argument(
        "--keep-faceless",
        action="store_true",
        help="keep a centre crop when no face is found (synthetic smoke tests only)",
    )
    audio = sub.add_parser("audio", help="compute log-mel features for an audio dataset")
    audio.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    configure_logging()
    cfg = load_config(args.config)
    if args.cmd == "faces":
        extract_faces(cfg, args.workers, args.overwrite, args.limit, args.keep_faceless)
    else:
        extract_audio(cfg)


if __name__ == "__main__":
    main()
