"""Cross-dataset, video-level test of the visual model on WildDeepfake (test split only).

WildDeepfake (Zi et al., ACM MM 2020; https://huggingface.co/datasets/xingjunm/WildDeepfake,
Apache-2.0, ungated author upload) holds face sequences extracted from real and deepfake videos
collected on the internet. Each ``*.tar.gz`` in ``real_test/`` or ``fake_test/`` is one source
video; images inside are grouped into sequences by their parent directory.

The DeepFakeFace-trained models never see this data: no training, threshold or calibration is
fitted on it. Up to ``frames_per_sequence`` frames are sampled uniformly from each sequence, sent
through the same face pipeline as the API and scored; frame scores are then aggregated per
sequence with every method the API offers, so frame-level and video-level AUC can be compared.

    python -m deepfake_detector.benchmarks.wilddeepfake --tars dl/ --models m1.keras m2.keras --out results/
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from ..config import FaceConfig, load_config
from ..evaluation.metrics import eer_from_scores, roc_auc
from ..inference.fusion import AGGREGATIONS, aggregate_frames
from ..utils.logging import configure_logging
from .image_ops import equalize, to_model_input
from .run_asvspoof import _write_json, run_metadata

logger = logging.getLogger(__name__)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SOURCE = "https://huggingface.co/datasets/xingjunm/WildDeepfake (Zi et al., ACM MM 2020)"


def _frame_key(name: str) -> tuple:
    stem = PurePosixPath(name).stem
    digits = "".join(c for c in stem if c.isdigit())
    return (int(digits) if digits else 0, stem)


def read_tar(path: Path, label: int, frames_per_sequence: int, face_cfg: dict) -> list[dict]:
    """Return one record per sampled frame: sequence id, label, face crop (RGB uint8)."""
    from ..preprocessing.pipeline import FaceExtractor

    extractor = FaceExtractor(FaceConfig(**face_cfg))
    seqs: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    out: list[dict] = []
    with tarfile.open(path, "r:*") as tar:  # the WildDeepfake ".tar.gz" files are plain tar archives
        members = [m for m in tar.getmembers() if m.isfile() and PurePosixPath(m.name).suffix.lower() in IMAGE_EXTS]
        for m in members:
            seqs[str(PurePosixPath(m.name).parent)].append(m)
        for seq, items in sorted(seqs.items()):
            items.sort(key=lambda m: _frame_key(m.name))
            idx = np.linspace(0, len(items) - 1, min(frames_per_sequence, len(items))).round().astype(int)
            for i in sorted(set(idx.tolist())):
                data = tar.extractfile(items[i]).read()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                crop = extractor.crop_image(img)
                rgb = cv2.cvtColor(cv2.resize(crop.crop_bgr, (face_cfg["crop_size"],) * 2), cv2.COLOR_BGR2RGB)
                out.append(
                    {
                        "sequence": f"{path.parent.name}/{path.name}:{seq}",
                        "video": f"{path.parent.name}/{path.name}",
                        "label": label,
                        "face_found": int(crop.face_found),
                        "src_px": int(min(img.shape[:2])),
                        "rgb": rgb,
                    }
                )
    return out


def _job(args: tuple) -> list[dict]:
    return read_tar(*args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tars", required=True, help="directory with real_test/ and fake_test/ tar.gz files")
    parser.add_argument("--models", nargs="+", required=True, help="name=path.keras[:eqPX]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    configure_logging(json_logs=True)
    cfg = load_config(args.config)
    face = {**vars(cfg.face), "crop_size": args.image_size}
    jobs = [(p, 0 if "real" in p.parent.name else 1, args.frames_per_sequence, face)
            for p in sorted(Path(args.tars).rglob("*.tar.gz"))]  # fmt: skip
    if not jobs:
        raise SystemExit(f"no tar.gz files under {args.tars}")
    t0 = time.perf_counter()
    records: list[dict] = []
    with get_context("fork").Pool(args.workers) as pool:  # before TensorFlow is imported
        for n, recs in enumerate(pool.imap_unordered(_job, jobs), 1):
            records.extend(recs)
            if n % 20 == 0:
                logger.info("tars", extra={"done": n, "total": len(jobs), "frames": len(records)})
    prep_s = time.perf_counter() - t0
    images = np.stack([r.pop("rgb") for r in records])
    y_frame = np.array([r["label"] for r in records])
    seq_ids = sorted({r["sequence"] for r in records})
    seq_index = {s: i for i, s in enumerate(seq_ids)}
    seq_of = np.array([seq_index[r["sequence"]] for r in records])
    y_seq = np.zeros(len(seq_ids), dtype=int)
    for r in records:
        y_seq[seq_index[r["sequence"]]] = r["label"]

    from ..inference.scorers import load_keras_model
    from .audio_models import logit_model

    report: dict = {
        "meta": run_metadata(),
        "dataset": "WildDeepfake test split (cross-dataset; models trained on DeepFakeFace)",
        "source": SOURCE,
        "licence": "Apache-2.0 (Hugging Face dataset card)",
        "n_videos": len(jobs),
        "n_videos_real": sum(1 for j in jobs if j[1] == 0),
        "n_sequences": len(seq_ids),
        "n_sequences_real": int((y_seq == 0).sum()),
        "n_frames": len(records),
        "frames_per_sequence_max": args.frames_per_sequence,
        "face_detected_rate": float(np.mean([r["face_found"] for r in records])),
        "median_source_px": {
            "real": float(np.median([r["src_px"] for r in records if r["label"] == 0])),
            "fake": float(np.median([r["src_px"] for r in records if r["label"] == 1])),
        },
        "preprocess_seconds": round(prep_s, 1),
        "systems": {},
    }
    for spec in args.models:
        name, _, rest = spec.partition("=")
        path, _, eq = rest.partition(":eq")
        lm = logit_model(load_keras_model(Path(path)))
        z = []
        for s in range(0, len(images), 64):
            chunk = images[s : s + 64]
            if eq:
                chunk = np.stack([equalize(im, int(eq)) for im in chunk])
            z.append(np.asarray(lm(to_model_input(chunk), training=False)).reshape(-1))
        z = np.concatenate(z).astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-z))
        frame_eer = eer_from_scores(-z[y_frame == 0], -z[y_frame == 1])[0]
        entry = {"frame_level": {"roc_auc": roc_auc(y_frame, z), "eer": frame_eer}, "sequence_level": {}}
        for agg in AGGREGATIONS:
            s_scores = np.array([aggregate_frames(p[seq_of == i], agg) for i in range(len(seq_ids))])
            entry["sequence_level"][agg] = {
                "roc_auc": roc_auc(y_seq, s_scores),
                "eer": eer_from_scores(-s_scores[y_seq == 0], -s_scores[y_seq == 1])[0],
            }
        report["systems"][name] = entry
        logger.info("wilddeepfake", extra={"system": name, "frame_auc": entry["frame_level"]["roc_auc"]})
    _write_json(Path(args.out) / "cross_dataset_wilddeepfake.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "meta"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
