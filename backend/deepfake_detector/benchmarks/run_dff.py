"""DeepFakeFace image benchmark for the visual branch (runs in GitHub Actions, CPU only).

    python -m deepfake_detector.benchmarks.run_dff prepare  --zips dff/ --work work/
    python -m deepfake_detector.benchmarks.run_dff probe    --work work/
    python -m deepfake_detector.benchmarks.run_dff finetune --work work/ [--exclude insight]
    python -m deepfake_detector.benchmarks.run_dff evaluate --work work/ --out results/
    python -m deepfake_detector.benchmarks.run_dff robustness --work work/ --out results/
    python -m deepfake_detector.benchmarks.run_dff gradcam  --work work/ --out results/
    python -m deepfake_detector.benchmarks.run_dff latency  --work work/ --out results/

Systems: ``meta_lr`` (file metadata only: format, resolution, size, face-detector output - a
shortcut probe), ``effb0_probe`` (frozen ImageNet EfficientNet-B0 + logistic regression),
``effb0_ft`` (EfficientNet-B0 fine-tuned end to end on the top 30% of layers) and
leave-one-generator-out variants that never see one manipulation type in training.
Thresholds and calibration are fitted on the identity-disjoint validation split.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np

from ..config import FaceConfig, load_config
from ..evaluation.calibration import fit_platt
from ..evaluation.metrics import (
    binary_report,
    brier_score,
    eer_from_scores,
    expected_calibration_error,
    reliability_bins,
    roc_auc,
)
from ..utils.logging import configure_logging
from . import dff
from .audio_models import LogisticRegression
from .image_ops import ROBUSTNESS_CONDITIONS, apply_condition, augment, to_model_input
from .run_asvspoof import _write_json, run_metadata

logger = logging.getLogger(__name__)
IMAGE_SIZE = 224
BACKBONE = "efficientnet_b0"
WEIGHTS: str | None = "imagenet"  # tests set None (no download)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -60, 60)))


def _records(work: Path) -> list[dff.ImageRecord]:
    return dff.read_index(work / "index.csv")


def _split(records, split):
    idx = [i for i, r in enumerate(records) if r.split == split]
    return idx, np.array([records[i].label for i in idx]), np.array([records[i].subset for i in idx])


# --------------------------------------------------------------------------- prepare
def cmd_prepare(args, cfg) -> None:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    all_records = []
    for subset in dff.SUBSETS:
        recs = dff.index_zip(Path(args.zips) / f"{subset}.zip", subset)
        logger.info("indexed", extra={"subset": subset, "n": len(recs), "examples": [r.member for r in recs[:2]]})
        all_records.extend(recs)
    dff.assign_splits(all_records, args.val_fraction, args.test_fraction, cfg.data.seed)
    chosen = dff.select_groups(
        all_records, {"train": args.train_groups, "val": args.val_groups, "test": args.test_groups}, cfg.data.seed
    )
    face = FaceConfig(**{**vars(cfg.face), "crop_size": IMAGE_SIZE})
    t0 = time.perf_counter()
    chosen = dff.extract_crops(chosen, args.zips, work / "crops", face, workers=args.workers)
    logger.info("crops done", extra={"n": len(chosen), "seconds": round(time.perf_counter() - t0, 1)})
    dff.write_index(chosen, work / "index.csv")
    audit = {
        "dataset": "DeepFakeFace (DFF)",
        "source": "https://huggingface.co/datasets/OpenRL/DeepFakeFace",
        "licence": "dataset card: openrail (README body also shows apache-2.0); real photos from IMDB-WIKI "
        "(academic research use)",
        "selection": {k: getattr(args, k) for k in ("train_groups", "val_groups", "test_groups")},
        "split_fractions": {"val": args.val_fraction, "test": args.test_fraction},
        "face": {"detector": face.detector, "crop_size": face.crop_size, "margin": face.margin, "align": face.align},
        "crop_reencode": "JPEG q=95 for every crop (real and fake)",
        "crop_seconds": round(time.perf_counter() - t0, 1),
        **dff.audit(chosen, all_records),
        "meta": run_metadata(),
    }
    _write_json(work / "audit.json", audit)


# --------------------------------------------------------------------------- probe baselines
def _feature_model():
    import keras

    from ..models.backbones import build_backbone

    inp = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = keras.layers.Rescaling(127.5, offset=127.5)(inp)
    return keras.Model(inp, build_backbone(BACKBONE, IMAGE_SIZE, WEIGHTS)(x, training=False))


def _batched(model, images_uint8: np.ndarray, batch: int = 64, fn=None) -> np.ndarray:
    out = []
    for s in range(0, len(images_uint8), batch):
        x = to_model_input(images_uint8[s : s + batch])
        out.append(np.asarray(model(x, training=False) if fn is None else fn(x)))
    return np.concatenate(out)


def meta_features(records: list[dff.ImageRecord]) -> np.ndarray:
    rows = []
    for r in records:
        rows.append([
            r.file_format == "jpeg", r.file_format == "png", np.log1p(r.width), np.log1p(r.height),
            np.log1p(r.file_bytes / max(1, r.width * r.height)), r.face_found, r.face_frac,
        ])  # fmt: skip
    return np.asarray(rows, dtype=np.float64)


def cmd_probe(args, cfg) -> None:
    work = Path(args.work)
    records = _records(work)
    images = dff.load_images(records, work / "crops", IMAGE_SIZE)
    t0 = time.perf_counter()
    feats = _batched(_feature_model(), images)
    logger.info("probe features", extra={"n": len(feats), "seconds": round(time.perf_counter() - t0, 1)})
    np.save(work / "effb0_features.npy", feats.astype(np.float32))
    tr, ytr, ttr = _split(records, "train")
    logits = {}
    meta = meta_features(records)
    logits["meta_lr"] = LogisticRegression().fit(meta[tr], ytr).decision_function(meta)
    logits["effb0_probe"] = LogisticRegression(l2=1e-2).fit(feats[tr], ytr).decision_function(feats)
    for held_out in dff.FAKE_TYPES:  # leave-one-generator-out: that fake type never seen in training
        keep = [i for i, t in zip(tr, ttr, strict=True) if t != held_out]
        y = np.array([records[i].label for i in keep])
        logits[f"effb0_probe_no_{held_out}"] = LogisticRegression(l2=1e-2).fit(feats[keep], y).decision_function(feats)
    (work / "logits").mkdir(exist_ok=True)
    for name, z in logits.items():
        np.save(work / "logits" / f"{name}.npy", z)


# --------------------------------------------------------------------------- fine-tuning
def cmd_finetune(args, cfg) -> None:
    import keras

    from ..models import build_frame_classifier
    from ..models.backbones import set_trainable

    work = Path(args.work)
    name = "effb0_ft" + (f"_no_{args.exclude}" if args.exclude else "")
    records = _records(work)
    images = dff.load_images(records, work / "crops", IMAGE_SIZE)
    tr, ytr, ttr = _split(records, "train")
    va, yva, _ = _split(records, "val")
    if args.exclude:
        keep = ttr != args.exclude
        tr, ytr, ttr = list(np.array(tr)[keep]), ytr[keep], ttr[keep]
    keras.utils.set_random_seed(cfg.data.seed)
    rng = np.random.default_rng(cfg.data.seed)
    model = build_frame_classifier(BACKBONE, IMAGE_SIZE, weights=WEIGHTS, dropout=0.3)
    real = np.array([i for i, y in zip(tr, ytr, strict=True) if y == 0])
    fake = np.array([i for i, y in zip(tr, ytr, strict=True) if y == 1])

    class Batches(keras.utils.PyDataset):
        def __init__(self, idx: np.ndarray, seed: int):
            super().__init__(workers=4, use_multiprocessing=False)
            self.idx, self.seed = idx, seed

        def __len__(self):
            return int(np.ceil(len(self.idx) / args.batch_size))

        def __getitem__(self, b):
            sel = self.idx[b * args.batch_size : (b + 1) * args.batch_size]
            r = np.random.default_rng((self.seed, b))
            x = np.stack([augment(images[i], r) for i in sel])
            y = np.array([records[i].label for i in sel], dtype=np.float32)
            return to_model_input(x), y

    def epoch_indices() -> np.ndarray:  # class-balanced: every real image + as many sampled fakes
        picked = np.r_[real, rng.choice(fake, min(len(fake), len(real)), replace=False)]
        return rng.permutation(picked)

    def val_auc() -> float:
        z = _batched(model, images[va]).reshape(-1)
        return roc_auc(yva, z)

    history, best_auc, best_w, best_epoch, bad = [], -1.0, None, 0, 0
    t_start = time.perf_counter()
    stages = [("head", 1e-3, 1), ("finetune", 1e-4, args.epochs)]
    epoch = 0
    for stage, lr, n_epochs in stages:
        set_trainable(model, stage, 0.3)
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.perf_counter()
            h = model.fit(Batches(epoch_indices(), cfg.data.seed + epoch), epochs=1, verbose=0)
            auc = val_auc()
            row = {"epoch": epoch, "stage": stage, "lr": lr, "loss": float(h.history["loss"][-1]), "val_auc": auc,
                   "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
            history.append(row)
            logger.info("epoch", extra=row)
            if auc > best_auc + 1e-4:
                best_auc, best_w, best_epoch, bad = auc, model.get_weights(), epoch, 0
            else:
                bad += 1
                if stage == "finetune" and bad >= args.patience:
                    break
    model.set_weights(best_w)
    out = work / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    from .audio_models import logit_model

    z = _batched(logit_model(model), images).reshape(-1)
    (work / "logits").mkdir(exist_ok=True)
    np.save(work / "logits" / f"{name}.npy", z)
    _write_json(out / "train.json", {"system": name, "history": history, "best_epoch": best_epoch,
                                     "best_val_auc": best_auc, "train_seconds": round(time.perf_counter() - t_start, 1),
                                     "params": int(model.count_params()), "excluded_type": args.exclude,
                                     "n_train_real": len(real), "n_train_fake": len(fake)})  # fmt: skip


# --------------------------------------------------------------------------- evaluation
NOTES = {
    "meta_lr": "Logistic regression on file metadata only (format, resolution, bytes/pixel, face box): shortcut probe",
    "effb0_probe": "Frozen ImageNet EfficientNet-B0 features + logistic regression",
    "effb0_ft": "EfficientNet-B0 fine-tuned (top 30% of layers, BatchNorm frozen), augmentation, balanced epochs",
}


def evaluate_system(z: np.ndarray, records, name: str) -> dict:
    va, yva, _ = _split(records, "val")
    te, yte, tte = _split(records, "test")
    zva, zte = z[va], z[te]
    _, thr_real = eer_from_scores(-zva[yva == 0], -zva[yva == 1])
    thr = -thr_real
    platt = fit_platt(zva, yva)
    held_out = name.split("_no_")[1] if "_no_" in name else None
    rep = binary_report(yte, zte, threshold=thr).to_dict()
    p_raw, p_cal = _sigmoid(zte), platt.transform(zte)
    out = {
        "description": NOTES.get(name.split("_no_")[0], name),
        "held_out_type": held_out,
        "val": {"roc_auc": roc_auc(yva, zva), "eer": eer_from_scores(-zva[yva == 0], -zva[yva == 1])[0]},
        "test": {
            "n_real": int((yte == 0).sum()),
            "n_fake": int((yte == 1).sum()),
            "roc_auc": rep["auc"],
            "pr_auc_fake": rep["pr_auc"],
            "eer": eer_from_scores(-zte[yte == 0], -zte[yte == 1])[0],
            "at_val_threshold": {
                k: rep[k]
                for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "confusion_matrix", "threshold")
            },
            "calibration": {
                "raw": {"ece": expected_calibration_error(yte, p_raw), "brier": brier_score(yte, p_raw)},
                "platt": {"ece": expected_calibration_error(yte, p_cal), "brier": brier_score(yte, p_cal)},
                "reliability_platt": reliability_bins(yte, p_cal, 10),
            },
        },
        "platt": platt.to_dict(),
    }
    per_type = {}
    for t in dff.FAKE_TYPES:
        m = (tte == "wiki") | (tte == t)
        y, s = yte[m], zte[m]
        per_type[t] = {
            "n_fake": int((y == 1).sum()),
            "roc_auc": roc_auc(y, s),
            "eer": eer_from_scores(-s[y == 0], -s[y == 1])[0],
            "detected_at_val_threshold": float(np.mean(s[y == 1] >= thr)),
            "seen_in_training": t != held_out,
        }
    out["test"]["per_type"] = per_type
    return out


def cmd_evaluate(args, cfg) -> None:
    work, out_dir = Path(args.work), Path(args.out)
    records = _records(work)
    results = {"meta": run_metadata(), "systems": {}}
    logits = {p.stem: np.load(p) for p in sorted((work / "logits").glob("*.npy"))}
    for name, z in logits.items():
        entry = evaluate_system(z, records, name)
        tj = work / "models" / name / "train.json"
        if tj.exists():
            entry["training"] = json.loads(tj.read_text())
        results["systems"][name] = entry
        logger.info("evaluated", extra={"system": name, "test_auc": entry["test"]["roc_auc"]})
    _write_json(out_dir / "metrics.json", results)
    te, _, _ = _split(records, "test")
    with gzip.open(out_dir / "test_scores.csv.gz", "wt", encoding="utf-8") as fh:
        names = list(logits)
        fh.write("group,subset,label," + ",".join(names) + "\n")
        for i in te:
            r = records[i]
            fh.write(f"{r.group},{r.subset},{r.label}," + ",".join(f"{logits[n][i]:.5f}" for n in names) + "\n")
    _write_json(out_dir / "data_audit.json", json.loads((work / "audit.json").read_text()))


# --------------------------------------------------------------------------- robustness
def _subset(records, n_per_class: int, seed: int) -> list[int]:
    te, yte, tte = _split(records, "test")
    rng = np.random.default_rng(seed)
    real = [i for i, y in zip(te, yte, strict=True) if y == 0]
    pick = list(rng.choice(real, min(n_per_class, len(real)), replace=False))
    per = n_per_class // len(dff.FAKE_TYPES)
    for t in dff.FAKE_TYPES:
        cand = [i for i, tt in zip(te, tte, strict=True) if tt == t]
        pick += list(rng.choice(cand, min(per, len(cand)), replace=False))
    return sorted(int(i) for i in pick)


def _load_ft(work: Path, name: str = "effb0_ft"):
    from ..inference.scorers import load_keras_model

    path = work / "models" / name / "model.keras"
    return load_keras_model(path) if path.exists() else None


def cmd_robustness(args, cfg) -> None:
    from .audio_models import logit_model

    work, out_dir = Path(args.work), Path(args.out)
    records = _records(work)
    metrics = json.loads((out_dir / "metrics.json").read_text())["systems"]
    pick = _subset(records, args.n_per_class, cfg.data.seed)
    images = dff.load_images([records[i] for i in pick], work / "crops", IMAGE_SIZE)
    y = np.array([records[i].label for i in pick])
    feats_all = np.load(work / "effb0_features.npy")
    tr, ytr, _ = _split(records, "train")
    probe = LogisticRegression(l2=1e-2).fit(feats_all[tr], ytr)
    fm = _feature_model()
    ft = _load_ft(work)
    systems = {"effb0_probe": lambda x: probe.decision_function(_batched(fm, x))}
    if ft is not None:
        lm = logit_model(ft)
        systems["effb0_ft"] = lambda x: _batched(lm, x).reshape(-1)
    report = {"meta": run_metadata(), "subset": {"n": len(pick), "n_real": int((y == 0).sum())}, "conditions": {}}
    for cond in ROBUSTNESS_CONDITIONS:
        rng = np.random.default_rng(cfg.data.seed)
        xs = np.stack([apply_condition(img, cond, rng) for img in images])
        entry = {}
        for name, fn in systems.items():
            z = fn(xs)
            thr = metrics[name]["test"]["at_val_threshold"]["threshold"]
            pred = z >= thr
            entry[name] = {
                "roc_auc": roc_auc(y, z),
                "eer": eer_from_scores(-z[y == 0], -z[y == 1])[0],
                "fpr_at_val_threshold": float(np.mean(pred[y == 0])),
                "fnr_at_val_threshold": float(np.mean(~pred[y == 1])),
            }
        report["conditions"][cond] = entry
        logger.info("robustness", extra={"condition": cond, **{k: v["roc_auc"] for k, v in entry.items()}})
    _write_json(out_dir / "robustness.json", report)


# --------------------------------------------------------------------------- explainability
def cmd_gradcam(args, cfg) -> None:
    """Aggregate Grad-CAM statistics (no dataset images are written: they show real people)."""
    import cv2

    from ..explain.gradcam import compute_gradcam

    work, out_dir = Path(args.work), Path(args.out)
    records = _records(work)
    ft = _load_ft(work)
    if ft is None:
        raise SystemExit("effb0_ft model missing")
    pick = _subset(records, args.n_per_class, cfg.data.seed + 7)
    images = dff.load_images([records[i] for i in pick], work / "crops", IMAGE_SIZE)
    groups: dict[str, list[np.ndarray]] = {}
    t0 = time.perf_counter()
    for idx, img in zip(pick, images, strict=True):
        heat = compute_gradcam(ft, to_model_input(img[None])[0])
        groups.setdefault(records[idx].subset, []).append(heat)
    seconds = (time.perf_counter() - t0) / len(pick)
    h = w = IMAGE_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    centre = (np.abs(yy - h / 2) < h / 4) & (np.abs(xx - w / 2) < w / 4)
    lower = yy > h * 0.55  # mouth / jaw region of an aligned crop
    stats = {"seconds_per_image": round(seconds, 4), "per_subset": {}}
    tiles = []
    for subset in dff.SUBSETS:
        hs = np.stack(groups.get(subset, [np.zeros((h, w), np.float32)]))
        mass = hs.sum(axis=(1, 2)) + 1e-9
        stats["per_subset"][subset] = {
            "n": len(groups.get(subset, [])),
            "centre_mass": round(float(np.mean(hs[:, centre].sum(axis=1) / mass)), 4),
            "lower_face_mass": round(float(np.mean(hs[:, lower].sum(axis=1) / mass)), 4),
            "centre_area_fraction": round(float(centre.mean()), 4),
        }
        mean = hs.mean(axis=0)
        tile = cv2.applyColorMap(np.uint8(255 * mean / max(mean.max(), 1e-9)), cv2.COLORMAP_VIRIDIS)
        cv2.putText(tile, subset, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cv2.imwrite(str(out_dir / "gradcam_mean_heatmaps.png"), np.hstack(tiles))
    _write_json(out_dir / "gradcam_stats.json", stats)


# --------------------------------------------------------------------------- latency
def _time(fn, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(t)
    return {"mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "n": reps}  # fmt: skip


def cmd_latency(args, cfg) -> None:
    import zipfile

    import cv2
    import tensorflow as tf

    from ..explain.gradcam import compute_gradcam
    from ..preprocessing.pipeline import FaceExtractor
    from .audio_models import logit_model

    work, out_dir = Path(args.work), Path(args.out)
    records = _records(work)
    ft = _load_ft(work)
    report: dict = {"meta": run_metadata(), "threads": os.cpu_count()}
    te = [r for r in records if r.split == "test"][:100]
    extractor = FaceExtractor(FaceConfig(**{**vars(cfg.face), "crop_size": IMAGE_SIZE}))
    raw = []
    for r in te[:50]:
        with zipfile.ZipFile(Path(args.zips) / f"{r.subset}.zip") as zf:
            raw.append(cv2.imdecode(np.frombuffer(zf.read(r.member), np.uint8), cv2.IMREAD_COLOR))
    it = iter(range(10**9))
    report["face_detect_align_crop"] = _time(lambda: extractor.crop_image(raw[next(it) % len(raw)]), 100)
    images = dff.load_images(te, work / "crops", IMAGE_SIZE)
    lm = logit_model(ft)
    x1 = to_model_input(images[:1])
    lm(x1, training=False)
    report["keras_batch1"] = _time(lambda: lm(x1, training=False), 30)
    x16 = to_model_input(images[:16])
    report["keras_batch16_per_image_ms"] = round(_time(lambda: lm(x16, training=False), 5)["mean_ms"] / 16, 2)
    report["gradcam_per_image"] = _time(lambda: compute_gradcam(ft, x1[0]), 5)
    keras_z = _batched(lm, images).reshape(-1)
    exports = {}
    for name, quant in (("tflite_fp32", False), ("tflite_int8_dynamic", True)):
        conv = tf.lite.TFLiteConverter.from_keras_model(lm)
        if quant:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = conv.convert()
        (work / "models" / "effb0_ft" / f"{name}.tflite").write_bytes(blob)
        interp = tf.lite.Interpreter(model_content=blob, num_threads=os.cpu_count())
        d = interp.get_input_details()[0]
        interp.resize_tensor_input(d["index"], [1, IMAGE_SIZE, IMAGE_SIZE, 3])
        interp.allocate_tensors()
        oi = interp.get_output_details()[0]["index"]

        def run(x, interp=interp, d=d, oi=oi):
            interp.set_tensor(d["index"], x.astype(np.float32))
            interp.invoke()
            return float(interp.get_tensor(oi).reshape(-1)[0])

        z = np.array([run(to_model_input(img[None])) for img in images])
        exports[name] = {
            "file_bytes": len(blob),
            "batch1": _time(lambda run=run: run(x1), 30),
            "max_abs_prob_diff_vs_keras": float(np.max(np.abs(_sigmoid(z) - _sigmoid(keras_z)))),
        }
    report["exports"] = exports
    report["keras_file_bytes"] = (work / "models" / "effb0_ft" / "model.keras").stat().st_size
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    _write_json(out_dir / "latency.json", report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--zips", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--train-groups", type=int, default=5000)
    p.add_argument("--val-groups", type=int, default=800)
    p.add_argument("--test-groups", type=int, default=2500)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p = sub.add_parser("probe")
    p.add_argument("--work", required=True)
    p = sub.add_parser("finetune")
    p.add_argument("--work", required=True)
    p.add_argument("--exclude", choices=dff.FAKE_TYPES, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    for name in ("evaluate", "robustness", "gradcam", "latency"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--n-per-class", type=int, default=600)
        p.add_argument("--zips", default=None)
    args = parser.parse_args(argv)
    configure_logging(json_logs=False)
    cfg = load_config(args.config)
    {"prepare": cmd_prepare, "probe": cmd_probe, "finetune": cmd_finetune, "evaluate": cmd_evaluate,
     "robustness": cmd_robustness, "gradcam": cmd_gradcam, "latency": cmd_latency}[args.cmd](args, cfg)  # fmt: skip


if __name__ == "__main__":
    main()
