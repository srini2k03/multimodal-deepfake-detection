"""ASVspoof 2019 LA benchmark for the audio branch (runs in GitHub Actions, CPU only).

    python -m deepfake_detector.benchmarks.run_asvspoof features --archive LA.zip --work work/
    python -m deepfake_detector.benchmarks.run_asvspoof train    --work work/ --system cnn
    python -m deepfake_detector.benchmarks.run_asvspoof train    --work work/ --system cnn_trim
    python -m deepfake_detector.benchmarks.run_asvspoof evaluate --archive LA.zip --work work/ --out results/
    python -m deepfake_detector.benchmarks.run_asvspoof robustness --archive LA.zip --work work/ --out results/
    python -m deepfake_detector.benchmarks.run_asvspoof latency  --archive LA.zip --work work/ --out results/

Model selection uses the dev split only (same attacks as train, new speakers); every number
on the eval split (new speakers, eleven unseen attacks) is computed once with the selected
model and the dev-chosen operating threshold.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..config import AudioConfig, Config, load_config
from ..evaluation.calibration import fit_platt
from ..evaluation.metrics import (
    binary_report,
    brier_score,
    eer_from_scores,
    expected_calibration_error,
    min_tdcf,
    reliability_bins,
    roc_auc,
)
from ..preprocessing.audio import clip_features, frames_per_clip, resample, trim_silence
from ..utils.logging import configure_logging
from .asvspoof import (
    ATTACKS,
    KNOWN_IN_EVAL,
    SPLITS,
    FeatureSplit,
    LAArchive,
    audit_split,
    extract_split,
    load_split,
    speaker_overlap,
    stratified_subset,
)
from .audio_models import (
    LogisticRegression,
    logmel_stats_features,
    predict_logits,
    silence_features,
    train_cnn,
)

logger = logging.getLogger(__name__)
CNN_SYSTEMS = {"cnn": "raw", "cnn_trim": "trim"}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def run_metadata() -> dict:
    import importlib.metadata as md

    versions = {}
    for pkg in ("numpy", "tensorflow", "tensorflow-cpu", "keras", "soundfile"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "runner": f"{platform.system()} {platform.machine()} cpus={os.cpu_count()}",
        "python": platform.python_version(),
        "packages": versions,
    }


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- features + audit
def cmd_features(args, cfg: Config) -> None:
    work = Path(args.work)
    feats = work / "features"
    splits = {}
    for split in SPLITS:
        t0 = time.perf_counter()
        splits[split] = extract_split(
            args.archive, split, cfg.audio, feats, workers=args.workers, limit=args.limit, seed=cfg.audio_train.seed
        )
        logger.info("split done", extra={"split": split, "seconds": round(time.perf_counter() - t0, 1)})
    archive = Path(args.archive)
    audit = {
        "dataset": "ASVspoof 2019 LA",
        "source": "https://datashare.ed.ac.uk/handle/10283/3336 (doi:10.7488/ds/2555)",
        "licence": "Open Data Commons Attribution License (ODC-By) 1.0",
        "archive": {"name": archive.name, "bytes": archive.stat().st_size if archive.is_file() else None},
        "limit_per_split": args.limit,
        "audio": {"sample_rate": cfg.audio.sample_rate, "clip_seconds": cfg.audio.clip_seconds},
        "splits": {s: audit_split(fs) for s, fs in splits.items()},
        "speaker_overlap": speaker_overlap(splits),
        "attacks": ATTACKS,
        "known_attacks_in_eval": KNOWN_IN_EVAL,
        "meta": run_metadata(),
    }
    if archive.is_file() and not args.skip_hash:
        audit["archive"]["sha256"] = sha256(archive)
    _write_json(work / "audit.json", audit)


# --------------------------------------------------------------------------- training
def cmd_train(args, cfg: Config) -> None:
    work = Path(args.work)
    variant = CNN_SYSTEMS[args.system]
    train, dev = load_split(work / "features", "train"), load_split(work / "features", "dev")
    if args.epochs:
        cfg.audio_train.epochs = args.epochs
    res = train_cnn(
        train.features[variant],
        train.labels,
        dev.features[variant],
        dev.labels,
        cfg.audio,
        cfg.audio_train,
        run_dir=work / "models" / args.system,
    )
    _write_json(
        work / "models" / args.system / "train.json",
        {
            "system": args.system,
            "variant": variant,
            "history": res.history,
            "best_epoch": res.best_epoch,
            "best_val_eer": res.best_val_eer,
            "train_seconds": res.seconds,
            "params": int(res.model.count_params()),
            "audio": vars(cfg.audio),
            "audio_train": vars(cfg.audio_train),
        },
    )


def _load_cnn(work: Path, system: str):
    from ..inference.scorers import load_keras_model

    path = work / "models" / system / "model.keras"
    return load_keras_model(path) if path.exists() else None


# --------------------------------------------------------------------------- evaluation
def split_report(logits: np.ndarray, fs: FeatureSplit, thr_logit: float, platt, asv: dict | None) -> dict:
    """Every metric for one system on one split. ``logits`` are spoof logits."""
    y = fs.labels
    bona, spoof = -logits[y == 0], -logits[y == 1]
    eer, _ = eer_from_scores(bona, spoof)
    out = {"n": int(y.size), "n_bonafide": int((y == 0).sum()), "n_spoof": int((y == 1).sum()), "eer": eer}
    if asv is not None and {"target", "nontarget", "spoof"} <= asv.keys():
        out["min_tdcf"] = min_tdcf(bona, spoof, asv["target"], asv["nontarget"], asv["spoof"])["min_tdcf"]
    rep = binary_report(y, logits, threshold=thr_logit).to_dict()
    out["at_dev_threshold"] = {
        k: rep[k] for k in ("accuracy", "precision", "recall", "f1", "fpr", "fnr", "confusion_matrix", "threshold")
    }
    out["roc_auc"] = rep["auc"]
    out["pr_auc_spoof"] = rep["pr_auc"]
    raw_p = _sigmoid(logits)
    cal_p = platt.transform(logits)
    out["calibration"] = {
        "raw": {"ece": expected_calibration_error(y, raw_p), "brier": brier_score(y, raw_p)},
        "platt": {"ece": expected_calibration_error(y, cal_p), "brier": brier_score(y, cal_p)},
        "reliability_platt": reliability_bins(y, cal_p, 10),
    }
    attacks = fs.attacks
    per_attack = {}
    for a in sorted(set(attacks) - {"-"}):
        mask = attacks == a
        a_eer, _ = eer_from_scores(bona, -logits[mask])
        per_attack[a] = {
            "n": int(mask.sum()),
            "eer": a_eer,
            "detected_at_dev_threshold": float(np.mean(logits[mask] >= thr_logit)),
            "seen_in_training": a in {"A01", "A02", "A03", "A04", "A05", "A06"} or a in KNOWN_IN_EVAL,
        }
    out["per_attack"] = per_attack
    return out


def _system_logits(work: Path, splits: dict[str, FeatureSplit]) -> dict[str, dict[str, np.ndarray]]:
    """Spoof logits of every system on every split (linear baselines are fitted here: seconds)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    tr = splits["train"]
    lr_sil = LogisticRegression().fit(silence_features(tr), tr.labels)
    out["silence_lr"] = {s: lr_sil.decision_function(silence_features(fs)) for s, fs in splits.items()}
    stats = {s: logmel_stats_features(fs.features["raw"]) for s, fs in splits.items()}
    lr_mel = LogisticRegression().fit(stats["train"], tr.labels)
    out["logmel_lr"] = {s: lr_mel.decision_function(stats[s]) for s in splits}
    for system, variant in CNN_SYSTEMS.items():
        model = _load_cnn(work, system)
        if model is None:
            logger.warning("model missing, skipped", extra={"system": system})
            continue
        out[system] = {s: predict_logits(model, fs.features[variant]) for s, fs in splits.items() if s != "train"}
        if system == "cnn":  # cross-condition: untrimmed model on silence-trimmed audio
            out["cnn_on_trimmed_audio"] = {
                s: predict_logits(model, fs.features["trim"]) for s, fs in splits.items() if s != "train"
            }
    return out


SYSTEM_NOTES = {
    "silence_lr": "Logistic regression on duration + leading/trailing silence only (shortcut probe)",
    "logmel_lr": "Logistic regression on per-band log-mel mean/std",
    "cnn": "Log-mel CNN, untrimmed audio (repository audio branch)",
    "cnn_trim": "Log-mel CNN trained and tested on silence-trimmed audio",
    "cnn_on_trimmed_audio": "Untrimmed-trained CNN tested on silence-trimmed audio (cross-condition)",
}


def cmd_evaluate(args, cfg: Config) -> None:
    work, out_dir = Path(args.work), Path(args.out)
    splits = {s: load_split(work / "features", s) for s in SPLITS}
    archive = LAArchive(args.archive)
    asv = {s: archive.asv(s) for s in ("dev", "eval")}
    archive.close()
    logits = _system_logits(work, splits)
    results: dict = {"meta": run_metadata(), "systems": {}}
    for system, per_split in logits.items():
        dev_logits = per_split["dev"]
        dev = splits["dev"]
        _, thr_bona = eer_from_scores(-dev_logits[dev.labels == 0], -dev_logits[dev.labels == 1])
        thr_logit = -thr_bona  # operating point chosen on dev, applied unchanged to eval
        platt = fit_platt(dev_logits, dev.labels)
        entry = {"description": SYSTEM_NOTES.get(system, system), "dev_threshold_logit": thr_logit}
        entry["platt"] = platt.to_dict()
        for split in ("dev", "eval"):
            entry[split] = split_report(per_split[split], splits[split], thr_logit, platt, asv.get(split))
        train_json = work / "models" / system / "train.json"
        if train_json.exists():
            t = json.loads(train_json.read_text())
            entry["training"] = {k: t[k] for k in ("best_epoch", "best_val_eer", "train_seconds", "params", "history")}
        results["systems"][system] = entry
        logger.info("evaluated", extra={"system": system, "eval_eer": entry["eval"]["eer"]})
    _write_json(out_dir / "metrics.json", results)
    # eval scores (spoof logits) so every number can be recomputed from the repository
    ev = splits["eval"]
    names = list(logits)
    with gzip.open(out_dir / "eval_scores.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write("utt,attack,label," + ",".join(names) + "\n")
        for i, row in enumerate(ev.rows):
            vals = ",".join(f"{logits[n]['eval'][i]:.5f}" for n in names)
            fh.write(f"{row['utt']},{row['attack']},{row['label']},{vals}\n")
    shutil.copy2(work / "audit.json", out_dir / "data_audit.json")


# --------------------------------------------------------------------------- robustness
def _add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = float(np.mean(x.astype(np.float64) ** 2)) + 1e-12
    noise = rng.standard_normal(x.size) * np.sqrt(p / (10 ** (snr_db / 10)))
    return (x + noise).astype(np.float32)


def _codec(x: np.ndarray, sr: int, codec: str, bitrate: str) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not available")
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = {"libmp3lame": "mp3", "libopus": "ogg"}[codec]
    enc = subprocess.run(
        [exe, "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-", "-c:a", codec, "-b:a", bitrate,
         "-f", fmt, "-"],
        input=pcm, capture_output=True, check=True, timeout=60,
    )  # fmt: skip
    dec = subprocess.run(
        [exe, "-v", "error", "-i", "-", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-"],
        input=enc.stdout, capture_output=True, check=True, timeout=60,
    )  # fmt: skip
    return np.frombuffer(dec.stdout, dtype="<i2").astype(np.float32) / 32768.0


CONDITIONS = [
    "clean",
    "noise_snr30",
    "noise_snr20",
    "noise_snr10",
    "noise_snr5",
    "telephone_8k",
    "mp3_32k",
    "opus_16k",
    "gain_minus20db",
    "truncate_1s",
    "truncate_2s",
    "trim_silence",
    "prepend_1s_silence",
]


def perturb(x: np.ndarray, sr: int, condition: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if condition == "clean":
        return x
    if condition.startswith("noise_snr"):
        return _add_noise(x, float(condition.removeprefix("noise_snr")), rng)
    if condition == "telephone_8k":
        return resample(resample(x, sr, 8000), 8000, sr)
    if condition == "mp3_32k":
        return _codec(x, sr, "libmp3lame", "32k")
    if condition == "opus_16k":
        return _codec(x, sr, "libopus", "16k")
    if condition == "gain_minus20db":
        return (x * 0.1).astype(np.float32)
    if condition.startswith("truncate_"):
        return x[: int(float(condition.removeprefix("truncate_").removesuffix("s")) * sr)]
    if condition == "trim_silence":
        return trim_silence(x, sr)
    if condition == "prepend_1s_silence":
        floor = rng.standard_normal(sr).astype(np.float32) * 1e-4  # ~ -80 dBFS noise floor
        return np.r_[floor, x].astype(np.float32)
    raise ValueError(condition)


_rob: dict = {}


def _rob_init(archive_path: str, cfg_dict: dict) -> None:
    _rob["archive"] = LAArchive(archive_path)
    _rob["cfgs"] = {v: AudioConfig(**{**cfg_dict, "trim_silence": v == "trim"}) for v in ("raw", "trim")}


def _rob_job(job: tuple[int, str, str]) -> tuple[int, dict[str, np.ndarray]]:
    i, utt, condition = job
    cfgs = _rob["cfgs"]
    sr = cfgs["raw"].sample_rate
    x = _rob["archive"].decode(utt, sr)
    seed = int(hashlib.sha1(f"{utt}:{condition}".encode()).hexdigest()[:8], 16)
    y = perturb(x, sr, condition, seed)
    return i, {v: clip_features(y, c).astype(np.float16) for v, c in cfgs.items()}


def cmd_robustness(args, cfg: Config) -> None:
    work, out_dir = Path(args.work), Path(args.out)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    archive = LAArchive(args.archive)
    trials = archive.trials("eval")
    archive.close()
    pick = stratified_subset(trials, per_attack=args.per_attack, bonafide=args.bonafide, seed=cfg.audio_train.seed)
    sub = [trials[i] for i in pick]
    labels = np.array([t.label for t in sub])
    models = {s: _load_cnn(work, s) for s in CNN_SYSTEMS}
    models = {s: m for s, m in models.items() if m is not None}
    conditions = [c for c in CONDITIONS if not (c in ("mp3_32k", "opus_16k") and shutil.which("ffmpeg") is None)]
    report: dict = {"meta": run_metadata(), "subset": {"n": len(sub), "n_bonafide": int((labels == 0).sum())}}
    report["conditions"] = {}
    # TensorFlow is already loaded (and running threads) here: forking now can deadlock the
    # children, so the decode/perturb workers are spawned fresh.
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_rob_init, initargs=(args.archive, vars(cfg.audio))) as pool:
        for cond in conditions:
            t0 = time.perf_counter()
            shape = (len(sub), cfg.audio.n_mels, frames_per_clip(cfg.audio))
            feats = {v: np.zeros(shape, np.float16) for v in ("raw", "trim")}
            jobs = [(i, t.utt, cond) for i, t in enumerate(sub)]
            for i, f in pool.imap_unordered(_rob_job, jobs, chunksize=32):
                for v in feats:
                    feats[v][i] = f[v]
            entry = {}
            for system, model in models.items():
                logits = predict_logits(model, feats[CNN_SYSTEMS[system]])
                thr = metrics["systems"][system]["dev_threshold_logit"]
                eer, _ = eer_from_scores(-logits[labels == 0], -logits[labels == 1])
                pred = logits >= thr
                entry[system] = {
                    "eer": eer,
                    "roc_auc": roc_auc(labels, logits),
                    "fpr_at_dev_threshold": float(np.mean(pred[labels == 0])),
                    "fnr_at_dev_threshold": float(np.mean(~pred[labels == 1])),
                }
            entry["seconds"] = round(time.perf_counter() - t0, 1)
            report["conditions"][cond] = entry
            logger.info("robustness", extra={"condition": cond, **{k: v for k, v in entry.items() if k != "seconds"}})
    _write_json(out_dir / "robustness.json", report)


# --------------------------------------------------------------------------- latency + export
def _tflite_convert(model, quantize: bool) -> bytes:
    import tensorflow as tf

    from .audio_models import logit_model

    # from_keras_model freezes the weights correctly for Keras 3 models (a hand-built
    # concrete function produced a weight-less 24 KB graph that output NaN).
    conv = tf.lite.TFLiteConverter.from_keras_model(logit_model(model))
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range int8 weights
    return conv.convert()


def _tflite_runner(blob: bytes):
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=blob, num_threads=os.cpu_count())
    detail = interp.get_input_details()[0]
    interp.resize_tensor_input(detail["index"], [1, *detail["shape"][1:]])
    interp.allocate_tensors()
    inp, out = detail["index"], interp.get_output_details()[0]["index"]

    def run(x: np.ndarray) -> float:
        interp.set_tensor(inp, x.astype(np.float32))
        interp.invoke()
        return float(interp.get_tensor(out).reshape(-1)[0])

    return run


def _timeit(fn, reps: int) -> dict:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    return {"mean_ms": round(float(a.mean()), 3), "p50_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "n": reps}  # fmt: skip


def cmd_latency(args, cfg: Config) -> None:
    import tensorflow as tf

    from .audio_models import logit_model

    work, out_dir = Path(args.work), Path(args.out)
    export_dir = work / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    archive = LAArchive(args.archive)
    trials = archive.trials("eval")
    pick = stratified_subset(trials, per_attack=args.per_attack, bonafide=args.bonafide, seed=cfg.audio_train.seed + 1)
    sub = [trials[i] for i in pick]
    labels = np.array([t.label for t in sub])
    sr = cfg.audio.sample_rate
    report: dict = {"meta": run_metadata(), "n_utterances": len(sub), "threads": os.cpu_count()}
    report["tensorflow_intra_op_threads"] = tf.config.threading.get_intra_op_parallelism_threads()
    # stage timings on a fixed set of utterances
    n_stage = min(200, len(sub))
    waves = [archive.decode(t.utt, sr) for t in sub[:n_stage]]
    report["mean_duration_s"] = round(float(np.mean([w.size / sr for w in waves])), 3)
    i_iter = iter(range(10**9))
    report["stages"] = {
        "decode_flac": _timeit(lambda: archive.decode(sub[next(i_iter) % n_stage].utt, sr), 200),
        "log_mel_4s": _timeit(lambda: clip_features(waves[next(i_iter) % n_stage], cfg.audio), 200),
    }
    archive.close()
    feats = np.stack([clip_features(w, cfg.audio) for w in waves[:64]])[..., None].astype(np.float32)
    for system in CNN_SYSTEMS:
        model = _load_cnn(work, system)
        if model is None:
            continue
        lm = logit_model(model)
        x1 = feats[:1]
        lm(x1, training=False)
        batch64 = _timeit(lambda m=lm: m(feats[:64], training=False), 10)
        entry = {
            "params": int(model.count_params()),
            "keras_batch1": _timeit(lambda m=lm, x=x1: m(x, training=False), 100),
            "keras_batch64_per_clip_ms": round(batch64["mean_ms"] / 64, 3),
        }
        keras_logits = predict_logits(model, feats[..., 0])
        model.save(export_dir / f"{system}.keras")
        entry["keras_file_bytes"] = (export_dir / f"{system}.keras").stat().st_size
        for name, quant in (("tflite_fp32", False), ("tflite_int8_dynamic", True)):
            blob = _tflite_convert(model, quant)
            (export_dir / f"{system}_{name}.tflite").write_bytes(blob)
            run = _tflite_runner(blob)
            lite_logits = np.array([run(f[None]) for f in feats])
            entry[name] = {
                "file_bytes": len(blob),
                "batch1": _timeit(lambda r=run, x=x1: r(x), 100),
                "max_abs_logit_diff_vs_keras": float(np.max(np.abs(lite_logits - keras_logits))),
                "max_abs_prob_diff_vs_keras": float(np.max(np.abs(_sigmoid(lite_logits) - _sigmoid(keras_logits)))),
            }
        report[system] = entry
    # quality parity of the exported models on a stratified subset of eval
    parity_feats = {}
    archive = LAArchive(args.archive)
    for v in ("raw", "trim"):
        c = AudioConfig(**{**vars(cfg.audio), "trim_silence": v == "trim"})
        parity_feats[v] = np.stack([clip_features(archive.decode(t.utt, sr), c) for t in sub])[..., None]
    archive.close()
    for system, variant in CNN_SYSTEMS.items():
        if system not in report:
            continue
        model = _load_cnn(work, system)
        k = predict_logits(model, parity_feats[variant][..., 0])
        keras_eer = eer_from_scores(-k[labels == 0], -k[labels == 1])[0]
        report[system]["parity_subset"] = {"n": len(sub), "keras_eer": keras_eer}
        for name in ("tflite_fp32", "tflite_int8_dynamic"):
            run = _tflite_runner((export_dir / f"{system}_{name}.tflite").read_bytes())
            q = np.array([run(f[None].astype(np.float32)) for f in parity_feats[variant]])
            report[system]["parity_subset"][f"{name}_eer"] = eer_from_scores(-q[labels == 0], -q[labels == 1])[0]
            report[system]["parity_subset"][f"{name}_max_abs_prob_diff"] = float(
                np.max(np.abs(_sigmoid(q) - _sigmoid(k)))
            )
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    _write_json(out_dir / "latency.json", report)


# --------------------------------------------------------------------------- cross-dataset
ITW_SOURCE = "https://huggingface.co/datasets/mueller91/In-The-Wild (Mueller et al., Interspeech 2022)"
_itw: dict = {}


def _itw_init(zip_path: str, cfg_dict: dict) -> None:
    import zipfile

    _itw["zip"] = zipfile.ZipFile(zip_path)
    _itw["cfgs"] = {v: AudioConfig(**{**cfg_dict, "trim_silence": v == "trim"}) for v in ("raw", "trim")}


def _itw_job(job: tuple[int, str]) -> tuple[int, dict[str, np.ndarray], dict[str, float]]:
    import io

    import soundfile as sf

    from ..preprocessing.audio import silence_profile

    i, member = job
    cfg = _itw["cfgs"]["raw"]
    data, sr = sf.read(io.BytesIO(_itw["zip"].read(member)), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != cfg.sample_rate:
        x = resample(x, sr, cfg.sample_rate)
    stats = silence_profile(x, cfg.sample_rate, cfg.trim_top_db)
    return i, {v: clip_features(x, c).astype(np.float16) for v, c in _itw["cfgs"].items()}, stats


def cmd_itw(args, cfg: Config) -> None:
    """Cross-dataset test: ASVspoof-trained systems on In-the-Wild (real politicians/celebrities)."""
    import csv
    import io
    import zipfile

    work, out_dir = Path(args.work), Path(args.out)
    with zipfile.ZipFile(args.zip) as zf:
        meta_name = next(n for n in zf.namelist() if n.endswith("meta.csv"))
        base = meta_name.rsplit("/", 1)[0] + "/" if "/" in meta_name else ""
        rows = list(csv.DictReader(io.StringIO(zf.read(meta_name).decode("utf-8"))))
    labels = np.array([0 if r["label"].lower().startswith("bona") else 1 for r in rows])
    members = [base + r["file"] for r in rows]
    shape = (len(rows), cfg.audio.n_mels, frames_per_clip(cfg.audio))
    feats = {v: np.zeros(shape, np.float16) for v in ("raw", "trim")}
    stats: list[dict] = [{}] * len(rows)
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(args.workers, initializer=_itw_init, initargs=(args.zip, vars(cfg.audio))) as pool:
        for i, f, st in pool.imap_unordered(_itw_job, list(enumerate(members)), chunksize=64):
            feats["raw"][i], feats["trim"][i], stats[i] = f["raw"], f["trim"], st
    seconds = round(time.perf_counter() - t0, 1)

    class _FS:  # minimal FeatureSplit stand-in for the silence features
        def __init__(self, st):
            self.st = st

        def column(self, k):
            return np.array([s[k] for s in self.st], dtype=np.float64)

    train = load_split(work / "features", "train")
    systems: dict[str, np.ndarray] = {}
    systems["silence_lr"] = (
        LogisticRegression().fit(silence_features(train), train.labels).decision_function(silence_features(_FS(stats)))
    )
    lr_mel = LogisticRegression().fit(logmel_stats_features(train.features["raw"]), train.labels)
    systems["logmel_lr"] = lr_mel.decision_function(logmel_stats_features(feats["raw"]))
    for system, variant in CNN_SYSTEMS.items():
        model = _load_cnn(work, system)
        if model is not None:
            systems[system] = predict_logits(model, feats[variant])
    asv_metrics = json.loads((out_dir / "metrics.json").read_text())["systems"]
    keys = ("duration_s", "lead_silence_s", "trail_silence_s")
    report: dict = {
        "meta": run_metadata(),
        "dataset": "In-the-Wild audio deepfakes",
        "source": ITW_SOURCE,
        "licence": "Hugging Face card: cc-by-sa-4.0; project page: CC BY-NC 4.0 (non-commercial use assumed)",
        "n": len(rows),
        "n_bonafide": int((labels == 0).sum()),
        "n_spoof": int((labels == 1).sum()),
        "n_speakers": len({r["speaker"] for r in rows}),
        "feature_seconds": seconds,
        "per_class_median": {
            name: {k: float(np.median([s[k] for s, y in zip(stats, labels, strict=True) if y == c])) for k in keys}
            for c, name in ((0, "bonafide"), (1, "spoof"))
        },
        "systems": {},
    }
    for name, z in systems.items():
        thr = asv_metrics[name]["dev_threshold_logit"]
        rep = binary_report(labels, z, threshold=thr).to_dict()
        a, b = asv_metrics[name]["platt"]["a"], asv_metrics[name]["platt"]["b"]  # fitted on ASVspoof dev only
        report["systems"][name] = {
            "eer": eer_from_scores(-z[labels == 0], -z[labels == 1])[0],
            "roc_auc": rep["auc"],
            "pr_auc_spoof": rep["pr_auc"],
            "at_asvspoof_dev_threshold": {k: rep[k] for k in ("accuracy", "fpr", "fnr", "f1", "confusion_matrix")},
            "ece_raw_sigmoid": expected_calibration_error(labels, _sigmoid(z)),
            "ece_asvspoof_dev_platt": expected_calibration_error(labels, _sigmoid(a * z + b)),
        }
        logger.info("in-the-wild", extra={"system": name, "eer": report["systems"][name]["eer"]})
    _write_json(out_dir / "cross_dataset_in_the_wild.json", report)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features")
    p.add_argument("--archive", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--limit", type=int, default=None, help="stratified subset per split (smoke runs)")
    p.add_argument("--skip-hash", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--work", required=True)
    p.add_argument("--system", choices=sorted(CNN_SYSTEMS), required=True)
    p.add_argument("--epochs", type=int, default=None)
    p = sub.add_parser("evaluate")
    p.add_argument("--archive", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    for name in ("robustness", "latency"):
        p = sub.add_parser(name)
        p.add_argument("--archive", required=True)
        p.add_argument("--work", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--per-attack", type=int, default=250)
        p.add_argument("--bonafide", type=int, default=1000)
        p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p = sub.add_parser("itw", help="cross-dataset evaluation on In-the-Wild")
    p.add_argument("--zip", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args(argv)
    configure_logging(json_logs=True)  # JSON lines keep per-epoch metrics visible in CI logs
    cfg = load_config(args.config)
    {"features": cmd_features, "train": cmd_train, "evaluate": cmd_evaluate, "robustness": cmd_robustness,
     "latency": cmd_latency, "itw": cmd_itw}[args.cmd](args, cfg)  # fmt: skip


if __name__ == "__main__":
    main()
