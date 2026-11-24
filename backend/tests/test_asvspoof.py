"""ASVspoof 2019 LA benchmark code, exercised on a synthetic archive with the real layout.

The synthetic "spoofs" are noisier tones with less leading silence than the "bona fide"
tones. Nothing here says anything about detection quality; it checks the plumbing.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from deepfake_detector.benchmarks.asvspoof import (
    LAArchive,
    audit_split,
    extract_split,
    load_split,
    parse_asv_scores,
    parse_protocol,
    speaker_overlap,
    stratified_subset,
)
from deepfake_detector.benchmarks.audio_models import LogisticRegression, silence_features, spec_augment, val_subset
from deepfake_detector.config import AudioConfig
from deepfake_detector.evaluation.metrics import det_curve, eer_from_scores, min_tdcf

sf = pytest.importorskip("soundfile")

SR = 16000
SPLIT_CODES = {
    "train": ("T", "trn", ["A01", "A02"]),
    "dev": ("D", "trl", ["A01", "A02"]),
    "eval": ("E", "trl", ["A07", "A08"]),
}


def _utterance(rng: np.random.Generator, spoof: bool) -> np.ndarray:
    dur = rng.uniform(1.2, 2.5)
    t = np.arange(int(dur * SR)) / SR
    speech = 0.3 * np.sin(2 * np.pi * rng.uniform(120, 220) * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    speech += (0.08 if spoof else 0.01) * rng.standard_normal(t.size)
    lead = np.zeros(int((0.05 if spoof else 0.4) * SR))
    return np.r_[lead, speech, np.zeros(int(0.1 * SR))].astype(np.float32)


def make_la_zip(path: Path, per_split: int = 16, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        uid = 1000000
        for split, (code, kind, attacks) in SPLIT_CODES.items():
            lines = []
            for i in range(per_split):
                spoof = i % 2 == 1
                speaker = f"LA_{split[:1].upper()}{i % 3:03d}"
                attack = attacks[(i // 2) % len(attacks)] if spoof else "-"
                utt = f"LA_{code}_{uid}"
                uid += 1
                buf = io.BytesIO()
                sf.write(buf, _utterance(rng, spoof), SR, format="FLAC")
                zf.writestr(f"LA/ASVspoof2019_LA_{split}/flac/{utt}.flac", buf.getvalue())
                lines.append(f"{speaker} {utt} - {attack} {'spoof' if spoof else 'bonafide'}")
            zf.writestr(f"LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{kind}.txt", "\n".join(lines))
            if split != "train":
                asv = [f"bonafide target {rng.normal(3, 1):.3f}" for _ in range(40)]
                asv += [f"bonafide nontarget {rng.normal(-3, 1):.3f}" for _ in range(40)]
                asv += [f"A07 spoof {rng.normal(1, 2):.3f}" for _ in range(40)]
                zf.writestr(
                    f"LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.{split}.gi.trl.scores.txt", "\n".join(asv)
                )
    return path


@pytest.fixture(scope="module")
def la_zip(tmp_path_factory) -> Path:
    return make_la_zip(tmp_path_factory.mktemp("la") / "LA.zip")


@pytest.fixture(scope="module")
def audio_cfg() -> AudioConfig:
    return AudioConfig(clip_seconds=1.0)


@pytest.fixture(scope="module")
def features(la_zip, audio_cfg, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("work") / "features"
    for split in ("train", "dev", "eval"):
        extract_split(la_zip, split, audio_cfg, out, workers=2)
    return out


def test_parse_protocol_and_asv_scores():
    trials = parse_protocol("LA_0079 LA_T_1138215 - - bonafide\nLA_0079 LA_T_1271820 - A01 spoof\n\n")
    assert [(t.utt, t.attack, t.label, t.key) for t in trials] == [
        ("LA_T_1138215", "-", 0, "bonafide"),
        ("LA_T_1271820", "A01", 1, "spoof"),
    ]
    with pytest.raises(ValueError):
        parse_protocol("A B - - maybe")
    asv = parse_asv_scores("bonafide target 1.5\nbonafide nontarget -2\nA07 spoof 0.5\n")
    assert asv["target"].tolist() == [1.5] and asv["spoof"].tolist() == [0.5]


def test_archive_index_and_decode(la_zip):
    arc = LAArchive(la_zip)
    assert set(arc.protocols) == {"train", "dev", "eval"} and set(arc.asv_scores) == {"dev", "eval"}
    trials = arc.trials("eval")
    assert len(trials) == 16 and {t.attack for t in trials} == {"-", "A07", "A08"}
    x = arc.decode(trials[0].utt, 16000)
    assert x.dtype == np.float32 and x.size > SR
    assert arc.decode(trials[0].utt, 8000).size == pytest.approx(x.size / 2, abs=2)
    arc.close()


def test_stratified_subset_is_seeded():
    trials = parse_protocol("\n".join(f"S U{i} - {'-' if i % 3 == 0 else 'A0' + str(i % 3)} "
                                      f"{'bonafide' if i % 3 == 0 else 'spoof'}" for i in range(60)))  # fmt: skip
    a = stratified_subset(trials, per_attack=5, bonafide=4, seed=1)
    assert a == stratified_subset(trials, per_attack=5, bonafide=4, seed=1)
    assert len(a) == 14 and a == sorted(a)


def test_feature_cache_and_audit(features, audio_cfg):
    splits = {s: load_split(features, s) for s in ("train", "dev", "eval")}
    tr = splits["train"]
    assert tr.features["raw"].shape == (16, audio_cfg.n_mels, 101) and tr.features["raw"].dtype == np.float16
    assert np.isfinite(tr.features["trim"].astype(np.float32)).all()
    assert not np.allclose(tr.features["raw"][0], tr.features["trim"][0])
    audit = audit_split(tr)
    assert audit["n_bonafide"] == 8 and audit["n_spoof"] == 8 and audit["attacks"] == {"A01": 4, "A02": 4}
    lead = audit["per_class"]
    assert lead["bonafide"]["lead_silence_s"]["median"] > lead["spoof"]["lead_silence_s"]["median"]
    assert speaker_overlap(splits) == {"dev&eval": 0, "dev&train": 0, "eval&train": 0}


def test_silence_only_baseline_learns_the_shortcut(features):
    tr, ev = load_split(features, "train"), load_split(features, "eval")
    lr = LogisticRegression().fit(silence_features(tr), tr.labels)
    s = lr.decision_function(silence_features(ev))
    eer, _ = eer_from_scores(-s[ev.labels == 0], -s[ev.labels == 1])
    assert eer < 0.2  # the synthetic data leaks the label through leading silence by design


def test_det_curve_matches_official_definition():
    tar, non = np.array([0.9, 0.8, 0.4]), np.array([0.1, 0.5, 0.3])
    frr, far, thr = det_curve(tar, non)
    assert frr[0] == 0 and far[0] == 1 and frr[-1] == 1 and far[-1] == 0
    assert thr.size == 7
    eer, _ = eer_from_scores(tar, non)
    assert eer == pytest.approx(1 / 3)


def test_min_tdcf_bounds():
    rng = np.random.default_rng(0)
    asv_t, asv_n, asv_s = rng.normal(3, 1, 500), rng.normal(-3, 1, 500), rng.normal(1, 2, 500)
    perfect = min_tdcf(rng.normal(5, 0.1, 200), rng.normal(-5, 0.1, 800), asv_t, asv_n, asv_s)
    useless = min_tdcf(rng.normal(0, 1, 200), rng.normal(0, 1, 800), asv_t, asv_n, asv_s)
    assert perfect["min_tdcf"] < 0.2
    assert 0.9 < useless["min_tdcf"] <= 1.0 + 1e-9  # a CM that accepts everything costs at most 1


def test_spec_augment_and_val_subset():
    rng = np.random.default_rng(0)
    x = np.ones((4, 16, 50), np.float32)
    spec_augment(x, rng, max_f=4, max_t=10)
    assert (x == 0).any() and (x == 1).any()
    y = np.array([0] * 5 + [1] * 20)
    idx = val_subset(y, max_spoof=6, seed=0)
    assert (y[idx] == 0).sum() == 5 and (y[idx] == 1).sum() == 6


@pytest.mark.tf
def test_benchmark_cli_end_to_end(la_zip, tmp_path, monkeypatch):
    """features -> train (1 epoch) -> evaluate -> robustness -> latency on the synthetic archive."""
    from deepfake_detector.benchmarks import run_asvspoof as bench

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("audio:\n  clip_seconds: 1.0\naudio_train:\n  epochs: 1\n  batch_size: 8\n")
    work, out = tmp_path / "work", tmp_path / "results"
    common = ["--config", str(cfg_path)]
    bench.main([*common, "features", "--archive", str(la_zip), "--work", str(work), "--workers", "2"])
    for system in ("cnn", "cnn_trim"):
        bench.main([*common, "train", "--work", str(work), "--system", system])
    bench.main([*common, "evaluate", "--archive", str(la_zip), "--work", str(work), "--out", str(out)])
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["systems"]) == {"silence_lr", "logmel_lr", "cnn", "cnn_trim", "cnn_on_trimmed_audio"}
    ev = metrics["systems"]["cnn"]["eval"]
    assert 0.0 <= ev["eer"] <= 1.0 and "min_tdcf" in ev and set(ev["per_attack"]) == {"A07", "A08"}
    assert (out / "eval_scores.csv.gz").exists() and (out / "data_audit.json").exists()
    small = ["--per-attack", "2", "--bonafide", "3", "--workers", "2"]
    monkeypatch.setattr(bench, "CONDITIONS", ["clean", "noise_snr10", "truncate_1s", "prepend_1s_silence"])
    bench.main([*common, "robustness", "--archive", str(la_zip), "--work", str(work), "--out", str(out), *small])
    rob = json.loads((out / "robustness.json").read_text())
    assert set(rob["conditions"]) == {"clean", "noise_snr10", "truncate_1s", "prepend_1s_silence"}
    bench.main([*common, "latency", "--archive", str(la_zip), "--work", str(work), "--out", str(out), *small])
    lat = json.loads((out / "latency.json").read_text())
    assert lat["cnn"]["tflite_fp32"]["max_abs_prob_diff_vs_keras"] < 1e-3
    assert lat["stages"]["log_mel_4s"]["mean_ms"] > 0
