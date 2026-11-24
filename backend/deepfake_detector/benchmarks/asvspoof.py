"""ASVspoof 2019 logical-access (LA) data: protocols, zip-streamed decoding and cached features.

The archive (``LA.zip``, Edinburgh DataShare, ODC-By 1.0) is read in place: FLAC members
are decoded straight from the zip, so the 7.6 GB archive is never extracted. Features are
written once per split to float16 memory-maps next to an ``index.csv`` that carries every
utterance's metadata (speaker, attack, label, duration, silence profile) for the audit.

Splits are the official ones: train / dev / eval have disjoint speakers, train and dev use
attacks A01-A06, eval uses A07-A19 (A16 and A19 re-use the A04 and A06 systems; the other
eleven attacks never appear in training).
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..config import AudioConfig
from ..preprocessing.audio import clip_features, frames_per_clip, resample, silence_profile

logger = logging.getLogger(__name__)

SPLITS = ("train", "dev", "eval")
BONAFIDE, SPOOF = 0, 1  # repository convention: 1 = fake
FEATURE_VARIANTS = ("raw", "trim")

# Attack descriptions from the ASVspoof 2019 evaluation plan / LA database paper (Wang et al., 2020).
ATTACKS = {
    "A01": "TTS: neural waveform model",
    "A02": "TTS: vocoder",
    "A03": "TTS: vocoder",
    "A04": "TTS: waveform concatenation",
    "A05": "VC: vocoder",
    "A06": "VC: spectral filtering",
    "A07": "TTS: vocoder + GAN",
    "A08": "TTS: neural waveform",
    "A09": "TTS: vocoder",
    "A10": "TTS: neural waveform",
    "A11": "TTS: Griffin-Lim",
    "A12": "TTS: neural waveform",
    "A13": "TTS+VC: waveform concatenation + filtering",
    "A14": "TTS+VC: vocoder",
    "A15": "TTS+VC: neural waveform",
    "A16": "TTS: waveform concatenation (same system as A04)",
    "A17": "VC: waveform filtering",
    "A18": "VC: vocoder",
    "A19": "VC: spectral filtering (same system as A06)",
}
KNOWN_IN_EVAL = {"A16": "A04", "A19": "A06"}

_PROTOCOL_RE = re.compile(r"ASVspoof2019\.LA\.cm\.(train|dev|eval)\.(?:trn|trl)\.txt$")
_ASV_SCORES_RE = re.compile(r"ASVspoof2019\.LA\.asv\.(dev|eval)\.gi\.trl\.scores\.txt$")
_FLAC_RE = re.compile(r"ASVspoof2019_LA_(train|dev|eval)/flac/(LA_[TDE]_\d+)\.flac$")


@dataclass(frozen=True)
class Trial:
    speaker: str
    utt: str
    attack: str  # "-" for bona fide
    label: int  # 0 bona fide, 1 spoof

    @property
    def key(self) -> str:
        return "bonafide" if self.label == BONAFIDE else "spoof"


def parse_protocol(text: str) -> list[Trial]:
    """Parse a CM protocol: ``SPEAKER UTT - ATTACK KEY`` per line."""
    trials = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        speaker, utt, attack, key = parts[0], parts[1], parts[3], parts[4]
        if key not in ("bonafide", "spoof"):
            raise ValueError(f"Unexpected key '{key}' in protocol line: {line!r}")
        trials.append(Trial(speaker, utt, attack, BONAFIDE if key == "bonafide" else SPOOF))
    return trials


def parse_asv_scores(text: str) -> dict[str, np.ndarray]:
    """Organisers' ASV scores: ``SOURCE KEY SCORE`` with KEY in target / nontarget / spoof."""
    out: dict[str, list[float]] = defaultdict(list)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-2] in ("target", "nontarget", "spoof"):
            out[parts[-2]].append(float(parts[-1]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


class LAArchive:
    """Index of an ASVspoof 2019 LA zip (or a directory with the same layout)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        names = self._zip.namelist() if self._zip else [str(p.relative_to(self.path)) for p in self.path.rglob("*")]
        self.protocols: dict[str, str] = {}
        self.asv_scores: dict[str, str] = {}
        self.audio: dict[str, str] = {}
        for name in names:
            if m := _PROTOCOL_RE.search(name):
                self.protocols[m.group(1)] = name
            elif m := _ASV_SCORES_RE.search(name):
                self.asv_scores[m.group(1)] = name
            elif m := _FLAC_RE.search(name):
                self.audio[m.group(2)] = name
        missing = [s for s in SPLITS if s not in self.protocols]
        if missing:
            raise FileNotFoundError(f"{self.path}: CM protocol(s) not found for {missing}")

    def read_bytes(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def trials(self, split: str) -> list[Trial]:
        return parse_protocol(self.read_bytes(self.protocols[split]).decode("utf-8"))

    def asv(self, split: str) -> dict[str, np.ndarray] | None:
        name = self.asv_scores.get(split)
        return parse_asv_scores(self.read_bytes(name).decode("utf-8")) if name else None

    def decode(self, utt: str, sample_rate: int) -> np.ndarray:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(self.read_bytes(self.audio[utt])), dtype="float32", always_2d=True)
        signal = data.mean(axis=1)
        return resample(signal, sr, sample_rate) if sr != sample_rate else signal

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def stratified_subset(trials: list[Trial], per_attack: int, bonafide: int, seed: int) -> list[int]:
    """Indices of a seeded subset with ``bonafide`` bona fide trials and ``per_attack`` per attack."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(trials):
        groups[t.attack].append(i)
    picked: list[int] = []
    for attack, idx in sorted(groups.items()):
        n = bonafide if attack == "-" else per_attack
        picked.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    return sorted(picked)


# --------------------------------------------------------------------------- feature cache
_worker: dict = {}


def _init_worker(archive_path: str, cfg_dict: dict) -> None:
    _worker["archive"] = LAArchive(archive_path)
    base = AudioConfig(**cfg_dict)
    _worker["cfg"] = {
        "raw": AudioConfig(**{**cfg_dict, "trim_silence": False}),
        "trim": AudioConfig(**{**cfg_dict, "trim_silence": True}),
    }
    _worker["base"] = base


def _extract_one(job: tuple[int, str]) -> tuple[int, dict[str, np.ndarray], dict[str, float]]:
    i, utt = job
    archive: LAArchive = _worker["archive"]
    base: AudioConfig = _worker["base"]
    signal = archive.decode(utt, base.sample_rate)
    stats = silence_profile(signal, base.sample_rate, base.trim_top_db)
    feats = {v: clip_features(signal, c).astype(np.float16) for v, c in _worker["cfg"].items()}
    return i, feats, stats


@dataclass
class FeatureSplit:
    """Memory-mapped features of one split plus its index rows."""

    split: str
    rows: list[dict[str, str]]
    features: dict[str, np.ndarray]  # variant -> (N, n_mels, frames) float16 memmap

    @property
    def labels(self) -> np.ndarray:
        return np.array([int(r["label"]) for r in self.rows], dtype=np.int64)

    @property
    def attacks(self) -> np.ndarray:
        return np.array([r["attack"] for r in self.rows])

    def column(self, name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in self.rows], dtype=np.float64)


INDEX_FIELDS = ["utt", "speaker", "attack", "label", "duration_s", "lead_silence_s", "trail_silence_s", "peak_db"]


def extract_split(
    archive_path: str | Path,
    split: str,
    cfg: AudioConfig,
    out_dir: str | Path,
    *,
    workers: int = 4,
    limit: int | None = None,
    seed: int = 0,
) -> FeatureSplit:
    """Decode every utterance of ``split`` once and cache raw + silence-trimmed log-mels."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = LAArchive(archive_path)
    trials = archive.trials(split)
    missing = [t.utt for t in trials if t.utt not in archive.audio]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} utterances missing from the archive, e.g. {missing[:3]}")
    if limit and limit < len(trials):
        keep = stratified_subset(trials, per_attack=max(1, limit // 8), bonafide=max(1, limit // 4), seed=seed)
        trials = [trials[i] for i in keep]
    archive.close()

    shape = (len(trials), cfg.n_mels, frames_per_clip(cfg))
    maps = {
        v: np.lib.format.open_memmap(out / f"{split}_{v}.npy", mode="w+", dtype=np.float16, shape=shape)
        for v in FEATURE_VARIANTS
    }
    stats: list[dict[str, float] | None] = [None] * len(trials)
    jobs = [(i, t.utt) for i, t in enumerate(trials)]
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(archive_path), vars(cfg))) as pool:
        for n, (i, feats, st) in enumerate(pool.imap_unordered(_extract_one, jobs, chunksize=64), 1):
            for v in FEATURE_VARIANTS:
                maps[v][i] = feats[v]
            stats[i] = st
            if n % 5000 == 0:
                logger.info("features", extra={"split": split, "done": n, "total": len(jobs)})
    for m in maps.values():
        m.flush()

    rows = []
    for t, st in zip(trials, stats, strict=True):
        assert st is not None
        rows.append(
            {
                "utt": t.utt,
                "speaker": t.speaker,
                "attack": t.attack,
                "label": str(t.label),
                **{k: f"{st[k]:.4f}" for k in ("duration_s", "lead_silence_s", "trail_silence_s", "peak_db")},
            }
        )
    with (out / f"{split}_index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("cached split", extra={"split": split, "n": len(rows), "shape": shape})
    return load_split(out, split)


def load_split(feature_dir: str | Path, split: str) -> FeatureSplit:
    d = Path(feature_dir)
    with (d / f"{split}_index.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    feats = {v: np.load(d / f"{split}_{v}.npy", mmap_mode="r") for v in FEATURE_VARIANTS}
    return FeatureSplit(split, rows, feats)


# --------------------------------------------------------------------------- audit
def _describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "p05": round(float(q[0]), 4),
        "p25": round(float(q[1]), 4),
        "median": round(float(q[2]), 4),
        "p75": round(float(q[3]), 4),
        "p95": round(float(q[4]), 4),
    }


def audit_split(fs: FeatureSplit) -> dict:
    """Counts, speakers, attacks and per-class duration / silence statistics of one split."""
    labels = fs.labels
    speakers_by_class = {c: sorted({r["speaker"] for r in fs.rows if int(r["label"]) == c}) for c in (0, 1)}
    per_class = {}
    for c, name in ((BONAFIDE, "bonafide"), (SPOOF, "spoof")):
        mask = labels == c
        per_class[name] = {
            "n": int(mask.sum()),
            **{k: _describe(fs.column(k)[mask]) for k in ("duration_s", "lead_silence_s", "trail_silence_s")},
        }
    return {
        "split": fs.split,
        "n": len(fs.rows),
        "n_bonafide": int((labels == BONAFIDE).sum()),
        "n_spoof": int((labels == SPOOF).sum()),
        "n_speakers": len({r["speaker"] for r in fs.rows}),
        "n_speakers_bonafide": len(speakers_by_class[0]),
        "attacks": dict(sorted(Counter(r["attack"] for r in fs.rows if r["attack"] != "-").items())),
        "per_class": per_class,
    }


def speaker_overlap(splits: dict[str, FeatureSplit]) -> dict[str, int]:
    """Number of speakers shared between each pair of splits (official splits: all zero)."""
    spk = {s: {r["speaker"] for r in fs.rows} for s, fs in splits.items()}
    names = sorted(spk)
    return {f"{a}&{b}": len(spk[a] & spk[b]) for i, a in enumerate(names) for b in names[i + 1 :]}


def iter_batches(array: np.ndarray, batch: int) -> Iterator[np.ndarray]:
    for start in range(0, len(array), batch):
        yield np.asarray(array[start : start + batch], dtype=np.float32)
