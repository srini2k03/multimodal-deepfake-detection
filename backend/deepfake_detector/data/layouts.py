"""Discover videos in benchmark dataset layouts and assign leakage-free splits.

Supported layouts (``data.dataset``):

``faceforensics`` (FaceForensics++)::

    <root>/original_sequences/youtube/<c23>/videos/000.mp4            -> real
    <root>/manipulated_sequences/<Method>/<c23>/videos/000_003.mp4    -> fake

``celebdf`` (Celeb-DF v1/v2)::

    <root>/Celeb-real/id0_0000.mp4       -> real
    <root>/YouTube-real/00000.mp4        -> real
    <root>/Celeb-synthesis/id0_id1_0000.mp4 -> fake
    <root>/List_of_testing_videos.txt    -> official test list (optional)

``generic``::

    <root>/real/**/*.mp4   <root>/fake/**/*.mp4

Splits are assigned per *identity group* (union of the source/target ids that appear
in a video name) so that a person never appears in both train and test.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
REAL, FAKE = 0, 1
FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")


@dataclass
class VideoRecord:
    video_id: str
    path: str
    label: int  # 0 = real, 1 = fake
    source: str
    group: str
    split: str = "train"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _files(directory: Path, exts: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _ffpp_ids(stem: str) -> list[str]:
    return stem.split("_") if re.fullmatch(r"\d+(_\d+)?", stem) else [stem]


def _celebdf_ids(stem: str, folder: str) -> list[str]:
    if folder == "YouTube-real":
        return [f"yt{stem}"]
    ids = re.findall(r"id\d+", stem)
    return ids or [stem]


def discover_faceforensics(root: Path, compression: str = "c23") -> list[tuple[VideoRecord, list[str]]]:
    out: list[tuple[VideoRecord, list[str]]] = []
    for sub in ("youtube", "actors"):
        for p in _files(root / "original_sequences" / sub / compression / "videos", VIDEO_EXTS):
            out.append((VideoRecord(f"real_{sub}__{p.stem}", str(p), REAL, f"original/{sub}", ""), _ffpp_ids(p.stem)))
    manip_root = root / "manipulated_sequences"
    methods = sorted(d.name for d in manip_root.iterdir() if d.is_dir()) if manip_root.is_dir() else []
    for method in methods:
        for p in _files(manip_root / method / compression / "videos", VIDEO_EXTS):
            out.append((VideoRecord(f"{method}__{p.stem}", str(p), FAKE, method, ""), _ffpp_ids(p.stem)))
    return out


def discover_celebdf(root: Path) -> list[tuple[VideoRecord, list[str]]]:
    out: list[tuple[VideoRecord, list[str]]] = []
    for folder, label in (("Celeb-real", REAL), ("YouTube-real", REAL), ("Celeb-synthesis", FAKE)):
        for p in _files(root / folder, VIDEO_EXTS):
            out.append((VideoRecord(f"{folder}__{p.stem}", str(p), label, folder, ""), _celebdf_ids(p.stem, folder)))
    return out


def discover_generic(root: Path, exts: set[str] = VIDEO_EXTS) -> list[tuple[VideoRecord, list[str]]]:
    out: list[tuple[VideoRecord, list[str]]] = []
    for folder, label in (("real", REAL), ("fake", FAKE)):
        for p in _files(root / folder, exts):
            rel = p.relative_to(root / folder).with_suffix("")
            vid = f"{folder}__" + "__".join(rel.parts)
            out.append((VideoRecord(vid, str(p), label, folder, ""), [vid]))
    return out


def _hash_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _official_celebdf_test(root: Path) -> set[str]:
    listing = root / "List_of_testing_videos.txt"
    if not listing.exists():
        return set()
    test = set()
    for line in listing.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            test.add(str((root / parts[1]).resolve()))
    return test


def assign_splits(
    items: list[tuple[VideoRecord, list[str]]],
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
    forced_test: set[str] | None = None,
) -> list[VideoRecord]:
    """Group by connected identities, then split groups deterministically by hash."""
    uf = _UnionFind()
    for _, ids in items:
        for other in ids[1:]:
            uf.union(ids[0], other)
        uf.find(ids[0])
    forced_test = forced_test or set()
    forced_groups = {uf.find(ids[0]) for rec, ids in items if str(Path(rec.path).resolve()) in forced_test}
    records = []
    for rec, ids in items:
        rec.group = uf.find(ids[0])
        if forced_test:
            # With an official test list, the remaining groups are split into train/val only.
            if rec.group in forced_groups:
                rec.split = "test"
            else:
                rec.split = "val" if _hash_fraction(rec.group, seed) < val_fraction else "train"
        else:
            u = _hash_fraction(rec.group, seed)
            rec.split = "test" if u < test_fraction else "val" if u < test_fraction + val_fraction else "train"
        records.append(rec)
    return records


def discover(
    dataset: str, root: str | Path, *, compression: str = "c23", val_fraction: float = 0.1,
    test_fraction: float = 0.1, seed: int = 42,
) -> list[VideoRecord]:
    root = Path(root)
    forced: set[str] = set()
    if dataset == "faceforensics":
        items = discover_faceforensics(root, compression)
    elif dataset == "celebdf":
        items = discover_celebdf(root)
        forced = _official_celebdf_test(root)
    elif dataset == "generic":
        items = discover_generic(root)
    else:
        raise ValueError(f"Unknown dataset layout: {dataset}")
    return assign_splits(items, val_fraction, test_fraction, seed, forced)


def write_manifest(records: list[VideoRecord], path: str | Path, extra_fields: list[str] | None = None,
                   extras: dict[str, dict] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(VideoRecord)] + (extra_fields or [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            row.update((extras or {}).get(rec.video_id, {}))
            writer.writerow(row)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def class_weights(labels: list[int] | tuple[int, ...]) -> dict[int, float]:
    """Inverse-frequency class weights normalised so the mean weight is 1."""
    total = len(labels)
    if total == 0:
        return {REAL: 1.0, FAKE: 1.0}
    counts = {c: sum(1 for y in labels if int(y) == c) for c in (REAL, FAKE)}
    return {c: (total / (2.0 * n)) if n else 0.0 for c, n in counts.items()}
