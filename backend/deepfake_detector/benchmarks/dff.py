"""DeepFakeFace (DFF) image benchmark data: indexing, identity-grouped splits and face crops.

DFF (Song et al., 2023; https://huggingface.co/datasets/OpenRL/DeepFakeFace, licence tag
``openrail``) has four zips of 30,000 images each: ``wiki`` (real photos selected from
IMDB-WIKI), ``inpainting`` (Stable Diffusion inpainting), ``insight`` (InsightFace face
swap) and ``text2img`` (Stable Diffusion v1.5). The fakes keep the IMDB-WIKI relative paths
of the real image they were derived from, so a real photo and its three fakes share a key.

Leakage control: every image is assigned to a *group* (the IMDB-WIKI person id parsed from
the file name, or the shared relative path when no id parses) and whole groups go to
train / val / test by a seeded hash. A person, and every fake derived from their photo,
therefore appears in exactly one split. This is image-level data: there is no video,
audio or temporal information.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from multiprocessing import get_context
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from ..config import FaceConfig

logger = logging.getLogger(__name__)

REAL, FAKE = 0, 1
SUBSETS = {"wiki": REAL, "inpainting": FAKE, "insight": FAKE, "text2img": FAKE}
FAKE_TYPES = ("inpainting", "insight", "text2img")
HF_URL = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/{subset}.zip"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_WIKI_ID = re.compile(r"^(\d+)_\d{3,4}-\d{1,2}-\d{1,2}_\d{4}$")
_IMDB_ID = re.compile(r"^(nm\d+)_")


@dataclass
class ImageRecord:
    subset: str
    label: int
    member: str
    key: str
    person: str
    group: str
    split: str = ""
    # filled in by crop extraction
    crop: str = ""
    file_format: str = ""
    width: int = 0
    height: int = 0
    file_bytes: int = 0
    face_found: int = 0
    face_frac: float = 0.0


def person_id(stem: str) -> str:
    """IMDB-WIKI identity from a file stem (``10000217_1981-05-05_2009`` -> ``wiki:10000217``)."""
    if m := _IMDB_ID.match(stem):
        return f"imdb:{m.group(1)}"
    if m := _WIKI_ID.match(stem):
        return f"wiki:{m.group(1)}"
    return ""


def relative_key(member: str, subset: str) -> str:
    """Path inside the subset with the subset folder(s) and the extension removed."""
    parts = [p for p in PurePosixPath(member).with_suffix("").parts if p not in ("", ".")]
    while parts and parts[0].lower() in (subset.lower(), *SUBSETS):
        parts = parts[1:]
    return "/".join(parts)


def index_zip(path: str | Path, subset: str) -> list[ImageRecord]:
    with zipfile.ZipFile(path) as zf:
        names = [
            n
            for n in zf.namelist()
            if PurePosixPath(n).suffix.lower() in IMAGE_EXTS and "__MACOSX" not in n and not n.endswith("/")
        ]
    out = []
    for n in sorted(names):
        key = relative_key(n, subset)
        pid = person_id(PurePosixPath(key).name)
        out.append(ImageRecord(subset, SUBSETS[subset], n, key, pid, pid or f"key:{key}"))
    return out


def _hash_fraction(key: str, seed: int) -> float:
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / float(16**12)


def assign_splits(records: list[ImageRecord], val: float, test: float, seed: int) -> None:
    for r in records:
        u = _hash_fraction(r.group, seed)
        r.split = "test" if u < test else "val" if u < test + val else "train"


def select_groups(
    records: list[ImageRecord], n_groups: dict[str, int], seed: int, require_all: bool = True
) -> list[ImageRecord]:
    """Seeded sample of whole groups per split; by default only groups with all four subsets,
    so every selected real photo comes with its inpainting / face-swap / text2img fakes."""
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_group[r.group].append(r)
    rng = np.random.default_rng(seed)
    chosen: list[ImageRecord] = []
    for split, n in n_groups.items():
        groups = sorted(
            g
            for g, rs in by_group.items()
            if rs[0].split == split and (not require_all or {r.subset for r in rs} == set(SUBSETS))
        )
        if n and n < len(groups):
            groups = sorted(rng.choice(groups, n, replace=False).tolist())
        for g in groups:
            chosen.extend(sorted(by_group[g], key=lambda r: (r.subset, r.member)))
    return chosen


def sniff_format(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "other"


# --------------------------------------------------------------------------- crops
_w: dict = {}


def _init(zip_dir: str, face_cfg: dict, out_dir: str, jpeg_quality: int) -> None:
    from ..preprocessing.pipeline import FaceExtractor

    _w["zips"] = {s: zipfile.ZipFile(Path(zip_dir) / f"{s}.zip") for s in SUBSETS}
    _w["extractor"] = FaceExtractor(FaceConfig(**face_cfg))
    _w["out"] = Path(out_dir)
    _w["q"] = jpeg_quality


def _crop_one(job: tuple[int, str, str]) -> tuple[int, dict]:
    i, subset, member = job
    data = _w["zips"][subset].read(member)
    meta = {"file_format": sniff_format(data), "file_bytes": len(data)}
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return i, {**meta, "crop": ""}
    h, w = image.shape[:2]
    crop = _w["extractor"].crop_image(image)
    face_frac = float(crop.box[2] * crop.box[3] / (w * h)) if crop.box else 0.0
    name = f"{i:07d}.jpg"
    # Every crop (real and fake) is re-encoded as JPEG at the same quality so the
    # container format of the source file cannot be used as a cue.
    cv2.imwrite(str(_w["out"] / name), crop.crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, _w["q"]])
    return i, {**meta, "crop": name, "width": w, "height": h, "face_found": int(crop.face_found),
               "face_frac": round(face_frac, 4)}  # fmt: skip


def extract_crops(
    records: list[ImageRecord], zip_dir: str | Path, out_dir: str | Path, face_cfg: FaceConfig, *,
    workers: int = 4, jpeg_quality: int = 95,
) -> list[ImageRecord]:  # fmt: skip
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, r.subset, r.member) for i, r in enumerate(records)]
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(str(zip_dir), vars(face_cfg), str(out), jpeg_quality)) as pool:
        for n, (i, meta) in enumerate(pool.imap_unordered(_crop_one, jobs, chunksize=32), 1):
            for k, v in meta.items():
                setattr(records[i], k, v)
            if n % 5000 == 0:
                logger.info("crops", extra={"done": n, "total": len(jobs)})
    failed = sum(1 for r in records if not r.crop)
    if failed:
        logger.warning("undecodable images skipped", extra={"n": failed})
    return [r for r in records if r.crop]


def write_index(records: list[ImageRecord], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(ImageRecord)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)


def read_index(path: str | Path) -> list[ImageRecord]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    types = {f.name: f.type for f in fields(ImageRecord)}
    out = []
    for row in rows:
        kw = {}
        for k, v in row.items():
            t = types[k]
            kw[k] = int(v) if t in (int, "int") else float(v) if t in (float, "float") else v
        out.append(ImageRecord(**kw))
    return out


def load_images(records: list[ImageRecord], crop_dir: str | Path, size: int) -> np.ndarray:
    """Decode crops into a uint8 ``(N, size, size, 3)`` RGB array."""
    out = np.zeros((len(records), size, size, 3), np.uint8)
    for i, r in enumerate(records):
        img = cv2.imread(str(Path(crop_dir) / r.crop))
        if img.shape[0] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return out


def audit(records: list[ImageRecord], all_records: list[ImageRecord] | None = None) -> dict:
    """Counts, formats, resolutions, face-detection rates and group overlap per split/subset."""

    def describe(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        if a.size == 0:
            return {}
        q = np.percentile(a, [5, 50, 95])
        return {"p05": round(float(q[0]), 1), "median": round(float(q[1]), 1), "p95": round(float(q[2]), 1)}

    out: dict = {"selected": {}}
    for split in ("train", "val", "test"):
        rs = [r for r in records if r.split == split]
        per_subset = {}
        for s in SUBSETS:
            ss = [r for r in rs if r.subset == s]
            per_subset[s] = {
                "n": len(ss),
                "formats": dict(Counter(r.file_format for r in ss)),
                "width": describe([r.width for r in ss]),
                "height": describe([r.height for r in ss]),
                "kb": describe([r.file_bytes / 1024 for r in ss]),
                "face_detected_rate": round(float(np.mean([r.face_found for r in ss])), 4) if ss else None,
            }
        out["selected"][split] = {
            "n": len(rs),
            "n_real": sum(r.label == REAL for r in rs),
            "n_fake": sum(r.label == FAKE for r in rs),
            "n_groups": len({r.group for r in rs}),
            "per_subset": per_subset,
        }
    groups = {s: {r.group for r in records if r.split == s} for s in ("train", "val", "test")}
    out["group_overlap"] = {
        "train&val": len(groups["train"] & groups["val"]),
        "train&test": len(groups["train"] & groups["test"]),
        "val&test": len(groups["val"] & groups["test"]),
    }
    if all_records is not None:
        keys = defaultdict(set)
        for r in all_records:
            keys[r.subset].add(r.key)
        out["full_dataset"] = {
            "n_per_subset": {s: len(keys[s]) for s in SUBSETS},
            "fake_keys_matching_a_real_key": {s: len(keys[s] & keys["wiki"]) for s in FAKE_TYPES},
            "with_parsed_person_id": round(float(np.mean([bool(r.person) for r in all_records])), 4),
            "n_groups": len({r.group for r in all_records}),
            "example_members": {s: [r.member for r in all_records if r.subset == s][:3] for s in SUBSETS},
        }
    return out
