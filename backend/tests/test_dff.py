"""DeepFakeFace benchmark plumbing on synthetic zips with the DFF layout (no downloads)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepfake_detector.benchmarks import dff
from deepfake_detector.benchmarks.image_ops import ROBUSTNESS_CONDITIONS, apply_condition, augment, to_model_input

from .conftest import synthetic_face


def make_dff(root: Path, n_people: int = 24) -> Path:
    """Four zips; every person has one real JPEG and three PNG 'fakes' sharing its path."""
    root.mkdir(parents=True, exist_ok=True)
    for subset in dff.SUBSETS:
        with zipfile.ZipFile(root / f"{subset}.zip", "w") as zf:
            for p in range(n_people):
                img = synthetic_face(160, seed=p)
                if subset != "wiki":
                    img = cv2.GaussianBlur(img, (0, 0), 2.0)  # a learnable synthetic "artefact"
                ext = ".jpg" if subset == "wiki" else ".png"
                ok, buf = cv2.imencode(ext, img)
                assert ok
                zf.writestr(f"{subset}/{p % 10:02d}/{1000000 + p}_1980-01-01_2010{ext}", buf.tobytes())
    return root


def test_keys_person_ids_and_grouped_splits(tmp_path):
    root = make_dff(tmp_path / "dff")
    assert dff.person_id("10000217_1981-05-05_2009") == "wiki:10000217"
    assert dff.person_id("nm0000001_rm124825600_1899-5-10_1968") == "imdb:nm0000001"
    assert dff.person_id("random_name") == ""
    assert dff.relative_key("inpainting/17/123_1980-1-1_2000.png", "inpainting") == "17/123_1980-1-1_2000"
    recs = [r for s in dff.SUBSETS for r in dff.index_zip(root / f"{s}.zip", s)]
    assert len(recs) == 96 and {r.group for r in recs} == {f"wiki:{1000000 + p}" for p in range(24)}
    dff.assign_splits(recs, 0.2, 0.3, seed=0)
    by_group = {}
    for r in recs:
        by_group.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in by_group.values())  # a person never straddles splits
    chosen = dff.select_groups(recs, {"train": 0, "val": 0, "test": 0}, seed=0)
    assert len(chosen) == 96
    assert dff.sniff_format(b"\xff\xd8\xff\xe0") == "jpeg" and dff.sniff_format(b"\x89PNG\r\n\x1a\n") == "png"


def test_image_ops_shapes_and_conditions():
    img = synthetic_face(64)[:, :, ::-1].copy()
    rng = np.random.default_rng(0)
    for cond in ROBUSTNESS_CONDITIONS:
        out = apply_condition(img, cond, rng)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(apply_condition(img, "clean", rng), img)
    from deepfake_detector.benchmarks.image_ops import equalize

    eq = equalize(img, 16)
    assert eq.shape == img.shape and np.abs(eq.astype(int) - img).mean() > 0
    assert np.array_equal(equalize(img, 128), img)  # never up-samples
    assert augment(img, rng).shape == img.shape
    x = to_model_input(img[None])
    assert x.min() >= -1.0 and x.max() <= 1.0
    with pytest.raises(ValueError):
        apply_condition(img, "sepia_1", rng)


@pytest.mark.tf
def test_dff_cli_end_to_end(tmp_path, monkeypatch):
    from deepfake_detector.benchmarks import run_dff

    monkeypatch.setattr(run_dff, "IMAGE_SIZE", 64)
    monkeypatch.setattr(run_dff, "WEIGHTS", None)
    monkeypatch.setattr(run_dff, "ROBUSTNESS_CONDITIONS", ["clean", "jpeg_q50"])
    root = make_dff(tmp_path / "dff", n_people=30)
    work, out = tmp_path / "work", tmp_path / "out"
    groups = ["--train-groups", "0", "--val-groups", "0", "--test-groups", "0"]
    fractions = ["--val-fraction", "0.25", "--test-fraction", "0.35"]
    run_dff.main(["prepare", "--zips", str(root), "--work", str(work), "--workers", "2", *groups, *fractions])
    audit = json.loads((work / "audit.json").read_text())
    assert audit["group_overlap"] == {"train&val": 0, "train&test": 0, "val&test": 0}
    assert audit["full_dataset"]["fake_keys_matching_a_real_key"] == {"inpainting": 30, "insight": 30, "text2img": 30}
    run_dff.main(["probe", "--work", str(work)])
    run_dff.main(["finetune", "--work", str(work), "--epochs", "1", "--batch-size", "8"])
    run_dff.main(["finetune", "--work", str(work), "--epochs", "1", "--batch-size", "8", "--equalize-px", "32"])
    run_dff.main(["evaluate", "--work", str(work), "--out", str(out)])
    m = json.loads((out / "metrics.json").read_text())["systems"]
    assert {"meta_lr", "effnet_probe", "effnet_ft", "effnet_ft_eq32", "effnet_probe_no_insight"} <= set(m)
    fsm = m["effnet_ft"]["test"]["face_size_matched"]
    assert (
        len(fsm["bins"]) == 5
        and sum(b["n_real"] + b["n_fake"] for b in fsm["bins"])
        == m["effnet_ft"]["test"]["n_real"] + m["effnet_ft"]["test"]["n_fake"]
    )
    assert m["effnet_probe_no_insight"]["test"]["per_type"]["insight"]["seen_in_training"] is False
    assert (out / "test_scores.csv.gz").exists()
    run_dff.main(["robustness", "--work", str(work), "--out", str(out), "--n-per-class", "6"])
    assert set(json.loads((out / "robustness.json").read_text())["conditions"]) == {"clean", "jpeg_q50"}
    run_dff.main(["gradcam", "--work", str(work), "--out", str(out), "--n-per-class", "3"])
    assert (out / "gradcam_mean_heatmaps.png").exists()
    run_dff.main(["latency", "--work", str(work), "--out", str(out), "--zips", str(root)])
    lat = json.loads((out / "latency.json").read_text())
    assert lat["exports"]["tflite_fp32"]["max_abs_prob_diff_vs_keras"] < 1e-3


@pytest.mark.tf
def test_wilddeepfake_cross_dataset_cli(tmp_path):
    """Video-level aggregation over sequences read from tar.gz files with the WildDeepfake layout."""
    import io
    import tarfile

    from deepfake_detector.benchmarks import wilddeepfake
    from deepfake_detector.models import build_frame_classifier

    for split, n_videos in (("real_test", 2), ("fake_test", 2)):
        (tmp_path / "tars" / split).mkdir(parents=True)
        for v in range(n_videos):
            with tarfile.open(tmp_path / "tars" / split / f"{v}.tar.gz", "w:gz") as tar:
                for seq in range(2):
                    for f in range(5):
                        _ok, buf = cv2.imencode(".png", synthetic_face(96, seed=v * 100 + seq * 10 + f))
                        info = tarfile.TarInfo(f"{v}/{seq}/{f}.png")
                        info.size = len(buf)
                        tar.addfile(info, io.BytesIO(buf.tobytes()))
    model = build_frame_classifier("efficientnetv2_b0", 64, weights=None)
    model.save(tmp_path / "m.keras")
    out = tmp_path / "out"
    wilddeepfake.main(["--tars", str(tmp_path / "tars"), "--models", f"ft={tmp_path / 'm.keras'}",
                       f"ft_eq={tmp_path / 'm.keras'}:eq32", "--out", str(out), "--image-size", "64",
                       "--frames-per-sequence", "3", "--workers", "2"])  # fmt: skip
    rep = json.loads((out / "cross_dataset_wilddeepfake.json").read_text())
    assert rep["n_videos"] == 4 and rep["n_sequences"] == 8 and rep["n_frames"] == 24
    assert set(rep["systems"]) == {"ft", "ft_eq"}
    assert set(rep["systems"]["ft"]["sequence_level"]) == {"mean", "median", "max", "topk_mean", "confidence_weighted"}
