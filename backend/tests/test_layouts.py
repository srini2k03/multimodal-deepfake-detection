from pathlib import Path

from deepfake_detector.data.layouts import FAKE, REAL, class_weights, discover, read_manifest, write_manifest


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_faceforensics_layout_and_group_split(tmp_path):
    root = tmp_path / "ffpp"
    for vid in ("000", "003", "001", "002", "010", "011"):
        touch(root / "original_sequences/youtube/c23/videos" / f"{vid}.mp4")
    for method in ("Deepfakes", "FaceSwap"):
        for pair in ("000_003", "003_000", "001_002", "010_011"):
            touch(root / "manipulated_sequences" / method / "c23/videos" / f"{pair}.mp4")
    touch(root / "manipulated_sequences/Deepfakes/c40/videos/000_003.mp4")  # other compression ignored

    recs = discover("faceforensics", root, val_fraction=0.3, test_fraction=0.3, seed=1)
    assert len(recs) == 6 + 8
    assert sum(r.label == FAKE for r in recs) == 8
    by_id = {r.video_id: r for r in recs}
    # identities 000/003 always share one group (and thus one split)
    group = {
        by_id[v].group for v in ("real_youtube__000", "real_youtube__003", "Deepfakes__000_003", "FaceSwap__003_000")
    }
    assert len(group) == 1
    for g in {r.group for r in recs}:
        assert len({r.split for r in recs if r.group == g}) == 1


def test_celebdf_layout_with_official_test_list(tmp_path):
    root = tmp_path / "celeb"
    touch(root / "Celeb-real/id0_0000.mp4")
    touch(root / "Celeb-real/id1_0000.mp4")
    touch(root / "Celeb-real/id5_0001.mp4")
    touch(root / "YouTube-real/00001.mp4")
    touch(root / "Celeb-synthesis/id0_id1_0000.mp4")
    touch(root / "Celeb-synthesis/id5_id6_0003.mp4")
    (root / "List_of_testing_videos.txt").write_text("1 Celeb-real/id5_0001.mp4\n0 Celeb-synthesis/id5_id6_0003.mp4\n")
    recs = {r.video_id: r for r in discover("celebdf", root)}
    assert recs["Celeb-real__id5_0001"].split == "test"
    assert recs["Celeb-synthesis__id5_id6_0003"].split == "test"
    assert recs["Celeb-real__id0_0000"].split != "test"
    assert recs["YouTube-real__00001"].label == REAL
    assert recs["Celeb-synthesis__id0_id1_0000"].group == recs["Celeb-real__id1_0000"].group


def test_generic_layout_manifest_roundtrip(tmp_path):
    root = tmp_path / "gen"
    touch(root / "real/a.mp4")
    touch(root / "fake/sub/b.avi")
    touch(root / "fake/notes.txt")
    recs = discover("generic", root)
    assert sorted(r.video_id for r in recs) == ["fake__sub__b", "real__a"]
    out = tmp_path / "manifest.csv"
    write_manifest(recs, out, ["n_frames"], {"real__a": {"n_frames": 5}})
    rows = read_manifest(out)
    assert {r["video_id"]: r["n_frames"] for r in rows} == {"fake__sub__b": "", "real__a": "5"}


def test_class_weights_balanced_mean():
    w = class_weights([0, 0, 0, 1])
    assert w[1] == 2.0 and abs(w[0] - 2 / 3) < 1e-9
    assert class_weights([]) == {0: 1.0, 1: 1.0}
