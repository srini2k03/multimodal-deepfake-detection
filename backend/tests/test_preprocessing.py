import numpy as np
import pytest

from deepfake_detector.config import FaceConfig, SamplingConfig
from deepfake_detector.preprocessing.alignment import align_and_crop, center_crop, rotate_image, square_box
from deepfake_detector.preprocessing.face_detector import FaceBox, FaceDetector, HaarFaceDetector, build_face_detector
from deepfake_detector.preprocessing.pipeline import FaceExtractor
from deepfake_detector.preprocessing.video import probe_video, sample_frames, sample_indices


class FixedDetector(FaceDetector):
    """Deterministic detector so pipeline tests do not depend on cascade behaviour."""

    name = "fixed"

    def _detect(self, image_bgr):
        h, w = image_bgr.shape[:2]
        return [FaceBox(w // 4, h // 4, w // 2, h // 2, 0.99), FaceBox(0, 0, 45, 45, 0.5)]


@pytest.mark.parametrize(
    ("count", "n", "expected"),
    [(100, 4, [12, 37, 62, 87]), (3, 10, [0, 1, 2]), (0, 5, []), (10, 0, [])],
)
def test_uniform_sampling(count, n, expected):
    assert sample_indices(count, n) == expected


def test_stride_sampling_and_bad_strategy():
    assert sample_indices(50, 3, "stride", 10) == [0, 10, 20]
    with pytest.raises(ValueError):
        sample_indices(10, 2, "random")


def test_square_box_is_square_and_clipped():
    x1, y1, x2, y2 = square_box(FaceBox(10, 20, 40, 60), margin=0.3, image_shape=(200, 300, 3))
    assert (x2 - x1) == (y2 - y1)
    assert x1 >= 0 and y1 >= 0 and x2 <= 300 and y2 <= 200
    # near the border the box is shifted, not shrunk
    x1, y1, x2, y2 = square_box(FaceBox(0, 0, 50, 50), margin=0.2, image_shape=(200, 200, 3))
    assert (x1, y1) == (0, 0) and x2 - x1 == 70


def test_align_and_crop_output_size():
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    out = align_and_crop(img, FaceBox(100, 60, 80, 100), crop_size=64, margin=0.25, align=True)
    assert out.crop_bgr.shape == (64, 64, 3)
    assert out.crop_bgr.dtype == np.uint8
    assert center_crop(img, 48).shape == (48, 48, 3)


def test_rotate_image_preserves_shape():
    img = np.zeros((50, 80, 3), np.uint8)
    assert rotate_image(img, 10.0, (40, 25)).shape == img.shape


def test_haar_detector_on_blank_image_and_min_size():
    det = HaarFaceDetector(min_face_size=30)
    assert det.detect(np.zeros((120, 120, 3), np.uint8)) == []
    assert det.detect(np.zeros((0, 0, 3), np.uint8)) == []
    assert FixedDetector(min_face_size=50).detect(np.zeros((200, 200, 3), np.uint8))[0].w == 100


def test_build_face_detector_falls_back_to_haar():
    det = build_face_detector(FaceConfig(detector="dnn", dnn_prototxt="missing", dnn_weights="missing"))
    assert det.name == "haar"


def test_probe_and_sample_video(video_file):
    info = probe_video(video_file)
    assert info.frame_count == 20 and info.width == 128
    assert info.duration_s == pytest.approx(2.0, rel=0.05)
    _, frames = sample_frames(video_file, 5)
    assert [f.index for f in frames] == sample_indices(20, 5)
    assert frames[0].image_bgr.shape == (128, 128, 3)


def test_probe_invalid_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        probe_video(bad)


def test_extractor_crops_and_cache(video_file, tmp_path):
    ext = FaceExtractor(
        FaceConfig(crop_size=64),
        SamplingConfig(frames_per_video=4),
        detector=FixedDetector(min_face_size=10),
        cache_dir=tmp_path / "cache",
    )
    res = ext.extract_video(video_file)
    assert len(res.crops) == 4 and res.faces_found == 4
    assert all(c.crop_bgr.shape == (64, 64, 3) for c in res.crops)
    cached_dirs = list((tmp_path / "cache").iterdir())
    assert len(cached_dirs) == 1 and (cached_dirs[0] / "meta.json").exists()
    again = ext.extract_video(video_file)  # served from cache
    assert [c.frame_index for c in again.crops] == [c.frame_index for c in res.crops]
    assert np.array_equal(again.crops[0].crop_bgr, res.crops[0].crop_bgr)


def test_extractor_faceless_fallback(image_file):
    import cv2

    ext = FaceExtractor(FaceConfig(crop_size=48), detector=HaarFaceDetector(min_face_size=500))
    img = cv2.imread(str(image_file))
    crop = ext.crop_image(img)
    assert crop is not None and not crop.face_found and crop.crop_bgr.shape == (48, 48, 3)
    assert ext.crop_image(img, fallback_center=False) is None
