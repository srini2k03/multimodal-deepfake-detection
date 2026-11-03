"""Detector tests using the TensorFlow-free placeholder backend."""

import pytest

from deepfake_detector.inference.predictor import DEMO_WARNING, DeepfakeDetector, media_type_for


def test_media_type_detection():
    assert media_type_for("a.JPG") == "image"
    assert media_type_for("a.mp4") == "video"
    assert media_type_for("a.wav") == "audio"
    with pytest.raises(ValueError):
        media_type_for("a.exe")


def test_demo_mode_off_requires_weights(cfg):
    cfg.api.demo_mode = "off"
    with pytest.raises(FileNotFoundError):
        DeepfakeDetector(cfg)


def test_placeholder_image_prediction(cfg, image_file):
    det = DeepfakeDetector(cfg)
    assert det.mode == "demo"
    assert det.info()["models"]["xception"]["status"] == "placeholder"
    res = det.predict_file(image_file)
    assert res["media_type"] == "image" and res["mode"] == "demo"
    assert res["warning"] == DEMO_WARNING
    assert 0.0 <= res["probability_fake"] <= 1.0
    assert res["label"] in ("REAL", "FAKE")
    visual = res["modalities"]["visual"]
    assert set(visual["per_model"]) == {"xception", "inception_v3"}
    assert len(visual["frames"]) == 1
    assert res["gradcam"]["overlay"].startswith("data:image/png;base64,")
    assert res["gradcam"]["method"].startswith("pseudo-saliency")
    # deterministic
    assert det.predict_file(image_file)["probability_fake"] == res["probability_fake"]


def test_placeholder_video_prediction(cfg, video_file):
    res = DeepfakeDetector(cfg).predict_file(video_file)
    visual = res["modalities"]["visual"]
    assert visual["frames_analyzed"] == 4
    assert [f["index"] for f in visual["frames"]] == sorted(f["index"] for f in visual["frames"])
    assert res["modalities"]["audio"]["available"] is False  # MJPG avi has no audio track
    assert res["fusion"]["used"] == ["visual"]


def test_placeholder_audio_prediction(cfg, wav_file):
    res = DeepfakeDetector(cfg).predict_file(wav_file)
    audio = res["modalities"]["audio"]
    assert audio["available"] and len(audio["segments"]) == 3
    assert res["modalities"]["visual"] is None
    assert res["probability_fake"] == pytest.approx(audio["probability"], abs=1e-5)
