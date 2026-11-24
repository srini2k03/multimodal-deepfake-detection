"""Upload security, rate limiting, audio validation, calibration and modality-aware fusion."""

from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pytest

from deepfake_detector.api import create_app
from deepfake_detector.api.app import prepare_upload_dir
from deepfake_detector.api.ratelimit import RateLimiter
from deepfake_detector.inference.predictor import Calibration, DeepfakeDetector
from deepfake_detector.preprocessing.audio import write_wav

from .test_api import MockDetector, upload


def test_rate_limiter_window_and_retry_after():
    rl = RateLimiter(limit=2, window_s=10)
    assert rl.check("a", now=0)[0] and rl.check("a", now=1)[0]
    allowed, retry = rl.check("a", now=2)
    assert not allowed and retry == pytest.approx(8)
    assert rl.check("b", now=2)[0]  # per client
    assert rl.check("a", now=10.5)[0]  # first hit expired
    assert RateLimiter(limit=0).check("x")[0]  # disabled


def test_api_returns_429_with_retry_after(cfg, image_file):
    cfg.api.rate_limit_per_minute = 2
    client = create_app(cfg, detector=MockDetector()).test_client()
    assert upload(client, image_file).status_code == 200
    assert upload(client, image_file).status_code == 200
    res = upload(client, image_file)
    assert res.status_code == 429 and int(res.headers["Retry-After"]) >= 1
    assert res.get_json()["error"]["status"] == 429


def test_upload_dir_is_private_and_cleans_stale_files(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir()
    old, new = d / "upload_old.mp4", d / "upload_new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    assert prepare_upload_dir(d, 3600) == 1
    assert not old.exists() and new.exists()
    assert (d.stat().st_mode & 0o777) == 0o700


def test_client_filename_never_becomes_a_path(cfg, image_file, tmp_path):
    client = create_app(cfg, detector=MockDetector()).test_client()
    data = {"file": (io.BytesIO(image_file.read_bytes()), "../../etc/passwd.png")}
    res = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["file"]["name"] == "passwd.png"
    assert not any((tmp_path / "uploads").iterdir())  # deleted after the request


def test_corrupt_and_too_long_audio_rejected(cfg, tmp_path):
    client = create_app(cfg, detector=MockDetector()).test_client()
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF....not really a wav file")
    assert upload(client, bad).status_code == 422
    cfg.api.max_audio_seconds = 1
    long_wav = tmp_path / "long.wav"
    write_wav(long_wav, np.zeros(16000 * 3, np.float32), 16000)
    client = create_app(cfg, detector=MockDetector()).test_client()
    res = upload(client, long_wav)
    assert res.status_code == 413 and "limit is 1s" in res.get_json()["error"]["message"]


def test_calibration_prior_shift():
    cal = Calibration(a=1.0, b=0.0, prior_fake=0.9)
    # identity scaling fitted on a 90 % fake split: re-weighting to equal priors lowers P(fake)
    assert float(cal(0.9)) == pytest.approx(0.5, abs=1e-6)
    assert float(Calibration(1.0, 0.0, 0.5)(0.73)) == pytest.approx(0.73, abs=1e-6)


def test_model_card_calibration_is_ignored_for_untrained_models(cfg, wav_file):
    weights = cfg.api.weights_dir
    os.makedirs(weights, exist_ok=True)
    card = {"audio": {"calibration": {"a": 2.0, "b": -1.0, "prior_fake": 0.9, "source": "test"}}}
    with open(os.path.join(weights, "model_card.json"), "w") as fh:
        json.dump(card, fh)
    det = DeepfakeDetector(cfg)
    assert "audio" in det.info()["calibration"]
    res = det.predict_file(wav_file)
    audio = res["modalities"]["audio"]
    assert audio["calibrated"] is False and audio["probability"] == audio["raw_probability"]


def test_missing_audio_is_not_evidence(cfg, video_file):
    res = DeepfakeDetector(cfg).predict_file(video_file)
    assert res["modalities"]["audio"]["available"] is False
    assert res["fusion"]["used"] == ["visual"] and res["fusion"]["validated"] is False
    assert res["probability_fake"] == pytest.approx(res["modalities"]["visual"]["probability"], abs=1e-5)


def test_long_audio_is_truncated(cfg, tmp_path):
    cfg.audio.max_seconds = 2.0
    path = tmp_path / "long.wav"
    t = np.arange(16000 * 5) / 16000
    write_wav(path, 0.3 * np.sin(2 * np.pi * 220 * t), 16000)
    audio = DeepfakeDetector(cfg).predict_file(path)["modalities"]["audio"]
    assert audio["truncated"] and audio["analysed_seconds"] == pytest.approx(2.0) and audio["duration"] == 5.0
