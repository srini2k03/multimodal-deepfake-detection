"""API tests with a mocked detector (no TensorFlow needed)."""

import io
import time

import pytest

from deepfake_detector.api import create_app


class MockDetector:
    mode = "trained"

    def __init__(self):
        self.calls = []

    def info(self):
        return {"mode": self.mode, "warning": None, "models": {"xception": {"status": "trained"}}}

    def predict_file(self, path, media_type=None):
        self.calls.append((str(path), media_type))
        return {
            "media_type": media_type,
            "mode": self.mode,
            "warning": None,
            "label": "FAKE",
            "probability_fake": 0.91,
            "threshold": 0.5,
            "modalities": {
                "visual": {"probability": 0.91, "per_model": {"xception": {"probability": 0.9}}, "frames": []},
                "audio": None,
            },
            "fusion": {"method": "weighted", "used": ["visual"]},
            "gradcam": None,
            "processing_ms": 1.0,
        }


@pytest.fixture
def mock_detector():
    return MockDetector()


@pytest.fixture
def client(cfg, mock_detector):
    cfg.api.max_upload_mb = 1
    app = create_app(cfg, detector=mock_detector)
    app.testing = True
    return app.test_client()


def upload(client, path, name=None, query=""):
    with open(path, "rb") as fh:
        data = {"file": (io.BytesIO(fh.read()), name or path.name)}
    return client.post(f"/api/predict{query}", data=data, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok" and body["mode"] == "trained"
    assert r.headers["X-Request-ID"]


def test_request_id_propagates(client):
    assert client.get("/api/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"


def test_models(client):
    assert client.get("/api/models").get_json()["models"]["xception"]["status"] == "trained"


def test_predict_image(client, image_file, mock_detector, cfg):
    r = upload(client, image_file)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["label"] == "FAKE" and body["probability_fake"] == 0.91
    assert body["file"]["name"] == "face.png" and body["file"]["width"] == 200
    assert mock_detector.calls[0][1] == "image"
    # temp upload is removed after the request
    from pathlib import Path

    assert list(Path(cfg.api.upload_dir).iterdir()) == []


def test_predict_video_metadata(client, video_file):
    body = upload(client, video_file).get_json()
    assert body["media_type"] == "video" and body["file"]["frames"] == 20


def test_missing_file(client):
    r = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "No file" in r.get_json()["error"]["message"]


def test_unsupported_extension(client, tmp_path):
    p = tmp_path / "evil.exe"
    p.write_bytes(b"MZ")
    assert upload(client, p).status_code == 415


def test_corrupt_image_rejected(client, tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"definitely not a jpeg")
    r = upload(client, p)
    assert r.status_code == 422
    assert "image" in r.get_json()["error"]["message"]


def test_too_large(client, tmp_path):
    p = tmp_path / "big.png"
    p.write_bytes(b"0" * (2 * 1024 * 1024))
    r = upload(client, p)
    assert r.status_code == 413
    assert "too large" in r.get_json()["error"]["message"]


def test_unknown_route_is_json(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["error"]["status"] == 404


def test_async_job_flow(client, image_file):
    r = upload(client, image_file, query="?async=1")
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    for _ in range(100):
        body = client.get(f"/api/jobs/{job_id}").get_json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert body["status"] == "done"
    assert body["result"]["probability_fake"] == 0.91
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_detector_error_returns_json_500(cfg, image_file):
    class Boom(MockDetector):
        def predict_file(self, path, media_type=None):
            raise RuntimeError("kaboom")

    app = create_app(cfg, detector=Boom())
    r = upload(app.test_client(), image_file)
    assert r.status_code == 500
    assert r.get_json()["error"]["message"] == "Internal server error"


def test_end_to_end_with_placeholder_detector(cfg, image_file):
    """No mock: real DeepfakeDetector in TensorFlow-free demo mode."""
    app = create_app(cfg, eager=True)
    client = app.test_client()
    assert client.get("/api/health").get_json()["mode"] == "demo"
    body = upload(client, image_file).get_json()
    assert body["mode"] == "demo" and "DEMO MODE" in body["warning"]
    assert body["gradcam"]["overlay"].startswith("data:image/png;base64,")


def test_upload_removed_when_detector_fails_to_load(cfg, image_file, tmp_path):
    cfg.api.demo_mode = "off"  # no weights in cfg.api.weights_dir -> detector construction fails
    app = create_app(cfg)
    r = upload(app.test_client(), image_file)
    assert r.status_code == 500
    assert list((tmp_path / "uploads").iterdir()) == []
