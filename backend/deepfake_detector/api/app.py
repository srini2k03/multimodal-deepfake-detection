"""Flask REST API.

Endpoints
---------
``GET  /api/health``        liveness + model mode (trained / partial / demo)
``GET  /api/models``        loaded models, ensemble and fusion settings
``POST /api/predict``       multipart upload (field ``file``); ``?async=1`` returns a job id
``GET  /api/jobs/<job_id>`` async job status and result
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Blueprint, Flask, current_app, g, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .. import __version__
from ..config import Config, load_config
from ..inference.predictor import media_type_for
from ..utils.logging import configure_logging, request_id_var
from .jobs import JobManager
from .validation import ValidationError, check_content, check_extension

logger = logging.getLogger("deepfake_detector.api")
bp = Blueprint("api", __name__, url_prefix="/api")


def error_response(status: int, message: str, **extra: Any):
    body = {"error": {"status": status, "message": message, **extra}}
    rid = request_id_var.get()
    if rid:
        body["error"]["request_id"] = rid
    return jsonify(body), status


class DetectorHolder:
    """Lazily constructs the detector once (thread-safe) so app start-up stays fast."""

    def __init__(self, cfg: Config, detector=None) -> None:
        self.cfg = cfg
        self._detector = detector
        self._lock = threading.Lock()
        self.error: str | None = None

    def get(self):
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    from ..inference.predictor import DeepfakeDetector

                    try:
                        self._detector = DeepfakeDetector(self.cfg)
                    except Exception as exc:
                        self.error = str(exc)
                        raise
        return self._detector

    @property
    def loaded(self) -> bool:
        return self._detector is not None


def _holder() -> DetectorHolder:
    return current_app.extensions["dfd_detector"]


def _jobs() -> JobManager:
    return current_app.extensions["dfd_jobs"]


def _cfg() -> Config:
    return current_app.extensions["dfd_config"]


# --------------------------------------------------------------------------- routes
@bp.get("/health")
def health():
    holder = _holder()
    body: dict[str, Any] = {"status": "ok", "version": __version__, "models_loaded": holder.loaded}
    if holder.loaded:
        info = holder.get().info()
        body.update(mode=info["mode"], warning=info["warning"])
    if holder.error:
        body.update(status="degraded", error=holder.error)
    return jsonify(body)


@bp.get("/models")
def models():
    return jsonify(_holder().get().info())


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in ("1", "true", "yes", "on")


@bp.post("/predict")
def predict():
    cfg = _cfg()
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ValidationError("No file uploaded. Send multipart/form-data with a 'file' field.")
    ext = check_extension(upload.filename)
    media_type = media_type_for(f"x{ext}")

    upload_dir = Path(cfg.api.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=ext, prefix="upload_", dir=upload_dir)
    os.close(fd)
    tmp = Path(tmp_name)
    upload.save(tmp)

    def cleanup() -> None:
        tmp.unlink(missing_ok=True)

    try:
        meta = check_content(tmp, media_type, cfg.api.max_video_seconds)
    except Exception:
        cleanup()
        raise

    detector = _holder().get()
    filename = Path(upload.filename).name

    def run() -> dict[str, Any]:
        result = detector.predict_file(tmp, media_type)
        result["file"] = {"name": filename, "size_bytes": tmp.stat().st_size if tmp.exists() else None, **meta}
        return result

    if _truthy(request.args.get("async")) or _truthy(request.form.get("async")):
        job = _jobs().submit(run, on_done=cleanup)
        logger.info("job queued", extra={"job_id": job.id, "media_type": media_type})
        return jsonify({"job_id": job.id, "status": job.status, "status_url": f"/api/jobs/{job.id}"}), 202

    try:
        result = run()
    finally:
        cleanup()
    logger.info(
        "prediction",
        extra={
            "media_type": media_type,
            "label": result["label"],
            "p_fake": result["probability_fake"],
            "mode": result["mode"],
            "ms": result["processing_ms"],
        },
    )
    return jsonify(result)


@bp.get("/jobs/<job_id>")
def job_status(job_id: str):
    job = _jobs().get(job_id)
    if job is None:
        return error_response(404, f"Unknown job '{job_id}'")
    return jsonify(job.to_dict())


# --------------------------------------------------------------------------- factory
def create_app(cfg: Config | None = None, detector=None, *, eager: bool = False) -> Flask:
    """Application factory. Pass ``detector`` to inject a (mock) model in tests."""
    cfg = cfg or load_config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.api.max_upload_mb * 1024 * 1024
    app.json.sort_keys = False
    CORS(app, resources={r"/api/*": {"origins": cfg.api.cors_origins}}, expose_headers=["X-Request-ID"])

    app.extensions["dfd_config"] = cfg
    app.extensions["dfd_detector"] = DetectorHolder(cfg, detector)
    app.extensions["dfd_jobs"] = JobManager(cfg.api.async_workers, cfg.api.job_ttl_seconds)
    app.register_blueprint(bp)

    @app.before_request
    def _start() -> None:
        g.start = time.perf_counter()
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        g.rid_token = request_id_var.set(g.request_id)

    @app.after_request
    def _finish(response):
        response.headers["X-Request-ID"] = g.get("request_id", "")
        if request.path.startswith("/api/"):
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - g.get("start", time.perf_counter())) * 1000, 1),
                },
            )
        return response

    @app.teardown_request
    def _teardown(_exc) -> None:
        token = g.pop("rid_token", None)
        if token is not None:
            request_id_var.reset(token)

    @app.errorhandler(ValidationError)
    def _validation(exc: ValidationError):
        return error_response(exc.status, str(exc))

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        return error_response(413, f"File too large. Maximum upload size is {cfg.api.max_upload_mb} MB.")

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        return error_response(exc.code or 500, exc.description or exc.name)

    @app.errorhandler(ValueError)
    def _value(exc: ValueError):
        return error_response(422, str(exc))

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        logger.exception("unhandled error")
        return error_response(500, "Internal server error")

    if eager:
        app.extensions["dfd_detector"].get()
    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the development API server")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    configure_logging(json_logs=not args.debug)
    create_app(load_config(args.config), eager=True).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
