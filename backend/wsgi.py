"""WSGI entry point: ``gunicorn -c gunicorn.conf.py wsgi:app``."""

from deepfake_detector.api import create_app
from deepfake_detector.utils.logging import configure_logging

configure_logging()
app = create_app(eager=True)
