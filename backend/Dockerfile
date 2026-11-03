# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd --create-home --uid 1000 app

# Set INSTALL_TF=0 for a slim image that only serves the placeholder demo mode.
ARG INSTALL_TF=1
COPY requirements-base.txt requirements.txt ./
RUN if [ "$INSTALL_TF" = "1" ]; then pip install -r requirements.txt; else pip install -r requirements-base.txt; fi

COPY deepfake_detector ./deepfake_detector
COPY configs ./configs
COPY wsgi.py gunicorn.conf.py ./
RUN mkdir -p weights uploads && chown -R app:app /app
USER app

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:5000/api/health || exit 1
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
