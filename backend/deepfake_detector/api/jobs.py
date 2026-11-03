"""Minimal in-process async job queue (ThreadPoolExecutor + TTL eviction).

Suitable for a single API process. For horizontal scaling, swap this for Celery/RQ
behind the same interface.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    created: float = field(default_factory=time.time)
    finished: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "created": self.created,
            "finished": self.finished,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self, workers: int = 2, ttl_seconds: int = 3600) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="dfd-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds

    def _evict(self) -> None:
        cutoff = time.time() - self.ttl
        with self._lock:
            for jid in [j.id for j in self._jobs.values() if j.finished and j.finished < cutoff]:
                del self._jobs[jid]

    def submit(self, fn: Callable[[], dict[str, Any]], on_done: Callable[[], None] | None = None) -> Job:
        self._evict()
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            job.status = "running"
            try:
                job.result = fn()
                job.status = "done"
            except Exception as exc:
                logger.exception("job failed", extra={"job_id": job.id})
                job.error = str(exc)
                job.status = "error"
            finally:
                job.finished = time.time()
                if on_done:
                    on_done()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
