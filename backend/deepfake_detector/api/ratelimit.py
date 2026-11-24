"""In-process sliding-window rate limiter for the upload endpoint.

Good enough for the single-process deployment this repository ships (one gunicorn worker
holding the models). Behind several workers or hosts, use a shared store (e.g. Redis) or the
reverse proxy's limiter instead.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, max_clients: int = 10_000) -> None:
        self.limit = limit
        self.window = window_s
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Record a hit for ``key``; return ``(allowed, retry_after_seconds)``."""
        if self.limit <= 0:
            return True, 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            if len(self._hits) > self.max_clients:  # bound memory: drop idle clients
                cutoff = now - self.window
                for k in [k for k, q in self._hits.items() if not q or q[-1] < cutoff]:
                    del self._hits[k]
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(0.0, q[0] + self.window - now)
            q.append(now)
            return True, 0.0
