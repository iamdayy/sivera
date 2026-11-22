"""In-process sliding-window rate limiter for public endpoints.

Single-replica deployment (API_WORKERS le=1, advisory lock) makes an
in-process limiter sufficient; front a multi-replica future with a shared
store before scaling out.
"""
import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, limit_per_minute: int, window_sec: float = 60.0, clock=time.monotonic) -> None:
        if limit_per_minute < 1:
            raise ValueError("limit_per_minute must be >= 1")
        self.limit = limit_per_minute
        self.window = window_sec
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        self._prune(now)
        return True

    def _prune(self, now: float) -> None:
        # Bound memory: drop fully expired keys when the map grows large.
        if len(self._hits) <= 10000:
            return
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] > self.window]:
            self._hits.pop(key, None)
