"""Per-model latency statistics and the device_health payload."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

import numpy as np


class LatencyStats:
    """Recent latencies of one model. Written by either lane, read by the main thread."""

    def __init__(self, keep: int = 1000):
        self.recent: deque[float] = deque(maxlen=keep)
        self.count = 0
        self.total_ms = 0.0
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self.recent.append(ms)
            self.count += 1
            self.total_ms += ms

    def percentile(self, q: float) -> float:
        with self._lock:
            values = list(self.recent)
        return float(np.percentile(values, q)) if values else 0.0


class Health:
    MODELS = ("V1", "V2", "V3", "S1", "S2")

    def __init__(self):
        self.stats = {name: LatencyStats() for name in self.MODELS}
        self._counts_at_last_snapshot = {name: 0 for name in self.MODELS}
        self._last_snapshot_t: float | None = None

    @contextmanager
    def timed(self, model: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stats[model].add((time.perf_counter() - start) * 1000.0)

    def latency_ms(self) -> dict[str, dict[str, float]]:
        return {name: {"p50": round(s.percentile(50), 1), "p95": round(s.percentile(95), 1), "runs": s.count}
                for name, s in self.stats.items() if s.count}

    def rates(self, t: float) -> dict[str, float]:
        """Runs per second of capture time since the previous call (the device_health 'fps' field)."""
        if self._last_snapshot_t is None:
            self._last_snapshot_t = t
        span = max(1e-9, t - self._last_snapshot_t)
        rates = {name: round((s.count - self._counts_at_last_snapshot[name]) / span, 2)
                 for name, s in self.stats.items() if name in ("V1", "V2", "V3")}
        self._counts_at_last_snapshot = {name: s.count for name, s in self.stats.items()}
        self._last_snapshot_t = t
        return rates
