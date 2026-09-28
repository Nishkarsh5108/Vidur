"""The deferred lane: V1 and V3 work that may lag behind the camera by seconds or minutes.

Potholes and signs end up as work orders for the municipality, so they do not need real-time
results. Only average throughput matters here, which is what lets V1 (YOLOv8m) run on a Pi 5 CPU:
it looks at a few buffered frames per IMU jolt instead of at the video stream.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Higher runs first and is dropped last. V1 triggers carry the IMU evidence, sign jobs are cheap,
# and sweep frames are the first to go when the lane is behind.
PRIORITY_V1_TRIGGER = 2.0      # + the shock's priority in [0, 1]
PRIORITY_V3_SIGN = 1.5
PRIORITY_V1_SWEEP = 0.5


@dataclass
class Job:
    kind: str                  # "v1_trigger" | "v1_sweep" | "v3_sign"
    priority: float
    t: float                   # capture time the job is about
    payload: dict[str, Any] = field(default_factory=dict)


class DeferredQueue:
    """A bounded priority queue.

    When full, a new job evicts the lowest-priority queued job if it outranks it, and is dropped
    otherwise. With block=True (offline replay), put() waits for space instead, so every job runs.
    set_paused(True) stops get() from handing out work, e.g. inside a school zone in school hours.
    """

    def __init__(self, max_jobs: int):
        self.max_jobs = max_jobs
        self._heap: list[tuple[float, int, Job]] = []
        self._order = itertools.count()
        self._cond = threading.Condition()
        self._paused = False
        self._unfinished = 0
        self.dropped: dict[str, int] = {}
        self.max_depth = 0

    def __len__(self) -> int:
        with self._cond:
            return len(self._heap)

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        with self._cond:
            self._paused = paused
            self._cond.notify_all()

    def put(self, job: Job, block: bool = False) -> bool:
        with self._cond:
            if block:
                while len(self._heap) >= self.max_jobs and not self._paused:
                    self._cond.wait(0.1)
            if len(self._heap) >= self.max_jobs:
                # Lowest priority, newest among equals. Heap items are (-priority, order, job).
                lowest = max(self._heap, key=lambda item: (item[0], item[1]))
                if -lowest[0] >= job.priority:
                    self._count_drop(job)
                    return False
                self._heap.remove(lowest)
                heapq.heapify(self._heap)
                self._unfinished -= 1
                self._count_drop(lowest[2])
            heapq.heappush(self._heap, (-job.priority, next(self._order), job))
            self._unfinished += 1
            self.max_depth = max(self.max_depth, len(self._heap))
            self._cond.notify_all()
            return True

    def get(self, timeout: float = 0.1) -> Job | None:
        with self._cond:
            if self._paused or not self._heap:
                self._cond.wait(timeout)
            if self._paused or not self._heap:
                return None
            job = heapq.heappop(self._heap)[2]
            self._cond.notify_all()
            return job

    def task_done(self) -> None:
        with self._cond:
            self._unfinished -= 1
            self._cond.notify_all()

    def join(self, timeout: float | None = None) -> bool:
        """Waits until every accepted job has been run (or evicted). Returns False on timeout."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cond:
            while self._unfinished > 0:
                remaining = None if deadline is None else deadline - time.perf_counter()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(0.1 if remaining is None else min(0.1, remaining))
        return True

    def _count_drop(self, job: Job) -> None:
        self.dropped[job.kind] = self.dropped.get(job.kind, 0) + 1
        log.debug("deferred queue full: dropped a %s job (t=%.2f)", job.kind, job.t)


class DeferredLane(threading.Thread):
    """Worker thread that runs jobs from a DeferredQueue and hands (job, result) back to the agent.

    On Linux it can be pinned to its own cores (cpu_affinity), so V1 never steals V2's cores;
    threads the inference runtime starts from here inherit the pinning.
    """

    def __init__(self, jobs: DeferredQueue, handlers: dict[str, Callable[[Job], Any]],
                 cpu_affinity: list[int] | None = None):
        super().__init__(name="deferred-lane", daemon=True)
        self.jobs = jobs
        self.handlers = handlers
        self.cpu_affinity = cpu_affinity or []
        self.results: queue.Queue[tuple[Job, Any]] = queue.Queue()
        self.busy_s = 0.0
        self._stop_event = threading.Event()
        self._started_at = time.perf_counter()

    def run(self) -> None:
        if self.cpu_affinity and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(self.cpu_affinity))
        while not self._stop_event.is_set():
            job = self.jobs.get(timeout=0.1)
            if job is None:
                continue
            start = time.perf_counter()
            try:
                result = self.handlers[job.kind](job)
            except Exception:   # a failing job must not take the lane down
                log.exception("deferred %s job failed", job.kind)
                result = None
            self.busy_s += time.perf_counter() - start
            self.results.put((job, result))
            self.jobs.task_done()

    @property
    def utilisation(self) -> float:
        """Fraction of wall time spent running jobs since the lane started."""
        return self.busy_s / max(1e-9, time.perf_counter() - self._started_at)

    def drain(self, timeout: float | None = None) -> bool:
        """Unpauses the queue and waits until every queued job has run."""
        self.jobs.set_paused(False)
        return self.jobs.join(timeout)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5.0)
