"""Load policy: how often V2 runs, and when the deferred lane must yield the CPU."""

from __future__ import annotations

import time
from pathlib import Path

from edge.deferred import DeferredQueue

_THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")


def read_cpu_temp_c() -> float | None:
    """SoC temperature on a Raspberry Pi / Linux; None elsewhere."""
    try:
        return int(_THERMAL_ZONE.read_text()) / 1000.0
    except (OSError, ValueError):
        return None


class Governor:
    """Rules (docs/edge-deployment.md, "Combining the models"):

    - V2 runs at v2_fps, and at v2_fps_slow while the bus is stopped (the scene barely changes, and
      the freed CPU drains the V1 backlog) or while the CPU is hot.
    - The deferred lane pauses inside a school zone during school hours (V2 gets the whole CPU)
      and above the thermal limit.
    Pausing only applies when running in real time; an offline replay has no CPU to protect.
    """

    def __init__(self, realtime_cfg, deferred_cfg, throttle_c: float, realtime: bool, jobs: DeferredQueue | None):
        self.fast_period = 1.0 / realtime_cfg.v2_fps
        self.slow_period = 1.0 / realtime_cfg.v2_fps_slow
        self.stopped_speed = realtime_cfg.stopped_speed_mps
        self.pause_in_school_zone = deferred_cfg.pause_in_school_zone
        self.throttle_c = throttle_c
        self.realtime = realtime
        self.jobs = jobs
        self.cpu_temp_c: float | None = None
        self._temp_read_at = -1e9

    def temperature(self) -> float | None:
        """SoC temperature, re-read at most every 5 s."""
        now = time.monotonic()
        if now - self._temp_read_at >= 5.0:
            self.cpu_temp_c = read_cpu_temp_c()
            self._temp_read_at = now
        return self.cpu_temp_c

    @property
    def hot(self) -> bool:
        temp = self.temperature()
        return temp is not None and temp >= self.throttle_c

    def v2_period(self, speed: float | None) -> float:
        """speed None means unknown (no GPS): treated as moving."""
        stopped = speed is not None and speed < self.stopped_speed
        return self.slow_period if stopped or self.hot else self.fast_period

    def update(self, school_zone_active: bool) -> None:
        if self.jobs is None or not self.realtime:
            return
        self.jobs.set_paused((self.pause_in_school_zone and school_zone_active) or self.hot)
