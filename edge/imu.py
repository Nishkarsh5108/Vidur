"""Streaming IMU logic: 100 m IRI windows (S1) and shock detection (S2, or a jerk fallback) for V1 triggers."""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import numpy as np

from edge.capture import ImuSample
from edge.runners import s2_shock
from edge.runners.s1_iri import IriModel, iri_class, window_tensors
from edge.runners.s2_shock import ShockModel

log = logging.getLogger(__name__)

Timer = Callable[[str], AbstractContextManager]
JERK_LABELS = ("None", "Minor jolt", "Moderate jolt", "Severe jolt")


def _no_timer(_name: str) -> AbstractContextManager:
    return nullcontext()


class Resampler:
    """Linear interpolation of an irregular stream onto a uniform grid (S2 was trained at exactly 100 Hz)."""

    def __init__(self, hz: float = 100.0, max_gap_s: float = 0.5):
        self.dt = 1.0 / hz
        self.max_gap_s = max_gap_s
        self._prev: tuple[float, float] | None = None
        self._next_t: float | None = None

    def add(self, t: float, value: float) -> list[tuple[float, float]]:
        out = []
        if self._prev is None or t - self._prev[0] > self.max_gap_s:
            self._next_t = t     # (re)start the grid after a start-up or a logging gap
        elif t > self._prev[0]:
            t0, v0 = self._prev
            while self._next_t <= t:
                out.append((self._next_t, v0 + (value - v0) * (self._next_t - t0) / (t - t0)))
                self._next_t += self.dt
        if self._prev is None or t > self._prev[0]:
            self._prev = (t, value)
        return out


@dataclass
class Shock:
    t: float                        # time of the peak inside the window
    cls: int                        # 0-3 on S2's scale
    method: str                     # "s2" | "jerk"
    p2p: float                      # peak-to-peak vertical acceleration in the window, m/s2
    probs: list[float] | None = None
    smoothed: bool = False

    @property
    def label(self) -> str:
        # The jerk fallback shares S2's 0-3 scale for the backend, but not its pothole wording.
        return s2_shock.LABELS[self.cls] if self.method == "s2" else JERK_LABELS[self.cls]

    @property
    def priority(self) -> float:
        """How urgently V1 should look at this jolt: S2's pothole probability, or a scaled jerk."""
        if self.probs is not None:
            return float(self.probs[3])
        return min(1.0, self.p2p / 20.0)


class _Gap:
    """At most one hit per min_gap_s and per min_gap_m of travel."""

    def __init__(self, min_gap_s: float, min_gap_m: float):
        self.min_gap_s, self.min_gap_m = min_gap_s, min_gap_m
        self._last: tuple[float, float] | None = None

    def allow(self, t: float, odometer: float) -> bool:
        if self._last is not None and (t - self._last[0] < self.min_gap_s or odometer - self._last[1] < self.min_gap_m):
            return False
        self._last = (t, odometer)
        return True


class ShockStream:
    """Slides a 128-sample (1.28 s) window over vertical acceleration and reports shocks.

    Returns two lists per step: road_shock events (S2 smoothed class >= 2, or a jerk spike) and
    V1 triggers (S2 class 3 above the confidence gate, or a jerk spike). Both are rate-limited by
    time and distance, and ignored while the bus is stopped (door slams, boarding).
    """

    def __init__(self, cfg, model: ShockModel | None, min_speed_mps: float, timed: Timer = _no_timer):
        self.model = model
        self.timed = timed
        self.method = "s2" if model is not None else "jerk"
        self.hop = cfg.s2.hop_samples
        self.min_confidence = cfg.s2.min_confidence
        self.jerk = cfg.jerk
        self.min_speed_mps = min_speed_mps
        self.votes: deque[int] = deque(maxlen=cfg.s2.vote_window)
        self.resampler = Resampler(100.0)
        self.window: deque[tuple[float, float, float]] = deque(maxlen=s2_shock.WINDOW)   # t, az, speed
        self._since_hop = 0
        self._event_gap = _Gap(cfg.min_gap_s, cfg.min_gap_m)
        self._trigger_gap = _Gap(cfg.min_gap_s, cfg.min_gap_m)

    def add(self, sample: ImuSample, odometer: float) -> tuple[list[Shock], list[Shock]]:
        events, triggers = [], []
        for t, az in self.resampler.add(sample.t, sample.az):
            self.window.append((t, az, sample.speed))
            self._since_hop += 1
            if len(self.window) < s2_shock.WINDOW or self._since_hop < self.hop:
                continue
            self._since_hop = 0
            times, az_w, speed_w = (np.array(c) for c in zip(*self.window))
            if np.mean(speed_w) < self.min_speed_mps:
                self.votes.clear()
                continue
            event, trigger = self._grade(times, az_w, float(np.mean(speed_w)))
            if event is not None and self._event_gap.allow(event.t, odometer):
                events.append(event)
            if trigger is not None and self._trigger_gap.allow(trigger.t, odometer):
                triggers.append(trigger)
        return events, triggers

    def _grade(self, times: np.ndarray, az: np.ndarray, mean_speed: float) -> tuple[Shock | None, Shock | None]:
        """(road_shock event or None, V1 trigger or None) for one window."""
        centred = az - az.mean()
        p2p = float(centred.max() - centred.min())
        t_peak = float(times[int(np.argmax(np.abs(np.diff(az, prepend=az[0]))))])

        if self.model is not None:
            with self.timed("S2"):
                probs = self.model.predict(az, mean_speed)
            cls = s2_shock.gate(probs, self.min_confidence)
            self.votes.append(cls)
            smoothed = Counter(self.votes).most_common(1)[0][0]
            event = Shock(t_peak, smoothed, "s2", p2p, probs.tolist(), smoothed=True) if smoothed >= 2 else None
            trigger = Shock(t_peak, cls, "s2", p2p, probs.tolist()) if cls == 3 else None
            return event, trigger

        rms = float(np.sqrt(np.mean(centred ** 2)))
        crest = float(np.max(np.abs(centred)) / (rms + 1e-6))
        if p2p < self.jerk.p2p_trigger or crest < self.jerk.crest_min:
            return None, None
        shock = Shock(t_peak, 3 if p2p >= self.jerk.p2p_big else 2, "jerk", p2p)
        return shock, shock


@dataclass
class IriWindow:
    t_start: float
    t_end: float
    start: tuple[float, float]      # lat, lon
    end: tuple[float, float]
    length_m: float
    iri: float
    iri_raw: float
    iri_class: str
    mean_speed_mps: float
    n_samples: int


class IriStream:
    """Cuts the IMU stream into consecutive 100 m windows of travel and scores each with S1.

    The upload stride is 100 m (not training's 10 m), so no stretch of road is counted ten times.
    """

    MIN_SAMPLES = 20

    def __init__(self, model: IriModel, window_m: float, min_speed_mps: float, az_gravity_offset: float = 0.0,
                 timed: Timer = _no_timer):
        self.model = model
        self.timed = timed
        self.window_m = window_m
        self.min_speed_mps = min_speed_mps
        self.az_offset = az_gravity_offset
        self.dist = 0.0
        self.start_m = 0.0
        self._last_t: float | None = None
        self._rows: list[tuple[float, float, float, float, float, float, float, float, float, float, float]] = []

    def add(self, s: ImuSample) -> IriWindow | None:
        if s.speed < self.min_speed_mps:     # as in training: near-stationary samples are dropped
            self._last_t = None
            return None
        if self._last_t is not None:
            self.dist += s.speed * min(s.t - self._last_t, 0.5)
        self._last_t = s.t
        self._rows.append((self.dist, s.t, s.ax, s.ay, s.az + self.az_offset, s.wx, s.wy, s.wz, s.speed, s.lat, s.lon))
        if self.dist - self.start_m < self.window_m:
            return None
        return self._close_window()

    def _close_window(self) -> IriWindow | None:
        end_m = self.start_m + self.window_m
        rows = [r for r in self._rows if self.start_m <= r[0] < end_m]
        self._rows = [r for r in self._rows if r[0] >= end_m]
        start_m, self.start_m = self.start_m, end_m
        if len(rows) < self.MIN_SAMPLES:
            return None
        data = np.array(rows, dtype=np.float64)
        with self.timed("S1"):
            raw, ctx = window_tensors(data[:, 0], data[:, 2:8], data[:, 8], data[:, 1], start_m, self.window_m)
            iri, iri_raw = self.model.predict(raw, ctx)
        return IriWindow(
            t_start=float(data[0, 1]), t_end=float(data[-1, 1]),
            start=(float(data[0, 9]), float(data[0, 10])), end=(float(data[-1, 9]), float(data[-1, 10])),
            length_m=self.window_m, iri=iri, iri_raw=iri_raw, iri_class=iri_class(iri),
            mean_speed_mps=float(np.mean(data[:, 8])), n_samples=len(rows),
        )
