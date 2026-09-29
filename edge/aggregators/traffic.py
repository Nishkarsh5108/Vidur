"""traffic_sample events: V2 counts summarised every 10 s or 100 m of travel, whichever comes first."""

from __future__ import annotations

from collections import Counter
from typing import Any

from edge.runners.v2_idd import COUNTED_CLASSES
from edge.runners.yolo import Box


class TrafficAggregator:
    """Averages per-frame counts over a stretch of road.

    Per-frame counts averaged over a stretch stay meaningful at 2-4 FPS from a moving bus, where
    tracker IDs switch often. Unique track IDs are reported alongside as an upper bound.
    Bottlenecks are decided in the backend from these samples plus the bus's GPS speed, not from
    pixel motion (which on a moving bus is only relative to the bus).
    """

    def __init__(self, window_s: float, window_m: float):
        self.window_s, self.window_m = window_s, window_m
        self._reset()

    def _reset(self) -> None:
        self.t0: float | None = None
        self.odo0 = 0.0
        self.frames = 0
        self.sums: Counter[str] = Counter()
        self.maxes: Counter[str] = Counter()
        self.tracks: set[int] = set()
        self.speeds: list[float] = []
        self.confs: list[float] = []

    def add(self, t: float, boxes: list[Box], odometer: float, speed: float | None) -> dict[str, Any] | None:
        """Adds one frame's detections (speed None: no GPS). Returns a finished sample when the window closes."""
        if self.t0 is None:
            self.t0, self.odo0 = t, odometer
        counts = Counter(b.name for b in boxes if b.name in COUNTED_CLASSES)
        self.frames += 1
        self.sums.update(counts)
        for name, n in counts.items():
            self.maxes[name] = max(self.maxes[name], n)
        self.tracks.update(b.track_id for b in boxes if b.name in COUNTED_CLASSES and b.track_id is not None)
        self.confs.extend(b.conf for b in boxes if b.name in COUNTED_CLASSES)
        if speed is not None:
            self.speeds.append(speed)
        if t - self.t0 >= self.window_s or odometer - self.odo0 >= self.window_m:
            return self.close(t, odometer)
        return None

    def close(self, t: float, odometer: float) -> dict[str, Any] | None:
        if self.t0 is None or self.frames == 0:
            return None
        sample = {
            "t_mid": (self.t0 + t) / 2,
            "confidence": round(sum(self.confs) / len(self.confs), 3) if self.confs else None,
            "metadata": {
                "windowS": round(t - self.t0, 2),
                "distanceM": round(odometer - self.odo0, 1),
                "framesAnalysed": self.frames,
                "meanCounts": {c: round(self.sums[c] / self.frames, 2) for c in COUNTED_CLASSES},
                "maxCounts": {c: int(self.maxes[c]) for c in COUNTED_CLASSES},
                "uniqueTracks": len(self.tracks),
                "busSpeedMps": round(sum(self.speeds) / len(self.speeds), 2) if self.speeds else None,
            },
        }
        self._reset()
        return sample
