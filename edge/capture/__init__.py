"""Sensor capture. Everything is stamped with one clock (epoch seconds, GPS/UTC time)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImuSample:
    t: float
    ax: float   # lateral, m/s2
    ay: float   # longitudinal (forward)
    az: float   # vertical
    wx: float   # pitch rate, rad/s
    wy: float   # roll rate
    wz: float   # yaw rate
    lat: float     # NaN when the log has no position (GPS then comes from its own stream)
    lon: float
    speed: float   # m/s

    @property
    def has_position(self) -> bool:
        return self.lat == self.lat and self.lon == self.lon     # False for NaN


@dataclass
class GpsFix:
    """A GPS position, e.g. from the bus's AIS-140 unit or a per-frame GPS trail."""
    t: float
    lat: float
    lon: float
    speed: float           # m/s
    heading: float | None = None


@dataclass
class Frame:
    t: float
    image: np.ndarray        # BGR uint8, possibly downscaled (see scale)
    index: int
    scale: float = 1.0       # image size / original capture size
    orig_size: tuple[int, int] = (0, 0)   # original (width, height)
