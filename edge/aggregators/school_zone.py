"""pedestrian_zone_alert events: people near the road inside a school zone. Alerts never carry an image."""

from __future__ import annotations

from typing import Any

from edge.geo import Fix, SchoolZone
from edge.runners.yolo import Box
from edge.util import in_school_hours

_LEVELS = {"ADVISORY": 1, "CRITICAL": 2}


class SchoolZoneMonitor:
    """Checks the bus's GPS position against school polygons on every analysed frame.

    CRITICAL: school hours and at least one pedestrian in the crossing area; otherwise ADVISORY.
    Riders are reported but do not escalate the level (IDD boxes the rider of every motorbike).
    At most one alert per school per cooldown, except an escalation from ADVISORY to CRITICAL.
    The backend re-checks the zone and the school hours.
    """

    def __init__(self, zones: list[SchoolZone], hours: list[list[str]], days: list[int],
                 cooldown_s: float, crossing_roi: list[float]):
        self.zones = zones
        self.hours, self.days = hours, days
        self.cooldown_s = cooldown_s
        self.roi = crossing_roi
        self._last: dict[str, tuple[float, str]] = {}

    def zone_at(self, lat: float, lon: float) -> SchoolZone | None:
        return next((z for z in self.zones if z.contains(lat, lon)), None)

    def active(self, t: float, fix: Fix | None) -> bool:
        """Inside a school zone during school hours (the deferred lane pauses then)."""
        return fix is not None and in_school_hours(t, self.hours, self.days) and self.zone_at(fix.lat, fix.lon) is not None

    def update(self, t: float, fix: Fix | None, boxes: list[Box], frame_size: tuple[int, int]) -> dict[str, Any] | None:
        if fix is None:
            return None
        zone = self.zone_at(fix.lat, fix.lon)
        if zone is None:
            return None
        width, height = frame_size
        x1, y1, x2, y2 = self.roi[0] * width, self.roi[1] * height, self.roi[2] * width, self.roi[3] * height

        def in_roi(b: Box) -> bool:
            cx, bottom = (b.xyxy[0] + b.xyxy[2]) / 2, b.xyxy[3]
            return x1 <= cx <= x2 and y1 <= bottom <= y2

        pedestrians = sum(1 for b in boxes if b.name == "person" and in_roi(b))
        riders = sum(1 for b in boxes if b.name == "rider" and in_roi(b))
        if pedestrians + riders == 0:
            return None
        school_hours = in_school_hours(t, self.hours, self.days)
        level = "CRITICAL" if school_hours and pedestrians >= 1 else "ADVISORY"
        last = self._last.get(zone.osm_id)
        if last is not None and t - last[0] < self.cooldown_s and _LEVELS[level] <= _LEVELS[last[1]]:
            return None
        self._last[zone.osm_id] = (t, level)
        return {
            "level": level,
            "schoolOsmId": zone.osm_id,
            "schoolName": zone.name,
            "pedestrianCount": pedestrians,
            "riderCount": riders,
            "schoolHoursActive": school_hours,
        }
