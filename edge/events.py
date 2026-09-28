"""Builds events in the edge -> backend data contract (docs/dashboard-backend-design.md §3)."""

from __future__ import annotations

from typing import Any

from edge.geo import GpsTrack
from edge.util import iso_utc, uuid7


class EventFactory:
    def __init__(self, gps: GpsTrack, sensor: str = "cam_front"):
        self.gps = gps
        self.sensor = sensor

    def make(self, event_type: str, t: float, confidence: float | None, model: str, model_version: str,
             metadata: dict[str, Any], media_id: str | None = None, sensor: str | None = None,
             location_t: float | None = None) -> dict[str, Any]:
        """One event. t is the capture time; the location is looked up at location_t (default t)."""
        fix = self.gps.at(t if location_t is None else location_t)
        event: dict[str, Any] = {
            "eventId": uuid7(),
            "eventType": event_type,
            "capturedAt": iso_utc(t),
            "location": None if fix is None else {"latitude": round(fix.lat, 7), "longitude": round(fix.lon, 7)},
            "headingDeg": None if fix is None or fix.heading is None else round(fix.heading, 1),
            "speedMps": None if fix is None else round(fix.speed, 2),
            "confidence": None if confidence is None else round(float(confidence), 3),
            "source": {"model": model, "modelVersion": model_version, "sensor": sensor or self.sensor},
            "metadata": metadata,
        }
        if media_id is not None:
            event["mediaId"] = media_id
        return event
