"""Geometry and GPS state: distances, headings, the bus's recent trace, and school-zone polygons."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from pathlib import Path

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def point_in_ring(lat: float, lon: float, ring: list[tuple[float, float]]) -> bool:
    """Ray casting on a closed or open ring of (lat, lon) vertices."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        yi, xi = ring[i]
        yj, xj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class Fix:
    t: float
    lat: float
    lon: float
    speed: float          # m/s
    heading: float | None
    odometer: float       # metres driven since the start of the trip


class GpsTrack:
    """The bus's recent GPS trace: current state, an odometer, and position lookup at any recent time.

    Deferred results (V1, V3) arrive seconds or minutes after capture, so their events are located
    by looking the capture time up in this history rather than using the bus's current position.
    """

    def __init__(self, history_s: float = 900.0, store_every_s: float = 0.2, heading_min_move_m: float = 5.0):
        self.history_s = history_s
        self.store_every_s = store_every_s
        self.heading_min_move_m = heading_min_move_m
        self.fixes: deque[Fix] = deque()
        self.odometer = 0.0
        self.heading: float | None = None
        self._last: tuple[float, float, float, float] | None = None   # t, lat, lon, speed
        self._anchor: tuple[float, float] | None = None

    @property
    def current(self) -> Fix | None:
        return self.fixes[-1] if self.fixes else None

    def update(self, t: float, lat: float, lon: float, speed: float) -> None:
        if self._last is not None:
            dt = t - self._last[0]
            if dt <= 0:
                return
            # Integrate speed rather than GPS positions: fixes arrive at about 1 Hz, too coarse for 15 m rules.
            self.odometer += 0.5 * (speed + self._last[3]) * min(dt, 0.5)
        self._last = (t, lat, lon, speed)

        if self._anchor is None:
            self._anchor = (lat, lon)
        elif haversine_m(*self._anchor, lat, lon) >= self.heading_min_move_m:
            self.heading = bearing_deg(*self._anchor, lat, lon)
            self._anchor = (lat, lon)

        if not self.fixes or t - self.fixes[-1].t >= self.store_every_s:
            self.fixes.append(Fix(t, lat, lon, speed, self.heading, self.odometer))
            while self.fixes and t - self.fixes[0].t > self.history_s:
                self.fixes.popleft()

    def at(self, t: float) -> Fix | None:
        """Linearly interpolated fix at time t (clamped to the stored history)."""
        if not self.fixes:
            return None
        times = [f.t for f in self.fixes]
        i = bisect_left(times, t)
        if i <= 0:
            return self.fixes[0]
        if i >= len(self.fixes):
            return self.fixes[-1]
        a, b = self.fixes[i - 1], self.fixes[i]
        w = (t - a.t) / (b.t - a.t)
        return Fix(
            t=t,
            lat=a.lat + w * (b.lat - a.lat),
            lon=a.lon + w * (b.lon - a.lon),
            speed=a.speed + w * (b.speed - a.speed),
            heading=b.heading if b.heading is not None else a.heading,
            odometer=a.odometer + w * (b.odometer - a.odometer),
        )


@dataclass
class SchoolZone:
    osm_id: str
    name: str
    ring: list[tuple[float, float]]      # (lat, lon)
    bbox: tuple[float, float, float, float]   # min_lat, min_lon, max_lat, max_lon

    def contains(self, lat: float, lon: float) -> bool:
        min_lat, min_lon, max_lat, max_lon = self.bbox
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return False
        return point_in_ring(lat, lon, self.ring)


def _make_zone(osm_id: str, name: str, ring: list[tuple[float, float]]) -> SchoolZone:
    lats, lons = [p[0] for p in ring], [p[1] for p in ring]
    return SchoolZone(osm_id, name, ring, (min(lats), min(lons), max(lats), max(lons)))


def square_around(lat: float, lon: float, half_side_m: float) -> list[tuple[float, float]]:
    d_lat = half_side_m / 111_320.0
    d_lon = half_side_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return [(lat - d_lat, lon - d_lon), (lat - d_lat, lon + d_lon), (lat + d_lat, lon + d_lon), (lat + d_lat, lon - d_lon)]


def load_school_zones(path: str | Path, point_buffer_m: float = 120.0) -> list[SchoolZone]:
    """Reads school zones from GeoJSON: Polygons and MultiPolygons as drawn, Points as squares.

    Points become squares of +/- point_buffer_m, the same rule as the V2 SchoolGeofenceManager.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return zones_from_features(data["features"] if data.get("type") == "FeatureCollection" else [data], point_buffer_m)


def zones_from_features(features: list[dict], point_buffer_m: float = 120.0) -> list[SchoolZone]:
    zones = []
    for n, feature in enumerate(features):
        props = feature.get("properties") or {}
        osm_id = str(props.get("osm_id") or props.get("@id") or props.get("id") or feature.get("id") or n)
        name = props.get("name") or f"School {osm_id}"
        geom = feature.get("geometry") or {}
        kind, coords = geom.get("type"), geom.get("coordinates")
        if kind == "Point":
            zones.append(_make_zone(osm_id, name, square_around(coords[1], coords[0], point_buffer_m)))
        elif kind == "Polygon":
            zones.append(_make_zone(osm_id, name, [(lat, lon) for lon, lat in coords[0]]))
        elif kind == "MultiPolygon":
            for polygon in coords:
                zones.append(_make_zone(osm_id, name, [(lat, lon) for lon, lat in polygon[0]]))
    return zones
