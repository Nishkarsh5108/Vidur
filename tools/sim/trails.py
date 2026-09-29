"""Per-frame GPS trails for videos recorded without GPS, built from real GPS traces of real roads.

A trace is one of the team's own drives (phone logger CSV: time or timestamp, latitude, longitude,
speed). A trail places a video on a stretch of that trace, one row per video frame:

- time mode: the video plays back the trace's own motion (its real speeds, stops and turns);
- distance mode: the video moves along the trace at the speed of its own IMU log (BeamNG clips),
  so the GPS distance matches the distance the IMU model integrates.

The pairing of video and road is simulated; the road geometry and speeds are real.
"""

from __future__ import annotations

import csv
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from edge.geo import SchoolZone, bearing_deg, zones_from_features
from edge.util import iso_utc

EARTH_R = 6_371_000.0
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter")


@dataclass
class Trace:
    name: str
    t: np.ndarray        # epoch seconds of each distinct GPS fix
    lat: np.ndarray
    lon: np.ndarray
    speed: np.ndarray    # m/s
    dist: np.ndarray     # metres along the trace

    def at_time(self, t: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        return np.interp(t, self.t, self.lat), np.interp(t, self.t, self.lon)

    def at_dist(self, d: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        return np.interp(d, self.dist, self.lat), np.interp(d, self.dist, self.lon)

    def bbox(self, margin_m: float = 0.0, mask: np.ndarray | None = None) -> tuple[float, float, float, float]:
        lat, lon = (self.lat, self.lon) if mask is None else (self.lat[mask], self.lon[mask])
        dlat = margin_m / 111_320.0
        dlon = margin_m / (111_320.0 * math.cos(math.radians(float(lat.mean()))))
        return float(lat.min() - dlat), float(lon.min() - dlon), float(lat.max() + dlat), float(lon.max() + dlon)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def load_trace(path: str | Path) -> Trace:
    """The distinct GPS fixes of a phone-logger CSV (the logger repeats each fix at the IMU rate)."""
    df = pd.read_csv(path)
    if "time" in df:
        # Seconds since the epoch, independent of the datetime unit pandas picks (ns in pandas 2, us in pandas 3).
        t = (pd.to_datetime(df["time"], format="mixed", utc=True) - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
    else:
        t = df["timestamp"].astype(float)
        t = t / 1000.0 if t.iloc[0] > 1e11 else t
    df = df.assign(_t=t)
    df = df[df.latitude.notna() & df.longitude.notna() & df._t.notna()
            & (df.latitude.abs() > 1) & (df.longitude.abs() > 1)].sort_values("_t")
    moved = (df.latitude.diff() != 0) | (df.longitude.diff() != 0)
    fixes = df[moved].drop_duplicates("_t")
    lat, lon = fixes.latitude.to_numpy(), fixes.longitude.to_numpy()
    step = np.concatenate([[0.0], haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])])
    return Trace(Path(path).name, fixes._t.to_numpy(), lat, lon, fixes.speed.to_numpy(dtype=float), np.cumsum(step))


# ---------------------------------------------------------------- choosing a stretch of road

def pick_start(trace: Trace, duration_s: float, *, min_speed: float = 2.0, target_kmh: float | None = None,
               distance_m: float | None = None, zones: list[SchoolZone] | None = None,
               max_gap_s: float = 6.0) -> float:
    """Start time on the trace for a video of `duration_s`.

    The stretch must have GPS fixes no more than max_gap_s apart and the vehicle moving (min_speed m/s).
    With `distance_m` (distance mode) it must be at least that long. With `zones`, the stretch must enter
    a school zone in its middle 60%; otherwise the mean speed closest to target_kmh wins.
    """
    best, best_score = None, math.inf
    candidates = trace.t[trace.t <= trace.t[-1] - duration_s]
    for t0 in candidates[:: max(1, len(candidates) // 3000)]:
        if distance_m is not None:
            d0 = np.interp(t0, trace.t, trace.dist)
            end = np.searchsorted(trace.dist, d0 + distance_m)
            if end >= len(trace.t):
                break
            t1 = trace.t[end]
        else:
            t1 = t0 + duration_s
        inside = (trace.t >= t0) & (trace.t <= t1)
        if inside.sum() < 3 or np.diff(trace.t[inside]).max() > max_gap_s:
            continue
        speeds = trace.speed[inside]
        if speeds.min() < min_speed:
            continue
        if zones:
            mid = np.linspace(t0 + 0.2 * (t1 - t0), t0 + 0.8 * (t1 - t0), 40)
            lat, lon = trace.at_time(mid)
            hits = [i for i, (a, b) in enumerate(zip(lat, lon)) if any(z.contains(a, b) for z in zones)]
            if not hits:
                continue
            score = abs(np.mean(hits) / len(mid) - 0.5) - len(hits) / len(mid)   # centred, and long inside
        else:
            score = abs(speeds.mean() * 3.6 - target_kmh) if target_kmh else 0.0
        if score < best_score:
            best, best_score = float(t0), score
            if score == 0.0:
                break
    if best is None:
        raise ValueError(f"no stretch of {trace.name} fits a {duration_s:.0f} s video with these constraints")
    return best


# ---------------------------------------------------------------- per-frame trails

def trail_time_mode(trace: Trace, t0: float, n_frames: int, fps: float) -> list[dict]:
    """The video replays the trace's own motion from t0: its logged speed profile along its road geometry.

    Moving along the path at the logged speed, rather than interpolating fix positions over fix times,
    keeps the motion smooth when a logger stamps its fixes irregularly.
    """
    video_t = np.arange(n_frames) / fps
    speed = np.interp(t0 + video_t, trace.t, trace.speed)
    return trail_distance_mode(trace, t0, video_t, speed)


def trail_distance_mode(trace: Trace, t0: float, video_t: np.ndarray, speed: np.ndarray) -> list[dict]:
    """The video moves along the trace from t0 at its own speed profile (e.g. from its IMU log)."""
    dt = np.diff(video_t, prepend=video_t[0])
    d = np.interp(t0, trace.t, trace.dist) + np.cumsum(speed * dt)
    lat, lon = trace.at_dist(d)
    return _rows(trace, t0, video_t, lat, lon, speed, d)


def _rows(trace: Trace, t0: float, video_t, lat, lon, speed, d) -> list[dict]:
    ahead_lat, ahead_lon = trace.at_dist(d + 3.0)
    behind_lat, behind_lon = trace.at_dist(d - 3.0)
    rows = []
    for k in range(len(video_t)):
        rows.append({
            "frame": k,
            "video_time_s": round(float(video_t[k]), 4),
            "time": iso_utc(t0 + float(video_t[k])),
            "latitude": round(float(lat[k]), 7),
            "longitude": round(float(lon[k]), 7),
            "speed_mps": round(float(speed[k]), 3),
            "heading_deg": round(bearing_deg(behind_lat[k], behind_lon[k], ahead_lat[k], ahead_lon[k]), 1),
        })
    return rows


def write_trail(rows: list[dict], csv_path: Path, geojson_path: Path, properties: dict) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    line = [[r["longitude"], r["latitude"]] for r in rows[:: max(1, len(rows) // 400)]] + \
           [[rows[-1]["longitude"], rows[-1]["latitude"]]]
    geojson_path.write_text(json.dumps({"type": "Feature", "properties": properties,
                                        "geometry": {"type": "LineString", "coordinates": line}}), encoding="utf-8")


# ---------------------------------------------------------------- real school zones (OpenStreetMap)

def fetch_schools(bbox: tuple[float, float, float, float], cache: Path | None = None) -> list[dict]:
    """amenity=school from OpenStreetMap inside (south, west, north, east): polygons for mapped
    school grounds, points otherwise. Cached, because the public Overpass servers are rate-limited."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    s, w, n, e = bbox
    query = f"[out:json][timeout:90];(way[amenity=school]({s},{w},{n},{e});node[amenity=school]({s},{w},{n},{e}););out geom;"
    last_error = None
    for url in OVERPASS:
        try:
            request = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": "vidur-sih-demo/0.1 (hackathon fleet simulation)"})
            with urllib.request.urlopen(request, timeout=120) as response:
                elements = json.load(response)["elements"]
            break
        except Exception as exc:          # try the next public mirror
            last_error = exc
    else:
        raise RuntimeError(f"Overpass unavailable: {last_error}")
    features = []
    for el in elements:
        props = {"name": el.get("tags", {}).get("name") or "School (unnamed)", "osm_id": f"{el['type']}/{el['id']}"}
        if el["type"] == "node":
            geometry = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif len(el.get("geometry", [])) >= 4:
            ring = [[p["lon"], p["lat"]] for p in el["geometry"]]
            geometry = {"type": "Polygon", "coordinates": [ring if ring[0] == ring[-1] else ring + [ring[0]]]}
        else:
            continue
        features.append({"type": "Feature", "properties": props, "geometry": geometry})
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(features), encoding="utf-8")
    return features


def zone_features(schools: list[dict], point_buffer_m: float = 120.0, grounds_buffer_m: float = 75.0) -> list[dict]:
    """Geofence rectangles: +/- point_buffer_m around a school point (the V2 geofence rule), and the
    school grounds' extent plus grounds_buffer_m, because grounds usually sit back from the road."""
    out = []
    for f in schools:
        coords = f["geometry"]["coordinates"]
        if f["geometry"]["type"] == "Point":
            lat0, lon0, lat1, lon1 = coords[1], coords[0], coords[1], coords[0]
            margin = point_buffer_m
        else:
            lons, lats = [p[0] for p in coords[0]], [p[1] for p in coords[0]]
            lat0, lon0, lat1, lon1 = min(lats), min(lons), max(lats), max(lons)
            margin = grounds_buffer_m
        dlat = margin / 111_320.0
        dlon = margin / (111_320.0 * math.cos(math.radians((lat0 + lat1) / 2)))
        ring = [[lon0 - dlon, lat0 - dlat], [lon1 + dlon, lat0 - dlat], [lon1 + dlon, lat1 + dlat],
                [lon0 - dlon, lat1 + dlat], [lon0 - dlon, lat0 - dlat]]
        out.append({"type": "Feature", "properties": {**f["properties"], "zoneMarginM": margin},
                    "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return out


def school_zones(zone_feats: list[dict]) -> list[SchoolZone]:
    return zones_from_features(zone_feats)
