"""KPIs, road-health, traffic and GeoJSON helpers. Every number comes from the database:
an empty database gives 0, [] or null — never made-up values.

"Recent" windows (last 24 h, active vehicles, active alerts) use the server's `receivedAt`,
so a replayed recording with old `capturedAt` timestamps still shows up as live data.
"""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pymongo.collection import Collection
from pymongo.database import Database

from app.core.config import ALERT_EVENT_TYPES, ISSUE_RULES, OPEN_STATUSES, settings
from app.db.mongodb import serialize

TRAFFIC_CLASS_FIELDS = ["carCount", "busCount", "truckCount", "twoWheelerCount", "autorickshawCount", "bicycleCount"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _round(value, digits: int = 2):
    return round(value, digits) if isinstance(value, (int, float)) else None


def _group_count(collection: Collection, field: str, match: dict | None = None) -> list[dict]:
    pipeline = [
        {"$match": match or {}},
        {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    return [{"name": row["_id"], "count": row["count"]} for row in collection.aggregate(pipeline)]


def road_quality_summary(db: Database) -> dict:
    match = {"eventType": "road_quality", "metadata.iri": {"$type": "number"}}
    rows = list(db.events.aggregate([
        {"$match": match},
        {"$group": {"_id": None, "samples": {"$sum": 1}, "avgIri": {"$avg": "$metadata.iri"},
                    "minIri": {"$min": "$metadata.iri"}, "maxIri": {"$max": "$metadata.iri"}}},
    ]))
    stats = rows[0] if rows else {}
    return {
        "samples": stats.get("samples", 0),
        "avgIri": _round(stats.get("avgIri")),
        "minIri": _round(stats.get("minIri")),
        "maxIri": _round(stats.get("maxIri")),
        "byClass": _group_count(db.events, "metadata.iriClass", {"eventType": "road_quality"}),
    }


def traffic_summary(db: Database, hours: int = 24) -> dict:
    since = _now() - timedelta(hours=hours)
    group = {"_id": None, "samples": {"$sum": 1},
             "avgVehicleCount": {"$avg": "$metadata.vehicleCount"},
             "maxVehicleCount": {"$max": "$metadata.vehicleCount"}}
    for field in TRAFFIC_CLASS_FIELDS:
        group[f"avg_{field}"] = {"$avg": f"$metadata.{field}"}
    rows = list(db.events.aggregate([
        {"$match": {"eventType": "traffic_density", "receivedAt": {"$gte": since}}},
        {"$group": group},
    ]))
    stats = rows[0] if rows else {}
    return {
        "windowHours": hours,
        "samples": stats.get("samples", 0),
        "avgVehicleCount": _round(stats.get("avgVehicleCount")),
        "maxVehicleCount": stats.get("maxVehicleCount"),
        "avgByClass": {field: _round(stats.get(f"avg_{field}")) for field in TRAFFIC_CLASS_FIELDS},
    }


def kpis(db: Database) -> dict:
    now = _now()
    return {
        "generatedAt": now.isoformat(),
        "totalEvents": db.events.count_documents({}),
        "eventsLast24h": db.events.count_documents({"receivedAt": {"$gte": now - timedelta(hours=24)}}),
        "totalIssues": db.issues.count_documents({}),
        "openIssues": db.issues.count_documents({"status": {"$in": OPEN_STATUSES}}),
        "issuesByType": _group_count(db.issues, "issueType", {"status": {"$in": OPEN_STATUSES}}),
        "issuesByStatus": _group_count(db.issues, "status"),
        "activeAlerts": db.events.count_documents({
            "eventType": {"$in": ALERT_EVENT_TYPES},
            "receivedAt": {"$gte": now - timedelta(minutes=settings.ALERT_WINDOW_MINUTES)},
        }),
        "activeVehicles": db.vehicles.count_documents(
            {"lastSeenAt": {"$gte": now - timedelta(minutes=settings.ACTIVE_VEHICLE_MINUTES)}}),
        "totalVehicles": db.vehicles.count_documents({}),
        "eventsByModel": _group_count(db.events, "model"),
        "eventsByType": _group_count(db.events, "eventType"),
        "roadQuality": road_quality_summary(db),
        "traffic": traffic_summary(db),
    }


def road_health(db: Database) -> dict:
    road_types = ["pothole", "road_damage"]
    open_road = {"issueType": {"$in": road_types}, "status": {"$in": OPEN_STATUSES}}
    recent = db.events.find({"eventType": "road_quality"}).sort("capturedAt", -1).limit(50)
    return {
        "iri": road_quality_summary(db),
        "openRoadIssues": db.issues.count_documents(open_road),
        "openIssuesByType": _group_count(db.issues, "issueType", open_road),
        "openIssuesBySeverity": _group_count(db.issues, "severity", open_road),
        "roadIssuesByStatus": _group_count(db.issues, "status", {"issueType": {"$in": road_types}}),
        "recentReadings": [
            {"eventId": e["eventId"], "capturedAt": e["capturedAt"].isoformat(), "vehicleId": e["vehicleId"],
             "location": e["location"], "iri": e["metadata"].get("iri"), "iriClass": e["metadata"].get("iriClass")}
            for e in recent
        ],
    }


def traffic(db: Database, hours: int) -> dict:
    since = _now() - timedelta(hours=hours)
    samples = db.events.find({"eventType": "traffic_density", "receivedAt": {"$gte": since}}) \
        .sort("capturedAt", -1).limit(50)
    bottlenecks = db.issues.find({"issueType": "traffic_bottleneck", "status": {"$in": OPEN_STATUSES}}) \
        .sort("lastSeen", -1).limit(20)
    return {
        "summary": traffic_summary(db, hours),
        "openBottlenecks": serialize(list(bottlenecks)),
        "recentSamples": [
            {"eventId": e["eventId"], "capturedAt": e["capturedAt"].isoformat(), "vehicleId": e["vehicleId"],
             "location": e["location"], "metadata": e["metadata"]}
            for e in samples
        ],
    }


# ---------- GeoJSON ----------

def bbox_filter(bbox: str | None) -> dict:
    """'minLon,minLat,maxLon,maxLat' -> Mongo $geoWithin filter. Raises ValueError if malformed."""
    if not bbox:
        return {}
    parts = [float(p) for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be minLon,minLat,maxLon,maxLat")
    min_lon, min_lat, max_lon, max_lat = parts
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("bbox values out of range or min >= max")
    ring = [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]
    return {"location": {"$geoWithin": {"$geometry": {"type": "Polygon", "coordinates": [ring]}}}}


def feature_collection(docs) -> dict:
    features = []
    for doc in docs:
        props = serialize(doc)
        geometry = props.pop("location")
        features.append({"type": "Feature", "id": props.get("id"), "geometry": geometry, "properties": props})
    return {"type": "FeatureCollection", "features": features}


def school_zones() -> dict:
    """The school geofences tools/sim built (real OpenStreetMap polygons around the demo routes).

    Returns an empty FeatureCollection, not an error, if the scenario hasn't been built yet —
    the Safety view should show "no school zones loaded" rather than fail.
    """
    path = Path(settings.SCHOOL_ZONES_PATH)
    if not path.is_file():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- Traffic bottlenecks ----------
# The full design (docs/dashboard-backend-design.md §4) buckets by H3 cell and time-of-day, and compares
# against an 85th-percentile free-flow speed learned over many days. This backend has neither an H3
# dependency nor days of history, so it approximates both from *this session's own data*: a plain
# lat/lon grid in place of H3, and this session's 85th-percentile cell speed in place of a long-run
# free-flow baseline. That approximation is spelled out in every response ("note"), not hidden in a
# number that looks more authoritative than it is.

_CELL_M = 120.0                 # grid cell size in metres, similar to an H3 resolution-10 cell


def _cell_key(lat: float, lon: float) -> tuple[int, int]:
    lat_step = _CELL_M / 111_320.0
    lon_step = _CELL_M / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return round(lat / lat_step), round(lon / lon_step)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def traffic_bottlenecks(db: Database, hours: int = 24, limit: int = 20) -> dict:
    since = _now() - timedelta(hours=hours)
    samples = list(db.events.find(
        {"eventType": "traffic_density", "receivedAt": {"$gte": since}, "metadata.speedMps": {"$type": "number"}},
        {"location": 1, "metadata.speedMps": 1, "metadata.vehicleCount": 1, "vehicleId": 1, "capturedAt": 1},
    ))
    cells: dict[tuple[int, int], dict] = {}
    for s in samples:
        lon, lat = s["location"]["coordinates"]
        cell = cells.setdefault(_cell_key(lat, lon), {"lats": [], "lons": [], "speeds": [], "densities": [],
                                                       "vehicles": set(), "lastSeen": s["capturedAt"]})
        cell["lats"].append(lat)
        cell["lons"].append(lon)
        cell["speeds"].append(s["metadata"]["speedMps"])
        cell["densities"].append(s["metadata"].get("vehicleCount") or 0)
        cell["vehicles"].add(s["vehicleId"])
        cell["lastSeen"] = max(cell["lastSeen"], s["capturedAt"])

    if len(cells) < 2:
        return {"windowHours": hours, "cellsObserved": len(cells), "freeFlowSpeedMps": None, "bottlenecks": [],
                "note": "Not enough distinct locations yet to estimate a free-flow speed for comparison."}

    cell_mean_speeds = [sum(c["speeds"]) / len(c["speeds"]) for c in cells.values()]
    cell_mean_densities = [sum(c["densities"]) / len(c["densities"]) for c in cells.values()]
    free_flow = _percentile(cell_mean_speeds, 85)
    density_p75 = _percentile(cell_mean_densities, 75)

    ranked = []
    for (lat_i, lon_i), c in cells.items():
        n = len(c["speeds"])
        mean_speed = sum(c["speeds"]) / n
        mean_density = sum(c["densities"]) / n
        is_bottleneck = (n >= 2 and free_flow > 0 and mean_speed < 0.4 * free_flow and mean_density >= density_p75)
        if is_bottleneck:
            ranked.append({
                "location": {"latitude": round(sum(c["lats"]) / n, 6), "longitude": round(sum(c["lons"]) / n, 6)},
                "meanSpeedMps": round(mean_speed, 2), "freeFlowSpeedMps": round(free_flow, 2),
                "meanDensity": round(mean_density, 2), "sampleCount": n, "vehicleCount": len(c["vehicles"]),
                "score": round((free_flow - mean_speed) * mean_density, 2),
                "lastSeen": c["lastSeen"].isoformat(),
            })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return {
        "windowHours": hours, "cellsObserved": len(cells), "freeFlowSpeedMps": round(free_flow, 2),
        "densityP75": round(density_p75, 2), "bottlenecks": ranked[:limit],
        "note": "Free-flow speed is this session's own 85th-percentile cell speed, not a long-run baseline "
                "(docs/dashboard-backend-design.md §4 describes the full design).",
    }


# ---------- Possibly-missing signboards ----------

def possibly_missing_signs(db: Database, limit: int = 50) -> dict:
    """Open sign issues a *different, later trip* drove near again without reconfirming.

    "A different trip" is the key condition — not just a later timestamp: while the same bus is still
    driving past on the pass that first saw the sign, it keeps logging other events (traffic samples,
    IRI windows, ...) a few metres further on every few seconds, which would otherwise look exactly
    like "drove past and didn't see it" from a query on time and distance alone. Requiring the nearby
    event to come from a trip that never contributed to this issue rules that out: it only counts once
    the fleet has genuinely come back on a separate trip. That is evidence the sign may now be missing,
    not proof — the bus could simply not have looked (camera off, sign occluded that pass).
    Needs repeat visits to the same spot to say anything; on a single-pass-per-route demo it returns none.
    """
    rule = ISSUE_RULES["traffic_sign"]
    flagged = []
    for issue in db.issues.find({"issueType": "traffic_sign", "status": {"$in": OPEN_STATUSES}}):
        revisit = db.events.find_one({
            "capturedAt": {"$gt": issue["lastSeen"]},
            "tripId": {"$nin": issue.get("tripIds") or []},
            "location": {"$nearSphere": {"$geometry": issue["location"], "$maxDistance": rule["radius_m"]}},
        }, sort=[("capturedAt", 1)])
        if revisit is not None:
            flagged.append({
                "issueId": str(issue["_id"]),
                "location": {"longitude": issue["location"]["coordinates"][0],
                             "latitude": issue["location"]["coordinates"][1]},
                "lastConfirmed": issue["lastSeen"].isoformat(),
                "revisitedAt": revisit["capturedAt"].isoformat(),
                "revisitedBy": revisit["vehicleId"],
                "eventCount": issue["eventCount"],
                "condition": (issue.get("metadata") or {}).get("conditionVotes"),
            })
            if len(flagged) >= limit:
                break
    return {"count": len(flagged), "signs": flagged,
            "note": "Flags a sign only once a bus has driven past its location again without redetecting it. "
                    "Needs repeated passes over the same road; a single-pass demo will show none."}
