"""KPIs, road-health, traffic and GeoJSON helpers. Every number comes from the database:
an empty database gives 0, [] or null — never made-up values.

"Recent" windows (last 24 h, active vehicles, active alerts) use the server's `receivedAt`,
so a replayed recording with old `capturedAt` timestamps still shows up as live data.
"""
from datetime import datetime, timedelta, timezone

from pymongo.collection import Collection
from pymongo.database import Database

from app.core.config import ALERT_EVENT_TYPES, OPEN_STATUSES, settings
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
