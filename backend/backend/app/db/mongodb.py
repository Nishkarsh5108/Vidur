"""MongoDB connection, indexes and JSON serialization helpers."""
import logging
from datetime import datetime

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, GEOSPHERE, MongoClient
from pymongo.database import Database

from app.core.config import settings

log = logging.getLogger("fleet_sense.db")

_client: MongoClient | None = None


def connect() -> None:
    """Open the client and fail fast with a clear message if MongoDB is not reachable."""
    global _client
    _client = MongoClient(settings.MONGODB_URI, tz_aware=True, serverSelectionTimeoutMS=3000)
    try:
        _client.admin.command("ping")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach MongoDB at {settings.MONGODB_URI}. "
            "Start it first (e.g. `docker compose up -d`) and check MONGODB_URI in .env."
        ) from exc
    log.info("Connected to MongoDB %s / %s", settings.MONGODB_URI, settings.DATABASE_NAME)


def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_db() -> Database:
    if _client is None:
        connect()
    return _client[settings.DATABASE_NAME]


def create_indexes() -> None:
    """Idempotent: safe to run on every startup."""
    db = get_db()
    db.events.create_index("eventId", unique=True)
    db.events.create_index([("location", GEOSPHERE)])
    db.events.create_index([("capturedAt", DESCENDING)])
    db.events.create_index([("receivedAt", DESCENDING)])
    db.events.create_index([("eventType", ASCENDING), ("capturedAt", DESCENDING)])
    db.events.create_index([("vehicleId", ASCENDING), ("capturedAt", DESCENDING)])
    db.events.create_index("issueId")

    db.issues.create_index([("location", GEOSPHERE)])
    db.issues.create_index([("issueType", ASCENDING), ("status", ASCENDING)])
    db.issues.create_index([("lastSeen", DESCENDING)])

    db.vehicles.create_index([("lastSeenAt", DESCENDING)])
    db.trips.create_index("vehicleId")


def to_object_id(value: str) -> ObjectId | None:
    return ObjectId(value) if ObjectId.is_valid(value) else None


def serialize(value):
    """Make a Mongo document JSON-friendly: _id -> id, ObjectId -> str, datetime -> ISO string."""
    if isinstance(value, list):
        return [serialize(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, v in value.items():
            out["id" if key == "_id" else key] = serialize(v)
        return out
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value
