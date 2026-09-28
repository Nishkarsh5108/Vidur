"""Test setup. Tests run against a real MongoDB (needed for 2dsphere queries) using a separate
database `fleet_sense_test`, which is emptied before every test. Start MongoDB first."""
import math
import os
import uuid
from datetime import datetime, timedelta, timezone

os.environ["DATABASE_NAME"] = "fleet_sense_test"     # must be set before app.* is imported
os.environ["DEVICE_API_KEY"] = "test-key"

import pytest
from fastapi.testclient import TestClient
from pymongo import MongoClient

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
HEADERS = {"X-Device-Key": "test-key"}
BASE_LAT, BASE_LON = 28.6139, 77.2090


def _mongo_is_up() -> bool:
    try:
        MongoClient(MONGODB_URI, serverSelectionTimeoutMS=1500).admin.command("ping")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def client():
    if not _mongo_is_up():
        pytest.skip(f"MongoDB is not reachable at {MONGODB_URI} — start it (docker compose up -d) and re-run")
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_db(client):
    from app.db.mongodb import get_db
    db = get_db()
    for name in ("events", "issues", "vehicles", "trips"):
        db[name].delete_many({})     # keep the indexes, drop the data
    yield db


def offset(lat: float, lon: float, north_m: float = 0, east_m: float = 0) -> tuple[float, float]:
    """Move a point by some metres (good enough for tests)."""
    return (lat + north_m / 111_320,
            lon + east_m / (111_320 * math.cos(math.radians(lat))))


def make_event(event_type: str = "pothole", lat: float = BASE_LAT, lon: float = BASE_LON, **overrides) -> dict:
    event = {
        "eventId": str(uuid.uuid4()),
        "eventType": event_type,
        "capturedAt": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        "location": {"latitude": lat, "longitude": lon},
        "confidence": 0.9,
        "model": "road_model",
        "modelVersion": "1.0",
        "metadata": {"severity": 2},
    }
    event.update(overrides)
    return event


def post_batch(client, events, vehicle_id="BUS_T1", trip_id="TRIP_T1", device_id="EDGE_T1", headers=HEADERS):
    body = {"schemaVersion": "1.0", "deviceId": device_id, "vehicleId": vehicle_id,
            "tripId": trip_id, "events": events}
    return client.post("/api/v1/ingest", json=body, headers=headers)
