"""Batch ingestion: validate -> store (idempotent) -> aggregate into issues -> update vehicle/trip."""
from datetime import datetime, timezone

from pydantic import ValidationError
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from app.db.mongodb import serialize
from app.schemas.ingest import EventIn, EventResult, IngestBatch, IngestResult
from app.services.issue_service import attach_event


def _reason(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'event'}: {err['msg']}" for err in error.errors()
    )


def process_batch(db: Database, batch: IngestBatch) -> tuple[IngestResult, list[dict]]:
    """Process events one by one so that a single bad event never rejects the whole batch.

    Returns the per-event result and the WebSocket messages to broadcast.
    """
    received_at = datetime.now(timezone.utc)
    results: list[EventResult] = []
    messages: list[dict] = []
    stored: list[dict] = []

    for index, raw in enumerate(batch.events):
        raw_id = raw.get("eventId")
        try:
            event = EventIn.model_validate(raw)
        except ValidationError as exc:
            results.append(EventResult(index=index, eventId=str(raw_id) if raw_id else None,
                                       status="rejected", reason=_reason(exc)))
            continue

        doc = {
            "eventId": event.eventId,
            "eventType": event.eventType,
            "confidence": event.confidence,
            "deviceId": batch.deviceId,
            "vehicleId": batch.vehicleId,
            "tripId": batch.tripId,
            "capturedAt": event.capturedAt,
            "receivedAt": received_at,
            "location": event.location.to_geojson(),
            "model": event.model,
            "modelVersion": event.modelVersion,
            "metadata": event.metadata,
            "issueId": None,
        }
        try:
            db.events.insert_one(doc)          # unique index on eventId makes this idempotent
        except DuplicateKeyError:
            results.append(EventResult(index=index, eventId=event.eventId, status="duplicate"))
            continue

        issue, action = attach_event(db, doc)
        if issue is not None:
            db.events.update_one({"_id": doc["_id"]}, {"$set": {"issueId": issue["_id"]}})
            doc["issueId"] = issue["_id"]

        stored.append(doc)
        results.append(EventResult(index=index, eventId=event.eventId, status="accepted",
                                   issueId=str(issue["_id"]) if issue else None))
        messages.append({"type": "event", "data": serialize(doc)})
        if issue is not None:
            messages.append({"type": "issue_update", "data": {**serialize(issue), "action": action}})

    if stored:
        vehicle = _update_vehicle_and_trip(db, batch, stored, received_at)
        messages.append({"type": "vehicle_update", "data": serialize(vehicle)})

    counts = {s: sum(1 for r in results if r.status == s) for s in ("accepted", "duplicate", "rejected")}
    result = IngestResult(accepted=counts["accepted"], duplicates=counts["duplicate"],
                          rejected=counts["rejected"], results=results)
    return result, messages


def _update_vehicle_and_trip(db: Database, batch: IngestBatch, stored: list[dict], now: datetime) -> dict:
    latest = max(stored, key=lambda d: d["capturedAt"])
    first_captured = min(d["capturedAt"] for d in stored)

    db.vehicles.update_one(
        {"_id": batch.vehicleId},
        {
            "$set": {"deviceId": batch.deviceId, "lastSeenAt": now, "lastTripId": batch.tripId,
                     "lastLocation": latest["location"]},
            "$max": {"lastCapturedAt": latest["capturedAt"]},
            "$setOnInsert": {"firstSeenAt": now},
            "$inc": {"eventCount": len(stored)},
        },
        upsert=True,
    )
    if batch.tripId:
        db.trips.update_one(
            {"_id": batch.tripId},
            {
                "$set": {"vehicleId": batch.vehicleId, "deviceId": batch.deviceId, "lastSeenAt": now},
                "$min": {"startedAt": first_captured},
                "$max": {"endedAt": latest["capturedAt"]},
                "$inc": {"eventCount": len(stored)},
            },
            upsert=True,
        )
    return db.vehicles.find_one({"_id": batch.vehicleId})
