from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.db.mongodb import get_db, serialize, to_object_id

router = APIRouter(prefix="/api/v1", tags=["events"])


@router.get("/events")
def list_events(
    eventType: str | None = None,
    vehicleId: str | None = None,
    tripId: str | None = None,
    model: str | None = None,
    issueId: str | None = None,
    from_: datetime | None = Query(default=None, alias="from", description="capturedAt >= (ISO 8601)"),
    to: datetime | None = Query(default=None, description="capturedAt <= (ISO 8601)"),
    limit: int = Query(default=100, ge=1, le=1000),
    skip: int = Query(default=0, ge=0),
):
    query: dict = {}
    for field, value in (("eventType", eventType), ("vehicleId", vehicleId), ("tripId", tripId), ("model", model)):
        if value:
            query[field] = value
    if issueId:
        query["issueId"] = to_object_id(issueId) or "__invalid__"
    if from_ or to:
        query["capturedAt"] = {}
        if from_:
            query["capturedAt"]["$gte"] = from_
        if to:
            query["capturedAt"]["$lte"] = to

    db = get_db()
    items = db.events.find(query).sort("capturedAt", -1).skip(skip).limit(limit)
    return {"total": db.events.count_documents(query), "limit": limit, "skip": skip, "items": serialize(list(items))}


@router.get("/events/{event_id}")
def get_event(event_id: str):
    event = get_db().events.find_one({"eventId": event_id})
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return serialize(event)
