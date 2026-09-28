"""Issue aggregation: turn many detections of the same real-world problem into one issue.

Rule (MVP, deliberately simple): an event joins the NEAREST open issue that has
  * the same type,
  * a location within ISSUE_RULES[type]["radius_m"] metres,
  * lastSeen within ISSUE_RULES[type]["window_hours"] of the event.
Otherwise a new `candidate` issue is created at the event's location.
"""
import re
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.database import Database

from app.core.config import ISSUE_RULES, OPEN_STATUSES, PASS_FACTOR

_SAFE_KEY = re.compile(r"[a-z_]{1,32}")


def _severity(metadata: dict) -> int | None:
    value = metadata.get("severity")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def _condition(metadata: dict) -> str | None:
    """Traffic-sign condition ('damaged' / 'good'), used as a vote counter key."""
    value = metadata.get("condition")
    if isinstance(value, str) and _SAFE_KEY.fullmatch(value.strip().lower()):
        return value.strip().lower()
    return None


def _derived_fields(issue: dict) -> dict:
    """Counts, confidence and automatic status, recomputed after every change."""
    vehicle_count = len(issue.get("vehicleIds", []))
    trip_count = len(issue.get("tripIds", []))
    factor = PASS_FACTOR.get(max(trip_count, 1), 1.0)
    best = issue.get("maxEventConfidence")
    status = issue["status"]
    if status == "candidate" and (trip_count >= 2 or vehicle_count >= 2):
        status = "probable"   # seen on 2+ separate passes -> probably real
    return {
        "vehicleCount": vehicle_count,
        "tripCount": trip_count,
        "confidence": round(best * factor, 3) if best is not None else None,
        "status": status,
    }


def attach_event(db: Database, event: dict) -> tuple[dict | None, str | None]:
    """Attach a stored event to an existing issue or create a new one.

    Returns (issue_document, "created" | "updated"), or (None, None) if this event type
    does not form issues (e.g. traffic_density).
    """
    rule = ISSUE_RULES.get(event["eventType"])
    if rule is None:
        return None, None

    now = datetime.now(timezone.utc)
    captured = event["capturedAt"]
    metadata = event.get("metadata") or {}
    severity = _severity(metadata)
    condition = _condition(metadata)

    nearest_open = db.issues.find_one({
        "issueType": event["eventType"],
        "status": {"$in": OPEN_STATUSES},
        "lastSeen": {"$gte": captured - timedelta(hours=rule["window_hours"])},
        "location": {"$nearSphere": {"$geometry": event["location"], "$maxDistance": rule["radius_m"]}},
    })

    if nearest_open is None:
        issue = {
            "issueType": event["eventType"],
            "status": "candidate",
            "location": event["location"],
            "firstSeen": captured,
            "lastSeen": captured,
            "eventCount": 1,
            "vehicleIds": [event["vehicleId"]],
            "tripIds": [event["tripId"]] if event.get("tripId") else [],
            "maxEventConfidence": event["confidence"],
            "severity": severity,
            "metadata": {"model": event["model"]},
            "statusNote": None,
            "createdAt": now,
            "updatedAt": now,
        }
        if condition:
            issue["metadata"]["conditionVotes"] = {condition: 1}
        issue.update(_derived_fields(issue))
        issue["_id"] = db.issues.insert_one(issue).inserted_id
        return issue, "created"

    update: dict = {
        "$inc": {"eventCount": 1},
        "$addToSet": {"vehicleIds": event["vehicleId"]},
        "$max": {"lastSeen": captured, "maxEventConfidence": event["confidence"]},
        "$min": {"firstSeen": captured},
        "$set": {"updatedAt": now},
    }
    if event.get("tripId"):
        update["$addToSet"]["tripIds"] = event["tripId"]
    if severity is not None:
        update["$max"]["severity"] = severity
    if condition:
        update["$inc"][f"metadata.conditionVotes.{condition}"] = 1

    issue = db.issues.find_one_and_update(
        {"_id": nearest_open["_id"]}, update, return_document=ReturnDocument.AFTER
    )
    derived = _derived_fields(issue)
    db.issues.update_one({"_id": issue["_id"]}, {"$set": derived})
    issue.update(derived)
    return issue, "updated"


def update_status(db: Database, issue_id: ObjectId, status: str, note: str | None) -> dict | None:
    """Human review: verify / reject / resolve (or reset) an issue."""
    return db.issues.find_one_and_update(
        {"_id": issue_id},
        {"$set": {"status": status, "statusNote": note, "updatedAt": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
