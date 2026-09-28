from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.api.websocket import manager
from app.db.mongodb import get_db, serialize, to_object_id
from app.schemas.issue import IssueStatus, IssueUpdate
from app.services.issue_service import update_status

router = APIRouter(prefix="/api/v1", tags=["issues"])


@router.get("/issues")
def list_issues(
    issueType: str | None = None,
    status: IssueStatus | None = None,
    minConfidence: float = Query(default=0, ge=0, le=1),
    limit: int = Query(default=100, ge=1, le=1000),
    skip: int = Query(default=0, ge=0),
):
    query: dict = {}
    if issueType:
        query["issueType"] = issueType
    if status:
        query["status"] = status
    if minConfidence > 0:
        query["confidence"] = {"$gte": minConfidence}

    db = get_db()
    items = db.issues.find(query).sort("lastSeen", -1).skip(skip).limit(limit)
    return {"total": db.issues.count_documents(query), "limit": limit, "skip": skip, "items": serialize(list(items))}


@router.get("/issues/{issue_id}")
def get_issue(issue_id: str):
    oid = to_object_id(issue_id)
    db = get_db()
    issue = db.issues.find_one({"_id": oid}) if oid else None
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    events = db.events.find({"issueId": oid}).sort("capturedAt", -1).limit(100)
    return {**serialize(issue), "events": serialize(list(events))}


@router.patch("/issues/{issue_id}")
async def patch_issue(issue_id: str, body: IssueUpdate):
    """Human review from the dashboard: verify / reject / resolve an issue."""
    oid = to_object_id(issue_id)
    issue = await run_in_threadpool(update_status, get_db(), oid, body.status, body.note) if oid else None
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    data = serialize(issue)
    await manager.broadcast({"type": "issue_update", "data": {**data, "action": "status_changed"}})
    return data
