"""Health, KPIs, road health, traffic and GeoJSON map layers."""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.mongodb import get_db
from app.schemas.issue import IssueStatus
from app.services import analytics_service

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.get("/health")
def health():
    try:
        get_db().command("ping")
        db_ok = True
    except Exception:
        db_ok = False
    body = {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "unreachable",
        "databaseName": settings.DATABASE_NAME,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(body, status_code=200 if db_ok else 503)


@router.get("/kpis")
def kpis():
    return analytics_service.kpis(get_db())


@router.get("/road-health")
def road_health():
    return analytics_service.road_health(get_db())


@router.get("/traffic")
def traffic(hours: int = Query(default=24, ge=1, le=24 * 30)):
    return analytics_service.traffic(get_db(), hours)


def _bbox(bbox: str | None) -> dict:
    try:
        return analytics_service.bbox_filter(bbox)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid bbox: {exc}")


@router.get("/map/events")
def map_events(
    eventType: str | None = None,
    vehicleId: str | None = None,
    bbox: str | None = Query(default=None, description="minLon,minLat,maxLon,maxLat"),
    limit: int = Query(default=500, ge=1, le=5000),
):
    query = _bbox(bbox)
    if eventType:
        query["eventType"] = eventType
    if vehicleId:
        query["vehicleId"] = vehicleId
    docs = get_db().events.find(query).sort("capturedAt", -1).limit(limit)
    return analytics_service.feature_collection(docs)


@router.get("/map/issues")
def map_issues(
    issueType: str | None = None,
    status: IssueStatus | None = None,
    bbox: str | None = Query(default=None, description="minLon,minLat,maxLon,maxLat"),
    limit: int = Query(default=1000, ge=1, le=5000),
):
    query = _bbox(bbox)
    if issueType:
        query["issueType"] = issueType
    if status:
        query["status"] = status
    docs = get_db().issues.find(query).sort("lastSeen", -1).limit(limit)
    return analytics_service.feature_collection(docs)
