from fastapi import APIRouter, HTTPException

from app.db.mongodb import get_db, serialize

router = APIRouter(prefix="/api/v1", tags=["vehicles"])


@router.get("/vehicles")
def list_vehicles():
    items = list(get_db().vehicles.find().sort("lastSeenAt", -1))
    return {"total": len(items), "items": serialize(items)}


@router.get("/vehicles/{vehicle_id}")
def get_vehicle(vehicle_id: str):
    db = get_db()
    vehicle = db.vehicles.find_one({"_id": vehicle_id})
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    trips = db.trips.find({"vehicleId": vehicle_id}).sort("startedAt", -1).limit(50)
    recent = db.events.find({"vehicleId": vehicle_id}).sort("capturedAt", -1).limit(20)
    return {**serialize(vehicle), "trips": serialize(list(trips)), "recentEvents": serialize(list(recent))}
