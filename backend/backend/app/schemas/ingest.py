"""The edge -> backend event contract (schemaVersion 1.0)."""
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    def to_geojson(self) -> dict:
        # GeoJSON order is [longitude, latitude]
        return {"type": "Point", "coordinates": [self.longitude, self.latitude]}


class EventIn(BaseModel):
    """One AI detection, already synchronized with GPS on the edge."""
    eventId: str = Field(min_length=1, max_length=128)
    eventType: str = Field(min_length=1, max_length=64)
    capturedAt: datetime
    location: Location
    # Detector confidence. Optional because measurement events (IRI, traffic counts) have none.
    confidence: float | None = Field(default=None, ge=0, le=1)
    model: str = Field(min_length=1, max_length=64)
    modelVersion: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("eventType")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("capturedAt")
    @classmethod
    def utc_and_not_future(cls, v: datetime) -> datetime:
        v = v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)
        if v > datetime.now(timezone.utc) + timedelta(seconds=settings.MAX_FUTURE_SECONDS):
            raise ValueError("capturedAt is in the future (check the edge clock)")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def none_to_empty(cls, v):
        return {} if v is None else v


class IngestBatch(BaseModel):
    """Envelope. `events` is kept as raw dicts so that one bad event does not reject the whole batch;
    each event is validated individually against EventIn in the service layer."""
    schemaVersion: Literal["1.0"] = "1.0"
    deviceId: str = Field(min_length=1, max_length=64)
    vehicleId: str = Field(min_length=1, max_length=64)
    tripId: str | None = Field(default=None, max_length=128)
    events: list[dict[str, Any]] = Field(min_length=1, max_length=settings.MAX_BATCH_SIZE)


class EventResult(BaseModel):
    index: int
    eventId: str | None = None
    status: Literal["accepted", "duplicate", "rejected"]
    reason: str | None = None
    issueId: str | None = None


class IngestResult(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    results: list[EventResult]
