"""Body of POST /api/v1/simulate/start."""
from pydantic import BaseModel, Field


class SimulateStart(BaseModel):
    rate: float = Field(default=1.0, ge=0, le=20, description="1 = real time (default), 0 = as fast as possible")
    clock: str = Field(default="now", max_length=64, description="'now', or an ISO 8601 start time")
    only: list[str] | None = Field(default=None, description="run only these vehicle ids (default: all)")
    maxConcurrent: int = Field(default=2, ge=1, le=8, description="bus processes loading their models at once")
    stagger: float = Field(default=6.0, ge=0, le=60, description="seconds between starting each next process")
    hud: bool = Field(default=False, description="also write an annotated video per bus")
