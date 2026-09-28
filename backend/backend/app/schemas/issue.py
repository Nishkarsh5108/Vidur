from typing import Literal

from pydantic import BaseModel, Field

IssueStatus = Literal["candidate", "probable", "verified", "rejected", "resolved"]


class IssueUpdate(BaseModel):
    """Body of PATCH /api/v1/issues/{id} — a human reviewer changes the status."""
    status: IssueStatus
    note: str | None = Field(default=None, max_length=500)
