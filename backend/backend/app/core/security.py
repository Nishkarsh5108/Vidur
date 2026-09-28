"""Simple API-key check for edge devices."""
import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


def require_device_key(x_device_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: the request must carry `X-Device-Key: <DEVICE_API_KEY>`."""
    if not x_device_key or not secrets.compare_digest(x_device_key, settings.DEVICE_API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-Device-Key")
