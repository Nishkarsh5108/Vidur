"""POST /api/v1/media (edge upload) and GET /api/v1/media/{id} (dashboard display).

Files are named by their sha256, so re-uploading the same crop (a retried batch) is a no-op rather
than a duplicate file. This intentionally does not touch MongoDB: a media id is just a filename: an
event's metadata carries `mediaId`, and the dashboard builds the URL itself as `/api/v1/media/{mediaId}`.
"""
import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.security import require_device_key

log = logging.getLogger("fleet_sense.media")
router = APIRouter(prefix="/api/v1", tags=["media"])

MEDIA_DIR = Path(settings.MEDIA_DIR)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
EXT_BY_TYPE = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


@router.post("/media", dependencies=[Depends(require_device_key)])
async def upload_media(file: UploadFile) -> dict:
    """Receive one crop from an edge device. Returns `{mediaId}` to reference from an event's metadata."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    data = await file.read(settings.MAX_MEDIA_BYTES + 1)
    if len(data) > settings.MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_MEDIA_BYTES} bytes")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    ext = EXT_BY_TYPE[file.content_type]
    media_id = hashlib.sha256(data).hexdigest()[:32] + ext
    path = MEDIA_DIR / media_id
    if not path.exists():
        path.write_bytes(data)
    return {"mediaId": media_id, "url": f"/api/v1/media/{media_id}"}


@router.get("/media/{media_id}")
def get_media(media_id: str) -> FileResponse:
    path = (MEDIA_DIR / media_id).resolve()
    # media_id is attacker-controllable (it ends up in a URL path); refuse anything that would
    # escape MEDIA_DIR (e.g. "../../secrets") before touching the filesystem.
    if MEDIA_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=604800, immutable"})
