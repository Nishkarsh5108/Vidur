from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.api.websocket import manager
from app.core.security import require_device_key
from app.db.mongodb import get_db
from app.schemas.ingest import IngestBatch, IngestResult
from app.services.event_service import process_batch

router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestResult, dependencies=[Depends(require_device_key)])
async def ingest(batch: IngestBatch) -> IngestResult:
    """Receive a batch of AI events from an edge device (header `X-Device-Key` required)."""
    # PyMongo is synchronous, so run the DB work in a worker thread, then broadcast on the event loop.
    result, messages = await run_in_threadpool(process_batch, get_db(), batch)
    for message in messages:
        await manager.broadcast(message)
    return result
