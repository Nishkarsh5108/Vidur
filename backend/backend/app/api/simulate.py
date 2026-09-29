"""POST /api/v1/simulate/start|stop, GET /api/v1/simulate/status — runs the demo fleet on this laptop."""
from fastapi import APIRouter, HTTPException, Request

from app.schemas.simulate import SimulateStart
from app.services import simulate_service

router = APIRouter(prefix="/api/v1/simulate", tags=["simulate"])


@router.get("/status")
def status():
    return simulate_service.status()


@router.post("/start")
def start(body: SimulateStart, request: Request):
    # The bus processes this spawns run on this same machine, so they always reach the backend via
    # loopback — even if a judge is viewing the dashboard from another laptop's browser, in which
    # case request.url.host would be this host's LAN IP, which is not guaranteed to route back to itself.
    backend_url = f"http://127.0.0.1:{request.url.port or 8000}"
    try:
        return simulate_service.start(backend_url, rate=body.rate, clock=body.clock, only=body.only,
                                      max_concurrent=body.maxConcurrent, stagger=body.stagger, hud=body.hud)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/stop")
def stop():
    return simulate_service.stop()
