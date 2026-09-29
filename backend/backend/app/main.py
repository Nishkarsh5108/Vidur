"""FleetSense backend: receives AI events from edge devices, aggregates issues, serves the dashboard.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import analytics, events, ingest, issues, media, simulate, vehicles, websocket
from app.core.config import REPO_ROOT
from app.db import mongodb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongodb.connect()
    mongodb.create_indexes()
    yield
    mongodb.close()


app = FastAPI(
    title="FleetSense Backend",
    description="Event ingestion, issue aggregation and GIS APIs for the SIH urban-intelligence platform.",
    version="1.0.0",
    lifespan=lifespan,
)

# The dashboard may run on another port/laptop during the demo.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for module in (ingest, events, issues, vehicles, analytics, media, simulate, websocket):
    app.include_router(module.router)

# The ops dashboard (dashboard/): a static, no-build-step page served straight off disk, so opening
# this one server is the whole demo. Mounted last so it never shadows an /api/* or /ws/* route above.
_dashboard_dir = REPO_ROOT / "dashboard"
if _dashboard_dir.is_dir():
    app.mount("/", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"name": "FleetSense Backend", "docs": "/docs", "health": "/api/v1/health", "live": "/ws/live"}
