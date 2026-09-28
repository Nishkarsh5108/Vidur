"""FleetSense backend: receives AI events from edge devices, aggregates issues, serves the dashboard.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analytics, events, ingest, issues, vehicles, websocket
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

for module in (ingest, events, issues, vehicles, analytics, websocket):
    app.include_router(module.router)


@app.get("/", include_in_schema=False)
def root():
    return {"name": "FleetSense Backend", "docs": "/docs", "health": "/api/v1/health", "live": "/ws/live"}
