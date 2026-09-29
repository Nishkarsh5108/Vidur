# Vidur

**Edge AI on public buses for road, traffic and pedestrian-safety intelligence.**
Smart India Hackathon 2026 · Problem statement PS26124 (Bharat Electronics Limited): *AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet.* The design documents call the platform **FleetSense**.

Each bus runs the models on board and uploads small events instead of video:
- potholes, confirmed by the camera and the IMU;
- road roughness for every 100 m;
- damaged signs and visible zebra crossings;
- traffic density;
- alerts when people are near the road in a school zone.

A backend turns those events into de-duplicated, located, confidence-scored issues for a city dashboard.

## Status (29 Sep 2026)

| Part | Status |
|---|---|
| Models: V1–V3 (vision) and S1–S2 (IMU) | Trained. Their as-built behaviour, metrics and caveats were checked against the model files: [docs/ml-models-spec.md](docs/ml-models-spec.md). |
| Edge agent ([edge/](edge/)) | **Implemented and tested.** All five models run in one pipeline, in replay on a laptop. |
| Backend ([backend/backend/](backend/backend/README.md)) | **Implemented and tested.** FastAPI + MongoDB: ingest, idempotency, issue aggregation, media store, bottleneck/missing-sign analytics, a simulate control API, REST/GeoJSON/WebSocket. 40 automated tests. |
| Ops dashboard ([dashboard/](dashboard/README.md)) | **Implemented.** A single map-first page served by the backend at `/`: live map, road health, traffic, infrastructure, safety and fleet views, a one-click "Simulate" button, light/dark theme. |
| Edge → backend integration | **Implemented and verified end to end** on a 4-bus simulated fleet ([tools/sim](tools/sim/README.md)): pothole, zebra, sign, road-shock, IRI and pedestrian-alert events all reach the backend, aggregate into issues, show up in its KPIs, and their photos display in the dashboard. |
| Raspberry Pi 5 deployment | **Designed and costed; not yet run on a Pi.** NCNN exports are ready. Pi figures are estimates and are labelled as such: [docs/edge-deployment.md](docs/edge-deployment.md). |

## How it works

```
ON THE BUS (edge agent)                                            BACKEND                     DASHBOARD (served at /)
camera ─► V2 road users + ByteTrack (continuous) ─► traffic, school-zone alerts ─┐
       └► V3 sign condition (once per sign, on crops)                             ├─► events ─► FastAPI + Mongo ─► live map, heatmaps,
IMU ───► S1 roughness (IRI) every 100 m                                           │  + photos    issues, KPIs,      review, fleet
     └► jolt ─► V1 pothole check on the frames just before it ────────────────────┘ (never video) simulate control
```

Only V2 runs on every frame it can keep up with. The expensive models run only when there is something for them to look at: V1 after an IMU jolt, V3 once per sign. That gating is what keeps the pipeline within a Raspberry Pi 5's CPU (about 40–60% estimated) and keeps the uplink to events and small crops. See [docs/edge-deployment.md](docs/edge-deployment.md).

## Models

| ID | Detects | Model | When it runs | Headline metric |
|---|---|---|---|---|
| V1 | Potholes, visible zebra crossings | YOLOv8m, 480×800 | On 3 buffered frames per IMU jolt | mAP50 0.674 (its validation split) |
| V2 | 13 Indian road-user and infrastructure classes, tracked | YOLOv8n + ByteTrack, 384×640 | Continuously, 3–4 FPS | mAP50 0.470 |
| V3 | Sign condition: damaged or good | YOLOv8s, 320 on crops | Once per tracked sign | mAP50 0.951 (its own split: close-up images) |
| S1 | Road roughness (IRI, m/km) | Two-branch 1D CNN, TFLite, 42 KB | Every 100 m of travel | MAE 1.65 m/km (simulator test set) |
| S2 | Shock grade from vertical acceleration | 1D CNN, TFLite, 221 KB | Every 0.1 s | Files not yet in this repository; a jerk threshold stands in |

None of the vision models has been evaluated on bus-mounted footage yet. The spec lists this and other caveats ([ml-models-spec.md §0](docs/ml-models-spec.md)).

## Quick start

```bash
conda create -n vidur python=3.11 -y && conda activate vidur
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU only
pip install -r edge/requirements.txt
python -m pytest                                                       # the edge test suite

# Road roughness and shocks along the team's 110 km field log (IMU only, about 40 s)
python -m edge --imu ML_models/IRI/iri_compliant/KMP_mathura.csv

# A recorded ride (video + IMU/GPS), with an annotated demo video
python -m edge --video recordings/ride1.mp4 --imu recordings/ride1.csv \
               --video-start 2026-10-05T08:02:11.000Z --hud out/ride1_hud.mp4
```

More options, the input formats and the Pi 5 steps are in [edge/README.md](edge/README.md).

To try the vision models on any image or video in the browser, with boxes drawn and GPU inference, use the [model lab](tools/model_lab/README.md): `python tools/model_lab/app.py`.

### End-to-end demo: the dashboard, backed by a live fleet

```bash
cd backend/backend && docker compose up -d && pip install -r requirements.txt && cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000            # see backend/backend/README.md for details

cd ../.. && python tools/sim/build_scenario.py              # once: pairs each demo video with a real road + real schools
```

Open **http://localhost:8000/** — that's the [dashboard](dashboard/README.md) — and click **Simulate**.
It runs one edge agent per demo video, each with a GPS trail built from a real recorded drive
([tools/sim](tools/sim/README.md)), streaming live events onto the map as they arrive: potholes, zebra
crossings, sign conditions with photos, road roughness, and a school-zone pedestrian alert. Verified
(29 Sep) end to end: 4 buses, 273 events, 20 aggregated issues, 25 photos uploaded and displayed
correctly, 4 real traffic bottlenecks detected from the session's own data.

## Repository layout

```
ML_models/   trained models and their original training/inference code, as delivered
edge/        the edge agent: model runners, two-lane scheduler, aggregators, replay, backend upload, tests
backend/     FastAPI + MongoDB backend: ingest, issue aggregation, media, simulate control, analytics (backend/backend/)
dashboard/   the ops dashboard: a static, no-build-step page the backend serves at "/"
tools/       model lab (run V1–V3 on any image/video) and the demo fleet simulator (tools/sim/)
docs/        specifications, deployment analysis, plans and decisions
exports/     NCNN / Hailo exports (generated by edge/scripts/export_models.py, not versioned)
recordings/, out/   demo recordings and run outputs (not versioned)
```

## Documentation

| Document | Contents |
|---|---|
| [docs/ml-models-spec.md](docs/ml-models-spec.md) | As-built specification of every model: inputs, outputs, metrics, known issues |
| [docs/edge-deployment.md](docs/edge-deployment.md) | Quantisation, how the models are combined, the Raspberry Pi 5 budget, when an AI HAT+ is worth it |
| [docs/dashboard-backend-design.md](docs/dashboard-backend-design.md) | Edge-to-backend data contract, backend processing, dashboard pages, demo plan |
| [docs/hello.md](docs/hello.md) | The backend's own, simpler event contract (what `edge/backend.py` actually sends) |
| [backend/backend/README.md](backend/backend/README.md) | Backend setup, full API reference, issue-aggregation rules, the bottleneck/missing-sign approximations |
| [dashboard/README.md](dashboard/README.md) | The ops dashboard: layout, live updates, the Simulate button, theming |
| [tools/sim/README.md](tools/sim/README.md) | How the demo fleet's GPS trails and school zones are built, and what's real vs. simulated in them |
| [docs/v1-successor-plan.md](docs/v1-successor-plan.md) | Plan for a nano road-surface model that can run continuously on the Pi |
| [docs/decisions.md](docs/decisions.md) | Decision log |
| [docs/sih2026-ps26124-roadmap.pdf](docs/sih2026-ps26124-roadmap.pdf) | Problem-statement requirements mapped to public datasets and features |

## Known gaps

- **S2's model files** (`ML_models/IRI/ml_model/road_classification/`) still need to be added. Until then, V1 is triggered by a jerk threshold calibrated on the team's field logs.
- **Evaluation:** the vision models have not been evaluated on footage from a bus-mounted camera.
- **Not yet covered:** cracks (planned in the V1 successor), faded lane markings, missing dividers, waterlogging, and hit-and-run / ANPR.
- **Raspberry Pi 5:** figures are estimates until measured on hardware.
- **Demo GPS is simulated, not recorded:** the demo videos have no GPS of their own, so their trails are built by pairing each one with a stretch of a real recorded drive ([tools/sim](tools/sim/README.md)) — real roads and real school locations, but a simulated pairing to the video. The dashboard's own map says so in a corner banner.
- **Traffic bottlenecks and possibly-missing signs are demo-scale approximations**, computed from the current session's own data rather than a long-run baseline or repeated-pass history — every API response carries a `note` field saying so ([backend/backend/README.md §4a–§4b](backend/backend/README.md)).
- **No dedicated tests for the dashboard's JavaScript** (only syntax/lint-level checks were possible in this environment — no headless browser). The backend endpoints it calls are tested; open it in a real browser to confirm the UI itself.

## Licensing

The YOLO models and the code that runs them use [Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. The dashboard vendors [MapLibre GL JS](https://maplibre.org/) (BSD-3-Clause) and uses CARTO's free basemap tiles and OpenStreetMap data, both attributed on the map itself.
