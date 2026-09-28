# FleetSense — Demo Dashboard & Backend Design

*28 Sep 2026. Consumes the models in [ml-models-spec.md](ml-models-spec.md). Goal: an end-to-end demo for the SIH finale that runs offline on one laptop, with anything not yet real clearly labelled as roadmap.*

---

## 0. Summary

- **Three parts:**

  | Part | Built with | Job |
  |---|---|---|
  | Edge agent | Python, on the bus (Pi), or on a laptop replaying a recorded ride | Runs the models and sends *events*, never video |
  | Backend | FastAPI + PostgreSQL/PostGIS + WebSocket | Turns raw detections into de-duplicated, located, reviewable **issues**, plus per-area **road and traffic statistics** |
  | Dashboard | React + MapLibre GL + deck.gl | Command-centre map, road-health, infrastructure, traffic, safety, review-queue and fleet-health pages |

- **The demo runs in replay mode.** A recorded ride (video + IMU + GPS on one clock) goes through the same edge agent and backend a live bus would use. It's reproducible on stage and needs no network or bus.
- **The PS asks for "confidence scores" and "minimizing bandwidth through intelligent edge processing".** So the dashboard shows model confidence and provenance on every item, and a live counter of bytes sent against the raw-video equivalent.
- **The backend owns everything that needs memory across frames, trips or buses:** de-duplication, confirmation, "missing" assets, bottlenecks. The edge only reports what it saw and where.

---

## 1. Architecture

```
┌──────────────────────── BUS / EDGE (Pi 5, or laptop in replay mode) ─────────────────────────┐
│ camera ─┐                                                                                    │
│ IMU ────┼─► capture (one monotonic clock, GPS time) ─► ring buffer (≈3 s frames, 10 s IMU)   │
│ GPS ────┘            │                                                                       │
│                      ├─► V2 IDD YOLOv8n + ByteTrack ──► traffic summariser (per 10 s / 100 m) │
│                      │        └─ "traffic sign" box ─► V3 sign condition ─► per-track vote   │
│                      ├─► S2 shock classifier (sliding) ─┐                                    │
│                      │                                  ├─► trigger ─► V1 RoadSense on       │
│                      ├─► S1 IRI (every 100 m)           │              buffered frames       │
│                      └─► school geofence (GPS) ─► pedestrian alert logic (V2 persons)        │
│                                          ▼                                                   │
│                     event builder ─► outbox (SQLite) ─► uploader (batch, retry, idempotent)  │
└──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               │ HTTPS  POST /api/v1/ingest (+ /media, JPEG crops)
┌──────────────────────────────── BACKEND (docker compose) ───────────────────────────────────┐
│ FastAPI ─► raw tables (events, telemetry, iri_windows, traffic_samples)                      │
│    │           └─► workers: issue clustering · confidence fusion · IRI and traffic grids ·   │
│    │                        bottleneck score · school-zone alerts · "expected but not seen"  │
│    ├─► REST (GeoJSON) for the dashboard           PostgreSQL 16 + PostGIS 3                  │
│    └─► WebSocket /ws/live (positions, new issues, alerts, KPIs)    media on local disk       │
└──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                               ▼
                        DASHBOARD  React + Vite + MapLibre GL + deck.gl
```

---

## 2. Edge agent

A single Python process in `edge/`. It is the **same code in live mode and in replay mode**; only the capture source changes.

| Module | Responsibility |
|---|---|
| `capture/` | **Live:** camera frames (picamera2/OpenCV), IMU (I²C), GPS (NMEA/gpsd, or the AIS-140 feed). **Replay:** a video file + sensor CSV aligned by a time offset, played at 1× or faster. Everything is stamped with one clock (GPS time). |
| `runners/` | One adapter per model that returns plain dataclasses. It hides the quirks listed in the spec: S1/S2 bound by tensor name, the S2 interleaved layout, V1 severity normalised by frame area, class `missing_zebra` renamed `zebra_crossing`, and V2 class 11 (ego vehicle) dropped. |
| `aggregators/` | Turns per-frame detections into events (rules below). |
| `outbox.py` | SQLite queue: `event_id` (primary key), payload, attempts, and a sent flag. Survives reboots and no-signal areas. |
| `uploader.py` | Every 5 s, POSTs up to 200 events. Exponential backoff. Uploads media first, then events. |
| `health.py` | Every 60 s: CPU temperature, FPS per runner, outbox depth, bytes sent. |

### Rates and aggregation rules

These are the rates the edge agent implements ([edge/](../edge/)). The reasoning is in [edge-deployment.md §2](edge-deployment.md).

| Source | Runs | Emits |
|---|---|---|
| V2 + ByteTrack | 3–4 FPS; 2 FPS while stopped or hot | `traffic_sample` every **10 s or 100 m**: mean and max counts per class, unique track IDs, bus speed |
| V3 | once per V2 `traffic sign` track, when it ends, on its best 3 crops (box area × sharpness) at 320 px | `sign_condition` **once per track**: majority condition, vote counts, JPEG crop |
| S2 (or a jerk threshold while S2's file is missing) | every 10 IMU samples | `road_shock` when the smoothed class is ≥ 2; at most one per 15 m |
| V1 | on 3 ring-buffer frames, 1.5, 1.0 and 0.6 s before each jolt. No continuous scan on a Pi 5 CPU (optional with a GPU or AI HAT+). | `pothole` (best box per jolt, located where the wheel hit it) and `zebra_crossing` (once per 30 m), each with a crop |
| S1 | every 100 m of travel (stride 100 m) | `iri_window` with start and end coordinates |
| Geofence + V2 persons | every frame, only inside a school polygon | `pedestrian_zone_alert` at most once per school per 5 min, **with no image** |
| GPS | 1 Hz | `telemetry` points, sent in batches |

**Privacy.** Before any crop leaves the device, blur faces and number plates (a small face/plate detector, or plain Gaussian blur over V2 `person`/`rider` boxes inside the crop). Pedestrian alerts never carry an image. This matters under the DPDP Act 2023, and because school-zone footage shows children.

---

## 3. Edge → backend data contract

### Envelope
`POST /api/v1/ingest`, `Content-Type: application/json`, header `X-Device-Key: <key>`.

```json
{
  "schemaVersion": "1.0",
  "deviceId": "edge-dl1pc0421",
  "vehicleId": "DL1PC0421",
  "tripId": "DL1PC0421-20261210T080211Z",
  "sentAt": "2026-12-10T08:14:03.120Z",
  "telemetry": [
    {"t": "2026-12-10T08:14:01.000Z", "lat": 28.61421, "lon": 77.21102, "speedMps": 7.4, "headingDeg": 92.0, "hdop": 0.9}
  ],
  "events": [
    {
      "eventId": "0192b7e2-5c1a-7d3e-9f10-3c2b1a0e4f55",
      "eventType": "pothole",
      "capturedAt": "2026-12-10T08:14:00.640Z",
      "location": {"latitude": 28.61408, "longitude": 77.21087, "accuracyM": 4.0},
      "headingDeg": 91.5,
      "speedMps": 7.2,
      "confidence": 0.71,
      "source": {"model": "V1-roadsense", "modelVersion": "yolov8m-800-2026-09-25", "sensor": "cam_front"},
      "metadata": {
        "bbox": [812, 604, 1011, 699], "frameSize": [1920, 1080],
        "bboxAreaFrac": 0.0091, "severity": 2,
        "trigger": "imu_shock", "imuShockClass": 3, "imuShockProb": 0.88
      },
      "mediaId": "m_7f3a…"
    }
  ]
}
```

**Rules:**
- `eventId` is a UUIDv7 generated at the edge. The backend ignores a duplicate `eventId`, so retries are safe.
- `capturedAt` is the capture time, not the send time.
- Keeping `eventType`, `confidence`, `vehicleId`, `location` and `metadata` makes this a **superset of the RoadSense payload**. The legacy `POST /api/events` (single event, `timestamp` field) is still accepted and mapped: `timestamp` becomes `capturedAt`, and the backend generates an `eventId`.
- **Media:** `POST /api/v1/media` (multipart, JPEG ≤ 60 KB, crop of the box plus a 20% margin, already blurred) returns `{mediaId}`. The edge uploads media before the events that reference it.
- **Response:** `202 {"accepted": n, "duplicates": m, "rejected": [{"eventId", "reason"}]}`.

### Event types and their `metadata`

| `eventType` | Emitted by | `metadata` fields | Media |
|---|---|---|---|
| `pothole` | V1 (camera) | `bbox`, `frameSize`, `bboxAreaFrac`, `severity` (1–3 by area fraction), `trigger` (`imu_shock` / `sweep`), `imuShockClass?`, `imuShockProb?` | crop |
| `zebra_crossing` | V1 class 1 | `bbox`, `frameSize` | crop |
| `sign_condition` | V2 → V3 | `condition` (`damaged` / `good`), `votes` `{damaged, good}`, `framesSeen`, `trackId`, `bbox`, `frameSize` | crop |
| `road_shock` | S2 | `class` (0–3), `label`, `probs[4]`, `smoothed` (bool) | — |
| `iri_window` | S1 | `start` `{lat, lon}`, `end` `{lat, lon}`, `lengthM`, `iri`, `iriRaw`, `iriClass` (`good` / `fair` / `poor` / `very_poor`), `meanSpeedMps`, `nSamples` | — |
| `traffic_sample` | V2 | `windowS`, `distanceM`, `meanCounts` `{car, bus, truck, autorickshaw, motorcycle, bicycle, person, rider}`, `maxCounts`, `uniqueTracks`, `busSpeedMps` | — |
| `pedestrian_zone_alert` | V2 + geofence | `level` (`ADVISORY` / `CRITICAL`), `schoolOsmId`, `schoolName`, `pedestrianCount`, `riderCount`, `schoolHoursActive` | **never** |
| `device_health` | edge | `cpuTempC`, `fps` `{V1, V2, V3}`, `outboxDepth`, `bytesSentToday`, `cameraSecondsToday` | — |

**Sizes.** A typical event is 0.5–1.5 KB of JSON; a crop is 20–60 KB. For comparison, one 1080p camera at about 4 Mb/s is about 1.8 GB per hour. The dashboard's bandwidth KPI is `cameraSeconds × 4 Mb/s ÷ 8` against the bytes actually sent.

---

## 4. Backend

### Stack
| Concern | Choice | Why |
|---|---|---|
| API | **FastAPI** (Python 3.11), Uvicorn | Same language as the ML code; async; built-in WebSocket; automatic OpenAPI docs to show judges |
| DB | **PostgreSQL 16 + PostGIS 3** | The roadmap names PostGIS; distance and point-in-polygon queries in SQL |
| ORM / migrations | SQLAlchemy 2 + GeoAlchemy2 + Alembic | |
| Spatial grid | `h3` Python library (resolution 10 ≈ 66 m edge; 11 ≈ 25 m), stored as a `bigint` column | Heatmaps and grouping without map-matching or a DB extension |
| Workers | An in-process asyncio loop every 10 s (upgrade to `arq` + Redis only if needed) | One less service on demo day |
| Media | Local disk `media/YYYY/MM/DD/<id>.jpg`, served by FastAPI | MinIO is optional |
| OSM data | Overpass, fetched once per city by a script: schools (`amenity=school`), marked crossings (`highway=crossing` with `crossing=marked/zebra` or `crossing:markings`), signals | Geofence and "expected" assets |
| Packaging | `docker compose` with `db`, `api`, `dashboard` | Whole demo runs offline on one laptop |

### Database schema (sketch)

```sql
CREATE TABLE vehicles (id text PRIMARY KEY, reg_no text, route_id text, device_key_hash text, last_seen timestamptz);
CREATE TABLE trips    (id text PRIMARY KEY, vehicle_id text REFERENCES vehicles, started_at timestamptz, ended_at timestamptz);

CREATE TABLE telemetry (
  vehicle_id text, trip_id text, t timestamptz, geom geography(Point,4326),
  speed_mps real, heading_deg real, PRIMARY KEY (vehicle_id, t));

CREATE TABLE events (                              -- raw, append-only
  event_id uuid PRIMARY KEY, vehicle_id text, trip_id text, event_type text,
  captured_at timestamptz, received_at timestamptz DEFAULT now(),
  geom geography(Point,4326), h3_r11 bigint, heading_deg real, speed_mps real,
  confidence real, model text, model_version text, metadata jsonb,
  media_id text, issue_id bigint);
CREATE INDEX ON events USING gist (geom);
CREATE INDEX ON events (event_type, captured_at);

CREATE TABLE issues (                              -- de-duplicated real-world things
  id bigserial PRIMARY KEY, issue_type text,       -- pothole | zebra_crossing | sign | zebra_expected_not_seen | sign_possibly_missing
  status text,                                     -- candidate | probable | verified | rejected | resolved
  geom geography(Point,4326), heading_deg real,
  first_seen timestamptz, last_seen timestamptz,
  n_events int, n_trips int, n_vehicles int,
  evidence jsonb,                                  -- {"camera": 4, "imu": 3, "fused": 2}
  confidence real, severity smallint, attrs jsonb, -- e.g. {"condition": "damaged", "votes": {...}}
  cover_media_id text, reviewed_by text, reviewed_at timestamptz, review_note text);
CREATE INDEX ON issues USING gist (geom);

CREATE TABLE iri_windows (
  id bigserial PRIMARY KEY, event_id uuid, vehicle_id text, trip_id text, captured_at timestamptz,
  geom geography(LineString,4326), iri real, iri_raw real, iri_class text, mean_speed_mps real);

CREATE TABLE traffic_samples (
  id bigserial PRIMARY KEY, event_id uuid, vehicle_id text, captured_at timestamptz,
  geom geography(Point,4326), h3_r10 bigint, counts jsonb, density_index real, bus_speed_mps real);

CREATE TABLE school_zones (osm_id bigint PRIMARY KEY, name text, geom geography(Polygon,4326));
CREATE TABLE expected_assets (osm_id bigint PRIMARY KEY, asset_type text, geom geography(Point,4326));
CREATE TABLE alerts (
  id bigserial PRIMARY KEY, alert_type text, level text, geom geography(Point,4326),
  school_osm_id bigint, vehicle_id text, created_at timestamptz, payload jsonb,
  ack_by text, ack_at timestamptz);
CREATE TABLE media (id text PRIMARY KEY, path text, bytes int, sha256 text, created_at timestamptz);
```

### Processing rules (the workers)

All rules are idempotent and run every 10 s over events not yet processed.

**1. Issue clustering.** For each new `pothole`, `zebra_crossing` or `sign_condition` event:
- Look for an open issue of the same type within a radius R: pothole 10 m; zebra crossing 20 m; sign 15 m **and** heading within ±45°, so signs facing opposite carriageways stay apart.
  ```sql
  … WHERE ST_DWithin(geom, $pt, R) …
  ```
- Attach the event to it, or create a new issue.
- Update `n_events`, `n_trips`, `last_seen`, and a running confidence-weighted centroid.

**2. Pothole status and confidence.** Rules, not a probability formula: frames from one pass aren't independent, so they aren't multiplied.

| Status | Condition |
|---|---|
| `candidate` | 1 pass, camera only or IMU only |
| `probable` | camera and IMU evidence within 15 m and 3 s on the same pass, **or** seen on ≥ 2 distinct trips |
| `verified` / `rejected` / `resolved` | set by a human in the review queue |

- Displayed confidence = max camera confidence × a pass factor (1 pass 0.6, 2 passes 0.85, ≥ 3 passes 1.0).
- Also shown: the raw evidence counts.
- A `road_shock` with no nearby camera event creates a pothole `candidate` with `evidence.imu` only. This covers night-time and blocked views.

**3. Pothole severity.** Take the maximum of:
- camera severity, from box area as a fraction of the frame (edge thresholds, tuned on your footage);
- IMU severity: S2 `Big Pothole` gives 3, `Med Pothole` gives 2;
- the containing S1 window's `iriClass` (poor gives 2, very poor gives 3).

**4. Sign issues.**
- `attrs.condition` is the majority of per-pass conditions.
- If the sign is `damaged` on ≥ 2 passes, it becomes `probable`.
- **Possibly missing** (needs repeated drives): a sign issue seen on ≥ 3 earlier passes but not seen on the last 2 passes, where the bus did drive past (telemetry within 30 m, same heading) → create `sign_possibly_missing` (`candidate`).

**5. Zebra crossings.**
- Visible crossings become `zebra_crossing` issues (an inventory, not a defect).
- **Expected but not seen:** an OSM marked crossing that the fleet passed ≥ 3 times (within 25 m, while the camera was on) with no `zebra_crossing` detection within 25 m → `zebra_expected_not_seen` (`candidate`).
- This is how the PS's "missing zebra crossing" is answered honestly.

**6. IRI road-condition layer.**
- Store each `iri_window` as a LineString taken from the bus's own GPS trace between the window's start and end. **No map-matching is needed**: the road is drawn where the bus drove.
- For the aggregate view, group windows by H3 r10 cell of their midpoint. Take the median IRI over the selected dates and show `n_passes`.

**7. Traffic density and bottlenecks** (replaces V2's pixel-speed stall logic).
- `density_index` = mean vehicles per frame, weighted by passenger-car units (PCU), plus `person` / `rider` counts kept separately.
- Keep a grid: (H3 r10 cell × 15-minute bucket) → median `density_index`, median `bus_speed_mps`.
- **Free-flow speed per cell:** the 85th-percentile bus speed in that cell during off-peak hours. Until there's enough history, use the posted speed limit or 40 km/h.
- **Bottleneck** when all of these hold:
  - bus speed < 40% of free-flow speed;
  - density index ≥ the 75th percentile for the city;
  - seen in ≥ 2 buckets or by ≥ 2 vehicles.
- Rank bottlenecks by (free-flow − observed speed) × density.
- Exclude bus stops: stopped time within 30 m of a known stop or depot, when available.

**8. School-zone alerts.**
- The server re-checks each alert: the point must be in `school_zones`, and school hours come from a configuration table (default 07:30–09:30 and 13:30–15:30 IST, Monday–Saturday).
- `CRITICAL` if school hours are on and `pedestrianCount ≥ 1`; otherwise `ADVISORY`.
- Riders are shown but don't escalate the level.
- New alerts are pushed on the WebSocket and need an operator acknowledgement.

**9. Route delay and origin–destination (O–D) analysis.**
- **Buildable now:** segment travel times from telemetry, compared with the median for the same hour-of-week. Route delay is the sum over the route's cells.
- **O–D analysis needs passenger boarding data** (ticketing / electronic ticketing machines, or GTFS with automatic passenger counts), which the fleet sensors don't provide. Show it as a roadmap slide, not a fake chart.

### API

**Ingest (edge-facing):**

| Method and path | Body / query | Returns |
|---|---|---|
| `POST /api/v1/ingest` | envelope (§3) | `202 {accepted, duplicates, rejected}` |
| `POST /api/v1/media` | multipart JPEG | `{mediaId}` |
| `POST /api/events` | legacy RoadSense single event | `200 {status}` |

**Dashboard-facing** (all map endpoints return GeoJSON `FeatureCollection` and take `bbox=minLon,minLat,maxLon,maxLat&from&to`):

| Method and path | Purpose |
|---|---|
| `GET /api/v1/vehicles` · `GET /api/v1/vehicles/{id}/track` | Fleet list; GPS trail |
| `GET /api/v1/issues?type&status&minConfidence` | Issue points |
| `GET /api/v1/issues/{id}` | Issue with its events, media URLs and history |
| `PATCH /api/v1/issues/{id}` | `{status, note}`: human verify, reject or resolve |
| `GET /api/v1/road/iri?mode=lines\|cells` | IRI polylines or H3 cells |
| `GET /api/v1/traffic/grid?hour=&dow=` | H3 cells with density and speed |
| `GET /api/v1/traffic/bottlenecks` | Ranked list |
| `GET /api/v1/alerts?level&ack=false` · `POST /api/v1/alerts/{id}/ack` | School-zone alerts |
| `GET /api/v1/kpis` | Kilometres surveyed, open issues by type, alerts today, bytes sent vs raw-video equivalent |
| `GET /api/v1/export/issues.{geojson,csv}` | Work-order export for the municipality |
| `WS /ws/live` | Messages `{type: "vehicle_position" \| "event" \| "issue_upsert" \| "alert" \| "kpi", data}` |

**Auth for the demo:** a per-device API key for ingest, and a single admin login for the dashboard (a JWT in an HTTP-only cookie). Serve over HTTPS if anything leaves the laptop.

---

## 5. Dashboard

**Stack:**
- React 18 + Vite + TypeScript.
- **MapLibre GL JS** as the base map, with a free vector style (e.g. OpenFreeMap) or cached OSM tiles for offline use. **deck.gl** layers on top.
- TanStack Query for REST, a small WebSocket store for live data, Recharts for charts, Tailwind + shadcn/ui for components.

### Pages

**1. Command Center** (landing page, the one judges see first)
- Live map: bus icons with 2-minute trails, new events popping in and fading after 10 s.
- KPI strip:
  - kilometres surveyed today;
  - open issues by type;
  - active alerts;
  - **"uplink: 38 MB vs 21 GB raw video (99.8% saved)"**.
- Live event feed: thumbnail, type, confidence chip, model name, time and location. Clicking an item flies the map to it.
- Alert toasts for `CRITICAL` school-zone alerts, with an acknowledge button.

**2. Road Health**
- Layers:
  - IRI polylines coloured Good / Fair / Poor / Very poor (< 4, 4–8, 8–14, ≥ 14 m/km);
  - pothole issues sized by severity and outlined by status;
  - a toggle for the evidence source (camera / IMU / both).
- Issue drawer:
  - crops over time (every pass), evidence counts;
  - the S2 shock class and the S1 IRI of the containing window;
  - model and version, confidence;
  - buttons: **Verify**, **Reject**, **Mark repaired**.
- Segment chart: IRI along the route (distance on x, IRI on y), with pothole markers.

**3. Infrastructure**
- Signs (damaged / good / possibly missing).
- Zebra crossings (seen / expected-not-seen).
- A **coverage layer** showing where the fleet has driven and how often, so "not seen" is never confused with "not driven".

**4. Traffic**
- H3 congestion heatmap with an hour-of-day and day-of-week slider.
- Bottleneck leaderboard, with the bus speed vs free-flow speed sparkline for each.
- Class-composition bar chart for the selected cell (cars, autos, two-wheelers, buses, trucks).

**5. Safety**
- School polygons, coloured by the number of alerts in the chosen period.
- Alert timeline.
- Pedestrian density during school hours.
- **No images** anywhere on this page.

**6. Review Queue**
- Table of `candidate` and `probable` issues, sorted by severity × confidence.
- Keyboard shortcuts: V verify, R reject, → next.
- Shows the human-in-the-loop step the PS implies with "confidence-scored reports".

**7. Fleet & Edge**
- Per device: last seen, CPU temperature, FPS per model, outbox depth, bytes per day.
- A "models deployed" table: name, version, size, runtime.

### Presentation rules
- Every map item shows **confidence, source model, and number of passes**. Nothing appears without provenance.
- Use one colour scale everywhere: sequential for IRI and density, categorical for issue types, and status shown as the outline style. Check it for colour-blind safety.
- Anything simulated (demo clock, replayed ride, OSM-derived "expected" assets) carries a visible **REPLAY / DEMO CLOCK** badge.

---

## 6. Demo plan

### Data to prepare (October–November)
- **Record your own rides.**
  - Phone or Pi with a camera (1080p, 15–30 FPS) + IMU (100 Hz) + GPS (1–10 Hz) on one clock.
  - Mount it at bus height if possible.
  - Drive **the same 5–10 km route 2–3 times**, so de-duplication, "probable" promotion and "expected-not-seen" have repeated passes to work with.
  - Include a school at opening or closing time, a few known potholes, and a stretch with damaged signs.
- **Fallback.** The existing `KMP_mathura.csv` and `delhi_mumbai_expressway.csv` logs (IMU + GPS at 100 Hz, no video) can drive the IRI and shock layers. Don't attach unrelated dashcam video to a real GPS track without labelling it as a simulation.
- **OSM extracts** for the demo city: schools and marked crossings, loaded once by `scripts/load_osm.py`.

### Stage script (about 6 minutes)
1. **Start the replay** at 1× (or 4×). The bus moves on the Command Center map; events stream in; the bandwidth counter climbs slowly next to the raw-video number.
2. **Pothole.** The IMU shock (S2) triggers V1 on the buffered frames; a camera-and-IMU event appears as `probable`. Open the drawer: crop, shock class, IRI window. **Verify** it, then export the work-order CSV.
3. **Road Health page.** The IRI-coloured route, and the IRI-along-route chart.
4. **Infrastructure page.** A damaged sign (V3) with vote counts; one OSM crossing marked "expected but not seen", next to the coverage layer that shows it was passed three times.
5. **Safety page.** A school-zone `CRITICAL` alert, on real time or a clearly labelled demo clock. Point out that no images are kept.
6. **Traffic page.** Heatmap and bottleneck ranking, and how it uses bus GPS speed rather than pixel motion.
7. **Fleet & Edge page.** The Pi's CPU temperature and per-model FPS: this is the "intelligent edge processing" slide, made live.

### Making it fail-safe
- The whole stack is local (`docker compose up`); no internet is needed (cached tiles).
- `make demo-reset` restores a DB snapshot taken just before the replay; `make demo-seed` pre-loads history for the heatmaps.
- Replay speed can be controlled from an admin menu; a recorded screen video is the last-resort backup.

---

## 7. Repository layout

```
Vidur/
├── ML_models/                 # as delivered (read-only for the app)
├── edge/                      # built (28 Sep)
│   ├── capture/ (replay.py, ring_buffer.py)                 # live.py (picamera2, I²C IMU, gpsd) to follow
│   ├── runners/ (v1_roadsense.py, v2_idd.py, v3_signs.py, s1_iri.py, s2_shock.py, yolo.py, tflite.py)
│   ├── aggregators/ (traffic.py, signs.py, potholes.py, school_zone.py)
│   ├── agent.py  deferred.py  governor.py  imu.py  outbox.py  health.py  hud.py
│   ├── config.yaml  config.pi5.yaml  scripts/ (export_models.py, benchmark.py)
│   └── tests/ (model input contracts, queue policies, aggregators, end-to-end replay)
├── backend/                   # to build
│   ├── app/ (main.py, api/, schemas/ (pydantic, mirrors §3), db/, workers/, ws.py)
│   ├── alembic/   scripts/load_osm.py   tests/
├── dashboard/  (Vite React TS: pages/, layers/, api/, ws/)       # to build
├── infra/docker-compose.yml   Makefile                           # to build
└── docs/
```

The edge agent writes each upload batch as one line of `envelopes.jsonl`, in exactly the §3 envelope format. An `uploader.py` that POSTs those lines with retries is the remaining edge piece, and it is only needed once the backend exists.

Write `backend/app/schemas` (pydantic) first and generate the TypeScript types from FastAPI's OpenAPI (`openapi-typescript`), so the edge, backend and dashboard can't drift apart.

---

## 8. Build plan

*Status 28 Sep: most of the edge column through November week 1–2 is already built: replay capture, all five runners with tests, the IMU-triggered V1, the V3 cascade with per-track votes, the traffic summariser, the school geofence and the NCNN exports. Still open on the edge: live capture, the uploader and the recorded rides. The backend and dashboard have not been started.*

| When | Edge | Backend | Dashboard |
|---|---|---|---|
| **By 30 Sep (PPT)** | — | — | Architecture diagram (§1) + screenshots of the existing IRI maps and model outputs |
| **Oct wk 1** | Replay capture; V2 and S1 runners with golden tests | Envelope schema, `/ingest`, `events` and `telemetry` tables, `docker compose` | Map with GPS trail and raw event points |
| **Oct wk 2** | V1 (IMU-triggered) and S2 runners; outbox and uploader | Media upload; WebSocket; `iri_windows` | Command Center feed and KPIs; IRI layer |
| **Oct wk 3** | V3 cascade and per-track votes | Issue clustering and pothole status; review endpoints | Road Health drawer; Review Queue |
| **Oct wk 4** | Traffic summariser; school geofence from OSM | Traffic grid and bottlenecks; alerts; OSM loader | Traffic and Safety pages |
| **Nov wk 1–2** | **Record 2–3 passes of the demo route**; Pi deployment (NCNN exports), health | Expected-not-seen and possibly-missing logic; export | Infrastructure page and coverage layer; Fleet page |
| **Nov wk 3–4** | Fix the spec's §10 model issues; tune thresholds on your footage | Demo seed and reset; load test (simulate 50 buses) | Polish, badges, rehearsal |
| **Dec (finale)** | Integration only | Integration only | Integration only |

### If time runs short, cut in this order (last row first)
- **Must:** ingest, events, the IRI layer, pothole issues with review, Command Center live feed, the bandwidth KPI.
- **Should:** signs, school-zone alerts, traffic heatmap.
- **Could:** bottleneck ranking, expected-not-seen, possibly-missing, the Fleet page.

---

## 9. Risks and decisions

| Risk / decision | Mitigation |
|---|---|
| The models haven't been validated on bus footage (spec §0 finding 7) | Record your own rides early (Nov wk 1 at the latest). Quote accuracy only on that footage. |
| The Pi can't run V1 in real time | IMU-triggered V1 only; or a laptop edge for the demo, stated openly |
| Clock skew between camera, IMU and GPS | Stamp everything with one monotonic clock plus a GPS-time offset. Reject events with `capturedAt` > 5 min in the future. |
| GPS noise near flyovers and in urban canyons | Send `hdop` / `accuracyM`; widen cluster radii when accuracy is poor; ignore points with accuracy > 25 m for issue creation |
| Privacy (faces, plates, children) | Blur on the edge; no pedestrian images; keep raw events 90 days and issues indefinitely |
| Map-matching complexity | Deliberately skipped: draw on the GPS trace and group with H3. Map-matching (OSRM/Valhalla) goes on the roadmap slide. |
| AGPL-3.0 (Ultralytics) | Fine for an open hackathon repo. A BEL product would need an Ultralytics enterprise licence or detectors under a permissive licence such as Apache-2.0. |
| Scale question from judges | Around 1,000 buses × about 1 event every 10 s is about 100 inserts/s: easy for one Postgres. Partition `telemetry` and `events` by day. Workers scale horizontally by H3 region. |
