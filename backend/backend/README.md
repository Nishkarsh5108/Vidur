# FleetSense Backend

FastAPI + MongoDB backend for the SIH urban-intelligence platform. It **receives AI events** from an edge
device (today: a laptop simulating the bus computer), **stores them without duplicates**, **groups nearby
detections into persistent issues**, and serves **REST, GeoJSON and WebSocket** data to the [ops dashboard](../../dashboard/README.md)
it also serves as static files, at `/`.

It does not itself run ML or process video — that happens in [edge/](../../edge/README.md). The one exception
is `/api/v1/simulate/*` (§7): the dashboard's "Simulate" button asks this API to *launch* the edge's own
`tools/sim/run_fleet.py` as a subprocess, so a hackathon judge can start the demo fleet from one page — the
backend supervises that process, it doesn't run any model itself.

```
Edge (video.mp4 + location.csv → sync → 5 ML models → event JSON)
        │  POST /api/v1/ingest   (X-Device-Key)         │  POST /api/v1/media (a JPEG crop)
        ▼                                                ▼
FastAPI ─ validate ─ store (events) ─ aggregate (issues) ─ update (vehicles, trips)
        │                         MongoDB: fleet_sense              media_store/ (JPEGs, by content hash)
        ├─ REST + GeoJSON + /simulate/*  → dashboard (served at "/" by this same app)
        └─ WS /ws/live                   → dashboard (live)
```

## 1. Setup (Windows PowerShell; bash equivalents in comments)

```powershell
cd backend/backend

# MongoDB, choose ONE option:
docker compose up -d                 # a) Docker
# b) a local MongoDB Community install listening on localhost:27017

python -m venv .venv
.\.venv\Scripts\Activate.ps1         # bash: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env               # bash: cp .env.example .env

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Dashboard:** http://localhost:8000/ (only appears once `dashboard/` exists at the repository root; otherwise this route falls back to a small JSON index)
- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health
- `--host 0.0.0.0` lets the friend's laptop reach this one. Use this laptop's LAN IP (from `ipconfig`) and allow port 8000 in Windows Firewall.
- To use the "Simulate" button, `SIMULATE_PYTHON` (in `.env`, defaults to whatever Python is running this server) must point at a Python that has `edge/requirements.txt` installed — the same interpreter you use to run `python -m edge` by hand. Build the demo scenario once first: `python tools/sim/build_scenario.py` (see [tools/sim/README.md](../../tools/sim/README.md)).

`.env`:
```
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=fleet_sense
DEVICE_API_KEY=demo-secret-key
```

## 2. Tests

```powershell
pytest -v                            # 40 automated tests, run against the database fleet_sense_test
python send_test_events.py           # live demo checks against the running server
```

To send real, model-shaped events instead of hand-built samples, run the actual edge pipeline against
this backend: `python -m edge --imu ... --backend http://localhost:8000` (see [edge/README.md](../../edge/README.md)),
or drive the whole demo fleet at once with [tools/sim/run_fleet.py](../../tools/sim/README.md).

The pytest suite needs MongoDB running. It uses a separate database, `fleet_sense_test`, so it never touches the demo data.

| Required test | Where |
|---|---|
| pothole, traffic_density, traffic_sign, pedestrian_alert | `test_ingest.py::test_four_sample_event_types_are_accepted`, `send_test_events.py` step 2 |
| duplicate event | `test_duplicate_event_is_stored_once`, step 3 |
| invalid latitude / longitude / confidence, missing eventType | `test_invalid_event_is_rejected_without_failing_the_batch`, step 4 |
| nearby potholes → same issue | `test_issues.py::test_nearby_potholes_join_the_same_issue`, step 6 |
| distant potholes → separate issues | `test_distant_potholes_create_separate_issues`, step 7 |
| same issue from different vehicles | `test_same_pothole_seen_by_two_vehicles`, step 8 |
| auth, PATCH, KPIs (including an empty database), GeoJSON, WebSocket | `test_ingest.py`, `test_issues.py`, `test_api.py` |
| media upload/fetch, content-addressing, auth, path traversal | `test_media.py` |
| bottleneck detection, the "different trip" rule for missing signs | `test_analytics_extra.py` |
| simulate start/stop/status, one-at-a-time guard | `test_simulate.py` (subprocess mocked — doesn't launch the real edge stack) |

To reset the demo data: `mongosh fleet_sense --eval "db.dropDatabase()"`, then restart uvicorn so the indexes are recreated.

## 3. Event contract: `POST /api/v1/ingest`

Header: `X-Device-Key: demo-secret-key`

```json
{
  "schemaVersion": "1.0",
  "deviceId": "EDGE_DEMO_01",
  "vehicleId": "BUS_DEMO_01",
  "tripId": "TRIP_001",
  "events": [{
    "eventId": "3f6c1c1e-8d0e-4b8e-9d6a-1f7a1c2b3d4e",
    "eventType": "pothole",
    "capturedAt": "2026-09-28T10:30:35Z",
    "location": {"latitude": 28.6139, "longitude": 77.2090},
    "confidence": 0.91,
    "model": "road_model",
    "modelVersion": "1.0",
    "metadata": {"severity": 2}
  }]
}
```

Response `200`:

```json
{
  "accepted": 1,
  "duplicates": 0,
  "rejected": 0,
  "results": [
    {"index": 0, "eventId": "3f6c…", "status": "accepted", "reason": null, "issueId": "6aba83…"}
  ]
}
```

**Validation rules**
- Required: `eventId`, `eventType`, `capturedAt`, `location` and `model`.
- Ranges: latitude −90…90, longitude −180…180, confidence 0…1.
- `confidence` is optional because measurement events such as IRI or traffic counts don't have one. `metadata` is optional.
- `capturedAt` more than 5 minutes in the future is rejected.

**Error handling**
- An invalid event is reported as `rejected` with a `reason`; the other events in the batch are still stored.
- A wrong or missing key returns `401`.
- A malformed envelope returns `422`: missing `deviceId` or `vehicleId`, empty `events`, or more than 500 events.

**Idempotency.** `eventId` has a unique index, so an event sent again is reported as `duplicate` and never stored twice. The edge should keep the same `eventId` when it retries.

**Event types.** Any lowercase string is accepted. The ones used today are `pothole`, `road_quality`, `road_damage`, `traffic_density`, `traffic_bottleneck`, `traffic_sign`, `zebra_crossing`, `infrastructure_issue`, `pedestrian_alert`, `incident` and `device_health`. Anything model-specific goes in `metadata`. Examples:
- traffic: `{"vehicleCount": 23, "carCount": 12, "busCount": 2, "truckCount": 3, "twoWheelerCount": 6}`
- IRI: `{"iri": 8.4, "iriClass": "poor"}`
- sign: `{"condition": "damaged"}`
- a photo (pothole, sign, zebra crossing): first `POST /api/v1/media` (multipart, auth) to get `{"mediaId": "<hash>.jpg"}`, then put `{"mediaId": "<hash>.jpg", "mediaUrl": "/api/v1/media/<hash>.jpg"}` in the event's `metadata`. The dashboard displays it via `mediaUrl` directly; there's no separate lookup.

## 4. Issue aggregation rule

These are configured in `ISSUE_RULES` in `app/core/config.py`:

| type | same issue if within | and last seen within |
|---|---|---|
| pothole, road_damage | 15 m | 30 days |
| traffic_sign, infrastructure_issue | 20 m | 30 days |
| zebra_crossing | 25 m | 30 days |
| incident | 50 m | 6 h |
| traffic_bottleneck | 100 m | 2 h |

**Matching**
- A new event joins the **nearest open issue** (`candidate`, `probable` or `verified`) of the same type that meets both limits. MongoDB finds it with a `$nearSphere` query on the 2dsphere index.
- If there's no match, the event creates a new `candidate` issue.

**Issue fields**
- `eventCount`: the number of events attached.
- `vehicleCount` and `tripCount`: the number of distinct vehicles and trips that saw it.
- `firstSeen` and `lastSeen`: the earliest and latest capture times.
- `severity`: the highest `metadata.severity` seen.
- `confidence`: the best event confidence × a pass factor (1 trip 0.6, 2 trips 0.85, 3 or more 1.0). Frames from one pass aren't independent evidence, but repeated passes are.
- Traffic signs also keep `metadata.conditionVotes`, for example `{"damaged": 2, "good": 1}`.

**Status**
- The backend promotes `candidate` to `probable` automatically once the issue is seen on 2 or more trips or by 2 or more vehicles.
- `verified`, `rejected` and `resolved` are set only by a person, through `PATCH /api/v1/issues/{id}`.
- Rejected and resolved issues are never matched again, so a pothole that reappears after repair opens a new issue.

Other event types (`traffic_density`, `road_quality`, `pedestrian_alert`, `device_health`) stay as events and feed `/kpis`, `/road-health` and `/traffic`.

### 4a. Traffic bottlenecks (computed on demand, not stored)

The full design (`docs/dashboard-backend-design.md` §4) buckets by H3 cell and compares against an
85th-percentile *free-flow* speed learned over many days of history. This MVP has neither an H3
dependency nor days of history, so `GET /api/v1/traffic/bottlenecks` approximates both from the
traffic this session has actually seen: a plain lat/lon grid (~120 m cells) stands in for H3, and this
session's own 85th-percentile cell speed stands in for a long-run free-flow baseline. A cell is flagged
when its mean bus speed is under 40% of that baseline *and* its density is at or above the 75th
percentile across cells, with at least 2 samples. Every response repeats this as a `note` field —
the intent is that the dashboard always shows the caveat next to the number, not that the number is
wrong to show.

### 4b. Possibly-missing signboards (computed on demand, not stored)

`GET /api/v1/infrastructure/missing-signs` flags an open `traffic_sign` issue only once a **different,
later trip** has driven within its matching radius (20 m) without producing an event that joined it.
"Different trip" is the important word: while the bus that first saw the sign is still driving past, it
keeps logging other events (traffic samples, IRI windows, ...) a few metres further on every few
seconds — a naive "any later nearby event" check would flag almost every sign as missing from that
alone. Requiring the nearby event's `tripId` to be one that never contributed to this issue rules that
out. On a demo where every bus drives its route once, this endpoint correctly returns none — it needs
the fleet to revisit a road, which needs either replaying the same route twice or running the real
fleet over days.

## 5. API

| Method | Path | Notes |
|---|---|---|
| POST | `/api/v1/ingest` | edge batches (auth) |
| GET | `/api/v1/events` | `eventType, vehicleId, tripId, model, issueId, from, to, limit, skip` |
| GET | `/api/v1/events/{eventId}` | |
| GET | `/api/v1/issues` | `issueType, status, minConfidence, limit, skip` |
| GET | `/api/v1/issues/{id}` | the issue plus its latest 100 events |
| PATCH | `/api/v1/issues/{id}` | `{"status": "verified", "note": "checked"}` |
| GET | `/api/v1/vehicles`, `/api/v1/vehicles/{id}` | the detail includes trips and recent events |
| GET | `/api/v1/kpis` | totals, open issues, by type/status/model, active alerts/vehicles, road and traffic summaries |
| GET | `/api/v1/road-health` | IRI stats and classes, open pothole/road-damage issues by severity |
| GET | `/api/v1/traffic?hours=24` | density averages per class, open bottlenecks, recent samples |
| GET | `/api/v1/traffic/bottlenecks?hours=24` | ranked hotspots: mean bus speed vs this session's own 85th-percentile "free-flow" speed — see the caveat in its own `note` field and in §4a |
| GET | `/api/v1/infrastructure/missing-signs` | open sign issues a *different, later trip* passed without reconfirming — empty until the fleet re-drives a route; see §4b |
| GET | `/api/v1/map/events`, `/api/v1/map/issues` | GeoJSON FeatureCollection; `bbox=minLon,minLat,maxLon,maxLat` |
| GET | `/api/v1/map/school-zones` | GeoJSON of the real OpenStreetMap school polygons `tools/sim/build_scenario.py` built; an empty FeatureCollection if that hasn't been run |
| POST | `/api/v1/media` | edge uploads one JPEG crop (auth, multipart `file`); returns `{"mediaId", "url"}`. Files are named by content hash, so a retried upload is a no-op. |
| GET | `/api/v1/media/{id}` | serves that crop (long-lived cache headers; the id is content-addressed, so it never needs to change) |
| GET \| POST | `/api/v1/simulate/status` \| `/start` \| `/stop` | supervises `tools/sim/run_fleet.py` as a subprocess — see §7 |
| GET | `/api/v1/health` | `503` if MongoDB is down |
| WS | `/ws/live` | `{"type": "event" \| "issue_update" \| "vehicle_update", "data": {...}}` |

"Recent" numbers (last 24 h, active vehicles in the last 10 min, active alerts in the last 30 min) use the server's `receivedAt`. A replayed recording with old `capturedAt` times still shows as live.

Quick WebSocket check from a browser console:

```js
const ws = new WebSocket("ws://localhost:8000/ws/live"); ws.onmessage = m => console.log(JSON.parse(m.data));
```

## 7. Running the demo fleet from the dashboard

`POST /api/v1/simulate/start` launches `tools/sim/run_fleet.py` (see [its own README](../../tools/sim/README.md))
as a background subprocess, with `--backend` pointed at `http://127.0.0.1:<this port>` — always loopback,
even if the dashboard itself was opened over a LAN IP, since the bus processes always run on this same
machine and loopback is the one address guaranteed to route back to it.

- Body (all optional): `{"rate": 1.0, "clock": "now", "only": ["BUS_IRI_01"], "maxConcurrent": 2, "stagger": 6.0, "hud": false}` — same meanings as `run_fleet.py`'s own flags.
- `409` if a simulation is already running; `422` if `tools/sim/build_scenario.py` hasn't been run yet (`SIMULATE_SCRIPT` doesn't exist).
- `GET /simulate/status` returns `{running, pid, startedAt, endedAt, returnCode, error, params, log}` — `log` is the subprocess's last ~200 output lines, so the dashboard can show real progress, not a spinner.
- `POST /simulate/stop` kills the whole process tree (`taskkill /T` on Windows), not just `run_fleet.py` itself — otherwise the bus processes it spawned would keep running.
- State lives in this process's memory only: restarting the backend forgets a running simulation (though the OS process keeps going until it finishes or is stopped by PID). That's the right amount of engineering for a laptop demo, not a production job queue.

## 8. Known limits (on purpose, for the MVP)

- `PATCH /issues` has no login; it's meant for the demo dashboard on a trusted network.
- There is one shared device key for all devices.
- Events in a batch are processed one after another. Two devices reporting the same new pothole at the same millisecond could create two issues. That is acceptable for the demo; a unique geo-cell key would fix it later.
- An issue's location is the location of its first event. It is not averaged.
- `/api/v1/simulate/*` has no auth. It runs arbitrary local subprocesses; treat it exactly like the device key — fine on a trusted demo network, not something to expose publicly.
- `/api/v1/traffic/bottlenecks` and `/api/v1/infrastructure/missing-signs` are computed fresh on every request (no caching, no stored index). Fine at demo scale; revisit if the events collection grows into the millions.
