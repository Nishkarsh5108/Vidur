# Fleet simulation

Turns the four demo videos (and the BeamNG pothole clip) into a small live fleet that streams real events into the [backend](../../backend/backend/README.md), so the dashboard has something to show without real buses.

**What's real and what's simulated.** We don't have real GPS logged with these videos, so each video is paired with a stretch of one of the team's own recorded drives around Delhi NCR (`ML_models_work/data/raw/binary_mannual_collection/*.csv`). **The road geometry, the speeds, and the school locations (from OpenStreetMap) are real.** The *pairing* of a given video to a given road is simulated — the dashcam footage was not actually recorded on that road. Every generated file says so in its own metadata, and the dashboard should label this data as a simulation, the same way the demo recordings from [dashboard-backend-design.md](../../docs/dashboard-backend-design.md) are labelled.

## How a bus gets its GPS

For **BUS_IRI_01** (the BeamNG clip, which already has synced IMU), the trail moves along the real road at the *IMU log's own speed profile*, so the distance the video appears to travel matches the distance the IRI model integrates from the IMU. This is "distance mode".

For the other three buses (no IMU), the trail replays the *real trace's own speed profile* along the real road ("time mode") — so a bus doesn't move at a constant, suspicious speed; it slows in traffic and speeds up on straights exactly as the original drive did.

`BUS_TRAFFIC_03` is deliberately matched to a stretch of its trace that passes within a real OpenStreetMap school polygon, so the school-zone pedestrian-alert logic has something to fire on.

## Files

| File | Role |
|---|---|
| `trails.py` | Loads a GPS trace, picks a stretch of road, and builds a per-frame trail (time or distance mode). Also fetches real `amenity=school` polygons from OpenStreetMap (Overpass), cached locally. |
| `demo_fleet.yaml` | Which video pairs with which trace, and any constraints (target speed, must pass a school, distance mode). |
| `build_scenario.py` | Runs the pairing once and writes the per-bus GPS CSVs, route GeoJSON, and a shared `school_zones.geojson`, into `recordings/fleet/` (not versioned — rebuild it locally). |
| `run_fleet.py` | Launches one `python -m edge` process per bus, all re-timed to "now" so the dashboard sees a live fleet, and streams their events to the backend. |

## Running it

```bash
# 1. Build the scenario once (needs internet, for OpenStreetMap; cached after the first run)
python tools/sim/build_scenario.py

# 2. Start the backend (see backend/backend/README.md), then:
python tools/sim/run_fleet.py --backend http://127.0.0.1:8000
```

Useful flags on `run_fleet.py`:
- `--rate 1` replays in real time (matches actual bus speeds); `--rate 0` (default) runs as fast as possible.
- `--clock 2026-09-29T08:20:00+05:30` starts the replay at a specific IST time — set it inside `07:30-09:30` to see the school-zone alert escalate from `ADVISORY` to `CRITICAL`.
- `--only BUS_TRAFFIC_03` runs a single bus.
- `--hud` also writes an annotated video per bus.
- `--max-concurrent` / `--stagger`: how many bus processes load their models at once, and how long to wait between starting each one. Four processes importing PyTorch/CUDA/Ultralytics at the exact same instant can exceed the Windows paging file even though steady-state memory is fine (`OSError: [WinError 1455]`); the defaults (2 concurrent, 6 s apart) avoid that on a 32 GB laptop. Lower `--max-concurrent` further on a smaller machine.

Each bus is a normal `edge/__main__.py` run under the hood — see [edge/README.md](../../edge/README.md) for what every flag does. Nothing here is special-cased: a real bus with a real GPS/IMU feed uses the exact same `edge` package.
