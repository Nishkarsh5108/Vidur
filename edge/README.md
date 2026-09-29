# Edge agent

The program that runs on each bus. It combines the five models (V1, V2, V3, S1, S2) into one pipeline and emits **events, never video**: potholes, zebra crossings, sign conditions, road roughness, shocks, traffic samples and school-zone alerts. The same code runs live on the bus or in **replay mode** on a recorded ride, which is how the demo runs.

The design is explained in [docs/edge-deployment.md](../docs/edge-deployment.md). The event format is the data contract in [docs/dashboard-backend-design.md §3](../docs/dashboard-backend-design.md).

## How it works

```
REAL-TIME LANE (never queues; always takes the newest frame)
  camera ─► V2 384×640 @ 3–4 FPS ─► ByteTrack ─► traffic_sample every 10 s / 100 m
                                             ├─► pedestrian_zone_alert (school geofence + hours)
                                             └─► a sign's track ends ─► its best 3 crops ─────────┐
  IMU ─► S2 or jerk trigger every 0.1 s ─► jolt ─► 3 buffered frames from before it ─────────────┤
      └─► S1 every 100 m ─► iri_window                                                           ▼
DEFERRED LANE (worker thread on its own cores; bounded priority queue)
  V3 at 320 px on sign crops ─► sign_condition  ·  V1 at 480×800 on triggered frames ─► pothole, zebra_crossing
```

## Setup

```bash
conda create -n vidur python=3.11 -y && conda activate vidur
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU only
pip install -r edge/requirements.txt
python -m pytest                                                                     # from the repository root
```

## Running a replay

Run from the repository root.

```bash
# IMU only (S1 roughness + shock events) on the team's 110 km field log: about 40 s on a laptop
python -m edge --imu ML_models/IRI/iri_compliant/KMP_mathura.csv

# A recorded ride: video + IMU/GPS log, with an annotated demo video
python -m edge --video recordings/ride1.mp4 --imu recordings/ride1.csv \
               --video-start 2026-10-05T08:02:11.000Z --hud out/ride1_hud.mp4

# Replay at real time, dropping frames the way a live camera would when the agent falls behind
python -m edge --video recordings/ride1.mp4 --imu recordings/ride1.csv --video-start ... --rate 1 --show

# A video with no IMU of its own, given a separate per-frame GPS trail, streamed live to the backend
python -m edge --video bus.mp4 --gps bus_gps.csv --clock now --backend http://127.0.0.1:8000 --rate 1
```

| Option | Meaning |
|---|---|
| `--video`, `--imu`, `--gps` | Any combination. Video alone gives V1–V3 with no map location. IMU alone gives S1 and shocks. `--gps` supplies position separately (e.g. from `tools/sim`) when the video or IMU log has none of its own. |
| `--video-start` / `--offset` | Aligns the video with the IMU log or GPS trail: the UTC time of the first frame, or its offset in seconds from the log's start. Default: taken from the GPS trail if it has one, else the IMU log. |
| `--clock` | `file` (default): keep the recording's own timestamps. `now`: re-time the whole replay to start at the current moment — for a live-looking demo. Or an exact ISO 8601 time, e.g. to land inside school hours. |
| `--rate` | `0` (default): as fast as possible, with no dropped jobs (reproducible). `1`: real time. `4`: 4× real time. |
| `--config` | Overlays merged over `config.yaml`, e.g. `edge/config.pi5.yaml` or `edge/config.laptop.yaml` |
| `--schools` | GeoJSON of school zones (polygons, or points turned into squares) |
| `--backend` | A FleetSense backend URL ([backend/backend](../backend/backend/README.md)); events are POSTed to `/api/v1/ingest` in the background, with retries, alongside the local `envelopes.jsonl` |
| `--device-key`, `--vehicle-id`, `--device-id`, `--trip-id` | Identify this run to the backend. `--device-key` also reads `$VIDUR_DEVICE_KEY`. |
| `--hud`, `--show` | Write or show the annotated video. It is labelled "REPLAY on <device> - not a Pi 5 measurement". |

**IMU log format.** Either the team's phone logger — `time,ax,ay,az,wx,wy,wz,latitude,longitude,altitude,speed,...` (`time` ISO 8601 or epoch seconds/ms; position and speed from the phone's own GPS) — or a BeamNG simulator clip built by [tools/make_sim_clip.py](../tools/make_sim_clip.py) (`sample_number,sim_time,speed_ms,ax,...,wz,video_time_s,...`; no position of its own, so pair it with `--gps`). Accelerations in m/s², gyro in rad/s, speed in m/s. Axes as in [ml-models-spec.md §2](../docs/ml-models-spec.md).

**GPS trail format** (`--gps`): `time,latitude,longitude,speed_mps[,heading_deg][,video_time_s]` — one row per reading; `tools/sim` writes one row per video frame.

**Recording a ride for the demo.**
- Record 1080p video at 15–30 FPS on a phone or camera, while the IMU logger runs on the same mount.
- Note the video's start time, or clap or tap the mount once in view of the camera so the two can be aligned.
- Put the recordings in `recordings/` (ignored by git).

**No GPS to go with a video?** [tools/sim](../tools/sim/README.md) builds a per-frame GPS trail for a video by pairing it with a stretch of a *real* GPS trace (real road, real speeds, and — via OpenStreetMap — real school zones), and can run several such buses at once against the backend for a live-looking demo fleet.

## Outputs

Each run writes to `out/run-<timestamp>/` (or `--out`):

| File | Contents |
|---|---|
| `envelopes.jsonl` | One `POST /api/v1/ingest` body per line, in the edge's own contract ([docs/dashboard-backend-design.md §3](../docs/dashboard-backend-design.md)): telemetry plus events. |
| `media/*.jpg` | Crops of potholes, crossings and signs (at most 60 KB each; people blurred). Alerts never carry images. |
| `summary.json` | Per-model latency (p50/p95), deferred-lane load, event counts, and uplink bytes against the raw-video equivalent |
| `backend_log.jsonl` | Only with `--backend`: the backend's response to every batch it accepted |

With `--backend`, events are also translated to the [backend's own event contract](../backend/backend/README.md#3-event-contract-post-apiv1ingest) (a different, simpler shape — see [backend.py](backend.py)) and POSTed there in real time.

## Raspberry Pi 5

```bash
python edge/scripts/export_models.py                       # on any PC: NCNN FP16 exports -> exports/
python edge/scripts/benchmark.py --config edge/config.pi5.yaml
python -m edge --config edge/config.pi5.yaml --video ... --imu ... --rate 1
```

[config.pi5.yaml](config.pi5.yaml) switches to the NCNN exports and pins the two lanes to cores 0–1 and 2–3, with 2 threads each. Validation steps and pass criteria are in [docs/edge-deployment.md §7](../docs/edge-deployment.md).

## Code map

| Path | Role |
|---|---|
| [agent.py](agent.py) | Wires the lanes together and turns results into events |
| [deferred.py](deferred.py) | Bounded priority queue and the worker thread |
| [governor.py](governor.py) | V2 rate; pausing in school zones and when hot |
| [imu.py](imu.py) | 100 m IRI windows, shock grading and V1 triggers |
| [runners/](runners/) | One adapter per model, each hiding that model's quirks (inputs bound by name, S2's interleaved layout, V1's class rename and severity) |
| [aggregators/](aggregators/) | Traffic samples, sign tracks and votes, pothole de-duplication, school zones |
| [capture/](capture/) | Replay source (video + IMU + GPS merged on one clock) and the frame ring buffer |
| [outbox.py](outbox.py), [events.py](events.py) | Envelopes, events (UUIDv7 IDs, capture time, location at capture) and blurred crops |
| [backend.py](backend.py) | Translates events to the FleetSense backend's contract and uploads them with retries |
| [scripts/](scripts/) | NCNN/Hailo export and latency benchmark |
| [tests/](tests/) | Model input contracts, queue policies, aggregators, end-to-end replay |

See also [tools/sim](../tools/sim/README.md) for building a demo fleet against the backend when your videos have no GPS of their own.
