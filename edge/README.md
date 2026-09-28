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
```

| Option | Meaning |
|---|---|
| `--video`, `--imu` | Either one or both. Video alone gives V1–V3 without GPS; IMU alone gives S1 and S2. |
| `--video-start` / `--offset` | Aligns the video with the IMU log: the UTC time of the first frame, or its offset in seconds from the log's start |
| `--rate` | `0` (default): as fast as possible, with no dropped jobs (reproducible). `1`: real time. `4`: 4× real time. |
| `--config` | Overlays merged over `config.yaml`, e.g. `edge/config.pi5.yaml` |
| `--schools` | GeoJSON of school zones (polygons, or points turned into 240 m squares) |
| `--hud`, `--show` | Write or show the annotated video. It is labelled "REPLAY on <device> - not a Pi 5 measurement". |

**IMU/GPS log format** (the team's phone logger): `time,ax,ay,az,wx,wy,wz,latitude,longitude,altitude,speed,...`
- `time`: ISO 8601 or epoch seconds.
- Accelerations in m/s², gyro in rad/s, speed in m/s.
- Axes as in [ml-models-spec.md §2](../docs/ml-models-spec.md).

**Recording a ride for the demo.**
- Record 1080p video at 15–30 FPS on a phone or camera, while the IMU logger runs on the same mount.
- Note the video's start time, or clap or tap the mount once in view of the camera so the two can be aligned.
- Put the recordings in `recordings/` (ignored by git).

## Outputs

Each run writes to `out/run-<timestamp>/` (or `--out`):

| File | Contents |
|---|---|
| `envelopes.jsonl` | One `POST /api/v1/ingest` body per line: telemetry plus events. Can be replayed into the backend verbatim. |
| `media/*.jpg` | Crops of potholes, crossings and signs (at most 60 KB each; people blurred). Alerts never carry images. |
| `summary.json` | Per-model latency (p50/p95), deferred-lane load, event counts, and uplink bytes against the raw-video equivalent |

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
| [capture/](capture/) | Replay source (video + IMU on one clock) and the frame ring buffer |
| [outbox.py](outbox.py), [events.py](events.py) | Envelopes, events (UUIDv7 IDs, capture time, location at capture) and blurred crops |
| [scripts/](scripts/) | NCNN/Hailo export and latency benchmark |
| [tests/](tests/) | Model input contracts, queue policies, aggregators, end-to-end replay |
