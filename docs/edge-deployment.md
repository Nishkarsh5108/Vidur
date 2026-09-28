# Edge deployment on a Raspberry Pi 5

*28 Sep 2026. How the five models are quantised, how they are combined into one pipeline, what that pipeline is expected to cost on a Raspberry Pi 5, and when an AI HAT+ is worth adding. The pipeline is implemented in [edge/](../edge/); the models are described in [ml-models-spec.md](ml-models-spec.md).*

> **Status.** The pipeline runs end to end in replay on a laptop, with all five models. It has **not yet been run on a Raspberry Pi 5.** Every Pi figure below is an estimate, marked **(est.)**, derived from published Pi 5 benchmarks as described in §4.1. Laptop measurements are marked as such and are not Pi results. §7 is the plan for replacing the estimates with measurements.

---

## 0. Summary

- **Target:** one Raspberry Pi 5 (8 GB, Active Cooler, CPU only) per bus, with one front camera, an IMU and GPS. The Raspberry Pi AI HAT+ (Hailo-8) is an optional upgrade that adds cameras and headroom. The pipeline does not depend on it.
- **Only one vision model runs continuously.**
  - V2 (YOLOv8n, about 5 GFLOPs per frame) runs at 3–4 FPS.
  - V1 (YOLOv8m, about 74 GFLOPs per frame) looks only at the three buffered frames before an IMU jolt.
  - V3 runs once per physical sign, on crops.
  - S1 and S2 take about 1 ms each.
- **Estimated load: about 40–60% of the Pi 5's CPU (est.).** That is inside the 60% ceiling that leaves room for capture, networking and heat.
- **Quantisation:**
  - NCNN FP16 with rectangular input sizes on the CPU.
  - INT8 only on a Hailo accelerator.
  - S1 and S2 stay as they are.
  - The rectangular input alone cut V1's time by **36%** (measured, laptop CPU, NCNN, same detections).
- **Gating saves more than precision.** The biggest saving comes from not running a model when there is nothing new for it to see: IMU triggers, one classification per sign, lower rates when the bus is stopped.

## 1. Why the models cannot all run on every frame

Scheduled naively, as the models were delivered, the Pi 5 would need more than all of its CPU:

| Model | FLOPs per run | Est. Pi 5 time per run | Naive rate | Share of the CPU |
|---|---|---|---|---|
| V2 YOLOv8n, 640×640 | 8.7 G | 80–120 ms (est.) | 3–5 FPS | 25–60% |
| V3 YOLOv8s, 640×640, full frames | 28.6 G | 240–360 ms (est.) | ≤ 2 FPS while a sign is in view | 50–70% while in view |
| V1 YOLOv8m, 800×800 | 123 G | 1.0–1.5 s (est.) | 0.5 FPS scan + triggers | 50–75% for the scan alone |

## 2. Combining the models

### 2.1 A gated cascade, not one merged network

The five models are combined at the pipeline level. Their weights are not merged into a single network:

1. **Each training set labels only its own classes.** Road-user frames (IDD) contain unlabelled potholes, and road-damage frames (RDD) contain unlabelled cars. A merged model would learn each as background unless every dataset were first pseudo-labelled by the other models.
2. **The models want different inputs at different rates.** V1 wants the road ahead, and only when the IMU felt something. V2 wants the full frame, all the time. V3 wants a close-up crop, once per sign. One network would have to run everything at the highest rate on the full frame.
3. **Gating is the point.** A cascade can decide *not* to run its expensive stages. A merged network always pays for everything.

### 2.2 Two lanes

```
REAL-TIME LANE (never queues; always takes the newest frame)
  camera ─► V2 384×640 @ 3–4 FPS ─► ByteTrack ─► traffic summary every 10 s / 100 m
                                             ├─► school-zone alerts (GPS geofence + school hours)
                                             └─► a sign's track ends ─► its best 3 crops ─────────┐
  IMU 100 Hz ─► S2, or the jerk trigger, every 0.1 s ─► jolt ─► 3 frames from the ring buffer ───┤
            └─► S1 every 100 m of travel ─► iri_window                                           ▼
DEFERRED LANE (worker thread on its own cores; bounded priority queue)
  V3 at 320 px on sign crops  ·  V1 at 480×800 on triggered frames, biggest jolt first
```

Code:
- [edge/agent.py](../edge/agent.py) wires the lanes together.
- [edge/deferred.py](../edge/deferred.py) holds the queue and the worker.
- [edge/governor.py](../edge/governor.py) holds the rate and pause rules.

### 2.3 Rules

| Rule | Why | Setting |
|---|---|---|
| V2 runs at 3–4 FPS, and at 2 FPS while the bus is stopped or the CPU is hot | Counting needs a steady rate. A stopped bus sees the same scene, so the freed CPU drains the V1 backlog. | `realtime.v2_fps`, `v2_fps_slow` |
| One V1 job per jolt: 3 frames from 1.5, 1.0 and 0.6 s before it | At 30 km/h the pothole was 12, 8 and 5 m ahead, in view. 3 frames instead of the whole 2 s window caps the cost. | `triggers.frame_offsets_s` |
| Jolts at least 1 s and 15 m apart, and none below 1 m/s | One pothole gives one job; door slams and boarding at a stop are ignored | `triggers.min_gap_*` |
| Bounded priority queue (20 jobs): V1 triggers > V3 signs > V1 scan frames. When full, the lowest-priority job is dropped. | Under load the lane sheds the least useful work first | `deferred.max_jobs` |
| Offline replays block instead of dropping | Reproducible results when replaying faster than real time | automatic when `--rate 0` |
| Lanes pinned to separate cores, with matching NCNN thread counts | V1 never steals V2's cores. On a laptop this cut V2's median latency from 185 ms to 62 ms while V1 ran. | [config.pi5.yaml](../edge/config.pi5.yaml) |
| The deferred lane pauses inside a school zone in school hours, and above 80 °C | Pedestrian detection gets the whole CPU when it matters. The Pi throttles at 80–85 °C. | `deferred.pause_in_school_zone`, `thermal.throttle_c` |
| V3 runs once per sign track, on its 3 best crops (box area × sharpness), majority vote | One sign seen at 4 FPS would otherwise be dozens of events | `signs.*` |

**Why deferring is legitimate.** Potholes and signs become work orders for the municipality, so results a few minutes late cost nothing. Only V2 feeds anything time-sensitive. So V1 and V3 need enough average throughput, not low per-frame latency.

### 2.4 The IMU trigger without S2

S2's model files are not in this repository yet (see [ml-models-spec.md §7](ml-models-spec.md)). Until they are added, the agent uses a jerk threshold instead: peak-to-peak vertical acceleration of at least 9 m/s², with a crest factor of at least 4, in a 1.28 s window.

On the team's own phone logs this fires:
- **0.26 times per km** on the KMP expressway (110 km);
- **1.2 times per km** on the Delhi–Mumbai expressway (2.6 km of IMU coverage).

The thresholds must be re-tuned on bus rides. S2 is used automatically once its file is present (`triggers.method: auto`).

## 3. Quantisation and export

| Model | Delivered as | Raspberry Pi 5 CPU | AI HAT+ (Hailo-8) | Avoid |
|---|---|---|---|---|
| V1 YOLOv8m | `.pt` | NCNN FP16, **480×800** | INT8 HEF, calibrated on road-heavy frames from bus footage. Retraining at 640 is the safest route. | INT8 as the CPU default |
| V2 YOLOv8n | `.pt` | NCNN FP16, **384×640** | INT8 HEF | |
| V3 YOLOv8s | `.pt` | NCNN FP16, **320×320**, on sign crops | INT8 HEF, calibrated on sign crops | Full frames at 640 |
| S1 IRI | TFLite, weights already INT8 (42 KB) | Keep as is: under 1 ms per 100 m | Stays on the CPU | Full-integer INT8: nothing to gain, and a regression feeding a calibration table |
| S2 shock | TFLite FP32 / FP16 | FP32 TFLite: about 1 ms per window | Stays on the CPU | INT8: nothing to gain |

```bash
python edge/scripts/export_models.py                                   # NCNN FP16 -> exports/, for the Pi 5
python edge/scripts/export_models.py --format hailo --data calib.yaml  # AI HAT+, on x86-64 Linux only
```

**Rectangular inputs (measured).**
- A 16:9 frame squeezed into an 800×800 input is 44% padding.
- This matters for *exported* models, whose input shape is fixed at export time. A `.pt` checkpoint already letterboxes 16:9 frames to 480×800 internally. That is why the spec's suggested `yolo export ... imgsz=800` would have cost the Pi so much.
- On the laptop CPU, with NCNN FP16:

| V1 export | Median time | Detections on the same frame |
|---|---|---|
| 800×800 (square) | 523 ms | zebra crossing 0.28 |
| 480×800 (rectangular) | 332 ms (−36%) | zebra crossing 0.27 |

**FP16.**
- The Pi 5's Cortex-A76 has native FP16 arithmetic, and NCNN uses it by default. The accuracy change is negligible.
- The NCNN files are about the same size as the `.pt` files (V1 50 MB), because Ultralytics already stores checkpoints in FP16. The halving is relative to FP32.

**INT8 on the CPU.**
- Ultralytics' NCNN export is FP16-only.
- INT8 on the Pi CPU is worth one experiment, on V1 only: NCNN's own `ncnn2table` / `ncnn2int8` tools, or LiteRT INT8 with XNNPACK.
- Keep it only if it is at least 1.5× faster *and* loses at most 2 mAP50 points on your own footage.

**Hailo.**
- INT8 only: `quantize=8, data=..., name=hailo8`.
- Compile on x86-64 Linux.
- Calibrate on 500–1,000 frames of your own bus footage, not on the training sets.
- Re-tune confidence thresholds afterwards.
- Ultralytics 8.4 loads Hailo exports through the same `YOLO()` call, so the edge agent only needs a config change.

**Accuracy gate before adopting any export.**
- mAP50 on your own clips must be within 1 point of the `.pt` model for FP16, or within 2 points for INT8.
- Ultralytics' `val` needs square inputs for exported models, so validate a square copy of the export, or compare predictions directly.

## 4. CPU budget on a Raspberry Pi 5

### 4.1 How the estimates are derived

- **Throughput.** Ultralytics' published Pi 5 NCNN benchmarks ([Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)) give:
  - YOLO26n: 5.4 GFLOPs in 67 ms, about **80 GFLOP/s**;
  - YOLO26s: 20.7 GFLOPs in 173 ms, about **120 GFLOP/s** (larger networks use the cores better).
- **Scaling.** Each model is scaled by its FLOPs at its real input size, plus about 10 ms per frame for resizing, NMS and tracking.
- **CPU share** = runs per second × time per run (with all four cores).
- **Why not use the laptop?** NCNN is tuned for ARM. On the x86 laptop it was no faster than PyTorch, so laptop numbers say nothing about the Pi.

### 4.2 Budget: one front camera, CPU only (est.)

| Work | Input | Time per run (est.) | Rate | Share of the CPU (est.) |
|---|---|---|---|---|
| V2 + ByteTrack | 384×640 | 60–90 ms | 3 FPS (4 if measurements allow) | 18–27% |
| V3 | 320×320 sign crops | 60–95 ms | 3 crops per sign, about 1 sign per 10 s | 2–3% |
| V1 | 480×800 | 0.62–0.93 s | 3 frames per jolt, 1 jolt per 15 s (a very bad road) | 12–19% |
| S1 + S2 (or the jerk trigger) | IMU windows | about 1 ms | 10 per s | about 1% |
| Capture, tracking, crops, upload | | | | 5–10% |
| **Total** | | | | **≈ 38–60%** |

### 4.3 Assumptions to check

- **Jolt rate.** One per 15 s is about 11 per km at 20 km/h: a very bad city road. The expressway logs gave 0.26–1.2 per km. When jolts come faster than V1 can process them, the queue absorbs the burst and drains while the bus is stopped.
- **Heat.** A bus cabin in summer is hot, and the Pi throttles at 80–85 °C. The Active Cooler is mandatory, and the governor slows V2 and pauses V1 above 80 °C.
- **Camera.** Use a CSI camera through `picamera2`, which costs nothing to decode. Decoding 1080p MJPEG from USB cameras uses a lot of CPU, so capture USB cameras at 720p.

### 4.4 What the CPU-only setup gives up

- **No continuous V1 scan.** Potholes the wheels do not hit (other lane, road edge) are only found by a later bus that does hit them. The fix without new hardware is the nano V1 successor ([v1-successor-plan.md](v1-successor-plan.md)). A 2 FPS scan with it costs about 5–6% of the CPU (est.).
- **V2 at 3–4 FPS.** Tracker IDs switch often at bus speeds. Traffic is therefore reported as per-frame counts averaged over each 100 m stretch, not as unique vehicles.
- **One camera.**

## 5. When the AI HAT+ is worth it

| Option | Approx. price | What changes |
|---|---|---|
| **Pi 5, CPU only** (baseline) | none | Everything in §4 |
| **AI HAT+ 26 TOPS (Hailo-8)** | about $110 | See below |
| AI HAT+ 13 TOPS (Hailo-8L) | cheaper | Probably enough for one camera with V1 at 640 (est.). Tight for V1 at 800 plus a second camera. |
| AI HAT+ 2 (Hailo-10H, 40 TOPS) | about $130 | Built for LLM and vision-language workloads. A [review](https://www.cnx-software.com/2026/01/20/raspberry-pi-ai-hat-2-review-a-40-tops-ai-accelerator-tested-with-computer-vision-llm-and-vlm-workloads/) found no vision speed-up over Hailo-8. Not the right buy. |
| Raspberry Pi AI Camera (Sony IMX500) | about $70 | Runs a YOLOv8n-sized model on the sensor itself. V2 could move there and free about 20–30% of the CPU. |

**AI HAT+ 26 TOPS (Hailo-8), in detail.**
- **What changes:**
  - All three YOLO models run in INT8 on the Hailo.
  - V1 drops from about 0.7 s to tens of milliseconds (est.), which allows a continuous V1 scan and V2 at 10+ FPS.
  - A second CSI camera fits.
  - CPU load drops to about 20–30% (est.).
- **Costs:**
  - The compile step needs x86-64 Linux.
  - INT8 needs calibration.
  - V1's 800 px input is off the standard path.
  - The HAT takes the Pi's only PCIe lane, so there is no NVMe drive.

**Decision** (see [decisions.md](decisions.md)):
- The accelerator is **not** needed to make edge inference feasible. It is needed for the multi-camera coverage the problem statement describes, or for a continuous V1 scan without first building the nano successor.
- Build and measure the CPU-only baseline first. It is the fallback if the Hailo compile fails, and "runs on a bare Pi 5" is the stronger claim.
- If a HAT is bought, the 26 TOPS model is the one. Train the V1 successor as YOLOv8n, which keeps about 100% of its mAP50 in Hailo INT8 against about 93% for YOLO26n (Ultralytics' Hailo-8L figures).

## 6. Measured so far (laptop, not a Pi)

| What | Result | Where |
|---|---|---|
| Test suite: S1/S2 input contracts, queue policies, aggregators, end-to-end replay | All pass | [edge/tests](../edge/tests/) |
| End-to-end replay on the laptop GPU (RTX 4050) | All five models run in one pipeline. Events match the data contract. | `python -m edge ...` |
| NCNN FP16 exports of V1, V2 and V3 at 480×800, 384×640 and 320 | Export, load and run | [export_models.py](../edge/scripts/export_models.py) |
| Rectangular vs square V1 export (NCNN, laptop CPU) | 332 vs 523 ms, same detections | §3 |
| Real-time replay at 1×, NCNN on the laptop CPU, 2 + 2 threads | V2 median 62 ms while V1 ran alongside. No dropped jobs. | §2.3 |
| Jerk trigger on the team's field logs | 0.26 per km (KMP, 110 km); 1.2 per km (Delhi–Mumbai) | §2.4 |
| S1 on the laptop CPU | About 1 ms per 100 m window | run summary |

## 7. Validation plan on a Raspberry Pi 5

1. **Hardware:**
   - Pi 5 (8 GB) with the Active Cooler;
   - Camera Module 3 on CSI;
   - an IMU (I²C or USB) and GPS, or the bus's AIS-140 feed;
   - on the bus, a 12/24 V to 5 V 5 A converter.
2. **Setup:** Raspberry Pi OS (64-bit), `pip install -r edge/requirements.txt`, and copy `exports/` from a PC.
3. **Per model:** `python edge/scripts/benchmark.py --config edge/config.pi5.yaml`.
4. **Whole pipeline:** replay a recorded ride for 30 minutes in real time (`python -m edge --config edge/config.pi5.yaml --video ... --imu ... --rate 1`), in a warm enclosure. Watch `vcgencmd measure_temp` and `vcgencmd get_throttled`.
5. **Accept if:**
   - V2's p95 time fits its frame period;
   - the deferred queue does not grow steadily, and no V1 trigger jobs are dropped;
   - average CPU use is at most 60%;
   - the temperature stays under 80 °C after 30 minutes.
6. **Record the results:** replace every (est.) in this document with the measured value.

## 8. Claims and their evidence

| Claim | Status | Evidence |
|---|---|---|
| The pipeline fits a Raspberry Pi 5 CPU at about 40–60% load | **Estimate** | §4 |
| Rectangular inputs cut V1's cost by about a third | **Measured** (laptop CPU, NCNN) | §3 |
| Only V2 runs continuously; V1 and V3 are gated | **Implemented and tested** | [edge/agent.py](../edge/agent.py), [edge/tests](../edge/tests/) |
| The uplink carries events and JPEG crops, never video | **Implemented.** Each run's `summary.json` reports bytes against the raw-video equivalent. | [edge/outbox.py](../edge/outbox.py) |
| An AI HAT+ adds cameras and a continuous V1 scan | **Estimate** | §5 |
