# FleetSense — As-Built ML Model Specifications

*Compiled 28 Sep 2026 from the code, checkpoints and exported models in `ML_models/`. Scope: the [roadmap](sih2026-ps26124-roadmap.pdf) minus hit-and-run / rash driving / ANPR.*

**How this was checked.** Everything below was read from the code and the model files themselves, not only from READMEs:

- The three YOLO checkpoints were unpickled to read their class maps, training arguments and final validation metrics.
- Every TFLite file was loaded to read tensor names, shapes and input order.
- The road-shock classifier's TFLite was run against its ONNX file to pin down the input memory layout.
- The IRI TFLite was run on the team's own field logs.

A number marked **(README)** comes only from a README or notebook text and couldn't be re-derived from the artefacts. **(est.)** marks an estimate.

The companion document [dashboard-backend-design.md](dashboard-backend-design.md) defines the event schema, backend and demo dashboard that consume these models. [edge-deployment.md](edge-deployment.md) covers how the models are quantised and combined to run on a Raspberry Pi 5; the implementation is in [edge/](../edge/).

---

## 0. Summary

| ID | Model | Folder | Sensor | What it does | Output | Format | Size | Headline metric |
|---|---|---|---|---|---|---|---|---|
| **V1** | RoadSense | `RoadSense_ML_Edge-master/` | front camera | Detects potholes and zebra crossings | boxes `pothole`, `missing_zebra`; pothole severity 1–3 from box area | Ultralytics `.pt`, YOLOv8m, 800 px | 52 MB | mAP50 **0.674** (both classes, val) |
| **V2** | IDD Traffic | `sih-2026-traffic-analytics-main/` | front camera | 13 Indian road-user and infrastructure classes + ByteTrack. Feeds a vehicle-density/bottleneck module and a school-zone pedestrian module. | tracked boxes, per-frame counts, alert level | `.pt`, YOLOv8n, 640 px | 6.2 MB | mAP50 **0.470**, recall 0.41 |
| **V3** | Sign Condition | `traffic_sign/` | front camera | Detects traffic signs and labels each one `Damaged` or `Good` | boxes | `.pt`, YOLOv8s, 640 px | 22.5 MB | mAP50 **0.951** on its own val split (not dashcam video) |
| **S1** | IRI regressor | `IRI/iri_compliant/` | IMU + GPS speed | International Roughness Index for every 100 m driven | IRI in m/km, calibrated range 1.21–15.57 | TFLite + JSON calibration table | 42 KB (25,769 params) | simulator test MAE **1.65 ± 0.06 m/km**, R² 0.52 (5 seeds) |
| **S2** | Road-shock classifier | `IRI/ml_model/road_classification/` | vertical accelerometer | Grades a 128-sample window | `Excellent` / `Patches` / `Med Pothole` / `Big Pothole` | TFLite FP16/FP32 + ONNX | 221 KB (FP16) | val accuracy 0.81–0.86 on 58 windows — **but see the label caveat in §S2** |

Plain (non-ML) logic that ships with these models: V1's box-area severity, and V2's `VehicleDensityTracker`, `PedestrianAlertSystem` and `SchoolGeofenceManager`.

### Findings that will break integration if ignored

1. **S1: the TFLite input order is the reverse of its usage doc.** Index 0 is `context_stats [1,13]` and index 1 is `raw_imu [1,400,6]`. The doc's Kotlin and Swift samples bind by position, so they feed the wrong tensors. Bind inputs by name or by rank.
2. **S2: the TFLite `vibration` tensor is channels-last `[1,128,2]`.** The doc and its Kotlin/Swift code write the two channels one after the other, as planar data, which gives wrong logits (verified against the ONNX model). The scaler constants in the doc are placeholders; the real ones are in §S2.
3. **S2's classes are quartiles of window jerk energy, not the human road labels.** The notebook overwrote the original labels. Treat its output as a relative shock grade, not a pothole detector.
4. **V1 doesn't run as committed.** *(Fixed 28 Sep: `severity.py` moved to `src/modules/`; pin changed to `ultralytics>=8.4`.)*
   - `src/infer.py` imports `src.modules.severity`, but the file sat at `src/modules/src/modules/severity.py`.
   - `requirements.txt` pinned `ultralytics==8.0.0`, which can't load a checkpoint saved with 8.4.163.
5. **V1's class `missing_zebra` actually fires on *visible* zebra crossings.** "Missing" has to be worked out in the backend.
6. **V2's bottleneck and pedestrian logic assume a fixed CCTV camera.**
   - Its "stalled vehicle" test uses pixel speed, which on a moving bus is only relative to the bus.
   - The school geofence is checked once, at start-up, against hard-coded coordinates.
7. **No model has seen bus-height footage or bus dynamics.**
   - The vision models were trained on car dashcam, handheld and web images.
   - Both IMU models were trained only in the BeamNG simulator, on cars.
8. **V1 (YOLOv8m at 800 px) won't run in real time on a Raspberry Pi CPU (est.).** Run it only on IMU-triggered frames (§8, as the edge agent does), or replace it with a nano model ([v1-successor-plan.md](v1-successor-plan.md)).

---

## 1. Coverage against the problem statement

| PS capability | Covered by | Status |
|---|---|---|
| Potholes | V1 (camera), S1 + S2 (IMU) | ✅ |
| Cracks / damaged surface | S1 (roughness) only | ⚠ No crack detector. V1 dropped RDD classes D00/D10/D20 on purpose. |
| Missing / faded zebra crossings | V1 class 1 (visible crossings) | ⚠ "Missing" needs backend logic. Faded and intact crossings are not separated. |
| Faded lane markings | — | ✗ |
| Damaged signboards | V3 | ✅ Binary condition only. No sign type. |
| Missing signboards | — | ✗ Needs a sign inventory built across trips (backend). |
| Vehicle density, classification, counting | V2 + ByteTrack | ✅ Per frame. ⚠ Counting logic is CCTV-style. |
| Traffic bottlenecks | V2 `VehicleDensityTracker` | ⚠ Image-space stall logic is invalid on a moving bus. Re-done in the backend with GPS speed. |
| Vulnerable pedestrians / school zones | V2 person/rider + OSM geofence + school hours | ✅ Logic exists. ⚠ Geofence is checked once. |
| Missing dividers | — | ✗ |
| Waterlogging | — | ✗ |
| Road roughness (IRI) | S1 | ✅ Simulator-trained. |

**Gaps against the plan:**

- V2 is trained on IDD only: no zebra-crossing, speed-bump or animal classes.
- V1 detects potholes but not cracks. Cracks are part of the planned nano successor ([v1-successor-plan.md](v1-successor-plan.md)).
- V3 is a full-frame detector with two conditions. The edge agent runs it on sign crops from V2 instead ([edge-deployment.md §2](edge-deployment.md)).
- A scene classifier for waterlogging, dividers and lane markings has not been started.

---

## 2. Conventions used below

- **Boxes** are `[x1, y1, x2, y2]` in pixels of the *original* frame (Ultralytics `boxes.xyxy`). Confidences are in [0, 1].
- **Frames** are OpenCV BGR `uint8` arrays of shape H×W×3. Ultralytics letterboxes them to the model's size internally; no manual resize is needed.
- **Time** should be the capture time in UTC. The current V1 wrapper stamps events when it formats them, not when the frame was captured (§V1).
- **IMU axes (S1)**, as defined by the simulator logger `IRI/iri_compliant/collect.py`:
  - `ax` = lateral, `ay` = longitudinal (forward), `az` = vertical.
  - `wx` = pitch rate (about the lateral axis), `wy` = roll rate, `wz` = yaw rate.
  - Units are m/s² and rad/s.
  - The usage doc's axis description contradicts itself in two places; the code is authoritative.

---

## 3. V1 — RoadSense (potholes + zebra crossings)

### Files
| File | Purpose |
|---|---|
| `models/roadsense_yolov8/weights/best.pt` | Weights (52 MB) |
| `road_detector.py` | Self-contained `RoadSenseDetector`. **Use this one.** |
| `src/infer.py` + `src/utils/formatter.py` | Same wrapper, split into modules. Currently broken (see issues). |
| `src/modules/src/modules/severity.py` | Box-area severity |
| `mock_backend.py`, `test_camera.py`, `tests/test_all.py` | Flask stub receiving `POST /api/events`, plus webcam test loops |

### Model (read from the checkpoint)
- **Architecture:** YOLOv8m (depth 0.67, width 0.75), about 26 M parameters, fine-tuned from `yolov8m.pt`.
- **Classes:** `{0: "pothole", 1: "missing_zebra"}`.
- **Training** (Kaggle, 2× T4, Ultralytics 8.4.163, saved 25 Sep 2026):
  - `imgsz=800`, `epochs=100`, `patience=25`, `batch=8`, `cos_lr=True`.
  - Loss weights `box=8.5`, `cls=1.5`.
  - Augmentation: `mosaic=1.0`, `mixup=0.15`, `degrees=10`.
  - The committed `src/train.py` omits `patience` and `degrees`, so it isn't the exact script that produced these weights.
- **Data (README):**
  - RDD2022-India: D40 pothole and D43 "crosswalk blur".
  - CrosswalkCDNet: `crosswalk`.
  - A Kaggle pothole set.
  - D00/D10/D20 (cracks) and D50 (manholes) were deliberately removed.
  - The script that merged these into `Master_Dataset` isn't in the repo.
- **Validation metrics (checkpoint):**

  | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|
  | 0.694 | 0.660 | 0.674 | 0.402 |

  - These are the two classes combined. No per-class numbers were saved.
  - The README's "v3" row (0.707 / 0.664 / 0.667) doesn't match the checkpoint; it may come from another epoch or run.

### Input
- One camera frame of any size (BGR numpy array or image path).
- Inference size is 800, restored from the checkpoint automatically.
- Confidence threshold: `0.40` in `src/infer.py` and `test_camera.py`; `0.50` is the class default in `road_detector.py`.

### Output
Raw Ultralytics output: `boxes.xyxy`, `boxes.conf`, `boxes.cls`.

The wrapper `predict(frame, current_lat, current_lon, vehicle_id)` returns **one dict per box per frame**, with no tracking or de-duplication:

```json
{
  "eventType": "pothole",            // or "missing_zebra"
  "confidence": 0.68,                // rounded to 2 decimals
  "timestamp": "2026-09-26T07:57:44Z",
  "vehicleId": "BUS_DEMO_01",
  "location": {"latitude": 28.6142, "longitude": 77.2110},
  "metadata": {"severity": 1}        // potholes only; {} for zebra
}
```

**Severity rule** (bounding-box area in pixels²; the `frame_shape` argument is ignored):

| Box area (px²) | Severity |
|---|---|
| ≤ 18,000 | 1 |
| 18,001 – 45,000 | 2 |
| > 45,000 | 3 |

### Known issues
1. **Import bug.** *(Fixed 28 Sep.)* `severity.py` was nested at `src/modules/src/modules/`, so `from src.modules.severity import …` in `src/infer.py` raised `ModuleNotFoundError`, and so did `tests/test_all.py`. `road_detector.py`, which computes severity inline, worked.
2. **Version pin.** *(Fixed 28 Sep.)* `requirements.txt` pinned `ultralytics==8.0.0`. The checkpoint's pickle references `ultralytics.nn.modules.block/conv/head`, a package layout that 8.0.0 doesn't have. Install Ultralytics 8.4.x.
3. **Severity depends on the camera's resolution.** 45,000 px² is 2.2% of a 1920×1080 frame but 14.6% of a 640×480 frame. Normalise by frame area (`area / (W·H)`) before thresholding, and send `bbox` and `frameSize` so the backend can re-derive severity.
4. **`missing_zebra` is misnamed.** Both sources for class 1 are boxes around *visible* crossings. A detector can't box something that isn't there. Treat class 1 as `zebra_crossing_seen`; "missing" is an absence in the backend (see the companion doc).
5. **Placeholder location, identity and time.** GPS defaults to a hard-coded (28.6142, 77.2110) and `vehicleId` to `BUS_DEMO_01`. The timestamp is taken when the event is formatted, not when the frame was captured.
6. **Unsupported README claims.**
   - "99.98% detection over 10 frames" assumes consecutive frames are independent, which they aren't. Don't repeat this in the PPT.
   - The Jetson and Hailo latency table isn't measured anywhere in the repo.
7. **Licence.** AGPL-3.0 (Ultralytics).

---

## 4. V2 — IDD Traffic model + density and school-zone modules

### Files
| File | Purpose |
|---|---|
| `models/idd_yolov8/weights/best.pt` | Weights (6.2 MB) |
| `src/infer.py` | Video/webcam pipeline: `model.track(..., tracker="bytetrack.yaml", persist=True)` → both modules → annotated video |
| `src/modules/vehicle_density.py` | `VehicleDensityTracker` |
| `src/modules/pedestrian_alert.py` | `PedestrianAlertSystem` |
| `src/utils/geofence.py` | `SchoolGeofenceManager` (OSM Overpass or offline mock) |
| `src/utils/dataset_conversion.py`, `extract_subset.py` | IDD → YOLO converter with stratified subsampling |

### Model (read from the checkpoint)
- **Architecture:** YOLOv8n (depth 0.33, width 0.25), about 3.0 M parameters, fine-tuned from `yolov8n.pt`.
- **Classes:**

  | ID | Class | Group in `configs/data.yaml` |
  |---|---|---|
  | 0 | person | vulnerable |
  | 1 | rider | vulnerable |
  | 2 | car | vehicle |
  | 3 | bus | vehicle |
  | 4 | truck | vehicle |
  | 5 | autorickshaw | vehicle |
  | 6 | motorcycle | vehicle |
  | 7 | bicycle | vehicle |
  | 8 | traffic light | infrastructure |
  | 9 | traffic sign | infrastructure |
  | 10 | vehicle fallback | other |
  | 11 | ego vehicle | other (drop: this is your own bonnet or dashboard) |
  | 12 | pole | infrastructure |

- **Training** (RTX 2050, Ultralytics 8.4.160, saved 24 Sep 2026):
  - `imgsz=640`, `epochs=30`, `patience=10`, `batch=8`, AMP.
  - Augmentation: `mosaic=1.0`, `mixup=0.1`, `degrees=5`, `hsv_s=0.6`.
- **Data:** IDD Detection, subsampled to 13,500 frames (10,800 train / 2,700 val).
  - Every frame containing a bicycle, traffic light, traffic sign or pole was kept.
  - **The split is by random frame, not by drive**, so neighbouring frames land in both train and val and the val scores are optimistic.
  - IDD labels outside the alias map (`animal`, `train`, `caravan`, `trailer`) were silently dropped, so the model learned them as background.
- **Validation metrics (checkpoint):**

  | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|
  | 0.710 | 0.410 | 0.470 | 0.289 |

- **Per-class mAP50 (README):**

  | Class | mAP50 | Recall |
  |---|---|---|
  | bus | 0.637 | 0.565 |
  | autorickshaw | 0.633 | 0.562 |
  | car | 0.617 | 0.534 |
  | truck | 0.599 | 0.521 |
  | motorcycle | 0.583 | 0.520 |
  | rider | 0.488 | 0.422 |
  | bicycle | 0.427 | 0.360 |
  | traffic sign | 0.410 | 0.370 |
  | person | 0.376 | **0.303** |
  | traffic light | 0.331 | 0.290 |
  | vehicle fallback | 0.076 | 0.061 |

  **Person recall of 0.30 is weak for a safety feature.**

### Input
- One frame (BGR) at 640, sent through `model.track(frame, persist=True, tracker="bytetrack.yaml")`.
- No confidence threshold is passed, so Ultralytics' tracker defaults apply.
- `persist=True` means **one tracker per camera stream**. Don't share a model object between cameras.

### Output
**Detector + tracker**, one entry per tracked box per frame. Boxes without a track ID are discarded.

```python
{"bbox": [x1, y1, x2, y2], "track_id": 17, "class_id": 2, "confidence": 0.81}
```

**`VehicleDensityTracker.update(detections, current_time)` → `VehicleDensityResult`:**

| Field | Meaning |
|---|---|
| `frame_timestamp` | Time passed in |
| `total_vehicles` | Tracked vehicles in *this frame*, classes 2–7 |
| `counts_by_class` | e.g. `{"car": 3, "bus": 1, ...}` |
| `zone_counts` | Vehicles whose box bottom-centre falls in each image polygon. Defaults: `Main_Corridor` (lower 60% of the frame) and `Bottleneck_Core` (a central trapezoid). |
| `zone_bottlenecks` | Per zone: `True` if ≥ 1 stalled vehicle or count ≥ `bottleneck_density_threshold` (4 in `infer.py`) |
| `stalled_vehicle_ids` | Tracks with dwell ≥ 6 s in a zone **and** smoothed image speed < 10 px/s |
| `active_vehicles` | Per-track state: first/last seen, dwell time, speed in px/s |

**`PedestrianAlertSystem.update(detections)` → `PedestrianAlertResult`:**

| Field | Meaning |
|---|---|
| `pedestrian_count`, `rider_count`, `total_vulnerable_count` | Class 0 and class 1 boxes whose bottom-centre lies in the crossing ROI (x 15–85%, y 55–85% of the frame) |
| `is_school_hours`, `active_school_window` | Local time within 07:30–09:30 or 13:30–15:30 |
| `in_school_geofence`, `matched_school_name` | Geofence result for the camera coordinates |
| `alert_level` | `NORMAL` / `ADVISORY` (people near a school outside school hours) / `CRITICAL` (people near a school during school hours) |
| `alert_message`, `vulnerable_track_ids` | Human-readable text and IDs |

**`SchoolGeofenceManager.check_coordinates(lat, lon)` → `(inside, school_name, zone)`:**
- Live mode queries Overpass for `amenity=school` in an area matched by city name.
- School points (nodes) become 120 m squares. School outlines (ways) are used as drawn, with no buffer. School relations are fetched but skipped.
- Mock mode has three Bengaluru schools.

### Known issues
1. **The stall and bottleneck logic is for a fixed camera.**
   - On a moving bus, a car travelling at the bus's speed has near-zero pixel speed and gets flagged as "stalled".
   - Zones are image regions, not road locations.
   - Replace this with segment-level density plus the bus's own GPS speed. The companion doc specifies how.
2. **The geofence is evaluated once, in the constructor**, using `camera_coords`, which `infer.py` hard-codes to (12.9716, 77.5946), inside the mock St. Joseph's school.
   - Per frame, you must call `check_coordinates` with the bus's current GPS position.
3. **The demo is wired to always be CRITICAL.** `infer.py` defaults to simulated school hours (08:15; `--school-hours` is on unless `--no-school-hours` is passed), with a camera that is always inside the geofence. So any person in the ROI gives CRITICAL. Say so in the demo, or wire it to real time and GPS.
4. **Riders count as vulnerable pedestrians.** IDD boxes the rider separately from the motorcycle, so every motorbike in the ROI raises the vulnerable count.
5. **Counts aren't unique.** `total_vehicles` counts per frame, not unique vehicles per road stretch.
6. **Hard-coded paths.** Tests and `configs/data.yaml` point to `E:\SIH_2026\…`.
7. **Licence.** AGPL-3.0.

---

## 5. V3 — Traffic-sign condition detector

### Files
| File | Purpose |
|---|---|
| `traffic_sign/best.pt` | Weights (22.5 MB) |
| `traffic_condition_monitor.ipynb` | Colab notebook: train, one test prediction, ONNX export |
| `Project README.pdf` | Dataset and ontology description |

There is **no inference wrapper and no tracking code**. You call Ultralytics directly.

### Model (read from the checkpoint)
- **Architecture:** YOLOv8s (depth 0.33, width 0.50), about 11 M parameters, fine-tuned from `yolov8s.pt`.
- **Classes:** `{0: "Damaged", 1: "Good"}`. Detection and condition happen in one pass; **the model doesn't say which sign it is.**
- **Training** (Colab T4, Ultralytics 8.4.163, saved 27 Sep 2026): `imgsz=640`, `epochs=50`, `batch=16`, default augmentation.
- **Data (README):**
  - IRTSD (pristine Indian signs) plus the Roboflow "Damaged Traffic Sign" dataset.
  - `Good` = healthy, ok.
  - `Damaged` = deformation, dirty, graffiti, knocked, occluded, perforation, stickers, worn.
  - Images were letterboxed to 640×640 by Roboflow. The train/val split came from the Roboflow export.
- **Validation metrics (checkpoint):**

  | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|
  | 0.937 | 0.919 | 0.951 | 0.708 |

  These are on the dataset's own split: mostly close-up images of single signs. Expect a large drop on bus dashcam video, where signs are small, blurred and seen at an angle. Edge-TSR measured 20–30% relative loss from still images to video.

### Input and output
- **Input:** frame (BGR), `imgsz=640`. A confidence of about 0.5 is a sensible starting point; tune it on your own footage.
- **Output:** boxes `xyxy`, `conf`, `cls`, where `cls` 0 = Damaged and 1 = Good.

### Integration notes
1. **Aggregate per physical sign.** Wrap the model in `model.track(..., persist=True)` and decide Damaged vs Good by majority vote over the whole track. Without this, one sign becomes about 30 events per second.
2. **Cheap cascade.** Run V3 only on frames where V2 reports a `traffic sign` (class 9), or at about 1–2 FPS.
3. **"Missing sign" is not possible from this model.** It needs a sign inventory built across trips (backend).
4. **Licence.** AGPL-3.0.

---

## 6. S1 — IRI regressor (IMU → road roughness)

### Files
Canonical copy: `IRI/iri_compliant/`. An identical copy is in `IRI/ml_model/iri_compliant/`.

| File | Purpose |
|---|---|
| `iri_background_model.tflite` | **Deploy this.** 42 KB, dynamic-range quantised. |
| `mobile_calibration_lut.json` | Isotonic calibration table: 81 `x_raw` → `y_calibrated` points |
| `best_iri_model.keras` | Keras source model (25,769 parameters). Loading it needs the custom `AsymmetricHuberLoss`, or `compile=False`. |
| `model.ipynb` | Training, evaluation and field-inference notebook |
| `iri_calculation.py` | Ground-truth IRI from simulator road profiles (not needed at runtime) |
| `collect.py` | BeamNG data logger. Defines the axis convention. |
| `model_usage_instructions.md` | Mobile integration guide. **Contains errors; see Known issues.** |

### What it is
- Continuous IRI in m/km for each **100 m of travel**, computed from a 6-axis IMU plus GPS speed.
- **Architecture:** two branches.
  - Raw branch on `[400, 6]`: DepthwiseConv1D(k5, ×2) → BN → MaxPool → Conv1D(48, k5) → BN → MaxPool → Conv1D(64, k3) → BN → global average + max pooling → Dropout(0.3).
  - Context branch on `[13]`: Dense(32) → BN → Dropout(0.2).
  - Fusion: Dense(64) → Dropout(0.2) → Dense(32) → Dense(1, softplus).
  - The model predicts `log1p(IRI)`.
- **Training data: BeamNG.tech simulator only.**
  - 3 cars (hopper, sunburst2, vivace), 19 trips.
  - Train: `east_coast_usa` and `west_coast_usa`. Val: `automation_test_track` trip 1. Test: trips 2–3.
  - 100 m windows at a 10 m stride: 9,082 train / 2,796 val / 1,903 test.
  - IMU logged at 100 Hz with gravity. Each window was re-sampled at a random 10–50 Hz with jitter, to make the model robust to phone sample rates.
- **Labels:**
  - Left and right LiDAR road-elevation profiles.
  - A curvature-adaptive correction removes the simulator's mesh-facet artefacts.
  - A 250 mm moving average is applied.
  - The ASTM E1926 "Golden Car" quarter-car model is run at 80 km/h over 100 m segments with a 20 m lead-in.
- **Loss and weighting:**
  - Asymmetric Huber loss (δ = 2; under-predictions weighted ×4).
  - IRI > 10 is oversampled.
  - Samples are weighted by inverse frequency on a speed × IRI 2-D histogram.

### Performance (simulator test set unless noted)
| Evaluation | Result |
|---|---|
| Notebook run (3 predictions ≥ 40 dropped) | MAE 1.69, RMSE 2.48, R² 0.558, r 0.759 m/km |
| After calibration (full test set) | MAE 1.67, RMSE 2.67 |
| 5-seed ablation (`iri_v3`, full model) | MAE **1.650 ± 0.061**, RMSE 2.583 ± 0.073, r 0.734, R² 0.521 |
| Error by severity (notebook) | Good 0–4: 1.17 · Fair 4–8: 1.65 · Poor 8–14: 2.49 · **Very poor > 14: 6.63** (severe roads are under-estimated) |
| vs. 9 baselines (Ridge … CatBoost, LSTM) | Best of all. CatBoost is next best (MAE 2.03). |
| AASHTO R 56 tolerance (±max(0.25 m/km, 10%)) | 18.4% of windows. The paper reports 77.9% within ±1.5 m/km. |
| Speed invariance (7 trips, 13–109 km/h, same track) | Spread between trips: std 1.24 m/km |
| **Real world, zero-shot** (PVS dataset, Brazil, 9 trips, 107 km) | Median IRI: asphalt-good 1.89, asphalt-regular 4.12, cobblestone/dirt 5.0–5.6 (regular vs bad not separated). 3% of good-asphalt windows read > 8 at 75–92 km/h (expansion joints). |
| 3-class mapping on PVS | Good < **3.8** ≤ Regular < **11.7** ≤ Bad gives balanced accuracy 0.61 and macro-F1 0.57 |

### Input contract (verified against the TFLite)

**TFLite tensors — bind by name, not position:**

| Index | Name | Shape | dtype |
|---|---|---|---|
| in 0 | `serving_default_context_stats:0` | `[1, 13]` | float32 |
| in 1 | `serving_default_raw_imu:0` | `[1, 400, 6]` | float32 |
| out | `StatefulPartitionedCall:0` | `[1, 1]` | float32 = log1p(IRI_raw) |

**Sensor stream the edge must provide:**
- 6-axis IMU at **20–100 Hz**.
- Speed in m/s (GPS or AIS-140).
- Latitude/longitude, for geotagging.
- One monotonic timestamp shared by all of the above.
- Axes as in §2:
  - `ax` lateral, `ay` forward, `az` vertical, in m/s².
  - **In training, `az` included gravity: about −9.81 m/s² at rest.**
  - Gyro in rad/s: `wx` pitch, `wy` roll, `wz` yaw.
  - The PVS validation script had to invert `az`, swap x↔y and convert deg/s to rad/s to match this. Any new sensor needs the same mapping into this frame.

**Pre-processing per window** (matches `process_trip_data` in training):

```text
1. Drop samples with speed < 1.0 m/s.
2. dist = cumsum(speed · dt). Cut 100 m windows (stride 10 m in training; 100 m is fine for upload).
   Skip windows with < 20 samples.
3. grid = linspace(start, start+100, 400).
   Linearly interpolate ax..wz and speed onto the grid, filling edges with the end values.
4. v_safe = max(v_grid, 5.0);  az_grid *= (22.22 / v_safe)**2      # always, as in training
5. raw_imu = [ax, ay, az_norm, wx, wy, wz]   → shape [400, 6], float32
6. context_stats = the 13 features below, computed on the grid (az = az_norm)
```

**The 13 context features, in order:**

| # | Name | Formula (NumPy, population std/var) |
|---|---|---|
| 0 | `speed_mean` | `mean(v)` |
| 1 | `speed_std` | `std(v)` |
| 2 | `rms_az` | `sqrt(mean(az²))` |
| 3 | `rms_ay` | `sqrt(mean(ay²))` |
| 4 | `var_az` | `var(az)` |
| 5 | `crest_factor_az` | `max\|az\| / (rms_az + 1e-6)` |
| 6 | `mcr_az` | `count(diff(sign(az)) ≠ 0) / 400` |
| 7 | `p2p_az` | `max(az) − min(az)` |
| 8 | `rms_wz` | `sqrt(mean(wz²))` |
| 9 | `rms_wy` | `sqrt(mean(wy²))` |
| 10 | `mean_abs_ax` | `mean(\|ax\|)` |
| 11 | `energy_ratio_1_4` | `P = \|rfft(az)\|²`, `f = rfftfreq(400, d=0.25) · speed_mean`; `sum P[1 ≤ f ≤ 4] / (sum P + 1e-6)`; 0 if `speed_mean ≤ 0` |
| 12 | `energy_ratio_4_15` | same, with `4 < f ≤ 15` |

### Output and post-processing
```text
iri_raw = expm1(y)
iri     = np.interp(iri_raw, lut.x_raw, lut.y_calibrated)   # clamps to [1.2128, 15.5720]
```

- Calibrated values sit on the table's steps, so many windows land on exactly the same value (for example 1.89, 2.60, 3.25, 12.98). On a map this is fine; don't treat the decimals as precision.
- Severity bins used across the notebooks and maps:

  | Good | Fair | Poor | Very poor |
  |---|---|---|---|
  | < 4 | 4–8 | 8–14 | ≥ 14 m/km |

- Attach the window's start and end coordinates, its mean speed and the number of samples.

### Known issues
1. **Input order in the usage doc is wrong.** The doc says input 0 is `raw_imu`; it is `context_stats`. Its Kotlin (`runForMultipleInputsOutputs(arrayOf(imu, ctx))`) and Swift (`copy(..., toInputAt: 0)`) samples bind the wrong tensors.
2. **The doc's sample code differs from training in two ways.**
   - It hard-codes the spectral ratios (features 11–12) to 0.35 and 0.45 instead of computing them.
   - It uses a shortened 12-point calibration table.
   - Use the formulas above and the full JSON table.
3. **Gravity convention mismatch.** Training `az` includes gravity (simulator mean −9.77). The team's own phone logs (`KMP_mathura.csv`, `delhi_mumbai_expressway.csv`) are gravity-removed linear acceleration at 100 Hz (mean `az` ≈ 0.04).
   - Measured on those logs, the effect on calibrated IRI is small: at most about 0.6 m/km median shift per speed band, and the same median at > 70 km/h.
   - Still, pick one convention. Either add −9.81 to vertical `az` at the edge, or retrain without gravity.
4. **The notebook's field-inference cell differs from training.** It applies the speed normalisation only when mean speed ≥ 15 km/h or a shock is detected. Training always applies it. Use the training behaviour.
5. **Simulator-only, car-only training.** Absolute IRI on a bus (different suspension, sensor mounted higher) is uncalibrated.
   - Present it as a relative road-condition index until it is checked against a reference stretch.
   - On the dashboard, aggregate several passes per road segment.
6. **Overlapping windows.** At a 10 m stride each point is covered by 10 windows. Upload at a 100 m stride, or aggregate per segment on the server, so a road isn't counted 10×.

---

## 7. S2 — Road-shock classifier (vertical-accelerometer window → 4 grades)

### Files (in `IRI/ml_model/road_classification/`)

> **These files are not in this repository yet.** They were read for this spec from the team's working copy. Until they are added, the edge agent triggers V1 with a jerk threshold instead ([edge-deployment.md §2.4](edge-deployment.md)) and switches to S2 automatically once `road_vision_final_float32.tflite` is present.

| File | Status |
|---|---|
| `road_vision_final_float16.tflite` (221 KB) / `_float32.tflite` (436 KB) | **Deploy these.** Converted from the ONNX file below. |
| `road_vision_final.onnx` | Source of the TFLite files (input `[1,2,128]`) |
| `context_scaler.pkl` | `StandardScaler` for the 4 context features (constants below) |
| `road_vision_final_85plus.pt` | PyTorch weights used by `vid.py`. **Different weights from the ONNX/TFLite**: on the same input they give different logits. |
| `final_road_model.pt` | Older variant with 3 context features (FC input 259). Not compatible. |
| `best_vision_model.pt`, `best_multiscale_model.pt` | Earlier experiments with other architectures. Unused. |
| `saved_model.pb` (+ empty `variables/`) | Incomplete TF SavedModel. Ignore. |
| `final_road_class.ipynb`, `vid.py`, `model_Usage_instructions.md` | Training notebook, video overlay renderer, mobile guide (**contains errors**) |

"vision" in the file names is misleading: this is a vibration model.

### Model
- **`RoadFinalNet`:**
  - Conv1d(2→64, k11) + BN + ReLU + Dropout1d(0.2)
  - Conv1d(64→128, k5) + BN + ReLU + Dropout1d(0.2)
  - Global average and max pooling (256 values), concatenated with 4 context features (260)
  - FC(256) + BN + ReLU + Dropout(0.5) → FC(4)
  - About 110 k parameters. Weights averaged with SWA; label smoothing 0.1.
- **Classes:** 0 `Excellent`, 1 `Patches`, 2 `Med Pothole`, 3 `Big Pothole`.
- **Data:** BeamNG simulator, 100 Hz, **286 windows in total**.
  - Each window is 128 samples centred on the largest |az| inside a hand-labelled range from `master_label.txt`.
  - Split 80/20 stratified at random: **58 val windows, no test set.**
- **Reported:** val accuracy 0.81 (macro-F1 0.80, κ 0.75) at the end of the notebook; 0.86 at the best epoch. The notebook's "Big Pothole recall 1.00" claim prints as 0.93 in the final cell.

### ⚠ What the labels actually mean
- `reassign_physical_labels()` **replaced the human labels** with the 25/50/75th percentiles of `std(sig)`, where `sig` is the window's per-sample difference, already z-scored.
- For a z-scored signal, `std = σ/(σ + 1e-6)`, where σ is the raw per-sample-difference spread. This is a monotone function of σ. So the classes are **quartiles of the raw jerk energy of the 286 simulator windows**, 25% each by construction.
- The context feature `RMS` is the same quantity. Its scaler scale is 6.0e-6, which blows those tiny differences back up. The model can therefore read the label almost directly from that feature.
- **Consequences:**
  - Read the output as "how violent was this 1.3 s window, relative to simulator driving".
  - It is not a validated pothole detector.
  - The class boundaries are fixed in raw m/s² at 100 Hz, so a different unit (g vs m/s²), sample rate or vehicle shifts every class.
  - Use S1 as the main IMU road metric and S2 as a shock trigger (§8), or retrain on the original labels.

### Input contract (verified: TFLite against ONNX)

| Index | Name | Shape | Layout |
|---|---|---|---|
| in 0 | `vibration` | `[1, 128, 2]` | **Channels-last.** The buffer is interleaved per time step: `c0[0], c1[0], c0[1], c1[1], …` |
| in 1 | `context` | `[1, 4]` | Scaled features, below |
| out | `Identity` | `[1, 4]` | Logits |

(The ONNX file takes `vibration` as `[1, 2, 128]` planar data. Transposing that gives identical logits from the TFLite.)

```text
az    : 128 consecutive vertical-acceleration samples, m/s², 100 Hz (as in training)
sig   = diff(az, prepend=az[0])                     # sig[0] = 0; gravity cancels here
sig   = (sig - mean(sig)) / (std(sig) + 1e-6)       # population std, float64
ch0   = |sig| ;  ch1 = np.gradient(sig)             # central difference, one-sided at the ends
rms   = sqrt(mean(sig**2))                          # NO +1e-6 here (training); vid.py/doc add it → skew
crest = max|sig| / (rms + 1e-6)
ctx_raw = [0.0, mean_speed_mps, rms, crest]         # VehID was 0 for every training window
ctx     = (ctx_raw - MEAN) / SCALE
MEAN  = [0.0, 12.464910954632682, 0.9999906116945316, 4.2572717833764555]
SCALE = [1.0,  6.405128570025835, 6.002955638863807e-06, 0.8918762876362021]
```

- Compute `rms` in **float64**. Its useful signal lives in the 6th–7th decimal place, and float32 rounding is about 2% of the scaler's scale.
- The doc's example constants (`[0, 12.5, 0.98, 3.45]` / `[1, 4.2, 0.31, 1.12]`) are placeholders and would break this feature.

### Output and post-processing (as in `vid.py`)
1. `p = softmax(logits)`, `cls = argmax(p)`.
2. If `cls ≥ 2` and `max(p) < 0.82`, set `cls = 0`.
3. Take a majority vote over the last 15 predictions. With a 10-sample hop at 100 Hz, that is about 1.5 s.
4. Training windows were *centred on the peak*. A sliding window at inference sees the peak at any position, which is a mild distribution shift.

---

## 8. Running them together on the edge

Implemented in [edge/](../edge/). The full analysis (quantisation, the two-lane design and the CPU budget) is in [edge-deployment.md](edge-deployment.md). As deployed on a Raspberry Pi 5 CPU (NCNN FP16; Pi figures are estimates until measured):

| Model | Input | FLOPs / run | Est. Pi 5 time | Rate |
|---|---|---|---|---|
| V2 YOLOv8n + ByteTrack | 384×640 (rectangular) | ~5.2 G | 60–90 ms | 3–4 FPS, 2 when stopped |
| V3 YOLOv8s | 320×320 crops of V2's sign boxes | ~7.2 G | 60–95 ms | 3 crops per tracked sign |
| V1 YOLOv8m | 480×800 (rectangular) | ~74 G | 0.6–0.9 s | 3 buffered frames per IMU jolt; no continuous scan on the CPU |
| S1 | 26 k params per 100 m | — | < 1 ms (1 ms measured on a laptop) | every 100 m |
| S2 (or jerk trigger) | ~110 k params per window | — | ~1 ms | every 10 samples |

- Export with `python edge/scripts/export_models.py`: NCNN FP16 at the rectangular sizes above. A square `imgsz=800` export, as first suggested here, was measured 36% slower for V1 with the same detections.
- The two TFLite files run with `ai-edge-litert` on Linux, or TensorFlow on Windows.
- The demo runs the same edge agent on a laptop in replay mode, and says so ([decisions.md](decisions.md), D4).

### Suggested fusion
- **Pothole (IMU trigger).** When S2 gives class 3 with p ≥ 0.82, or |jerk| crosses a threshold, run V1 on three buffered frames from 1.5, 1.0 and 0.6 s before the jolt.
- **Pothole status.** Camera + IMU within about 15 m → `probable`. Camera only, or IMU only → `candidate`.
- **Road condition.** S1 gives the per-segment roughness layer; S2 marks point shocks on top of it.
- **Signs.** V2 finds `traffic sign` → V3 on that frame → one event per track.
- **Traffic.** V2 per-frame counts are summarised every 10 s or 100 m on the edge. Bottleneck and school-zone decisions use GPS, on the edge and re-checked in the backend.

The event types and payload fields each model emits are defined in [dashboard-backend-design.md §3](dashboard-backend-design.md#3-edge--backend-data-contract).

---

## 9. What to freeze for the demo

| Model | Artefacts |
|---|---|
| V1 | `RoadSense_ML_Edge-master/models/roadsense_yolov8/weights/best.pt` + the `road_detector.py` logic (fix the severity normalisation) |
| V2 | `sih-2026-traffic-analytics-main/models/idd_yolov8/weights/best.pt` + `src/modules/*`, `src/utils/geofence.py` (after the GPS fixes) |
| V3 | `traffic_sign/best.pt` |
| S1 | `IRI/iri_compliant/iri_background_model.tflite` + `mobile_calibration_lut.json` |
| S2 | `IRI/ml_model/road_classification/road_vision_final_float32.tflite` (or FP16) + the scaler constants above |

**Not for integration.** These are research or superseded artefacts, kept for the paper:
- `IRI/ml_model/iri_v4` and `iri_v2/output/IRINet-v5.tflite` / `IRINet-v4_Baseline.tflite`: experimental IRINet-v5. It takes 19 context features, uses a `v^1.5` speed normalisation and adds a 4-class severity head. Its outputs are `[1,1]` + `[1,4]`. No calibration or evaluation in the repo is tied to it.
- `IRI/ml_model/iri` (v1): input `[1,100,7]` + `[1,7]`, with a wavelet-based labelling pipeline. Superseded.
- `IRI/ml_model/baseline`: 9 baseline models × 5 seeds, for the paper's comparison table.
- `iri_v3` / `iri_v3_pica_last`: ablation study.
- `AASHTO_compliance`, `pvs_comparision`, `review_work`, `speed_invariance_test`: evaluation scripts and outputs.

## 10. Fix list, ordered by demo impact

Items 1–3 are done in the edge agent ([edge/](../edge/), 28 Sep). The teammates' original wrappers in `ML_models/` keep their old behaviour, apart from V1's two packaging fixes.

1. ✅ **S1/S2 integration code:** bind TFLite inputs by name; use the S2 interleaved layout, the real scaler constants and `rms` without epsilon; compute S1's spectral features instead of hard-coding them. Covered by tests in `edge/tests/`.
2. ✅ **V2:** re-check the geofence on every frame with GPS; drop the pixel-speed "stall" logic in favour of the backend's GPS-based bottleneck score; stop counting riders as pedestrians near schools (or show them separately); tie school hours to the real clock, or label the demo clock.
3. ✅ **V1:** move `severity.py` to `src/modules/`; pin `ultralytics>=8.4`; normalise severity by frame area; rename class 1 to `zebra_crossing` in the events; stamp events with capture time and real GPS.
4. **All vision models:** record 500–1,000 frames from a bus-mounted camera and check precision and recall on them before quoting any accuracy in the pitch.
5. **S2:** retrain on the original `master_label.txt` labels, or present it only as a shock trigger.
6. **S1:** settle the gravity convention; run a reference drive over a known-good and a known-bad stretch to sanity-check absolute IRI on the bus.
