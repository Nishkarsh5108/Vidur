# FleetSense — ML Model Plan for Raspberry Pi Edge Inference

*Prepared 24 Sep 2026 for SIH 2026 PS26124. Scope = the [roadmap](sih2026-ps26124-roadmap.pdf) **minus** hit-and-run / rash-driving / ANPR (dropped as out of scope).*
*Everything marked "verified" was checked against the source on this date. Latency figures marked "est." are scaled from measured benchmarks and must be re-measured on your Pi.*

---

## 0. Summary

- **Four small models plus some plain logic cover every remaining capability.** Each model is a nano-size network (about 2–3 M parameters) that runs on the Pi 5's CPU through NCNN.
- **Only road damage has a genuinely ready-to-use model**: a YOLO26n distilled on RDD2022. Strong Indian vehicle models exist too, but they were trained on CCTV footage and are too large for the Pi, so they serve as **teachers** (they label our data), not as the deployed model. Signs, zebra crossings, dividers and waterlogging all need training of our own.
- **Gating saves the most compute, not quantization.** Gating means not running models when there is nothing new to see: IMU triggers, frame rates that follow bus speed, cropping to the road region, and classifying each tracked object once instead of on every frame.
- **Quantization on a CPU-only Pi 5:** FP16 NCNN is the default (almost no accuracy loss, half the file size). INT8 is **mandatory only if you add a Hailo HAT or the AI Camera**; on the CPU it is an optional experiment. Model size is not the constraint (a nano model is 5–10 MB and the Pi has 4–16 GB of RAM). Latency is.

| # | Model | What it does | Architecture | Starting weights | Pi input | Rate |
|---|---|---|---|---|---|---|
| **M1** | TrafficNet | Indian vehicle classes, pedestrians, animals, signs/boards, traffic lights, zebra crossings, speed bumps, roadwork | YOLO26n detect | COCO `yolo26n.pt`, then DriveIndia + IDD | 640×384 | 2–6 FPS, adaptive |
| **M2** | RoadNet | cracks D00/D10/D20 + potholes D40 | YOLO26n (optionally s) | **ready-made** `TamAko783/YOLO26n_RDD_FRDC_Distilled_v2` | road crop 640×256 | 1–4 FPS by speed, plus IMU burst |
| **M3** | SignCondition | condition of each sign: ok / faded / damaged / occluded (+ sign type later) | YOLO26n-cls | ImageNet cls weights | 96×96 crop | once per tracked sign |
| **M4** | SceneCls | waterlogged road · divider present/broken/absent · lane-marking visibility | YOLO26n-cls (MobileCLIP2-S0 as labeler/fallback) | pseudo-labels | 224×224 | 0.5–1 FPS |
| — | IMU model (existing) | roughness/IRI, pothole severity, trigger | unchanged | — | — | continuous |

Non-ML logic:
- ByteTrack tracking for counting and density.
- An OpenStreetMap (OSM) school-zone geofence.
- An inventory comparison across trips to find "missing" signs and zebra crossings.
- A paint-contrast score that marks zebra crossings as faded.

---

## 1. Coverage: PS capability → how it is handled

| PS capability | Approach | Components |
|---|---|---|
| Potholes, cracks, damaged roads | Camera detection fused with IMU roughness and severity | M2 + IMU |
| Faded / missing zebra crossings | M1 detects the crossing. A paint-contrast score flags **faded**. A crossing that OSM lists but no pass ever detects flags **missing**. | M1 + logic |
| Faded lane markings | Scene-level visibility class | M4 |
| **Missing dividers** (in the PS but absent from the roadmap table) | Scene-level divider state, aggregated per road segment across trips | M4 |
| Damaged / missing signboards | M1 detects the sign or board, M3 classifies its condition, and the trip-to-trip inventory comparison finds missing signs | M1 + M3 + logic |
| Vehicle density, classification, counting, bottlenecks | Detection + ByteTrack + bus speed from the AIS-140 GPS feed | M1 + logic |
| Vulnerable pedestrians / school zones | Person detection + OSM `amenity=school` geofence + school hours. **No age classification.** | M1 + logic |
| Waterlogging (stretch goal: weakest data) | Scene-level classifier | M4 |

---

## 2. Edge hardware and compute budget

### 2.1 Measured: Raspberry Pi 5, 640×640 input (Ultralytics benchmarks, verified)

| Model | **NCNN** | MNN | OpenVINO | LiteRT (TFLite) | ONNX |
|---|---|---|---|---|---|
| YOLO26n | **67 ms** | 92 ms | 105 ms | 123 ms | 126 ms |
| YOLO26s | **173 ms** | 237 ms | 282 ms | 360 ms | 356 ms |

- **NCNN is the runtime to use.** On the Pi 5, YOLO26n is about 15% faster than YOLO11n (ONNX: 128 vs 147 ms) and slightly more accurate.
- **Pi 4:** the Pi 5's CPU is 2–3× faster. The Pi 4's Cortex-A72 also lacks FP16 arithmetic and dot-product instructions. On a Pi 4, drop inputs to 320–416 or get a Pi 5.

### 2.2 Budget: Pi 5, CPU only, default configuration (est.)

| Workload | Input | ≈ms per pass | Rate | ≈CPU ms per second |
|---|---|---|---|---|
| M1 on front camera | 640×384 | ~40 | 4 FPS (2–6) | ~160 |
| M2 on front camera (n) | 640×256 road crop | ~27 | 2 FPS + bursts | ~55–80 |
| *M2 option: YOLO26s* | *640×256* | *~70* | *2 FPS* | *~140* |
| M3 per new sign track | 96×96 | ~1–3 | ~1 per s | ~3 |
| M4 | 224×224 | ~5–10 | 0.5–1 FPS | ~10 |
| M1 per extra camera (side/rear) | 512×288 | ~25 | 1–2 FPS | 25–50 each |
| **Total: front + 2 side cameras, M2 = n** | | | | **≈300 ms/s ≈ 30% of the CPU** |

- **Keep total load at or below 50–60%.** The rest is needed for frame capture, tracking, IMU processing, networking and heat. A bus cabin in an Indian summer is hot: the Pi 5 Active Cooler is mandatory, and the Pi starts throttling at around 80–85 °C.
- **Capture:**
  - Use CSI cameras through `picamera2`, which has no decode cost.
  - With USB cameras, capture at 1280×720 and 10–15 FPS. Decoding 1080p30 MJPEG from several USB cameras eats the CPU.

### 2.3 If you add an accelerator, the base model changes

| Option | Price (approx.) | Effect on the plan |
|---|---|---|
| **None (Pi 5 CPU)**: default | — | **YOLO26n** everywhere. It is the fastest model on the CPU, and because it needs no NMS post-processing, latency stays steady in dense traffic. |
| **AI HAT+ 26 TOPS (Hailo-8)** / 13 TOPS (Hailo-8L) | ~$110 (26 TOPS) | **INT8 only.** Switch M1/M2 to **YOLO11s / YOLOv8s**. Ultralytics' measured INT8 mAP50 retention on Hailo-8L: YOLOv8n ≈100%, YOLO11n ≈96%, YOLO26n ≈93%. You can afford "s" models. Compiling needs **x86-64 Linux**. |
| AI HAT+ 2 (Hailo-10H, 40 TOPS) | ~$130 | Built for LLM/VLM workloads; a review found **no vision speed-up over Hailo-8**. For this project the 26 TOPS AI HAT+ is the better buy. |
| Raspberry Pi AI Camera (Sony IMX500) | ~$70 | Runs **YOLO11n / YOLOv8n on the sensor** (~59 ms/frame) and uses no Pi CPU. YOLO26 is not supported. A Pi 5 takes at most 2 CSI cameras. |

Training works the same way for every target (Ultralytics). Switching the base architecture is a one-string change plus a few hours retraining a nano model. **Develop with YOLO26n, and retrain as YOLO11 only if the hardware decision requires it.**

---

## 3. What keeps inference cheap (highest impact first)

1. **Gating: don't run what you don't need.**
   - **M2's rate follows speed:** `fps ≈ speed_mps / 5`, clamped to 1–4. At 40 km/h this still gives each road patch 2–4 looks as it passes through the 5–25 m band ahead. M2 pauses when the bus stops (bus stops, signals).
   - **IMU burst:** keep a ring buffer of the last ~3 s of frames. When the IMU flags an anomaly, run M2 on the frames from **2.0 s to 0.3 s before impact**, when the defect was visible ahead of the wheel. The result is a visually confirmed event with an IMU severity score.
   - **M1's rate follows the scene:** higher when the bus is slow in dense traffic (counting matters), lower on empty roads.
   - **M4 runs at ≤1 FPS**, because road-scene attributes change slowly.
2. **Cascade instead of a bigger model.** M1 finds a sign; M3 classifies its crop **once per track**, using the best frame by size × sharpness (Laplacian variance), not every frame.
3. **Crop to the region that matters.** M2 sees only the road band ahead (about the central 70% of the width × the lower half of the frame, calibrated once per camera mount), resized to 640×256. That costs about 40% of a full 640² pass and gives more pixels per crack than a full-frame resize.
4. **Temporal voting beats a bigger model.** Aggregate results per track (signs) and per GPS cell (potholes, zebra crossings, dividers) over several frames and several trips. A 2026 edge-deployment study ([Edge-TSR](https://arxiv.org/abs/2606.17241)) measured a **20–30% relative accuracy drop** going from still-image benchmarks to live video. Track-level temporal stabilization recovered part of it.
5. **Distill from big teachers.** Train large models on Kaggle or the GPU, use them to pseudo-label merged datasets and your own footage, then train the nano student on the result (§7).
6. **Reduce numeric precision last.** FP16 costs nothing on the Pi 5 CPU. Use INT8 only where the runtime rewards it (§9).
7. **Upload events, not video.** An event is a JSON record plus one JPEG crop. This directly answers the PS's "minimizing bandwidth through intelligent edge processing".

---

## 4. Model-by-model plan

### M1 — TrafficNet (YOLO26n detect)

**Classes (17)**, mapped from DriveIndia's 24 and IDD's labels:

| M1 class | DriveIndia source | IDD source |
|---|---|---|
| person | Pedestrian | person |
| bicycle | Bicycle | bicycle |
| two_wheeler | Motorcycle | motorcycle (+ merge the rider box, see gotchas) |
| autorickshaw | Autorickshaw | autorickshaw |
| car | Car, Ambulance, Police vehicle | car |
| bus | Bus | bus |
| truck | Truck | truck |
| lcv | Commercial vehicle | vehicle fallback (review) |
| other_vehicle | Tractor, Pushcart, Construction vehicle | caravan, trailer |
| animal | Animal | animal |
| traffic_sign | Traffic sign | traffic sign |
| info_board | Route board | — |
| traffic_light | Traffic light | traffic light |
| roadwork | Traffic cone, Temporary traffic barrier | — |
| zebra_crossing | Zebra crossing | — |
| speed_bump_marked | Marked speed bump, Rumble strips | — |
| speed_bump_unmarked | Unmarked speed bump | — |

Unmarked speed breakers and roadwork are both **infrastructure / bottleneck signals** worth mapping. Potholes are left to M2, but DriveIndia's pothole boxes are reused as M2 training data.

**Ready-made check:**
- **COCO `yolo26n.pt`** detects person, car, bus, truck, motorcycle and bicycle out of the box. It has no autorickshaw, no zebra crossing, no speed bump, and only a generic "stop sign". Good enough for a day-1 demo only.
- **[iisc-aim/UVH-26](https://huggingface.co/iisc-aim/UVH-26)** (YOLO11-S/X, RT-DETRv2-S/X, DAMO-YOLO-T/L; 14 Indian vehicle classes; Apache-2.0) and **[iisc-aim/BMD-45](https://huggingface.co/iisc-aim/BMD-45)** (YOLOv12-S/X, RT-DETRv2-X, RF-DETR-X, D-FINE-X; same 14-class taxonomy): excellent Indian vehicle taxonomy, but trained on a **fixed CCTV viewpoint** and in S–X sizes. **Use the X models as teachers** to pseudo-label vehicle types on dashcam and bus frames.

**Data:**
- **DriveIndia** (primary): dashcam view, 24 classes, CC BY 4.0, access by request form (§6).
- **IDD Detection** (secondary).
- **Own bus frames**, pseudo-labeled (§7).

**Gotchas:**
- **Box conventions differ between datasets.** UVH/BMD "Two-wheeler" boxes include the rider; IDD labels rider and motorcycle as separate boxes. Harmonize before merging (union IDD's rider and motorcycle boxes into one `two_wheeler` box), and visually check ~50 images per dataset.
- **IDD has no zebra crossing, speed bump or roadwork labels.** If you merge IDD naively, the model learns that those objects are background. Either pseudo-label those classes on IDD with a DriveIndia-trained teacher first, or train M1 v1 on DriveIndia alone and add IDD in v2.
- **Oversample images containing rare classes** (animal, speed bumps, roadwork, info_board).

**Downstream logic:**
- **Density:** ByteTrack on M1's detections, per-class counts per 100 m GPS segment and per minute, then a **density index**. Base the index on per-frame counts averaged over the segment; this stays robust at 2–4 FPS from a moving bus, where tracker IDs switch often.
- **Bottleneck score:** low bus speed + high density + repeated across trips/buses (from the AIS-140 feed).
- **Pedestrian alert:** person tracks inside the road area, while inside a school geofence, during school hours.

**Targets:**
- mAP50 ≥ 0.60 on DriveIndia val. The paper's best YOLO baseline reaches 78.7% with an unspecified, larger model, so a nano will land lower.
- Person and two-wheeler recall ≥ 0.85 at precision ≥ 0.8 within ~30 m.

### M2 — RoadNet (YOLO26n; YOLO26s if the budget allows)

**Classes:** D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole. This is the standard CRDDC 4-class scheme, compatible with every public road-damage model and dataset.

**Ready-made ✅ (this is the one capability with a genuinely usable model):**
- **[TamAko783/YOLO26n_RDD_FRDC_Distilled_v2](https://huggingface.co/TamAko783/YOLO26n_RDD_FRDC_Distilled_v2)**
  - 2.4 M parameters.
  - mAP50 **0.638** / mAP50-95 0.337 on a 4,509-image held-out RDD val set.
  - Distilled from Co-DETR + RTMDet teachers built by the ORDDC'2024 winning team.
  - AGPL-3.0.
  - Its sibling **YOLO26s** scores mAP50 **0.692** (+5.4 points).
- **[dronefreak/rdd2022-*](https://huggingface.co/dronefreak/rdd2022-yolo26s)** (DetectionBench, RDD2022 test split). This is a different evaluation split, so the numbers are not directly comparable with TamAko783's.
  - YOLOv8n: 58.8 mAP50.
  - YOLO26s: 61.3.
  - RF-DETR-S: 64.7 and RF-DETR-M: 65.1, which make good teachers.
  - Per-class pothole AP50 is about 71–75.
- **Decision:** use TamAko783's n (and s) as **day-1 baselines and fine-tuning init**.

**Data:**
- **RDD2022**: 47,420 images. **India subset: 9,665 images with 3,187 pothole labels in the training split, more than any other country.** Indian images were shot at 960×720 on phones and squashed to 720×720.
- **[Unified Road Defect Dataset](https://huggingface.co/datasets/TamAko783/Unified_Road_Defect_Dataset)**: RDD2022 + UAV-PDD2023 + RoadDamageVision, already in YOLO format.
- **DriveIndia pothole boxes**: those images have no crack labels, so pseudo-label cracks with the teacher before mixing them in.
- **BharatPothole** ([iWatchRoad](https://arxiv.org/abs/2508.10945), NISER; 7,000+ Indian dashcam frames): release status unclear, so check or email the authors.
- **Own IMU-auto-labeled bus frames** (§7).

**Training:**
- Fine-tune from TamAko783's n weights on an **India-weighted mix**: oversample RDD-India ×2–3 and own data ×5.
- Add bus-view augmentation: perspective, scale 0.5, motion blur, rain/fog, JPEG artifacts.
- Validate **separately** on RDD-India val and on your own clips.

**Fusion with the IMU:** each event is `{GPS, time, visual class + conf, IMU severity (existing 2-level classifier), IRI}`.

| Evidence | Meaning | Confidence |
|---|---|---|
| Camera + IMU | Confirmed defect | High |
| Camera only | Defect outside the wheel path (other lanes) | Kept, lower |
| IMU only | Night or occluded view | Kept, lower |

**Targets:**
- Pothole AP50 ≥ 0.70 on RDD-India val.
- ≥ 80% of IMU-detected potholes visually confirmed on your own daylight clips.

### M3 — SignCondition (YOLO26n-cls, 96×96 crops)

**Why a crop classifier:** fading and damage are fine-grained appearance cues. A 96-px classifier costs about 1–3 ms and runs once per physical sign.

**Classes:**
- v1: `ok`, `faded` (discoloured, peeling), `damaged` (bent, broken, tilted post, vandalized), `occluded` (vegetation, objects).
- v2 (optional): sign type (IRTSD's 37 Indian classes + `other`).

**Ready-made:** none for Indian signs. Hugging Face only has GTSRB, LISA and Vietnam sign models.

**Data:**
- **[IRTSD-Datasetv1](https://ieee-dataport.org/documents/irtsd-datasetv1-indian-road-traffic-sign-detection-dataset)** (IEEE DataPort, **covered by your subscription**): 5,141 images, 37 classes, 90+ Indian cities. Use it for sign crops and types.
- **ITSDD** and **ICTS** (Indian Cautionary Traffic Signs), also on IEEE DataPort: optional extras.
- **Condition labels:**
  - "A novel Traffic Sign Dataset with Condition Annotations" (Sandhu et al., IEEE SSCI 2023): 970 image pairs, 9 condition categories. Request it from the authors.
  - [Damaged Signs Dataset](https://www.kaggle.com/datasets/danielvareta/damaged-signs-dataset) (Kaggle).
  - **Synthetic damage** on IRTSD crops: HSV fading, peel/scratch textures, perspective bends, graffiti strokes, leaf occlusion by cut-and-paste.
  - **Own crops:** Indian roads have plenty of faded and bent signs, so a few hundred real crops go a long way.

**Missing signs (logic, not ML):**
- Build a **sign inventory** by clustering M1's sign detections by GPS + heading across trips.
- A sign seen in ≥ k earlier passes but absent in the last N passes is flagged "possibly missing" for human review.
- Stretch reference: **MTSVD** (IIIT-H, [CueCAn paper](https://arxiv.org/abs/2303.02641)): 200 scenes where a sign *should* be but isn't (left curve, right curve, gap-in-median, side-road-left), plus 10k sign tracks.

### M4 — SceneCls (YOLO26n-cls 224×224; MobileCLIP2-S0 as teacher and fallback)

**Labels** (trained as 3 small heads or 3 tiny classifiers):
- Water: `dry` / `wet` / `waterlogged`
- Divider: `present` / `broken_gap` / `absent` / `n.a.`
- Lane markings: `visible` / `faded` / `none`

**Ready-made:** nothing that is both street-level and Pi-sized. Hugging Face flood models are satellite-based, or large classifiers trained on generic disaster photos.

**Bootstrap with zero-shot labels:**
1. Run **[MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0)** on the GPU with text prompts such as "a flooded road" or "a road with a concrete median divider". It has a small image encoder with roughly CLIP ViT-B/16-level zero-shot accuracy, under a research-use licence (`apple-amlr`, fine for the hackathon).
2. Spot-check the resulting pseudo-labels by hand.
3. Train YOLO26n-cls on them.
4. If the trained classifier underperforms, deploy MobileCLIP2-S0's image encoder + linear heads on the Pi instead. It takes tens of ms, which is fine at ≤1 FPS; measure it.

**Data:**
- **Waterlogging:**
  - **[UW-Bench](https://github.com/zhang-chenxu/LSM-Adapter)**: 7,677 images with pixel masks (5,584 train / 2,093 test), surveillance + handheld views, China, access by signed agreement. Turn mask area into image-level labels.
  - **[Roadway Flooding Image Dataset](https://www.kaggle.com/datasets/saurabhshahane/roadway-flooding-image-dataset)**: 441 images.
  - **[FRED](https://arxiv.org/abs/2605.22018)** (2026): vehicle-mounted flooded-road data, camera + LiDAR. Check access.
  - **IDD-AW**: Indian adverse weather; use rain and wet frames as negatives.
  - **Own monsoon footage.** Tamil Nadu and coastal Andhra still get the north-east monsoon in Oct–Dec.
  - ⚠ **Puddle-1000 (cited in the roadmap) is no longer downloadable**: its CloudStor host was decommissioned.
- **Divider:**
  - **Mapillary Vistas v2.0**: 25k street images worldwide including India, 124 classes including *Road Median*, crosswalk, lane markings, water and pothole.
  - **IDD Segmentation**: curb, guard rail, wall.
  - Auto-derive image labels: *median pixels inside the road-centre band = divider present*.
- **Markings:**
  - **[CeyMo](https://github.com/oshadajay/CeyMo)**: 2,887 images at 1920×1080 from Sri Lankan roads, the closest public match to Indian markings; 11 classes including pedestrian crossing; mirrored on Hugging Face as `dronefreak/CeyMo`.
  - Vistas' lane-marking classes.

**Status:** Phase-2 / stretch for the finale. It can still be demoed through the zero-shot fallback.

### Zebra crossing condition (logic on M1's output)

- **Faded:** for each `zebra_crossing` box, take a 1-D intensity profile across the stripes and compute a **paint-contrast score** (FFT periodicity peak × stripe/asphalt luminance ratio).
  - A low score means faded.
  - No training, ~1 ms, easy to explain.
  - Calibrate the threshold on ~100 labeled crops. Use a tiny classifier only if the heuristic fails.
- **Missing:** OSM crossings (`highway=crossing` with `crossing=marked` or `crossing:markings`), plus approaches to schools and signalized junctions, where no zebra crossing is detected over N passes, are flagged "expected but not seen".
- ⚠ **Roadmap correction:** the RDD "D43 crosswalk blur / D44 white-line blur" labels come from the Japan-only RDD2018/2019 era. **RDD2022's official label set is 4 classes.** Don't rely on them for Indian zebra crossings; use DriveIndia's `Zebra crossing` class and CeyMo.

---

## 5. Ready-made model survey (priority 1)

| Task | Best candidate | Size | Licence | Verdict |
|---|---|---|---|---|
| Road damage | [TamAko783/YOLO26n_RDD_FRDC_Distilled_v2](https://huggingface.co/TamAko783/YOLO26n_RDD_FRDC_Distilled_v2) (+ [YOLO26s sibling](https://huggingface.co/TamAko783/YOLO26s_RDD_FRDC_Distilled_v2)) | 2.4 M (9 M) | AGPL-3.0 | ✅ **Deploy as baseline, fine-tune** |
| Road damage | [dronefreak/rdd2022-*](https://huggingface.co/dronefreak/rdd2022-yolov8n) (YOLOv8n/s/m, YOLO26s/m, RF-DETR n/s/m) | 3–30 M | AGPL-3.0 (YOLO) | Benchmark reference; RF-DETR-M as teacher |
| Road damage | [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022) (5 classes incl. Repair) | s | MIT | Skip: needs a YOLOv12 fork |
| Indian vehicles | [iisc-aim/UVH-26](https://huggingface.co/iisc-aim/UVH-26) | S–X | Apache-2.0 | Teacher (CCTV view) |
| Indian vehicles | [iisc-aim/BMD-45](https://huggingface.co/iisc-aim/BMD-45) | S–X | Apache-2.0 | Teacher (CCTV view) |
| Pedestrians | `yolo26n.pt` (COCO) | 2.4 M | AGPL-3.0 | ✅ Works as-is; fine-tuned inside M1 |
| Indian signs | — | — | — | ✗ Train (IRTSD) |
| Zebra crossing | [Aleton/Zebra_crossing](https://huggingface.co/Aleton/Zebra_crossing) (YOLOv8-seg, undocumented data) | n | MIT | ✗ Untrustworthy; train (DriveIndia / CeyMo) |
| Waterlogging | [prithivMLmods/Flood-Image-Detection](https://huggingface.co/prithivMLmods/Flood-Image-Detection) (SigLIP2 classifier) | large | — | ✗ Too heavy and not dashcam-view; MobileCLIP2-S0 instead |
| Labeling only (GPU) | [facebook/sam3](https://huggingface.co/facebook/sam3) (text-prompted segmentation, gated access), YOLOE / YOLO-World (open-vocabulary, in Ultralytics) | large | SAM licence / AGPL | Auto-labeling only, never on the Pi |

---

## 6. Dataset catalogue and access actions

| Dataset | Size | Relevant labels | View | Licence / access | Used by | Action |
|---|---|---|---|---|---|---|
| **DriveIndia** (TiHAN-IITH) | 66,986 imgs, 24 classes | vehicles incl. autorickshaw, pedestrian, pothole, zebra, speed bumps, signs, route boards, cones | front + rear dashcam, 1080p, 3,400+ km | CC BY 4.0; **Google form + EULA** on the [TiAND page](https://tihan.iith.ac.in/TiAND.html) | M1 primary, M2 potholes | **Request today** |
| **IDD Detection** (IIIT-H) | 46,588 imgs (31,569 train / 10,225 val) | incl. autorickshaw, rider, animal, sign | dashcam | Research licence; register at [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in/) | M1 | **Register today** |
| IDD Segmentation / IDD-AW | ~10k / adverse weather | curb, guard rail; rain / fog / night | dashcam | same | M4 | with the above |
| **RDD2022** | 47,420 imgs; India 9,665 | D00/D10/D20/D40 | phone on dashboard | CC BY-SA 4.0; YOLO-format mirror [dronefreak/RDD2022](https://huggingface.co/datasets/dronefreak/RDD2022) | M2 | Download now |
| Unified Road Defect | RDD2022 + UAV-PDD2023 + RoadDamageVision | 4 classes | ground + drone | mixed (research) | M2 | Optional |
| BharatPothole (iWatchRoad) | 7,000+ frames | pothole | Indian dashcam | CC BY-NC-SA (paper); release unclear | M2 | Email authors |
| UVH-26 / BMD-45 (IISc) | 26,646 / 45,986 imgs | 14 Indian vehicle classes | CCTV | CC BY 4.0 on HF | M1 teacher | As needed |
| **IRTSD-Datasetv1** | 5,141 imgs, 37 classes | Indian signs | phone, 90+ cities | IEEE DataPort (**your subscription**) | M3, M1 sign boxes | Download |
| ITSDD, ICTS | — | Indian signs | — | IEEE DataPort | M3 | Optional |
| Sign condition (Sandhu 2023) | 970 image pairs, 9 classes | sign damage types | — | Request from authors | M3 | Email |
| MTSVD (IIIT-H) | 200 scenes + 10k sign tracks | missing-sign contexts | Indian dashcam video | [C4MTS challenge](https://cvit.iiit.ac.in/ncvpripg2023/c4mtschallenge/) | stretch | Later |
| **Mapillary Vistas v2.0** | 25,000 imgs | Road Median, crosswalk, lane markings, water, pothole | global street-level | CC BY-NC-SA; registration | M4 | Register |
| **CeyMo** | 2,887 imgs | 11 road-marking classes incl. pedestrian crossing | Sri Lanka dashcam | research; HF mirror | M1 (zebra), M4 | Download |
| **UW-Bench** | 7,677 imgs, masks | waterlogging | CCTV + handheld | Signed access agreement | M4 | Request |
| Roadway Flooding | 441 imgs | flooded roads | mixed | Kaggle / Mendeley | M4 | Download |
| FRED (2026) | — | flooded roads, camera + LiDAR | vehicle | check | M4 | Check |
| OpenStreetMap (Overpass) | — | schools, crossings, signals, dual carriageways | — | ODbL | logic | Script it |

**Corrections to the roadmap (verified 24 Sep 2026):**
1. **"IDD-117K"**: no such release could be found. The verified IDD Detection set is 46,588 images.
2. **RDD D43/D44**: legacy Japan-only labels, not a usable source for Indian zebra crossings.
3. **Puddle-1000**: no longer downloadable.
4. **DriveIndia**: real and CC BY 4.0, but only through a request form + EULA. Request it early.
5. **MTSVD**: confirmed real (IIIT-H, 2023).

---

## 7. Filling label gaps (the data engine)

1. **Cross-dataset pseudo-labeling.** This lets you merge datasets without the "missing label = background" problem.
   - Train a teacher per label space (YOLO26m or RF-DETR-M, on Kaggle).
   - Run it over the *other* datasets' images and add confident boxes (conf ≥ 0.5, no overlap with ground truth).
   - Train the nano student on the union.
   - TamAko783's results show this mainly **improves cross-domain robustness**, not in-domain mAP, and robustness is exactly what a bus camera needs.
2. **Own bus footage: the biggest accuracy lever.** Bus cameras sit about **2.5–3 m high**, versus about 1.3 m for the car dashcams in RDD and DriveIndia, so perspective and object scale differ. Add windshield glare and vibration on top.
   - Record 1080p at 10–15 FPS, **synced to GPS and your IMU app on the same clock**, on 3–5 routes (day, night, rain).
   - Sample 1 frame/s. Pre-label with the M1/M2 teachers plus SAM 3 text prompts ("pothole", "zebra crossing", "road divider"), then correct in CVAT or Label Studio.
   - 500–1,000 verified frames give a noticeable domain boost.
   - **Hold out whole routes for validation.** Never split random frames: neighbouring frames leak between train and val.
3. **IMU-driven auto-labeling (the roadmap's differentiator).**
   - For every IMU pothole event, pull the frames from 2.0–0.3 s before impact.
   - The teacher and SAM 3 propose a box, and a human verifies it (seconds per image).
   - The result becomes M2 training data.
   - Frames where M2 fires but the IMU saw nothing in the wheel path go to a **hard-negative review queue**.
4. **Zero-shot bootstrap for weak classes** (waterlogging, divider, markings). MobileCLIP2, SAM 3 and YOLOE on the GPU produce candidate labels; a human spot-checks them; the result is distilled into M4.

---

## 8. Training plan: RTX 4050 local first, Kaggle for the heavy jobs

**Local machine** (i5-13420H, 32 GB RAM, RTX 4050 Laptop 6 GB, 420 GB free on D:):
- **Runs locally:** all nano/small student models, all fine-tunes, and the M3/M4 classifiers.
- **Settings:**
  - `batch=-1` (auto-sizes to ~60% of VRAM), `amp=True`, `imgsz=640`, `workers=6`, `patience=30`, `close_mosaic=10`.
  - `cache=disk`, because caching ~40k images of 640 px in RAM needs ~40–50 GB, more than your 32 GB.
- **Rough time for ~40k images at 640 px** (est.; time epoch 1 and re-plan):
  - YOLO26n: ~5–8 min/epoch, so 100 epochs is about one overnight run.
  - YOLO26s: ~10–15 min/epoch.
- **Setup:** keep the laptop plugged in, on the Best Performance power plan, and well ventilated.
- ⚠ **Learning rate:** with `optimizer=auto`, Ultralytics **ignores `lr0`**. When fine-tuning with a lower learning rate, set the optimizer explicitly.

**Kaggle** (free GPU, ~30 h/week; check your quota):
- **Runs on Kaggle:** teachers (YOLO26m/l, or Apache-2.0 RF-DETR-M) and large pseudo-labeling runs.
- **Sessions are capped at 12 h:** write `last.pt` to the output folder and use `resume=True`.

**WSL2** (already installed): use it for the Hailo compile if you buy a HAT. Compiling needs x86-64 Linux, and Ultralytics doesn't officially support WSL for this. Fallbacks: a Kaggle notebook or a Linux live USB.

```bash
# M2: fine-tune the ready-made road-damage model on the India-weighted mix
yolo detect train model=YOLO26n_RDD_FRDC_Distilled_v2.pt data=roadnet.yaml imgsz=640 epochs=80 \
     optimizer=SGD lr0=0.003 batch=-1 cache=disk workers=6 close_mosaic=10 patience=30

# M1: DriveIndia (+ IDD) from COCO weights
yolo detect train model=yolo26n.pt data=trafficnet.yaml imgsz=640 epochs=150 batch=-1 cache=disk workers=6

# M3 / M4: classifiers (folder-per-class datasets)
yolo classify train model=yolo26n-cls.pt data=datasets/sign_condition imgsz=96 epochs=60
yolo classify train model=yolo26n-cls.pt data=datasets/scene_water imgsz=224 epochs=60
```

---

## 9. Quantization and export pipeline

Ultralytics ≥ 8.4 uses `quantize=16|8`; the older `half=True` / `int8=True` still work but are deprecated.

1. **FP32 reference.** Run `yolo val` on (a) the dataset val split and (b) your own held-out route clips, and record both results.
2. **Export for the target:**
   - **Pi 5 CPU (default):** `model.export(format="ncnn", imgsz=(384, 640), quantize=16)`.
     - Accuracy change ≈ 0 and the file shrinks by half.
     - NCNN may already use FP16 arithmetic internally on the Cortex-A76, so measure the speed change rather than assume it.
     - Ultralytics' NCNN export has **no INT8 option**.
   - **INT8 on the CPU (experiment only).** Keep it only if it is *both* faster than NCNN FP16 *and* loses ≤ 2 mAP50 points. The bar to beat is 67 ms for YOLO26n at 640². Options:
     - LiteRT static INT8: `format="litert", quantize=8, data=calib.yaml`, run with XNNPACK.
     - NCNN's native INT8 tools (`ncnn2table` + `ncnn2int8`, outside Ultralytics).
     - ONNX Runtime INT8.
   - **Hailo (if bought):** `format="hailo", name="hailo8", data=calib.yaml` on Linux x86-64. Use DFC v3.x for Hailo-8/8L and v5.x for Hailo-10H. INT8 only.
     - Use ≥ 1,024 diverse calibration images.
     - YOLO26 confidences shift down by about 0.05 on Hailo, so re-tune thresholds.
   - **AI Camera / IMX500 (if bought):** YOLO11n only, `format="imx", data=calib.yaml`.
3. **Calibration set.** Use 500–1,000 frames from **your own bus footage**, spread across routes, day, night and rain. Calibration data must match what the camera sees in deployment, not the training datasets.
4. **Quantization-aware training (QAT) only if post-training quantization loses > 2 mAP50 points.**
   - Command: `yolo train ... quantize=8` (fake-quantization fine-tune, 5–20 epochs, tiny learning rate; runs on your RTX 4050 through NVIDIA ModelOpt).
   - Caveat: Ultralytics QAT checkpoints export **only to ONNX (Q/DQ) or TensorRT**. Other formats re-calibrate and reject a QAT checkpoint.
   - Ultralytics' own numbers show YOLO26n barely needs QAT (post-training quantization loses only ≈0.008 mAP50-95); it matters more for s/m models.
5. **Gate before freezing.** Accept an export only if all three hold:
   1. mAP50 on your own clips is within 1 point of FP32 (FP16) or within 2 points (INT8).
   2. p95 latency on the Pi fits the §2.2 budget **with the full pipeline running**.
   3. CPU temperature stays under ~80 °C after 30 min.
6. **Only if still over budget:** lower the input size first (e.g. M1 at 512×288). Structured pruning (Torch-Pruning) + fine-tuning typically cuts 20–40% of FLOPs for about 1 mAP point, but it adds export risk, so it is not in the default plan.

---

## 10. Evaluation

- **Offline:** mAP50 / mAP50-95, plus per-class precision and recall **at the operating threshold**, on both the dataset val split *and* your own held-out routes.
- **Streaming:** run the full pipeline on recorded route videos. Score per **event** (deduplicated by GPS), not per frame; see Edge-TSR's 20–30% still-to-video drop.
- **On-device:** p50 / p95 latency per model, end-to-end FPS, CPU %, temperature, power.
- **Field KPIs for the pitch:**
  - False alerts per 10 km.
  - % of IMU pothole events with visual confirmation.
  - Sign-inventory repeatability across trips.
  - Uplink bytes per km, event-only upload vs raw video.

---

## 11. Timeline

| When | ML work |
|---|---|
| **By 30 Sep (PPT round)** | No training needed. (1) **Submit access requests today:** DriveIndia form + EULA, IDD registration, Mapillary Vistas, UW-Bench agreement. Download RDD2022 (HF mirror) and IRTSD (IEEE DataPort). (2) Run the ready-made YOLO26n RDD model and COCO YOLO26n on a few Indian dashcam clips for screenshots. (3) If you have a Pi 5, export to NCNN and put **real latency numbers** in the PPT. |
| Oct, weeks 1–2 | M2 India-weighted fine-tune. M1 v1 on whichever arrives first, IDD or DriveIndia. Pi benchmark harness. Camera + GPS + IMU recording rig. |
| Oct, weeks 3–4 | Record bus footage. Teacher pseudo-labels. Verify 500–1,000 frames. M1 v2 on DriveIndia (+ IDD). M3 v1. |
| Nov, weeks 1–2 | M4 (zero-shot bootstrap → classifier). Distillation round. Export / quantization experiments. Full pipeline running on the Pi. |
| Nov, weeks 3–4 | Field test, error analysis, one retrain, **freeze models**. Record demo videos. |
| Dec (36-h finale) | Integration and demo only; **no training**. |

Tier-1 for the finale demo:
- Pothole / defect fusion (M2 + IMU).
- Vehicle density (M1 + tracking).
- Signboards (M1 + M3).
- Pedestrian / school-zone alerts (M1 + geofence).

M4 (waterlogging, dividers, markings) is the stretch item.

---

## 12. Risks and open decisions

- **Hardware (decide first).** Pi 4 or Pi 5, how much RAM, and whether an AI HAT+ or AI Camera can be bought. This changes the base architecture (§2.3). *Default assumed here: Pi 5 (8 GB), CPU only.*
- **Licences.**
  - Ultralytics code and weights, including the ready-made RDD models, are **AGPL-3.0**. That's fine for SIH as long as your code is open. A BEL product would need an Ultralytics enterprise licence or a switch to Apache-2.0 detectors (RF-DETR, D-FINE).
  - IDD and Mapillary Vistas are non-commercial research licences.
  - MobileCLIP2 is research-use only.
- **Domain gap** (bus height, windshield glare, vibration): your own footage is not optional.
- **DriveIndia approval delay**: fall back to IDD for M1 v1.
- **Waterlogging data is weak**: keep it a stretch goal with the zero-shot fallback.
- **Thermal**: a Pi 5 in a bus cabin needs active cooling and a CPU budget of ≤ 60%.

---

## Sources

- Ultralytics Raspberry Pi guide (Pi 5 benchmarks): https://docs.ultralytics.com/guides/raspberry-pi/
- Ultralytics export (formats, `quantize`, QAT): https://docs.ultralytics.com/modes/export/
- Ultralytics Hailo export: https://docs.ultralytics.com/integrations/hailo
- Ultralytics Sony IMX500 export: https://docs.ultralytics.com/integrations/sony-imx500/
- YOLO26 overview: https://docs.ultralytics.com/models/yolo26 · https://www.ultralytics.com/blog/ultralytics-yolo26-the-new-standard-for-edge-first-vision-ai
- Raspberry Pi AI HATs: https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html
- AI HAT+ 2 review (CNX Software): https://www.cnx-software.com/2026/01/20/raspberry-pi-ai-hat-2-review-a-40-tops-ai-accelerator-tested-with-computer-vision-llm-and-vlm-workloads/
- RDD2022 paper: https://arxiv.org/abs/2209.08538 · repo: https://github.com/sekilab/RoadDamageDetector
- TamAko783 distilled YOLO26 road-damage models: https://huggingface.co/TamAko783/YOLO26n_RDD_FRDC_Distilled_v2
- DetectionBench RDD2022 models: https://huggingface.co/dronefreak/rdd2022-yolo26s
- UVH-26: https://huggingface.co/iisc-aim/UVH-26 · https://arxiv.org/abs/2511.02563
- BMD-45: https://huggingface.co/datasets/iisc-aim/BMD-45 · https://arxiv.org/abs/2604.24419
- DriveIndia: https://arxiv.org/abs/2507.19912 · access: https://tihan.iith.ac.in/TiAND.html
- IDD: https://idd.insaan.iiit.ac.in/ · stats: https://datasetninja.com/idd-detection
- IRTSD-Datasetv1: https://ieee-dataport.org/documents/irtsd-datasetv1-indian-road-traffic-sign-detection-dataset
- MTSVD / CueCAn: https://arxiv.org/abs/2303.02641
- Sign condition dataset (Sandhu et al. 2023): https://ieeexplore.ieee.org/document/10371993/
- Mapillary Vistas: https://www.mapillary.com/dataset/vistas
- CeyMo: https://github.com/oshadajay/CeyMo
- UW-Bench: https://arxiv.org/abs/2407.08109 · https://github.com/zhang-chenxu/LSM-Adapter
- Puddle-1000 status (inaccessible): https://arxiv.org/pdf/2504.05112
- Roadway Flooding Image Dataset: https://www.kaggle.com/datasets/saurabhshahane/roadway-flooding-image-dataset
- FRED: https://arxiv.org/abs/2605.22018
- iWatchRoad / BharatPothole: https://arxiv.org/abs/2508.10945
- Edge-TSR (streaming vs benchmark gap): https://arxiv.org/abs/2606.17241
- MobileCLIP2: https://huggingface.co/apple/MobileCLIP2-S0
- SAM 3: https://huggingface.co/facebook/sam3
