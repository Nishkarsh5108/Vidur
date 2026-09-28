# V1 successor: a nano road-surface model

*28 Sep 2026. Plan for replacing V1 (YOLOv8m: potholes and zebra crossings) with a model cheap enough to run continuously on a Raspberry Pi 5 CPU. V1 stays in the demo and in the edge agent until its successor beats it on the test set defined below.*

---

## 1. Why replace V1

- **Cost.** V1 needs about 74 GFLOPs per 480×800 frame. A YOLO26n on a 640×256 crop of the road ahead needs about 2.2 GFLOPs, about 30× less.
  - V1 then no longer has to be limited to IMU-triggered frames. A continuous 2 FPS scan would cost about 5–6% of a Pi 5's CPU (est., [edge-deployment.md §4](edge-deployment.md)).
  - That scan catches potholes the wheels never hit: the other lane and the road edge.
- **Coverage.** The successor adds cracks (RDD classes D00, D10 and D20). V1 left them out on purpose, but the problem statement asks for "damaged roads".
- **Reproducibility.** V1 cannot be retrained as it stands:
  - the script that merged its training data is not in the repository;
  - the committed `src/train.py` does not match the checkpoint's training arguments.
- **Not a verdict on V1's quality.** Its checkpoint metrics (mAP50 0.674 on its validation split) are real. Its known bugs were in the wrapper code, and have been fixed. The reason for replacing it is compute.

## 2. Target model

| | |
|---|---|
| Classes | `pothole`, `crack` (D00/D10/D20 merged, unless crack types are needed), `zebra_crossing` |
| Architecture | **YOLO26n** for the Pi 5 CPU. **YOLOv8n** if an AI HAT+ is bought: it keeps about 100% of its mAP50 in Hailo INT8, against about 93% for YOLO26n. |
| Input | The road ahead: the lower half of the frame and the central ~70% of its width (calibrated once per camera mount), resized to 640×256 |
| Starting weights | CPU path: [TamAko783/YOLO26n_RDD_FRDC_Distilled_v2](https://huggingface.co/TamAko783/YOLO26n_RDD_FRDC_Distilled_v2) (2.4 M parameters; mAP50 0.638 on a 4,509-image RDD validation set). Hailo path: [dronefreak/rdd2022](https://huggingface.co/dronefreak/rdd2022-yolo26s) YOLOv8n (58.8 mAP50 on the RDD2022 test split). |
| Est. cost on a Pi 5 | About 27 ms per crop (est.) |
| Integration | Same interface as [edge/runners/v1_roadsense.py](../edge/runners/v1_roadsense.py), plus a road-crop step. It is switched in by config. The continuous scan is enabled with `deferred.v1_sweep_fps`. |

## 3. Defining "as good"

- **Do not compare against V1's 0.674.** That figure comes from an unknown validation split with two classes, one of which (zebra crossings) is large and easy to detect. Numbers from different validation sets cannot be compared.
- **Build one frozen test set first:**
  - the RDD2022-India validation split;
  - 300–500 frames from your own rides, with potholes and crossings labelled.

  Never train on it, and split it by drive, not by frame.
- **Measure two things:**
  - per-frame pothole AP50;
  - **event recall** on recorded rides: the share of real potholes that end up reported.
- **Why event recall matters most.** At 30 km/h a pothole stays within 5–25 m ahead of the bus for about 2.4 s. A 2 FPS scan looks at it 4–5 times, so a nano model that is a few points worse per frame can still find more potholes than V1 limited to triggers.
- **Adopt the successor when all of these hold:**
  - pothole AP50 within about 3 points of V1 on the frozen test set;
  - event recall at least that of the V1 trigger pipeline on recorded rides;
  - pothole AP50 of at least 0.70 on the RDD-India validation split (target).

## 4. Steps

1. **Frozen test set.** Build it first; everything else is measured against it.
2. **Baselines.** Run V1 at 480×800 and the ready-made nano model, as they are, on the test set. If the nano model is already close on potholes, most of the work is done.
3. **Classes.** As in §2.
4. **Fill in missing labels, using V1 as the teacher.**
   - The road-damage images have no crossing labels, and the crossing datasets have no pothole labels.
   - Run V1 over both to pseudo-label the missing class (confidence ≥ 0.5), then check a few hundred by hand.
5. **Fine-tune** from the starting weights on an India-weighted mix: RDD-India ×2–3, own frames ×5.
   - Add augmentation that mimics a bus camera: perspective, scale, motion blur, rain, JPEG artefacts.
   - Crop the training images the same way inference will crop.
6. **Data hygiene.**
   - Split by drive or sequence, never by random frame. V2's validation scores are inflated by exactly that mistake.
   - Remove near-duplicates from the Kaggle pothole sets, which often repeat across splits.
   - Commit the list of datasets and the merge script.
7. **Export and deploy.**
   - Export: NCNN FP16 at 256×640, or Hailo INT8 ([edge-deployment.md §3](edge-deployment.md)).
   - Benchmark on the Pi, then switch it in by config.
   - Keep V1 as the fallback.

## 5. Data

- **[RDD2022](https://github.com/sekilab/RoadDamageDetector).**
  - The India subset has 9,665 images and 3,187 pothole labels in its training split, more than any other country.
  - The Indian images were shot at 960×720 and squashed to 720×720.
- **[Unified Road Defect Dataset](https://huggingface.co/datasets/TamAko783/Unified_Road_Defect_Dataset).** RDD2022 + UAV-PDD2023 + RoadDamageVision, already in YOLO format.
- **Crossing data.** CrosswalkCDNet and the other sources V1 was trained on. Ask V1's author for the exact list.
- **Your own rides.**
  - Frames from 1.5–0.6 s before each IMU jolt are weak pothole labels, the roadmap's "auto-labelling from IMU ground truth".
  - Bus cameras sit about 2.5–3 m high, against about 1.3 m for the car dashcams in RDD, so your own footage is the largest single accuracy lever.

## 6. Timeline

| When | What |
|---|---|
| Before 30 Sep | No training. At most, run the ready-made model on a few Indian clips for a screenshot. |
| October, week 1 | Record rides (also needed for INT8 calibration). Build the frozen test set. Measure the baselines. |
| October, weeks 2–3 | Pseudo-label, fine-tune, evaluate, export |
| After that | Adopt it if it meets §3; otherwise keep V1 and keep it trigger-only |
