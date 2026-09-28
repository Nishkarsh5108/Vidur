"""
End-to-End Traffic Analytics Inference Pipeline.

Connects:
1. YOLOv8 Multi-Class Detector + ByteTrack Object Tracker
2. Vehicle Density & Bottleneck Module (src/modules/vehicle_density.py)
3. Vulnerable Pedestrian & School-Zone Geofence Alert Module (src/modules/pedestrian_alert.py)
4. Unified Live HUD Visualizer with Video/Webcam Stream Support
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.modules.vehicle_density import VehicleDensityTracker
from src.modules.pedestrian_alert import PedestrianAlertSystem
from src.utils.geofence import SchoolGeofenceManager

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def run_pipeline(
    source: str = "0",
    weights: str = "models/idd_yolov8/weights/best.pt",
    coco_mode: bool = False,
    simulate_school_hours: bool = True,
    output_path: str = None,
    show: bool = False,
    max_frames: int = None
):
    """
    Executes the shared detector+tracker and passes output to both decoupled modules.
    """
    print("=" * 65)
    print("      SIH 2026 TRAFFIC ANALYTICS INFERENCE PIPELINE")
    print("=" * 65)

    # 1. Determine Model Weights
    weights_path = Path(weights)
    use_coco = coco_mode

    if not weights_path.is_file():
        if not use_coco:
            print(f"[!] Custom weights '{weights}' not found. Falling back to 'yolov8n.pt' in COCO mode.")
            use_coco = True
        weights_to_load = "yolov8n.pt"
    else:
        weights_to_load = str(weights_path)

    print(f"[*] Loading YOLO Detector: {weights_to_load} (COCO Mode: {use_coco})")
    if not HAS_YOLO:
        print("[!] Ultralytics not installed. Please run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(weights_to_load)

    # 2. Initialize Decoupled Modules
    print("[*] Initializing Vehicle Density & Bottleneck Tracker...")
    vehicle_tracker = VehicleDensityTracker(
        dwell_time_threshold=6.0,   # 6 seconds stationary = stall
        speed_threshold=10.0,       # px/s
        bottleneck_density_threshold=4
    )
    if use_coco:
        vehicle_tracker.enable_coco_mode()

    print("[*] Initializing Pedestrian & School-Zone Geofence System...")
    geofence_mgr = SchoolGeofenceManager(city_name="Bengaluru", use_mock=True)
    pedestrian_system = PedestrianAlertSystem(
        geofence_manager=geofence_mgr,
        camera_coords=(12.9716, 77.5946)  # St. Joseph's Indian High School, Bengaluru
    )
    if use_coco:
        pedestrian_system.enable_coco_mode()
    if simulate_school_hours:
        pedestrian_system.set_demo_school_hours(is_active=True)

    # 3. Open Video Source
    is_webcam = source.isdigit()
    cap_src = int(source) if is_webcam else source
    cap = cv2.VideoCapture(cap_src)

    if not cap.isOpened():
        print(f"[!] Could not open video source: {source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"[*] Input Source: {source} ({width}x{height} @ {fps:.1f} FPS)")

    # Configure default video ROIs based on frame dimensions
    vehicle_tracker.set_default_roi(width, height)
    pedestrian_system.set_crossing_roi(width, height)

    # Video Writer setup if output requested
    writer = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
        print(f"[*] Recording output to: {output_path}")

    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[*] End of video stream reached.")
                break

            frame_idx += 1
            cur_time = time.time()

            # Execute YOLOv8 + ByteTrack
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            # Extract tracker detections
            parsed_detections = []
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                cls_ids = results[0].boxes.cls.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, tid, cid, conf in zip(boxes, track_ids, cls_ids, confs):
                    parsed_detections.append({
                        "bbox": box.tolist(),
                        "track_id": int(tid),
                        "class_id": int(cid),
                        "confidence": float(conf)
                    })

            # Update Module 1: Vehicle Density & Bottlenecks
            v_res = vehicle_tracker.update(parsed_detections, current_time=cur_time)

            # Update Module 2: Vulnerable Pedestrian & School-Zone Alert
            p_res = pedestrian_system.update(parsed_detections, current_time=cur_time)

            # Composite Visual Overlays
            frame_annotated = vehicle_tracker.draw_overlay(frame, v_res, parsed_detections)
            frame_annotated = pedestrian_system.draw_overlay(frame_annotated, p_res, parsed_detections)

            # Pipeline FPS Tag
            elapsed = time.time() - t_start
            cur_fps = frame_idx / max(1e-3, elapsed)
            cv2.putText(frame_annotated, f"PIPELINE: {cur_fps:.1f} FPS", (width - 190, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if writer:
                writer.write(frame_annotated)

            if show:
                cv2.imshow("SIH 2026 Traffic Analytics - Dual Module HUD", frame_annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[*] User quit requested.")
                    break

            if max_frames and frame_idx >= max_frames:
                print(f"[*] Reached target frame limit ({max_frames}). Stopping.")
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"[+] Processed {frame_idx} frames in {time.time() - t_start:.2f}s ({cur_fps:.1f} FPS).")


def main():
    parser = argparse.ArgumentParser(description="SIH 2026 Traffic Analytics Inference Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video file path or webcam index ('0')")
    parser.add_argument("--weights", type=str, default="models/idd_yolov8/weights/best.pt", help="YOLO checkpoint path")
    parser.add_argument("--coco", action="store_true", help="Force COCO mode for testing with base yolov8n.pt")
    parser.add_argument("--school-hours", action="store_true", default=True, help="Simulate active school hours for demo")
    parser.add_argument("--no-school-hours", dest="school_hours", action="store_false")
    parser.add_argument("--output", type=str, default="demo_output.mp4", help="Path to save annotated video")
    parser.add_argument("--show", action="store_true", help="Display GUI preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        weights=args.weights,
        coco_mode=args.coco,
        simulate_school_hours=args.school_hours,
        output_path=args.output,
        show=args.show,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
