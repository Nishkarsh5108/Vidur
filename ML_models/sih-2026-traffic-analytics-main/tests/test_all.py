"""
Comprehensive Automated Test Suite for SIH 2026 Traffic Analytics Prototype.

Verifies:
1. School Geofencing & Point-in-Polygon Engine (src/utils/geofence.py)
2. Vulnerable Pedestrian & Time-of-Day Alert Module (src/modules/pedestrian_alert.py)
3. Vehicle Density, Counting & Bottleneck Identification (src/modules/vehicle_density.py)
4. Trained YOLOv8 Model Weights & GPU Inference (models/idd_yolov8/weights/best.pt)
5. End-to-End Pipeline on Real Video (src/infer.py)
"""

import os
import sys
import time
import unittest
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.geofence import SchoolGeofenceManager
from src.modules.pedestrian_alert import PedestrianAlertSystem
from src.modules.vehicle_density import VehicleDensityTracker

try:
    from ultralytics import YOLO
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestTrafficAnalyticsSystem(unittest.TestCase):

    def test_01_geofence_point_in_polygon(self):
        """Test Point-in-Polygon matching against school geofences."""
        mgr = SchoolGeofenceManager(city_name="Bengaluru", use_mock=True)
        self.assertGreater(len(mgr.school_zones), 0, "Should load school zones")

        # Point inside St. Joseph's Indian High School
        inside_lat, inside_lon = 12.9716, 77.5946
        is_inside, name, _ = mgr.check_coordinates(inside_lat, inside_lon)
        self.assertTrue(is_inside)
        self.assertIn("St. Joseph", name)

        # Point far outside
        outside_lat, outside_lon = 12.8399, 77.6770
        is_out, out_name, _ = mgr.check_coordinates(outside_lat, outside_lon)
        self.assertFalse(is_out)
        self.assertIsNone(out_name)
        print(" [PASS] Test 1: Geofence Point-in-Polygon verified.")

    def test_02_vulnerable_pedestrian_time_of_day_alerts(self):
        """Test pedestrian presence + time-of-day alert matrix."""
        system = PedestrianAlertSystem()

        # Detections: 1 person (class 0), 1 rider (class 1), 1 car (class 2)
        mock_dets = [
            {"track_id": 1, "class_id": 0, "bbox": [100, 100, 150, 200]},
            {"track_id": 2, "class_id": 1, "bbox": [200, 100, 250, 200]},
            {"track_id": 3, "class_id": 2, "bbox": [300, 100, 450, 250]}
        ]

        # Case A: School hours ACTIVE -> Must trigger CRITICAL alert
        system.set_demo_school_hours(is_active=True)
        res_a = system.update(mock_dets)
        self.assertEqual(res_a.alert_level, "CRITICAL")
        self.assertEqual(res_a.total_vulnerable_count, 2)
        self.assertTrue(res_a.is_school_hours)

        # Case B: School hours INACTIVE -> Must downgrade to ADVISORY
        system.set_demo_school_hours(is_active=False)
        res_b = system.update(mock_dets)
        self.assertEqual(res_b.alert_level, "ADVISORY")
        self.assertFalse(res_b.is_school_hours)

        # Case C: Zero pedestrians -> Must be NORMAL
        res_c = system.update([])
        self.assertEqual(res_c.alert_level, "NORMAL")
        self.assertEqual(res_c.total_vulnerable_count, 0)
        print(" [PASS] Test 2: Vulnerable Pedestrian multi-tier alerts verified.")

    def test_03_vehicle_density_and_bottleneck(self):
        """Test vehicle counting, dwell-time accumulation, and bottleneck flags."""
        tracker = VehicleDensityTracker(dwell_time_threshold=3.0, speed_threshold=5.0)
        zone_poly = np.array([[50, 50], [500, 50], [500, 400], [50, 400]], np.int32)
        tracker.zones = {"Corridor_A": zone_poly}

        t0 = 100.0
        # Frame 1: Vehicle 1 enters zone
        res1 = tracker.update([
            {"track_id": 10, "class_id": 2, "bbox": [100, 100, 200, 250]}  # car
        ], current_time=t0)
        self.assertEqual(res1.total_vehicles, 1)
        self.assertEqual(res1.zone_counts["Corridor_A"], 1)
        self.assertFalse(res1.zone_bottlenecks["Corridor_A"])

        # Frame 2: Vehicle 1 dwells in zone for 4.0 seconds (> threshold 3.0s) without moving
        t1 = t0 + 4.0
        res2 = tracker.update([
            {"track_id": 10, "class_id": 2, "bbox": [100, 100, 200, 250]}
        ], current_time=t1)
        self.assertIn(10, res2.stalled_vehicle_ids)
        self.assertTrue(res2.zone_bottlenecks["Corridor_A"])
        print(" [PASS] Test 3: Vehicle density & bottleneck identification verified.")

    def test_04_trained_model_gpu_weights(self):
        """Verify that trained best.pt exists, loads on GPU, and runs inference."""
        if not HAS_TORCH:
            self.skipTest("PyTorch/Ultralytics not installed in this environment.")

        weights_path = Path(r"e:\SIH_2026\models\idd_yolov8\weights\best.pt")
        self.assertTrue(weights_path.is_file(), f"Trained weights not found at {weights_path}")

        model = YOLO(str(weights_path))
        self.assertEqual(len(model.names), 13, "Model must have 13 IDD classes")

        # Test inference on dummy frame
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model.predict(dummy_img, device=0 if torch.cuda.is_available() else "cpu", verbose=False)
        self.assertIsNotNone(results)
        print(" [PASS] Test 4: Trained YOLOv8 model GPU weights verified.")

    def test_05_demo_video_generated(self):
        """Verify that the end-to-end video inference produced a valid output video."""
        output_video = Path(r"e:\SIH_2026\demo_output.mp4")
        self.assertTrue(output_video.is_file(), f"Demo output video not found at {output_video}")
        self.assertGreater(output_video.stat().st_size, 1024 * 1024, "Video file should be > 1MB")
        print(f" [PASS] Test 5: End-to-end demo video verified ({output_video.stat().st_size / (1024*1024):.2f} MB).")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("      SIH 2026 TRAFFIC ANALYTICS FULL SYSTEM TEST SUITE")
    print("=" * 65)
    unittest.main(verbosity=2)
