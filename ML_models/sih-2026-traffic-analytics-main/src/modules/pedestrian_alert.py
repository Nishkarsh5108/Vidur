"""

Vulnerable Pedestrian / School-Zone Crossing Detection Module.

Consumes tracker detections, filters for IDD 'person' and 'rider' classes,
evaluates school-zone geofence proximity (via OpenStreetMap Overpass API or mock),
and cross-references with time-of-day school hours to raise high-priority safety alerts.
"""

import time
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import cv2
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.geofence import SchoolGeofenceManager, SchoolZone


# IDD vulnerable classes
DEFAULT_PEDESTRIAN_CLASS_IDS = {0, 1}  # 0: person, 1: rider
DEFAULT_PEDESTRIAN_NAMES = {0: "person", 1: "rider"}

# COCO vulnerable classes
COCO_PEDESTRIAN_CLASS_IDS = {0}        # 0: person
COCO_PEDESTRIAN_NAMES = {0: "person"}


@dataclass
class PedestrianAlertResult:
    frame_timestamp: float
    simulated_time_str: str
    pedestrian_count: int
    rider_count: int
    total_vulnerable_count: int
    is_school_hours: bool
    active_school_window: Optional[str]
    in_school_geofence: bool
    matched_school_name: Optional[str]
    alert_level: str                         # "NORMAL", "ADVISORY", "CRITICAL"
    alert_message: str
    vulnerable_track_ids: List[int]


class PedestrianAlertSystem:
    def __init__(
        self,
        geofence_manager: Optional[SchoolGeofenceManager] = None,
        camera_coords: Optional[Tuple[float, float]] = None,
        school_windows: Optional[List[Tuple[str, str]]] = None,
        pedestrian_class_ids: Optional[set] = None,
        frame_crossing_roi: Optional[np.ndarray] = None,
        simulated_time: Optional[datetime.time] = None
    ):
        """
        geofence_manager: SchoolGeofenceManager instance
        camera_coords: (lat, lon) coordinates of camera/intersection (default: Bengaluru St. Joseph's)
        school_windows: List of ("HH:MM", "HH:MM") tuples for peak school commute
        frame_crossing_roi: Optional polygon in video frame representing school crossing / zebra zone
        simulated_time: datetime.time to override clock for live hackathon demos
        """
        self.geofence_mgr = geofence_manager or SchoolGeofenceManager(city_name="Bengaluru", use_mock=True)
        # Default camera placed at St. Joseph's Indian High School, Bengaluru (lat, lon)
        self.camera_coords = camera_coords or (12.9716, 77.5946)
        
        # Standard Indian School Opening & Dismissal Hours
        self.school_windows = school_windows or [
            ("07:30", "09:30"),  # Morning drop-off peak
            ("13:30", "15:30")   # Afternoon dismissal peak
        ]

        self.pedestrian_class_ids = pedestrian_class_ids or DEFAULT_PEDESTRIAN_CLASS_IDS
        self.frame_crossing_roi = frame_crossing_roi
        self.simulated_time = simulated_time

        # Pre-check camera location against school geofence
        self.camera_in_geofence, self.nearby_school_name, _ = self.geofence_mgr.check_coordinates(
            self.camera_coords[0], self.camera_coords[1]
        )

    def enable_coco_mode(self):
        """Switches filter to COCO classes for early testing before IDD weights."""
        self.pedestrian_class_ids = COCO_PEDESTRIAN_CLASS_IDS

    def set_crossing_roi(self, frame_width: int, frame_height: int):
        """Defines a default school-crossing corridor in the middle-lower region of frame."""
        self.frame_crossing_roi = np.array([
            [int(frame_width * 0.15), int(frame_height * 0.55)],
            [int(frame_width * 0.85), int(frame_height * 0.55)],
            [int(frame_width * 0.90), int(frame_height * 0.85)],
            [int(frame_width * 0.10), int(frame_height * 0.85)]
        ], np.int32)

    def set_demo_school_hours(self, is_active: bool = True):
        """Helper to force school hours active/inactive for presentations."""
        if is_active:
            self.simulated_time = datetime.time(8, 15)  # 08:15 AM
        else:
            self.simulated_time = datetime.time(22, 30) # 10:30 PM

    def is_current_time_in_school_hours(self, now: Optional[datetime.datetime] = None) -> Tuple[bool, Optional[str], str]:
        """
        Evaluates whether the given or current time falls into school commute hours.
        Returns: (is_active, active_window_str, current_time_str)
        """
        if self.simulated_time is not None:
            cur_time = self.simulated_time
            time_str = cur_time.strftime("%H:%M:%S") + " (Simulated Demo Time)"
        else:
            if now is None:
                now = datetime.datetime.now()
            cur_time = now.time()
            time_str = cur_time.strftime("%H:%M:%S")

        cur_hm = (cur_time.hour, cur_time.minute)

        for start_str, end_str in self.school_windows:
            s_h, s_m = map(int, start_str.split(":"))
            e_h, e_m = map(int, end_str.split(":"))
            if (s_h, s_m) <= cur_hm <= (e_h, e_m):
                window_str = f"{start_str} - {end_str}"
                return True, window_str, time_str

        return False, None, time_str

    def _point_in_crossing(self, point: Tuple[float, float]) -> bool:
        """Checks if pedestrian bottom-center is within designated crossing ROI."""
        if self.frame_crossing_roi is None:
            return True  # If no ROI specified, whole camera zone is treated as school vicinity
        return cv2.pointPolygonTest(self.frame_crossing_roi, (float(point[0]), float(point[1])), False) >= 0

    def update(
        self,
        detections: List[Dict[str, Any]],
        current_time: Optional[float] = None
    ) -> PedestrianAlertResult:
        """
        Process detections from one frame.
        Evaluates presence of 'person' and 'rider', checks geofence, and triggers alert.
        """
        if current_time is None:
            current_time = time.time()

        is_school_hours, active_window, time_str = self.is_current_time_in_school_hours()

        pedestrian_count = 0
        rider_count = 0
        vulnerable_ids = []

        for det in detections:
            cls_id = int(det.get("class_id", -1))
            if cls_id not in self.pedestrian_class_ids:
                continue

            track_id = det.get("track_id", 0)
            bbox = det["bbox"]
            footprint = ((bbox[0] + bbox[2]) / 2.0, bbox[3])

            # Check if pedestrian is inside crossing zone
            if self._point_in_crossing(footprint):
                if cls_id == 0:
                    pedestrian_count += 1
                elif cls_id == 1:
                    rider_count += 1
                vulnerable_ids.append(track_id)

        total_vulnerable = pedestrian_count + rider_count

        # Alert Decision Logic
        if not self.camera_in_geofence:
            alert_level = "NORMAL"
            alert_msg = "Location outside school geofence."
        elif total_vulnerable == 0:
            alert_level = "NORMAL"
            alert_msg = f"School Zone: {self.nearby_school_name} | No pedestrians detected."
        elif is_school_hours and total_vulnerable > 0:
            # BOTH CONDITIONS TRUE -> HIGH PRIORITY CRITICAL ALERT!
            alert_level = "CRITICAL"
            alert_msg = (
                f"VULNERABLE SCHOOL-ZONE CROSSING ALERT! "
                f"[{total_vulnerable} Pedestrian(s)/Rider(s) in {self.nearby_school_name} during peak hours ({active_window})]"
            )
        else:
            # Pedestrians in school zone, but outside operating hours
            alert_level = "ADVISORY"
            alert_msg = f"Advisory: {total_vulnerable} Pedestrian(s) near school outside peak hours."

        return PedestrianAlertResult(
            frame_timestamp=current_time,
            simulated_time_str=time_str,
            pedestrian_count=pedestrian_count,
            rider_count=rider_count,
            total_vulnerable_count=total_vulnerable,
            is_school_hours=is_school_hours,
            active_school_window=active_window,
            in_school_geofence=self.camera_in_geofence,
            matched_school_name=self.nearby_school_name,
            alert_level=alert_level,
            alert_message=alert_msg,
            vulnerable_track_ids=vulnerable_ids
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: PedestrianAlertResult,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Renders crossing zone polygon, pedestrian bounding boxes, and alert HUD."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw crossing zone polygon
        if self.frame_crossing_roi is not None:
            roi_color = (0, 0, 255) if result.alert_level == "CRITICAL" else (255, 200, 0)
            cv2.polylines(overlay, [self.frame_crossing_roi], isClosed=True, color=roi_color, thickness=2)
            cv2.putText(overlay, "School Crossing Geofence Zone",
                        (self.frame_crossing_roi[0][0], max(20, self.frame_crossing_roi[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

        # Draw pedestrian detections
        vul_set = set(result.vulnerable_track_ids)
        for det in detections:
            tid = det.get("track_id", 0)
            if tid not in vul_set:
                continue

            x1, y1, x2, y2 = map(int, det["bbox"])
            box_color = (0, 0, 255) if result.alert_level == "CRITICAL" else (0, 255, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

            tag = f"Pedestrian #{tid}"
            if result.alert_level == "CRITICAL":
                tag += " [VULNERABLE]"
            cv2.putText(overlay, tag, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # Draw Pedestrian & School Zone Alert HUD in bottom
        hud_h = 80
        hud_w = w - 40
        hud_y = h - hud_h - 10
        cv2.rectangle(overlay, (20, hud_y), (20 + hud_w, hud_y + hud_h), (20, 20, 20), -1)

        border_color = (0, 0, 255) if result.alert_level == "CRITICAL" else ((0, 200, 255) if result.alert_level == "ADVISORY" else (80, 80, 80))
        cv2.rectangle(overlay, (20, hud_y), (20 + hud_w, hud_y + hud_h), border_color, 2)

        # Title line
        status_tag = f"[{result.alert_level}]"
        cv2.putText(overlay, f"SCHOOL-ZONE GEOFENCE: {result.matched_school_name or 'None'} {status_tag}",
                    (35, hud_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2)

        # Detail line
        time_tag = f"Time: {result.simulated_time_str} | Peak Hours: {'ACTIVE' if result.is_school_hours else 'INACTIVE'}"
        count_tag = f"Vulnerable Count: {result.total_vulnerable_count}"
        cv2.putText(overlay, f"{time_tag} | {count_tag}",
                    (35, hud_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        return overlay


# Self-test when executed directly
if __name__ == "__main__":
    print("[*] Running standalone PedestrianAlertSystem verification...")
    system = PedestrianAlertSystem()
    
    # Test A: School hours active + 2 pedestrians -> Must be CRITICAL
    system.set_demo_school_hours(is_active=True)
    mock_detections = [
        {"track_id": 1, "class_id": 0, "bbox": [100, 100, 150, 200]},
        {"track_id": 2, "class_id": 1, "bbox": [200, 100, 250, 200]},
        {"track_id": 3, "class_id": 2, "bbox": [300, 100, 450, 250]}  # car, should be ignored
    ]
    res_a = system.update(mock_detections)
    print(f"Test A (Peak Hours + Pedestrians): Alert={res_a.alert_level} | Msg={res_a.alert_message}")
    assert res_a.alert_level == "CRITICAL", f"Expected CRITICAL, got {res_a.alert_level}"

    # Test B: School hours inactive (night) + 2 pedestrians -> Must be ADVISORY
    system.set_demo_school_hours(is_active=False)
    res_b = system.update(mock_detections)
    print(f"Test B (Off-Peak + Pedestrians): Alert={res_b.alert_level} | Msg={res_b.alert_message}")
    assert res_b.alert_level == "ADVISORY", f"Expected ADVISORY, got {res_b.alert_level}"

    # Test C: Peak hours + 0 pedestrians -> Must be NORMAL
    system.set_demo_school_hours(is_active=True)
    res_c = system.update([])
    print(f"Test C (Peak Hours + 0 Pedestrians): Alert={res_c.alert_level} | Msg={res_c.alert_message}")
    assert res_c.alert_level == "NORMAL", f"Expected NORMAL, got {res_c.alert_level}"

    print("[+] PedestrianAlertSystem unit tests PASSED successfully!")
