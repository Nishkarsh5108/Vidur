"""
Vehicle Density, Classification, Counting, and Bottleneck Identification Module.

Consumes tracker outputs (ByteTrack), filters for vehicle classes, tracks per-zone
occupancy, calculates dwell time and speed, and flags traffic bottlenecks.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import cv2


# IDD default vehicle classes
DEFAULT_VEHICLE_CLASS_IDS = {2, 3, 4, 5, 6, 7}  # car, bus, truck, autorickshaw, motorcycle, bicycle
DEFAULT_VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "bus",
    4: "truck",
    5: "autorickshaw",
    6: "motorcycle",
    7: "bicycle"
}

# COCO vehicle class mappings (for testing with baseline yolov8n.pt before IDD training)
COCO_VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
COCO_VEHICLE_CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


@dataclass
class VehicleTrackState:
    track_id: int
    class_id: int
    class_name: str
    first_seen: float
    last_seen: float
    positions: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)
    current_zone: Optional[str] = None
    zone_entry_time: Optional[float] = None
    dwell_time: float = 0.0
    smoothed_speed: float = 0.0  # pixels / second
    is_bottleneck_culprit: bool = False


@dataclass
class VehicleDensityResult:
    frame_timestamp: float
    total_vehicles: int
    counts_by_class: Dict[str, int]
    zone_counts: Dict[str, int]
    zone_bottlenecks: Dict[str, bool]
    stalled_vehicle_ids: List[int]
    active_vehicles: List[VehicleTrackState]


class VehicleDensityTracker:
    def __init__(
        self,
        zones: Optional[Dict[str, np.ndarray]] = None,
        vehicle_class_ids: Optional[set] = None,
        class_name_map: Optional[Dict[int, str]] = None,
        dwell_time_threshold: float = 8.0,       # seconds before flagged as bottleneck
        speed_threshold: float = 8.0,            # pixels/sec below which vehicle is stationary
        history_window_sec: float = 3.0,         # window for speed calculation
        bottleneck_density_threshold: int = 5    # vehicle count in zone that triggers congestion warning
    ):
        """
        zones: Dict of {'zone_name': np.array([[x1, y1], [x2, y2], ...], np.int32)}
        dwell_time_threshold: Seconds a vehicle can remain stationary before bottleneck alert
        speed_threshold: Speed threshold in px/sec
        """
        self.zones = zones or {}
        self.vehicle_class_ids = vehicle_class_ids or DEFAULT_VEHICLE_CLASS_IDS
        self.class_name_map = class_name_map or DEFAULT_VEHICLE_CLASS_NAMES
        self.dwell_time_threshold = dwell_time_threshold
        self.speed_threshold = speed_threshold
        self.history_window_sec = history_window_sec
        self.bottleneck_density_threshold = bottleneck_density_threshold

        # Persistent state: track_id -> VehicleTrackState
        self.tracks: Dict[int, VehicleTrackState] = {}
        self.max_lost_time = 3.0  # Prune tracks inactive for > 3s

    def enable_coco_mode(self):
        """Switches filter to COCO vehicle IDs for early demoing on base YOLOv8."""
        self.vehicle_class_ids = COCO_VEHICLE_CLASS_IDS
        self.class_name_map = COCO_VEHICLE_CLASS_NAMES

    def set_default_roi(self, frame_width: int, frame_height: int):
        """Sets up default zones if none were explicitly configured."""
        # Zone 1: Main corridor (bottom 60%)
        main_poly = np.array([
            [int(frame_width * 0.05), int(frame_height * 0.40)],
            [int(frame_width * 0.95), int(frame_height * 0.40)],
            [int(frame_width * 0.98), int(frame_height * 0.95)],
            [int(frame_width * 0.02), int(frame_height * 0.95)]
        ], np.int32)

        # Zone 2: Intersection / Critical Bottleneck Core
        core_poly = np.array([
            [int(frame_width * 0.25), int(frame_height * 0.50)],
            [int(frame_width * 0.75), int(frame_height * 0.50)],
            [int(frame_width * 0.80), int(frame_height * 0.85)],
            [int(frame_width * 0.20), int(frame_height * 0.85)]
        ], np.int32)

        self.zones = {
            "Main_Corridor": main_poly,
            "Bottleneck_Core": core_poly
        }

    def _get_bottom_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculates ground contact point (x_center, y_max) from [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    def _point_in_zone(self, point: Tuple[float, float], zone_poly: np.ndarray) -> bool:
        """Point-in-polygon test using OpenCV."""
        return cv2.pointPolygonTest(zone_poly, (float(point[0]), float(point[1])), False) >= 0

    def update(
        self,
        detections: List[Dict[str, Any]],
        current_time: Optional[float] = None
    ) -> VehicleDensityResult:
        """
        Process detections from one frame.
        detections: list of dicts with keys:
            - 'track_id': int
            - 'bbox': [x1, y1, x2, y2]
            - 'class_id': int
            - 'confidence': float (optional)
        """
        if current_time is None:
            current_time = time.time()

        active_track_ids = set()
        counts_by_class = {name: 0 for name in self.class_name_map.values()}
        zone_counts = {zone_name: 0 for zone_name in self.zones}
        zone_stalled_counts = {zone_name: 0 for zone_name in self.zones}
        stalled_ids = []

        for det in detections:
            cls_id = int(det.get("class_id", -1))
            if cls_id not in self.vehicle_class_ids:
                continue

            track_id = det.get("track_id")
            if track_id is None:
                continue

            active_track_ids.add(track_id)
            bbox = det["bbox"]
            footprint = self._get_bottom_center(bbox)
            cls_name = self.class_name_map.get(cls_id, "vehicle")
            counts_by_class[cls_name] = counts_by_class.get(cls_name, 0) + 1

            # Detect zone membership
            assigned_zone = None
            for zone_name, zone_poly in self.zones.items():
                if self._point_in_zone(footprint, zone_poly):
                    assigned_zone = zone_name
                    zone_counts[zone_name] = zone_counts.get(zone_name, 0) + 1
                    break

            # Update or create track state
            if track_id not in self.tracks:
                state = VehicleTrackState(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    first_seen=current_time,
                    last_seen=current_time,
                    positions=[(footprint[0], footprint[1], current_time)],
                    current_zone=assigned_zone,
                    zone_entry_time=current_time if assigned_zone else None,
                    dwell_time=0.0,
                    smoothed_speed=0.0
                )
                self.tracks[track_id] = state
            else:
                state = self.tracks[track_id]
                state.last_seen = current_time
                state.positions.append((footprint[0], footprint[1], current_time))

                # Handle zone transitions
                if assigned_zone != state.current_zone:
                    state.current_zone = assigned_zone
                    state.zone_entry_time = current_time if assigned_zone else None
                    state.dwell_time = 0.0
                elif state.current_zone and state.zone_entry_time:
                    state.dwell_time = current_time - state.zone_entry_time

                # Compute smoothed speed over history window
                # Prune positions older than history_window_sec
                cutoff = current_time - self.history_window_sec
                state.positions = [p for p in state.positions if p[2] >= cutoff]

                if len(state.positions) >= 2:
                    p_old = state.positions[0]
                    p_new = state.positions[-1]
                    dt = max(1e-3, p_new[2] - p_old[2])
                    dist = np.hypot(p_new[0] - p_old[0], p_new[1] - p_old[1])
                    speed_px_sec = dist / dt
                    # Exponential smoothing
                    state.smoothed_speed = 0.6 * speed_px_sec + 0.4 * state.smoothed_speed

                # Check bottleneck criteria
                is_stalled = (
                    state.current_zone is not None
                    and state.dwell_time >= self.dwell_time_threshold
                    and state.smoothed_speed < self.speed_threshold
                )
                state.is_bottleneck_culprit = is_stalled

                if is_stalled:
                    stalled_ids.append(track_id)
                    if state.current_zone:
                        zone_stalled_counts[state.current_zone] = zone_stalled_counts.get(state.current_zone, 0) + 1

        # Evaluate zone-level bottleneck status
        zone_bottlenecks = {}
        for zone_name in self.zones:
            has_bottleneck = (
                zone_stalled_counts.get(zone_name, 0) >= 1
                or zone_counts.get(zone_name, 0) >= self.bottleneck_density_threshold
            )
            zone_bottlenecks[zone_name] = has_bottleneck

        # Prune dead tracks
        dead_keys = [
            t_id for t_id, s in self.tracks.items()
            if (current_time - s.last_seen) > self.max_lost_time
        ]
        for dk in dead_keys:
            del self.tracks[dk]

        active_vehicles = [self.tracks[tid] for tid in active_track_ids if tid in self.tracks]

        return VehicleDensityResult(
            frame_timestamp=current_time,
            total_vehicles=len(active_track_ids),
            counts_by_class=counts_by_class,
            zone_counts=zone_counts,
            zone_bottlenecks=zone_bottlenecks,
            stalled_vehicle_ids=stalled_ids,
            active_vehicles=active_vehicles
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        result: VehicleDensityResult,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Renders zone boundaries, bounding boxes, dwell timers, and bottleneck alerts."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw zones
        for zone_name, poly in self.zones.items():
            is_blocked = result.zone_bottlenecks.get(zone_name, False)
            color = (0, 0, 220) if is_blocked else (0, 200, 100)  # Red if bottleneck, green if normal
            cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2)

            # Zone label
            cnt = result.zone_counts.get(zone_name, 0)
            tag = f"{zone_name}: {cnt} veh"
            if is_blocked:
                tag += " [BOTTLENECK!]"
            cv2.putText(overlay, tag, (poly[0][0], max(20, poly[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw vehicle detections
        track_map = {v.track_id: v for v in result.active_vehicles}
        for det in detections:
            tid = det.get("track_id")
            if tid not in track_map:
                continue

            state = track_map[tid]
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = (0, 0, 255) if state.is_bottleneck_culprit else (0, 220, 255)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid} {state.class_name}"
            if state.dwell_time > 1.0:
                label += f" | {state.dwell_time:.1f}s"
            if state.is_bottleneck_culprit:
                label += " [STALLED]"

            cv2.putText(overlay, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Draw Vehicle Density Summary HUD in top-left
        hud_h = 100
        hud_w = 340
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 80), 1)

        cv2.putText(overlay, f"TRAFFIC DENSITY: {result.total_vehicles} Active Vehicles",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        class_summary = " | ".join([f"{k[:3]}:{v}" for k, v in result.counts_by_class.items() if v > 0])
        if not class_summary:
            class_summary = "None detected"
        cv2.putText(overlay, f"Classes: {class_summary}",
                    (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        bottleneck_count = sum(1 for b in result.zone_bottlenecks.values() if b)
        b_color = (0, 50, 255) if bottleneck_count > 0 else (0, 255, 100)
        b_text = f"Bottlenecks: {bottleneck_count} Zone(s) Alerted" if bottleneck_count > 0 else "Flow Status: Normal (No Stalls)"
        cv2.putText(overlay, b_text,
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 2)

        return overlay


# Self-test when executed directly
if __name__ == "__main__":
    print("[*] Running standalone VehicleDensityTracker verification...")
    tracker = VehicleDensityTracker(
        dwell_time_threshold=4.0,
        speed_threshold=5.0
    )
    # Define a 640x480 test zone
    zone_poly = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], np.int32)
    tracker.zones = {"Intersection_Zone": zone_poly}

    # Simulate vehicle 1 moving through
    t0 = 1000.0
    # Frame 1: Vehicle enters zone
    res1 = tracker.update([
        {"track_id": 101, "class_id": 2, "bbox": [200, 200, 260, 280]}  # car
    ], current_time=t0)
    assert res1.total_vehicles == 1
    assert res1.zone_counts["Intersection_Zone"] == 1
    assert res1.zone_bottlenecks["Intersection_Zone"] is False
    print(f"Test 1 (Entry): Count={res1.total_vehicles}, Bottleneck={res1.zone_bottlenecks['Intersection_Zone']}")

    # Frame 2: Vehicle 1 remains stationary for 5 seconds (exceeding dwell_time_threshold 4.0s)
    t1 = t0 + 5.0
    res2 = tracker.update([
        {"track_id": 101, "class_id": 2, "bbox": [200, 200, 260, 280]}  # identical position
    ], current_time=t1)
    assert 101 in res2.stalled_vehicle_ids, "Vehicle 101 should be marked as stalled"
    assert res2.zone_bottlenecks["Intersection_Zone"] is True, "Zone should be flagged as bottleneck"
    print(f"Test 2 (Stalled > 4s): Stalled IDs={res2.stalled_vehicle_ids}, Bottleneck={res2.zone_bottlenecks['Intersection_Zone']}")

    print("[+] VehicleDensityTracker unit tests PASSED successfully!")

