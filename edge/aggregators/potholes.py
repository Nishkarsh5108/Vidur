"""pothole and zebra_crossing events from V1 results, one per stretch of road."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from edge.capture import Frame
from edge.runners.v1_roadsense import severity
from edge.runners.yolo import Box


class RouteDeduper:
    """Accepts at most one event of a kind per `gap_m` metres of travel (1-D, along the odometer)."""

    def __init__(self, keep: int = 64):
        self._seen: dict[str, deque[float]] = {}
        self.keep = keep

    def accept(self, kind: str, odometer: float, gap_m: float) -> bool:
        seen = self._seen.setdefault(kind, deque(maxlen=self.keep))
        if any(abs(odometer - x) < gap_m for x in seen):
            return False
        seen.append(odometer)
        return True


@dataclass
class Detection:
    frame: Frame
    box: Box

    @property
    def bbox_original(self) -> list[float]:
        """Box in the original capture's pixels (ring-buffer frames may be downscaled)."""
        return [round(v / self.frame.scale, 1) for v in self.box.xyxy]


def best_of(results: list[tuple[Frame, list[Box]]], name: str) -> Detection | None:
    """Most confident detection of one class across a job's frames."""
    candidates = [Detection(frame, box) for frame, boxes in results for box in boxes if box.name == name]
    return max(candidates, key=lambda d: d.box.conf, default=None)


def pothole_metadata(det: Detection, thresholds: tuple[float, float], trigger: str,
                     shock: Any | None) -> dict[str, Any]:
    height, width = det.frame.image.shape[:2]
    frame_area = float(width * height)
    metadata = {
        "bbox": det.bbox_original,
        "frameSize": list(det.frame.orig_size),
        "bboxAreaFrac": round(det.box.area / frame_area, 5),
        "severity": severity(det.box.area, frame_area, thresholds),
        "trigger": trigger,                      # "imu_shock" | "sweep"
    }
    if shock is not None:
        metadata.update({
            "imuShockClass": shock.cls,
            "imuShockProb": None if shock.probs is None else round(float(shock.probs[3]), 3),
            "imuShockMethod": shock.method,
            "imuPeakToPeak": round(shock.p2p, 2),
        })
    return metadata
