"""V1 RoadSense (YOLOv8m): potholes and visible zebra crossings.

At about 74 GFLOPs per 480x800 frame, V1 is too heavy to run continuously on a Pi 5 CPU. The agent
runs it on the few buffered frames before an IMU jolt (see edge/agent.py and docs/edge-deployment.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edge.runners.yolo import Box, YoloModel

MODEL_NAME = "V1-roadsense"
MODEL_VERSION = "yolov8m-800-2026-09-25"

# The checkpoint calls class 1 "missing_zebra", but its training boxes are all visible crossings:
# a detector cannot box something that is not there. "Missing" is worked out in the backend.
CLASS_RENAME = {"pothole": "pothole", "missing_zebra": "zebra_crossing"}


def severity(box_area: float, frame_area: float, thresholds: tuple[float, float]) -> int:
    """1-3 from box area as a fraction of the frame, so severity no longer depends on camera resolution.

    The default thresholds equal V1's original 18,000 / 45,000 px2 on a 1920x1080 frame.
    """
    frac = box_area / frame_area
    return 3 if frac > thresholds[1] else 2 if frac > thresholds[0] else 1


class RoadSenseRunner:
    def __init__(self, weights: str | Path, imgsz: list[int], conf: float, device: str = "auto", threads: int = 0):
        self.model = YoloModel(weights, imgsz, device, conf, threads)

    def detect(self, image: np.ndarray) -> list[Box]:
        boxes = self.model.predict(image)
        for box in boxes:
            box.name = CLASS_RENAME.get(box.name, box.name)
        return boxes
