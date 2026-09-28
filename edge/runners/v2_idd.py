"""V2 IDD traffic model (YOLOv8n) + ByteTrack: Indian road users, signs, lights and poles.

This is the only vision model that runs continuously. Its per-frame counts feed the traffic summary,
its person/rider boxes feed the school-zone logic, and its "traffic sign" tracks feed V3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edge.runners.yolo import Box, YoloModel

MODEL_NAME = "V2-idd-traffic"
MODEL_VERSION = "yolov8n-640-2026-09-24"

COUNTED_CLASSES = ("car", "bus", "truck", "autorickshaw", "motorcycle", "bicycle", "person", "rider")
SIGN_CLASS = "traffic sign"
DROPPED_CLASSES = {"ego vehicle"}   # class 11 is the bus's own bonnet or dashboard


class TrafficRunner:
    def __init__(self, weights: str | Path, imgsz: list[int], tracker: str, device: str = "auto", threads: int = 0):
        self.model = YoloModel(weights, imgsz, device, threads=threads)
        self.tracker = tracker

    def track(self, image: np.ndarray) -> list[Box]:
        return [b for b in self.model.track(image, self.tracker) if b.name not in DROPPED_CLASSES]
