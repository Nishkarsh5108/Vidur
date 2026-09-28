"""V3 sign condition (YOLOv8s): damaged vs good.

V3 was trained on letterboxed, mostly close-up sign images, so the agent feeds it crops around V2's
sign boxes at 320 px instead of whole frames at 640 px. That is about 4x cheaper and closer to what it
saw in training. Check its accuracy on crops from your own footage before quoting it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edge.runners.yolo import YoloModel

MODEL_NAME = "V3-sign-condition"
MODEL_VERSION = "yolov8s-640-2026-09-27"


class SignConditionRunner:
    def __init__(self, weights: str | Path, imgsz: int, conf: float, device: str = "auto", threads: int = 0):
        self.model = YoloModel(weights, imgsz, device, conf, threads)

    def classify(self, crop: np.ndarray) -> tuple[str, float] | None:
        """("damaged" | "good", confidence) from the most confident box in the crop, or None."""
        boxes = self.model.predict(crop)
        if not boxes:
            return None
        best = max(boxes, key=lambda b: b.conf)
        return best.name.lower(), best.conf
