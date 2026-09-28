"""Annotated video for demos: V2 boxes plus the agent's live state. Never part of what is uploaded."""

from __future__ import annotations

import cv2
import numpy as np

from edge.runners.yolo import Box

_GROUPS = {
    "person": (0, 140, 255), "rider": (0, 140, 255),                       # vulnerable: orange
    "traffic sign": (255, 200, 0), "traffic light": (255, 200, 0), "pole": (255, 200, 0),   # infrastructure: cyan
}
_VEHICLE = (80, 200, 80)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(image: np.ndarray, boxes: list[Box], status: list[str], ticker: list[str], label: str) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    s = width / 1280.0
    for b in boxes:
        color = _GROUPS.get(b.name, _VEHICLE)
        x1, y1, x2, y2 = (int(v) for v in b.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, int(2 * s)))
        cv2.putText(out, f"{b.name} #{b.track_id}", (x1, max(12, y1 - 4)), _FONT, 0.45 * s, color, max(1, int(s)))

    bar_h = int(28 * s) * len(status) + int(10 * s)
    cv2.rectangle(out, (0, 0), (width, bar_h), (0, 0, 0), -1)
    for i, line in enumerate(status):
        cv2.putText(out, line, (int(10 * s), int(24 * s) + i * int(28 * s)), _FONT, 0.6 * s, (255, 255, 255), max(1, int(s)))

    for i, line in enumerate(reversed(ticker)):
        y = height - int(12 * s) - i * int(26 * s)
        (tw, th), _ = cv2.getTextSize(line, _FONT, 0.55 * s, max(1, int(s)))
        cv2.rectangle(out, (int(6 * s), y - th - int(6 * s)), (int(14 * s) + tw, y + int(6 * s)), (0, 0, 0), -1)
        cv2.putText(out, line, (int(10 * s), y), _FONT, 0.55 * s, (0, 255, 255), max(1, int(s)))

    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55 * s, max(1, int(s)))
    x, y = width - tw - int(12 * s), bar_h + int(24 * s)
    cv2.rectangle(out, (x - int(6 * s), y - th - int(6 * s)), (width, y + int(6 * s)), (0, 0, 0), -1)
    cv2.putText(out, label, (x, y), _FONT, 0.55 * s, (80, 80, 255), max(1, int(s)))
    return out
