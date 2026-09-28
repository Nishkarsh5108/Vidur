"""Image helpers shared by the sign tracker and the media store."""

from __future__ import annotations

import cv2
import numpy as np

XYXY = tuple[float, float, float, float]


def crop_with_margin(image: np.ndarray, xyxy: XYXY, margin: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop of the box grown by `margin` x its size on every side, clamped to the image.
    Returns the crop and its top-left corner in the image."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    height, width = image.shape[:2]
    left, top = max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h))
    right, bottom = min(width, int(np.ceil(x2 + margin * w))), min(height, int(np.ceil(y2 + margin * h)))
    return image[top:bottom, left:right], (left, top)


def shift(boxes: list[XYXY], dx: float, dy: float, scale: float = 1.0) -> list[XYXY]:
    """Scales boxes, then moves them by (-dx, -dy): maps full-frame boxes into a resized crop."""
    return [(x1 * scale - dx, y1 * scale - dy, x2 * scale - dx, y2 * scale - dy) for x1, y1, x2, y2 in boxes]


def blur(image: np.ndarray, boxes: list[XYXY]) -> np.ndarray:
    """Gaussian-blurs each box in place (people and riders, before a crop leaves the device)."""
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
        if x2 - x1 > 2 and y2 - y1 > 2:
            k = max(3, (min(x2 - x1, y2 - y1) // 3) | 1)
            image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (k, k), 0)
    return image


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian: higher means less motion blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
