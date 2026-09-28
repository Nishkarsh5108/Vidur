"""A short, time-indexed history of camera frames, so a jolt felt by the IMU can be looked at afterwards."""

from __future__ import annotations

from collections import deque

import cv2

from edge.capture import Frame


class FrameRingBuffer:
    """Keeps the last `seconds` of frames at `fps`, downscaled to at most `max_width` pixels wide.

    5 s at 10 FPS and 1280x720 is about 140 MB, fine on a Pi 5 with 8 GB.
    """

    def __init__(self, seconds: float, fps: float, max_width: int):
        self.period = 1.0 / fps
        self.max_width = max_width
        self.frames: deque[Frame] = deque(maxlen=max(2, int(round(seconds * fps)) + 1))

    def push(self, frame: Frame) -> None:
        if self.frames and frame.t - self.frames[-1].t < self.period - 1e-6:
            return
        image, scale = frame.image, 1.0
        height, width = image.shape[:2]
        if width > self.max_width:
            scale = self.max_width / width
            image = cv2.resize(image, (self.max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
        self.frames.append(Frame(frame.t, image, frame.index, scale, frame.orig_size or (width, height)))

    def nearest(self, t: float, tolerance: float) -> Frame | None:
        best = min(self.frames, key=lambda f: abs(f.t - t), default=None)
        return best if best is not None and abs(best.t - t) <= tolerance else None

    def pick(self, times: list[float], tolerance: float = 0.25) -> list[Frame]:
        """The stored frame nearest each requested time, without repeats."""
        picked: dict[int, Frame] = {}
        for t in times:
            frame = self.nearest(t, tolerance)
            if frame is not None:
                picked.setdefault(frame.index, frame)
        return sorted(picked.values(), key=lambda f: f.t)
