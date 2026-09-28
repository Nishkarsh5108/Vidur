"""Replay a recorded ride (video + IMU/GPS log) through the same code path a live bus uses."""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2

from edge.capture import Frame, ImuSample
from edge.util import parse_time

log = logging.getLogger(__name__)

# Column names accepted in the IMU/GPS CSV (the team's phone logger writes the first of each).
_COLUMNS = {
    "time": ("time", "timestamp", "t"),
    "lat": ("latitude", "lat"),
    "lon": ("longitude", "lon", "lng"),
    "speed": ("speed", "speed_mps", "speed_ms"),
}
_AXES = ("ax", "ay", "az", "wx", "wy", "wz")


class ImuCsvReader:
    """Streams rows of an IMU + GPS log.

    Expected header (the team's logger): time,ax,ay,az,wx,wy,wz,latitude,longitude,altitude,speed,...
    time is ISO 8601 or epoch seconds, accelerations m/s2, gyro rad/s, speed m/s.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        self._index = self._resolve_columns([h.strip() for h in header])

    def _resolve_columns(self, header: list[str]) -> dict[str, int]:
        index = {}
        for key, names in {**_COLUMNS, **{a: (a,) for a in _AXES}}.items():
            found = next((header.index(n) for n in names if n in header), None)
            if found is None:
                raise ValueError(f"{self.path.name}: missing column {names[0]!r} (header: {header})")
            index[key] = found
        return index

    def first_time(self) -> float:
        return next(iter(self)).t

    def __iter__(self) -> Iterator[ImuSample]:
        ix = self._index
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                try:
                    yield ImuSample(
                        t=parse_time(row[ix["time"]]),
                        ax=float(row[ix["ax"]]), ay=float(row[ix["ay"]]), az=float(row[ix["az"]]),
                        wx=float(row[ix["wx"]]), wy=float(row[ix["wy"]]), wz=float(row[ix["wz"]]),
                        lat=float(row[ix["lat"]]), lon=float(row[ix["lon"]]),
                        speed=float(row[ix["speed"]]),
                    )
                except (ValueError, IndexError):
                    continue   # a torn or partial row; the logger writes these on shutdown


class VideoReader:
    """Frames of a video file, timestamped as start_t + index / fps."""

    def __init__(self, path: str | Path, start_t: float):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.start_t = start_t
        self.index = 0
        self.done = False

    @property
    def next_time(self) -> float:
        return self.start_t + self.index / self.fps

    def read(self) -> Frame | None:
        ok, image = self.cap.read()
        if not ok:
            self.done = True
            return None
        frame = Frame(t=self.next_time, image=image, index=self.index, orig_size=(image.shape[1], image.shape[0]))
        self.index += 1
        return frame

    def skip(self) -> None:
        """Advances one frame without decoding it (used when replay falls behind real time)."""
        if not self.cap.grab():
            self.done = True
        self.index += 1

    def close(self) -> None:
        self.cap.release()


class ReplaySource:
    """Merges video frames and IMU samples into one time-ordered stream.

    rate = 1 replays in real time, 4 at 4x, and 0 as fast as possible. When pacing (rate > 0)
    and the consumer falls more than max_lag_s behind, frames are skipped without decoding,
    exactly like a live camera that always hands over its newest frame. IMU samples are never skipped.
    """

    def __init__(self, video: VideoReader | None, imu: Iterable[ImuSample] | None,
                 rate: float = 0.0, max_lag_s: float = 0.5, max_seconds: float | None = None):
        if video is None and imu is None:
            raise ValueError("need a video, an IMU log, or both")
        self.video, self.imu = video, imu
        self.rate, self.max_lag_s, self.max_seconds = rate, max_lag_s, max_seconds
        self.frames_skipped = 0
        self.start_t: float | None = None
        self.end_t: float | None = None

    def __iter__(self) -> Iterator[tuple[str, Frame | ImuSample]]:
        imu_iter = iter(self.imu) if self.imu is not None else iter(())
        pending_imu = next(imu_iter, None)
        video = self.video
        wall0 = time.perf_counter()
        while True:
            video_t = video.next_time if video is not None and not video.done else None
            imu_t = pending_imu.t if pending_imu is not None else None
            if video_t is None and imu_t is None:
                break
            is_frame = imu_t is None or (video_t is not None and video_t <= imu_t)
            t = video_t if is_frame else imu_t

            if self.start_t is None:
                self.start_t = t
            if self.max_seconds is not None and t - self.start_t > self.max_seconds:
                break

            if self.rate > 0:
                lag = time.perf_counter() - (wall0 + (t - self.start_t) / self.rate)
                if lag < 0:
                    time.sleep(-lag)
                elif is_frame and lag > self.max_lag_s:
                    video.skip()
                    self.frames_skipped += 1
                    continue

            self.end_t = t
            if is_frame:
                frame = video.read()
                if frame is not None:
                    yield "frame", frame
            else:
                yield "imu", pending_imu
                pending_imu = next(imu_iter, None)

        if video is not None:
            video.close()
