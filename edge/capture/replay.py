"""Replay a recorded ride (video + IMU + GPS) through the same code path a live bus uses."""

from __future__ import annotations

import csv
import heapq
import logging
import math
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2

from edge.capture import Frame, GpsFix, ImuSample
from edge.util import parse_time

log = logging.getLogger(__name__)

_AXES = ("ax", "ay", "az", "wx", "wy", "wz")
_ABSOLUTE_TIME = ("time", "timestamp", "t")
_RELATIVE_TIME = ("video_time_s", "sim_time")      # seconds from the start of the recording
_LAT, _LON = ("latitude", "lat"), ("longitude", "lon", "lng")
_SPEED = ("speed", "speed_mps", "speed_ms")


def _find(header: list[str], names: tuple[str, ...]) -> int | None:
    return next((header.index(n) for n in names if n in header), None)


def _epoch(value: str) -> float:
    """ISO 8601, epoch seconds or epoch milliseconds -> epoch seconds."""
    t = parse_time(value)
    return t / 1000.0 if t > 1e11 else t


class ImuCsvReader:
    """Streams rows of an IMU log.

    Accepted layouts:
    - the team's phone logger: time,ax,ay,az,wx,wy,wz,latitude,longitude,altitude,speed,...
      (time ISO 8601 or epoch seconds/milliseconds; position and speed from the phone's GPS);
    - a BeamNG clip from tools/make_sim_clip.py: sample_number,sim_time,speed_ms,ax,...,wz,video_time_s,...
      (relative time, no position). Its times are placed at `time_base` (the video's start).
    Accelerations m/s2, gyro rad/s, speed m/s. `shift` moves every timestamp by that many seconds.
    """

    def __init__(self, path: str | Path, time_base: float | None = None, shift: float = 0.0):
        self.path = Path(path)
        self.shift = shift
        with open(self.path, newline="", encoding="utf-8") as fh:
            header = [h.strip() for h in next(csv.reader(fh))]
        self._axes = {a: _find(header, (a,)) for a in _AXES}
        missing = [a for a, i in self._axes.items() if i is None]
        self._speed = _find(header, _SPEED)
        if missing or self._speed is None:
            raise ValueError(f"{self.path.name}: missing columns {missing or ['speed']} (header: {header})")
        self._abs = _find(header, _ABSOLUTE_TIME)
        self._rel = None if self._abs is not None else _find(header, _RELATIVE_TIME)
        if self._abs is None and self._rel is None:
            raise ValueError(f"{self.path.name}: needs a time column {_ABSOLUTE_TIME + _RELATIVE_TIME}")
        self.relative = self._abs is None
        # video_time_s is already 0 at the video's first frame; sim_time is made relative to its first row.
        self._rel_from_first_row = self.relative and header[self._rel] != "video_time_s"
        self.time_base = time_base
        self._lat, self._lon = _find(header, _LAT), _find(header, _LON)
        self.has_position = self._lat is not None and self._lon is not None

    def first_time(self) -> float:
        return next(iter(self)).t

    def __iter__(self) -> Iterator[ImuSample]:
        if self.relative and self.time_base is None:
            raise ValueError(f"{self.path.name} has relative times: give a time base (the video's start)")
        ax = self._axes
        rel0 = None
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                try:
                    if self.relative:
                        rel = float(row[self._rel])
                        if rel0 is None:
                            rel0 = rel if self._rel_from_first_row else 0.0
                        t = self.time_base + rel - rel0
                    else:
                        t = _epoch(row[self._abs])
                    yield ImuSample(
                        t=t + self.shift,
                        ax=float(row[ax["ax"]]), ay=float(row[ax["ay"]]), az=float(row[ax["az"]]),
                        wx=float(row[ax["wx"]]), wy=float(row[ax["wy"]]), wz=float(row[ax["wz"]]),
                        lat=float(row[self._lat]) if self.has_position else math.nan,
                        lon=float(row[self._lon]) if self.has_position else math.nan,
                        speed=float(row[self._speed]),
                    )
                except (ValueError, IndexError):
                    continue   # a torn or partial row; the logger writes these on shutdown


class GpsCsvReader:
    """Streams a GPS trail: time,latitude,longitude,speed_mps[,heading_deg][,video_time_s].

    tools/sim writes one row per video frame, so `video_start()` recovers when the video began.
    """

    def __init__(self, path: str | Path, shift: float = 0.0):
        self.path = Path(path)
        self.shift = shift
        with open(self.path, newline="", encoding="utf-8") as fh:
            header = [h.strip() for h in next(csv.reader(fh))]
        self._t, self._lat, self._lon = _find(header, _ABSOLUTE_TIME), _find(header, _LAT), _find(header, _LON)
        self._speed, self._heading = _find(header, _SPEED), _find(header, ("heading_deg", "heading"))
        self._video_t = _find(header, ("video_time_s",))
        if None in (self._t, self._lat, self._lon):
            raise ValueError(f"{self.path.name}: needs time, latitude and longitude columns (header: {header})")

    def video_start(self) -> float | None:
        """Time of video frame 0 (before any shift), if the trail records video times."""
        if self._video_t is None:
            return None
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            row = next(reader)
        return _epoch(row[self._t]) - float(row[self._video_t])

    def first_time(self) -> float:
        return next(iter(self)).t

    def __iter__(self) -> Iterator[GpsFix]:
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                try:
                    heading = row[self._heading] if self._heading is not None else ""
                    yield GpsFix(
                        t=_epoch(row[self._t]) + self.shift,
                        lat=float(row[self._lat]), lon=float(row[self._lon]),
                        speed=float(row[self._speed]) if self._speed is not None else 0.0,
                        heading=float(heading) if heading not in ("", "nan") else None,
                    )
                except (ValueError, IndexError):
                    continue


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
    """Merges video frames, IMU samples and GPS fixes into one time-ordered stream.

    rate = 1 replays in real time, 4 at 4x, and 0 as fast as possible. When pacing (rate > 0)
    and the consumer falls more than max_lag_s behind, frames are skipped without decoding,
    exactly like a live camera that always hands over its newest frame. Sensor samples are never skipped.
    """

    def __init__(self, video: VideoReader | None, imu: Iterable[ImuSample] | None,
                 gps: Iterable[GpsFix] | None = None, rate: float = 0.0, max_lag_s: float = 0.5,
                 max_seconds: float | None = None):
        if video is None and imu is None and gps is None:
            raise ValueError("need a video, an IMU log or a GPS trail")
        self.video = video
        self.sensors = [s for s in (imu, gps) if s is not None]
        self.rate, self.max_lag_s, self.max_seconds = rate, max_lag_s, max_seconds
        self.frames_skipped = 0
        self.start_t: float | None = None
        self.end_t: float | None = None

    def __iter__(self) -> Iterator[tuple[str, Frame | ImuSample | GpsFix]]:
        sensors = heapq.merge(*(iter(s) for s in self.sensors), key=lambda item: item.t)
        pending = next(sensors, None)
        video = self.video
        wall0 = time.perf_counter()
        while True:
            video_t = video.next_time if video is not None and not video.done else None
            sensor_t = pending.t if pending is not None else None
            if video_t is None and sensor_t is None:
                break
            is_frame = sensor_t is None or (video_t is not None and video_t <= sensor_t)
            t = video_t if is_frame else sensor_t

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
                yield ("gps" if isinstance(pending, GpsFix) else "imu"), pending
                pending = next(sensors, None)

        if video is not None:
            video.close()
