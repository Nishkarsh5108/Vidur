"""Where events go: ingest envelopes appended to envelopes.jsonl, and JPEG crops in media/.

Each line of envelopes.jsonl is a complete `POST /api/v1/ingest` body, so an uploader (or the
backend team) can replay a run verbatim. Nothing here ever stores or sends video.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge.imaging import XYXY, blur, crop_with_margin, shift
from edge.util import iso_utc, uuid7


class MediaStore:
    """JPEG crops of at most max_bytes, with people blurred before anything leaves the device."""

    def __init__(self, directory: Path, max_bytes: int = 60_000):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self.files = 0

    def save_crop(self, image: np.ndarray, xyxy: XYXY, margin: float, blur_boxes: list[XYXY] = ()) -> str:
        """Crops a box plus margin, blurs blur_boxes (image coordinates) and stores the JPEG. Returns its mediaId."""
        crop, (left, top) = crop_with_margin(image, xyxy, margin)
        return self.save(crop.copy(), shift(list(blur_boxes), left, top))

    def save(self, image: np.ndarray, blur_boxes: list[XYXY] = ()) -> str:
        data = self._encode(blur(image, list(blur_boxes)))
        media_id = "m_" + uuid7().replace("-", "")
        (self.directory / f"{media_id}.jpg").write_bytes(data)
        self.bytes_written += len(data)
        self.files += 1
        return media_id

    def _encode(self, image: np.ndarray) -> bytes:
        for _ in range(6):
            for quality in (85, 70, 55, 40):
                ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok and len(buf) <= self.max_bytes:
                    return buf.tobytes()
            image = cv2.resize(image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        return buf.tobytes()


class Outbox:
    """Batches telemetry points and events into ingest envelopes every `every_s` seconds of capture time."""

    def __init__(self, directory: Path, device_id: str, vehicle_id: str, every_s: float = 5.0):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "envelopes.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.device_id, self.vehicle_id = device_id, vehicle_id
        self.every_s = every_s
        self.trip_id: str | None = None
        self.events: list[dict[str, Any]] = []
        self.telemetry: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.bytes_written = 0
        self.envelopes = 0
        self._last_flush: float | None = None

    def start_trip(self, t: float) -> None:
        if self.trip_id is None:
            self.trip_id = f"{self.vehicle_id}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(t))}"

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.counts[event["eventType"]] += 1

    def add_telemetry(self, t: float, lat: float, lon: float, speed: float, heading: float | None) -> None:
        self.telemetry.append({"t": iso_utc(t), "lat": round(lat, 7), "lon": round(lon, 7), "speedMps": round(speed, 2),
                               "headingDeg": None if heading is None else round(heading, 1)})

    def maybe_flush(self, t: float) -> None:
        if self._last_flush is None:
            self._last_flush = t
        elif t - self._last_flush >= self.every_s:
            self.flush(t)

    def flush(self, t: float) -> None:
        self._last_flush = t
        if not self.events and not self.telemetry:
            return
        envelope = {
            "schemaVersion": "1.0",
            "deviceId": self.device_id,
            "vehicleId": self.vehicle_id,
            "tripId": self.trip_id,
            "sentAt": iso_utc(t),
            "telemetry": self.telemetry,
            "events": self.events,
        }
        line = json.dumps(envelope, separators=(",", ":")) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
        self.bytes_written += len(line.encode("utf-8"))
        self.envelopes += 1
        self.events, self.telemetry = [], []
