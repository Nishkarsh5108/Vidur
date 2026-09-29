"""Sends events (and their photos) to the FleetSense backend (backend/backend), in the contract of
docs/hello.md.

The edge keeps its own richer event format (envelopes.jsonl). This module translates each event to
the backend's shape and, in the same background thread, uploads any crop it references to the
backend's own media store so the dashboard can actually show it — then posts the batch with retries,
so a slow or unreachable backend never stalls the cameras. Event IDs are stable, so a retried batch is
reported as duplicates by the backend instead of being stored twice.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MAX_BATCH = 500        # the backend's limit per request

# edge event type -> backend eventType (docs/hello.md). Types not listed keep their name;
# the backend accepts any lowercase type and stores it as an event.
EVENT_TYPES = {
    "iri_window": "road_quality",
    "traffic_sample": "traffic_density",
    "sign_condition": "traffic_sign",
    "pedestrian_zone_alert": "pedestrian_alert",
}
MODEL_NAMES = {
    "V1-roadsense": "roadsense_yolov8m",
    "V2-idd-traffic": "idd_yolov8n",
    "V3-sign-condition": "sign_yolov8s",
    "S1-iri": "iri_cnn",
    "S2-road-shock": "shock_cnn",
    "jerk-threshold": "jerk_threshold",
    "edge-agent": "edge_agent",
}
_VEHICLES = ("car", "bus", "truck", "autorickshaw", "motorcycle", "bicycle")
_TRAFFIC_FIELDS = {"car": "carCount", "bus": "busCount", "truck": "truckCount", "motorcycle": "twoWheelerCount",
                   "autorickshaw": "autorickshawCount", "bicycle": "bicycleCount"}


def to_backend(event: dict[str, Any]) -> dict[str, Any] | None:
    """One edge event in the backend's shape, or None if it cannot be placed on the map (no GPS).

    `mediaId` is deliberately left out here: the edge's id names a local JPEG (out/.../media/*.jpg)
    that means nothing to the backend. BackendUploader fills in the backend's own id, if any, after
    uploading the file — see _prepare().
    """
    if event.get("location") is None:
        return None
    source = event["source"]
    metadata = dict(event.get("metadata") or {})
    for key in ("headingDeg", "speedMps"):
        if event.get(key) is not None:
            metadata[key] = event[key]
    metadata["sensor"] = source.get("sensor")

    event_type = EVENT_TYPES.get(event["eventType"], event["eventType"])
    if event_type == "traffic_density":
        means = metadata.get("meanCounts", {})
        metadata.update({field: means.get(cls, 0.0) for cls, field in _TRAFFIC_FIELDS.items()})
        metadata["vehicleCount"] = round(sum(means.get(cls, 0.0) for cls in _VEHICLES), 2)
        metadata["pedestrianCount"] = means.get("person", 0.0)
        metadata["riderCount"] = means.get("rider", 0.0)
        metadata["countMethod"] = "mean_per_frame"      # average vehicles in view; uniqueTracks is the upper bound

    out = {
        "eventId": event["eventId"],
        "eventType": event_type,
        "capturedAt": event["capturedAt"],
        "location": event["location"],
        "model": MODEL_NAMES.get(source["model"], source["model"].lower()),
        "modelVersion": source.get("modelVersion"),
        "metadata": metadata,
    }
    if event.get("confidence") is not None:
        out["confidence"] = event["confidence"]
    return out


def _multipart_body(field: str, filename: str, content_type: str, data: bytes) -> tuple[bytes, str]:
    """A minimal single-file multipart/form-data body, built without the `requests` dependency."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, boundary


class BackendUploader(threading.Thread):
    """Converts, uploads media for, and posts events to the backend, one queued batch at a time.

    Both the media upload and the ingest POST happen in this background thread, so `submit()` (called
    from the capture/IMU loop) only ever enqueues plain Python dicts and returns immediately.
    """

    def __init__(self, url: str, device_key: str, device_id: str, vehicle_id: str, trip_id: str,
                 log_dir: Path, media_dir: Path | None = None, timeout_s: float = 10.0):
        super().__init__(name="backend-uploader", daemon=True)
        self.base_url = url.rstrip("/")
        self.endpoint = f"{self.base_url}/api/v1/ingest"
        self.media_endpoint = f"{self.base_url}/api/v1/media"
        self.device_key = device_key
        self.envelope = {"schemaVersion": "1.0", "deviceId": device_id, "vehicleId": vehicle_id, "tripId": trip_id}
        self.media_dir = media_dir
        self.timeout_s = timeout_s
        self.log_path = log_dir / "backend_log.jsonl"
        self.log_path.write_text("", encoding="utf-8")
        self.batches: queue.Queue[list[dict] | None] = queue.Queue()
        self.stats: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.skipped_no_location = 0
        self._media_cache: dict[str, str | None] = {}     # local mediaId -> backend mediaId (or None: failed)
        self._closing = threading.Event()

    def submit(self, events: list[dict[str, Any]]) -> None:
        """Enqueues raw edge events for conversion, media upload and delivery in the background thread."""
        for i in range(0, len(events), MAX_BATCH):
            self.batches.put(events[i:i + MAX_BATCH])

    def run(self) -> None:
        while (raw_batch := self.batches.get()) is not None:
            batch = self._prepare(raw_batch)
            if batch:
                self._send_with_retry(batch)

    def _prepare(self, raw_events: list[dict[str, Any]]) -> list[dict]:
        prepared = []
        for raw in raw_events:
            converted = to_backend(raw)
            if converted is None:
                self.skipped_no_location += 1
                continue
            local_id = raw.get("mediaId")
            if local_id:
                backend_id = self._media_for(local_id)
                if backend_id:
                    converted["metadata"]["mediaId"] = backend_id
                    converted["metadata"]["mediaUrl"] = f"{self.base_url}/api/v1/media/{backend_id}"
            prepared.append(converted)
        return prepared

    def _media_for(self, local_id: str) -> str | None:
        """Uploads out/.../media/<local_id>.jpg once and remembers the result, so a retried batch
        (same eventId, same mediaId) never re-uploads the same photo."""
        if local_id in self._media_cache:
            return self._media_cache[local_id]
        backend_id = self._upload_media(local_id)
        self._media_cache[local_id] = backend_id
        return backend_id

    def _upload_media(self, local_id: str) -> str | None:
        if self.media_dir is None:
            return None
        path = self.media_dir / f"{local_id}.jpg"
        if not path.is_file():
            log.warning("media upload: %s not found in %s", path.name, self.media_dir)
            return None
        content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        body, boundary = _multipart_body("file", path.name, content_type, path.read_bytes())
        request = urllib.request.Request(
            self.media_endpoint, data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Device-Key": self.device_key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.load(response)["mediaId"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError) as exc:
            log.warning("media upload failed for %s: %s", path.name, exc)
            self.stats["media_upload_failed"] += 1
            return None

    def _send_with_retry(self, batch: list[dict]) -> None:
        delay = 1.0
        while True:
            try:
                self._post(batch)
                return
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in (401, 422):          # not fixed by retrying: wrong key or malformed envelope
                    log.error("backend refused a batch of %d events (HTTP %d): %s", len(batch), exc.code, body)
                    self.stats[f"http_{exc.code}"] += len(batch)
                    return
                log.warning("backend HTTP %d, retrying in %.0f s", exc.code, delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                log.warning("backend unreachable (%s), retrying in %.0f s", getattr(exc, "reason", exc), delay)
            self.stats["retries"] += 1
            if self._closing.is_set() and delay > 8:
                log.error("giving up on %d events after retries; they are still in envelopes.jsonl", len(batch))
                self.stats["undelivered"] += len(batch)
                return
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    def _post(self, batch: list[dict]) -> None:
        body = json.dumps({**self.envelope, "events": batch}).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=body, method="POST",
                                         headers={"Content-Type": "application/json", "X-Device-Key": self.device_key})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.load(response)
        for key in ("accepted", "duplicates", "rejected"):
            self.stats[key] += result.get(key, 0)
        for item in result.get("results", []):
            if item.get("status") == "rejected":
                self.reasons[item.get("reason") or "?"] += 1
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
        if result.get("rejected"):
            log.warning("backend rejected %d of %d events: %s", result["rejected"], len(batch),
                        "; ".join(sorted(self.reasons)[:3]))

    def close(self, timeout_s: float = 60.0) -> dict[str, Any]:
        """Sends everything still queued, then stops. Returns the delivery counts."""
        self._closing.set()
        self.batches.put(None)
        self.join(timeout_s)
        return {"endpoint": self.endpoint, **self.stats, "skippedNoLocation": self.skipped_no_location,
                "rejectReasons": dict(self.reasons)}
