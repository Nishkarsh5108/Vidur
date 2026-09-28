"""The edge agent: V1, V2, V3, S1 and S2 combined into one two-lane pipeline.

    REAL-TIME LANE (this thread; never queues, always takes the newest frame)
      camera -> V2 384x640 @ 4 FPS -> ByteTrack -> traffic summary, school-zone alerts
                                               -> a sign's track ends -> its best 3 crops -> V3 job
      IMU    -> S2 (or jerk) every 10 samples -> jolt -> 3 buffered frames from before it -> V1 job
             -> S1 every 100 m -> iri_window
    DEFERRED LANE (worker thread on its own cores; bounded priority queue)
      V3 at 320 px on sign crops  .  V1 at 480x800 on triggered frames, biggest jolt first

The models are not merged into one network: each wants a different input (full frame, road ahead,
close-up crop) at a different rate, and gating decides when the expensive ones run at all.
"""

from __future__ import annotations

import json
import logging
import os
import queue
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from edge import __version__, hud
from edge.aggregators.potholes import RouteDeduper, best_of, pothole_metadata
from edge.aggregators.school_zone import SchoolZoneMonitor
from edge.aggregators.signs import SignTracker, sign_metadata, vote
from edge.aggregators.traffic import TrafficAggregator
from edge.capture import Frame, ImuSample
from edge.capture.replay import ReplaySource
from edge.capture.ring_buffer import FrameRingBuffer
from edge.config import Section, repo_path
from edge.deferred import (PRIORITY_V1_SWEEP, PRIORITY_V1_TRIGGER, PRIORITY_V3_SIGN, DeferredLane, DeferredQueue,
                           Job)
from edge.events import EventFactory
from edge.geo import GpsTrack, load_school_zones
from edge.governor import Governor
from edge.health import Health
from edge.imaging import XYXY
from edge.imu import IriStream, IriWindow, Shock, ShockStream
from edge.outbox import MediaStore, Outbox
from edge.runners import s1_iri, s2_shock, v1_roadsense, v2_idd, v3_signs
from edge.runners.s1_iri import IriModel
from edge.runners.s2_shock import ShockModel
from edge.util import IST, parse_time

log = logging.getLogger(__name__)

_PEOPLE = {"person", "rider"}
_TICKER_TYPES = {"pothole", "zebra_crossing", "sign_condition", "road_shock", "pedestrian_zone_alert"}


class EdgeAgent:
    def __init__(self, cfg: Section, *, use_video: bool, use_imu: bool, realtime: bool, out_dir: Path,
                 hud_path: Path | None = None, hud_fps: float = 30.0, show: bool = False):
        self.cfg = cfg
        self.realtime = realtime
        self.out_dir = out_dir
        self.health = Health()
        self.gps = GpsTrack()
        self.events = EventFactory(self.gps)
        self.outbox = Outbox(out_dir, cfg.device.id, cfg.device.vehicle_id, cfg.output.envelope_every_s)
        self.media = MediaStore(out_dir / "media", cfg.output.jpeg_max_bytes)
        self.ticker: deque[str] = deque(maxlen=5)
        self.devices: dict[str, str] = {}
        models = cfg.models

        self.iri: IriStream | None = None
        self.shocks: ShockStream | None = None
        if use_imu:
            iri_model = IriModel(repo_path(models.s1.tflite), repo_path(models.s1.calibration))
            self.iri = IriStream(iri_model, cfg.iri.window_m, cfg.iri.min_speed_mps, cfg.iri.az_gravity_offset,
                                 self.health.timed)
            self.shocks = ShockStream(cfg.triggers, self._load_s2(), cfg.realtime.stopped_speed_mps, self.health.timed)
            self.devices.update(S1="cpu (TFLite)", S2="cpu (TFLite)" if self.shocks.model else "jerk threshold")

        self.vision = use_video
        self.jobs: DeferredQueue | None = None
        if use_video:
            compute = cfg.device.compute
            log.info("loading V2, V1 and V3 ...")
            self.v2 = v2_idd.TrafficRunner(repo_path(models.v2.weights), models.v2.imgsz, models.v2.tracker, compute,
                                           cfg.realtime.threads)
            self.v1 = v1_roadsense.RoadSenseRunner(repo_path(models.v1.weights), models.v1.imgsz, models.v1.conf,
                                                   compute, cfg.deferred.threads)
            self.v3 = v3_signs.SignConditionRunner(repo_path(models.v3.weights), models.v3.imgsz, models.v3.conf,
                                                   compute, cfg.deferred.threads)
            self.devices.update(V1=self.v1.model.device, V2=self.v2.model.device, V3=self.v3.model.device)
            rt = cfg.realtime
            self.ring = FrameRingBuffer(rt.ring_buffer_s, rt.ring_buffer_fps, rt.ring_buffer_max_width)
            self.traffic = TrafficAggregator(cfg.traffic.window_s, cfg.traffic.window_m)
            s = cfg.signs
            self.signs = SignTracker(s.crops_per_track, s.crop_margin, s.track_timeout_s, s.min_box_px)
            self.dedupe = RouteDeduper()
            self.jobs = DeferredQueue(cfg.deferred.max_jobs)
            self.lane = DeferredLane(self.jobs, {"v1_trigger": self._run_v1, "v1_sweep": self._run_v1,
                                                 "v3_sign": self._run_v3}, cfg.deferred.cpu_affinity)
            self.lane.start()

        self.school: SchoolZoneMonitor | None = None
        zones_path = repo_path(cfg.school_zones.geojson)
        if zones_path is not None:
            z = cfg.school_zones
            zones = load_school_zones(zones_path, z.point_buffer_m)
            self.school = SchoolZoneMonitor(zones, z.hours, z.days, z.alert_cooldown_s, z.crossing_roi)
            log.info("loaded %d school zones from %s", len(zones), zones_path)

        self.governor = Governor(cfg.realtime, cfg.deferred, cfg.thermal.throttle_c, realtime, self.jobs)

        self.hud_path, self.hud_fps, self.show = hud_path, hud_fps, show
        self._hud_writer: cv2.VideoWriter | None = None
        self._boxes: list = []
        self._people: deque[tuple[float, list[XYXY]]] = deque(maxlen=64)   # recent V2 person/rider boxes
        self._last_iri: IriWindow | None = None
        self._next_v2_t = float("-inf")
        self._next_sweep_t = float("-inf")
        self._last_telemetry_t: float | None = None
        self._last_health_t: float | None = None
        self._first_frame_t: float | None = None
        self._last_frame_t: float | None = None
        self._triggers_without_frames = 0
        self._stop_requested = False

    # ------------------------------------------------------------------ main loop

    def run(self, source: ReplaySource) -> dict[str, Any]:
        affinity = self.cfg.realtime.cpu_affinity
        if affinity and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(affinity))
        last_t = None
        try:
            for kind, item in source:
                if kind == "imu":
                    self._on_imu(item)
                else:
                    self._on_frame(item)
                self._collect()
                self._periodic(item.t)
                last_t = item.t
                if self._stop_requested:
                    break
        except KeyboardInterrupt:
            log.warning("interrupted: finishing queued work and writing the summary")
        return self._finish(source, last_t)

    def _on_imu(self, s: ImuSample) -> None:
        self.outbox.start_trip(s.t)
        self.gps.update(s.t, s.lat, s.lon, s.speed)
        if self._last_telemetry_t is None or s.t - self._last_telemetry_t >= 1.0:
            self._last_telemetry_t = s.t
            self.outbox.add_telemetry(s.t, s.lat, s.lon, s.speed, self.gps.heading)
        if self.iri is not None:
            window = self.iri.add(s)
            if window is not None:
                self._emit_iri(window)
        if self.shocks is not None:
            shock_events, triggers = self.shocks.add(s, self.gps.odometer)
            for shock in shock_events:
                self._emit_shock(shock)
            for shock in triggers:
                self._trigger_v1(shock)

    def _on_frame(self, f: Frame) -> None:
        self.outbox.start_trip(f.t)
        if self._first_frame_t is None:
            self._first_frame_t = f.t
        self._last_frame_t = f.t
        self.ring.push(f)
        fix = self.gps.current
        speed = fix.speed if fix is not None else None     # None: no GPS (video-only replay)

        if f.t >= self._next_v2_t:
            self._next_v2_t = f.t + self.governor.v2_period(speed)
            with self.health.timed("V2"):
                boxes = self.v2.track(f.image)
            self._boxes = boxes
            self._people.append((f.t, [b.xyxy for b in boxes if b.name in _PEOPLE]))
            sample = self.traffic.add(f.t, boxes, self.gps.odometer, speed)
            if sample is not None:
                self._emit_traffic(sample)
            self.signs.update(f.t, f.image, boxes)
            if self.school is not None:
                alert = self.school.update(f.t, fix, boxes, (f.image.shape[1], f.image.shape[0]))
                if alert is not None:
                    self._publish(self.events.make("pedestrian_zone_alert", f.t, None, v2_idd.MODEL_NAME,
                                                   v2_idd.MODEL_VERSION, alert))

        for track in self.signs.pop_ended(f.t):
            self._enqueue(Job("v3_sign", PRIORITY_V3_SIGN, track.last_t, {"track": track}))

        sweep_fps = self.cfg.deferred.v1_sweep_fps
        moving = speed is None or speed >= self.cfg.realtime.stopped_speed_mps
        if sweep_fps > 0 and f.t >= self._next_sweep_t and moving:
            self._next_sweep_t = f.t + 1.0 / sweep_fps
            latest = self.ring.frames[-1] if self.ring.frames else None
            if latest is not None:
                self._enqueue(Job("v1_sweep", PRIORITY_V1_SWEEP, latest.t,
                                  {"frames": [latest], "people": {latest.index: self._people_near(latest.t)},
                                   "shock": None}))

        school_active = self.school is not None and self.school.active(f.t, fix)
        self.governor.update(school_active)
        if self.hud_path is not None or self.show:
            self._draw_hud(f)

    def _trigger_v1(self, shock: Shock) -> None:
        if not self.vision:
            return
        frames = self.ring.pick([shock.t - offset for offset in self.cfg.triggers.frame_offsets_s])
        if not frames:
            self._triggers_without_frames += 1
            return
        people = {frame.index: self._people_near(frame.t) for frame in frames}
        self._enqueue(Job("v1_trigger", PRIORITY_V1_TRIGGER + shock.priority, shock.t,
                          {"frames": frames, "people": people, "shock": shock}))
        self.ticker.append(f"{_clock(shock.t)}  jolt ({shock.method}, {shock.p2p:.1f} m/s2): V1 queued on {len(frames)} frames")

    def _enqueue(self, job: Job, block: bool | None = None) -> None:
        # Offline replays block instead of dropping, so every job runs and results are reproducible.
        self.jobs.put(job, block=(not self.realtime) if block is None else block)

    def _people_near(self, t: float, tolerance: float = 0.5) -> list[XYXY]:
        best = min(self._people, key=lambda entry: abs(entry[0] - t), default=None)
        return best[1] if best is not None and abs(best[0] - t) <= tolerance else []

    # ------------------------------------------------------------------ deferred lane (worker thread)

    def _run_v1(self, job: Job) -> list:
        results = []
        for frame in job.payload["frames"]:
            with self.health.timed("V1"):
                results.append((frame, self.v1.detect(frame.image)))
        return results

    def _run_v3(self, job: Job) -> list[tuple[str, float]]:
        votes = []
        for _score, _order, candidate in job.payload["track"].crops:
            with self.health.timed("V3"):
                result = self.v3.classify(candidate.crop)
            if result is not None:
                votes.append(result)
        return votes

    # ------------------------------------------------------------------ results -> events

    def _collect(self) -> None:
        if not self.vision:
            return
        while True:
            try:
                job, result = self.lane.results.get_nowait()
            except queue.Empty:
                return
            if result is None:
                continue
            if job.kind == "v3_sign":
                self._on_sign_result(job, result)
            else:
                self._on_v1_result(job, result)

    def _on_v1_result(self, job: Job, results: list) -> None:
        cfg = self.cfg.potholes
        shock: Shock | None = job.payload["shock"]
        pothole = best_of(results, "pothole")
        if pothole is not None:
            # A confirmed jolt locates the pothole where the wheel hit it; a sweep frame only knows where the bus was.
            location_t = shock.t if shock is not None else pothole.frame.t
            kind = "pothole_imu" if shock is not None else "pothole_sweep"
            if self.dedupe.accept(kind, self._route_m(location_t), cfg.dedupe_m):
                media = self.media.save_crop(pothole.frame.image, pothole.box.xyxy, cfg.crop_margin,
                                             self._blur_boxes(job, pothole.frame))
                metadata = pothole_metadata(pothole, tuple(cfg.severity_area_frac),
                                            "imu_shock" if shock is not None else "sweep", shock)
                self._publish(self.events.make("pothole", pothole.frame.t, pothole.box.conf, v1_roadsense.MODEL_NAME,
                                               v1_roadsense.MODEL_VERSION, metadata, media, location_t=location_t))
        zebra = best_of(results, "zebra_crossing")
        if zebra is not None:
            if self.dedupe.accept("zebra", self._route_m(zebra.frame.t), cfg.zebra_dedupe_m):
                media = self.media.save_crop(zebra.frame.image, zebra.box.xyxy, cfg.crop_margin,
                                             self._blur_boxes(job, zebra.frame))
                metadata = {"bbox": zebra.bbox_original, "frameSize": list(zebra.frame.orig_size)}
                self._publish(self.events.make("zebra_crossing", zebra.frame.t, zebra.box.conf, v1_roadsense.MODEL_NAME,
                                               v1_roadsense.MODEL_VERSION, metadata, media))

    def _route_m(self, t: float) -> float:
        """Metres along the route at time t. Without GPS (video-only runs), assume 10 m/s."""
        fix = self.gps.at(t)
        return fix.odometer if fix is not None else t * 10.0

    @staticmethod
    def _blur_boxes(job: Job, frame: Frame) -> list[XYXY]:
        """People near this frame, mapped into the (possibly downscaled) ring-buffer image."""
        return [tuple(v * frame.scale for v in box) for box in job.payload["people"].get(frame.index, [])]

    def _on_sign_result(self, job: Job, results: list[tuple[str, float]]) -> None:
        verdict = vote(results)
        if verdict is None:
            return
        condition, confidence, votes = verdict
        track = job.payload["track"]
        best = track.best()
        media = self.media.save(best.crop.copy(), best.blur_boxes)
        self._publish(self.events.make("sign_condition", best.t, confidence, v3_signs.MODEL_NAME,
                                       v3_signs.MODEL_VERSION, sign_metadata(track, condition, votes), media))

    def _emit_iri(self, w: IriWindow) -> None:
        self._last_iri = w
        metadata = {
            "start": {"lat": round(w.start[0], 7), "lon": round(w.start[1], 7)},
            "end": {"lat": round(w.end[0], 7), "lon": round(w.end[1], 7)},
            "lengthM": w.length_m, "iri": round(w.iri, 2), "iriRaw": round(w.iri_raw, 2), "iriClass": w.iri_class,
            "meanSpeedMps": round(w.mean_speed_mps, 2), "nSamples": w.n_samples,
        }
        self._publish(self.events.make("iri_window", w.t_end, None, s1_iri.MODEL_NAME, s1_iri.MODEL_VERSION,
                                       metadata, sensor="imu", location_t=(w.t_start + w.t_end) / 2))

    def _emit_shock(self, shock: Shock) -> None:
        metadata = {
            "class": shock.cls, "label": shock.label, "smoothed": shock.smoothed, "method": shock.method,
            "probs": None if shock.probs is None else [round(p, 3) for p in shock.probs],
            "peakToPeak": round(shock.p2p, 2),
        }
        if shock.method == "s2":
            model, version, confidence = s2_shock.MODEL_NAME, s2_shock.MODEL_VERSION, max(shock.probs)
        else:
            model, version, confidence = "jerk-threshold", "v1", None
        self._publish(self.events.make("road_shock", shock.t, confidence, model, version, metadata, sensor="imu"))

    def _emit_traffic(self, sample: dict[str, Any]) -> None:
        self._publish(self.events.make("traffic_sample", sample["t_mid"], None, v2_idd.MODEL_NAME,
                                       v2_idd.MODEL_VERSION, sample["metadata"]))

    def _emit_health(self, t: float) -> None:
        metadata: dict[str, Any] = {
            "cpuTempC": self.governor.temperature(),
            "fps": self.health.rates(t),
            "outboxDepth": len(self.outbox.events),
            "bytesSentToday": self.outbox.bytes_written + self.media.bytes_written,
            "cameraSecondsToday": round(self._camera_seconds(), 1),
            "latencyMs": self.health.latency_ms(),
            "compute": self.devices,
        }
        if self.jobs is not None:
            metadata["deferred"] = {"depth": len(self.jobs), "dropped": dict(self.jobs.dropped),
                                    "paused": self.jobs.paused, "utilisation": round(self.lane.utilisation, 3)}
        self._publish(self.events.make("device_health", t, None, "edge-agent", __version__, metadata, sensor="system"))

    def _publish(self, event: dict[str, Any]) -> None:
        self.outbox.add_event(event)
        if event["eventType"] in _TICKER_TYPES:
            self.ticker.append(f"{_clock_iso(event['capturedAt'])}  {_describe(event)}")

    def _periodic(self, t: float) -> None:
        self.outbox.maybe_flush(t)
        if self._last_health_t is None:
            self._last_health_t = t
        elif t - self._last_health_t >= self.cfg.output.health_every_s:
            self._last_health_t = t
            self._emit_health(t)

    # ------------------------------------------------------------------ shutdown and summary

    def _finish(self, source: ReplaySource, last_t: float | None) -> dict[str, Any]:
        end_t = source.end_t or last_t or 0.0
        if self.vision:
            for track in self.signs.pop_all():
                self._enqueue(Job("v3_sign", PRIORITY_V3_SIGN, track.last_t, {"track": track}), block=True)
            self.lane.drain(timeout=None if not self.realtime else 60.0)
            self._collect()
            self.lane.stop()
            sample = self.traffic.close(end_t, self.gps.odometer)
            if sample is not None:
                self._emit_traffic(sample)
        self._emit_health(end_t)
        self.outbox.flush(end_t)
        if self._hud_writer is not None:
            self._hud_writer.release()
        if self.show:
            cv2.destroyAllWindows()
        summary = self._summary(source)
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _camera_seconds(self) -> float:
        if self._first_frame_t is None or self._last_frame_t is None:
            return 0.0
        return self._last_frame_t - self._first_frame_t

    def _summary(self, source: ReplaySource) -> dict[str, Any]:
        camera_s = self._camera_seconds()
        raw_video_bytes = camera_s * self.cfg.output.raw_video_mbps * 1e6 / 8
        uplink = self.outbox.bytes_written + self.media.bytes_written
        summary: dict[str, Any] = {
            "mode": f"real time x{source.rate:g}" if self.realtime else "offline (as fast as possible)",
            "compute": self.devices,
            "captureSeconds": round((source.end_t or 0) - (source.start_t or 0), 1),
            "cameraSeconds": round(camera_s, 1),
            "distanceKm": round(self.gps.odometer / 1000.0, 3),
            "latencyMs": self.health.latency_ms(),
            "events": dict(self.outbox.counts),
            "uplink": {"envelopeBytes": self.outbox.bytes_written, "mediaBytes": self.media.bytes_written,
                       "mediaFiles": self.media.files, "totalBytes": uplink,
                       "rawVideoBytes": round(raw_video_bytes),
                       "savedPct": round(100 * (1 - uplink / raw_video_bytes), 2) if raw_video_bytes else None},
            "outDir": str(self.out_dir),
        }
        if self.vision:
            summary["deferred"] = {"maxDepth": self.jobs.max_depth, "dropped": dict(self.jobs.dropped),
                                   "utilisation": round(self.lane.utilisation, 3),
                                   "framesSkipped": source.frames_skipped,
                                   "triggersWithoutFrames": self._triggers_without_frames}
        if self._last_iri is not None:
            summary["lastIriWindow"] = asdict(self._last_iri)
        return summary

    # ------------------------------------------------------------------ helpers

    def _load_s2(self) -> ShockModel | None:
        method = self.cfg.triggers.method
        path = repo_path(self.cfg.models.s2.tflite)
        if method == "jerk":
            return None
        if path.exists():
            return ShockModel(path)
        if method == "s2":
            raise FileNotFoundError(f"triggers.method is 's2' but {path} does not exist")
        log.warning("S2 model not found at %s: using the jerk-threshold trigger instead", path)
        return None

    def _draw_hud(self, f: Frame) -> None:
        fix = self.gps.current
        speed_kmh = fix.speed * 3.6 if fix is not None else 0.0
        iri = f"IRI {self._last_iri.iri:.1f} m/km ({self._last_iri.iri_class})" if self._last_iri else "IRI -"
        v2 = self.health.stats["V2"]
        status = [
            f"{_clock(f.t)} IST   {speed_kmh:4.0f} km/h   {self.gps.odometer / 1000:6.2f} km   {iri}",
            f"V2 p50 {v2.percentile(50):4.0f} ms   deferred queue {len(self.jobs)}   lane busy {self.lane.utilisation:4.0%}"
            f"   uplink {(self.outbox.bytes_written + self.media.bytes_written) / 1e6:.2f} MB",
        ]
        label = f"REPLAY on {self.devices.get('V2', 'cpu')} - not a Pi 5 measurement"
        image = hud.draw(f.image, self._boxes, status, list(self.ticker), label)
        if self.hud_path is not None:
            if self._hud_writer is None:
                self.hud_path.parent.mkdir(parents=True, exist_ok=True)
                self._hud_writer = cv2.VideoWriter(str(self.hud_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                   self.hud_fps, (image.shape[1], image.shape[0]))
            self._hud_writer.write(image)
        if self.show:
            cv2.imshow("Vidur edge agent", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._stop_requested = True


def _clock(t: float) -> str:
    return datetime.fromtimestamp(t, IST).strftime("%H:%M:%S")


def _clock_iso(stamp: str) -> str:
    return _clock(parse_time(stamp))


def _describe(event: dict[str, Any]) -> str:
    m, kind = event["metadata"], event["eventType"]
    conf = event.get("confidence")
    if kind == "pothole":
        return f"pothole  severity {m['severity']}  conf {conf:.2f}  ({m['trigger']})"
    if kind == "zebra_crossing":
        return f"zebra crossing  conf {conf:.2f}"
    if kind == "sign_condition":
        return f"sign {m['condition']}  votes {m['votes']['damaged']}D/{m['votes']['good']}G"
    if kind == "road_shock":
        return f"road shock: {m['label']} ({m['method']})"
    if kind == "pedestrian_zone_alert":
        return f"{m['level']}: people near {m['schoolName']}"
    return kind
