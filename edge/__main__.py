"""Command line: replay a recorded ride through the edge agent, optionally sending events to the backend.

    python -m edge --imu ride.csv                                     # IMU only: S1 + shocks
    python -m edge --video ride.mp4 --imu ride.csv --video-start 2026-10-05T08:02:11.000Z --hud out/ride.mp4
    python -m edge --video bus.mp4 --gps bus_gps.csv --clock now --backend http://127.0.0.1:8000 --rate 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from edge.agent import EdgeAgent
from edge.backend import BackendUploader
from edge.capture.replay import GpsCsvReader, ImuCsvReader, ReplaySource, VideoReader
from edge.config import load_config, repo_path
from edge.util import parse_time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m edge", description="Replay a recorded ride through the Vidur edge agent.")
    p.add_argument("--video", help="front-camera video of the ride")
    p.add_argument("--imu", help="IMU log: the phone logger's CSV (with GPS) or a BeamNG clip from tools/make_sim_clip.py")
    p.add_argument("--gps", help="GPS trail CSV (time,latitude,longitude,speed_mps,...), e.g. from tools/sim")
    p.add_argument("--video-start", help="UTC time of the video's first frame, ISO 8601. Default: from the GPS "
                                         "trail, else the IMU log's first timestamp plus --offset")
    p.add_argument("--offset", type=float, default=0.0, help="seconds from the IMU log's start to the first video frame")
    p.add_argument("--clock", default="file",
                   help="'file': keep the recording's timestamps (default); 'now': re-time the recording so it "
                        "starts now (live demo); or an ISO 8601 time to start at")
    p.add_argument("--rate", type=float, default=0.0,
                   help="replay rate: 1 = real time (frames are skipped if the agent falls behind, as on the bus), "
                        "4 = 4x, 0 = as fast as possible with no drops (default)")
    p.add_argument("--config", nargs="*", default=[], help="config overlays merged over edge/config.yaml, in order")
    p.add_argument("--device", help="override device.compute: auto | cpu | cuda:0")
    p.add_argument("--vehicle-id", help="override device.vehicle_id")
    p.add_argument("--device-id", help="override device.id")
    p.add_argument("--trip-id", help="default: <vehicle>-<start time>")
    p.add_argument("--schools", help="GeoJSON of school zones (overrides school_zones.geojson)")
    p.add_argument("--backend", help="FleetSense backend URL, e.g. http://127.0.0.1:8000: events are POSTed to /api/v1/ingest")
    p.add_argument("--device-key", default=os.environ.get("VIDUR_DEVICE_KEY", "demo-secret-key"),
                   help="X-Device-Key for the backend (default: $VIDUR_DEVICE_KEY or demo-secret-key)")
    p.add_argument("--out", help="output folder (default: out/run-<timestamp>)")
    p.add_argument("--hud", help="write an annotated demo video to this .mp4 path")
    p.add_argument("--show", action="store_true", help="show the annotated video in a window (q quits)")
    p.add_argument("--max-seconds", type=float, help="stop after this many seconds of recording")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not (args.video or args.imu or args.gps):
        p.error("give --video, --imu and/or --gps")

    cfg = load_config(*args.config)
    for flag, key in ((args.device, "compute"), (args.vehicle_id, "vehicle_id"), (args.device_id, "id")):
        if flag:
            cfg.device[key] = flag
    if args.schools:
        cfg.school_zones["geojson"] = args.schools

    imu = ImuCsvReader(args.imu) if args.imu else None
    gps = GpsCsvReader(args.gps) if args.gps else None
    start = recording_start(args, imu, gps)
    if imu is not None and imu.relative:
        imu.time_base = start                  # a BeamNG clip's time 0 is the video's first frame
    shift = 0.0 if args.clock == "file" else (time.time() if args.clock == "now" else parse_time(args.clock)) - start
    for reader in (imu, gps):
        if reader is not None:
            reader.shift = shift
    video = VideoReader(args.video, start + shift) if args.video else None

    vehicle_id = cfg.device.vehicle_id
    trip_id = args.trip_id or f"{vehicle_id}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(start + shift))}"
    out_dir = repo_path(args.out or f"{cfg.output.dir}/run-{time.strftime('%Y%m%d-%H%M%S')}")
    agent = EdgeAgent(cfg, use_video=video is not None, use_imu=imu is not None, use_gps=gps is not None,
                      realtime=args.rate > 0, out_dir=out_dir, trip_id=trip_id,
                      hud_path=Path(args.hud) if args.hud else None,
                      hud_fps=video.fps if video is not None else 30.0, show=args.show)

    uploader = None
    if args.backend:
        uploader = BackendUploader(args.backend, args.device_key, cfg.device.id, vehicle_id, trip_id, out_dir,
                                   media_dir=out_dir / "media")
        uploader.start()
        agent.outbox.listeners.append(uploader.submit)
        logging.info("sending events to %s as %s / %s", uploader.endpoint, vehicle_id, trip_id)

    summary = agent.run(ReplaySource(video, imu, gps, rate=args.rate, max_seconds=args.max_seconds))
    if uploader is not None:
        summary["backend"] = uploader.close()
    print(format_summary(summary))
    return 0


def recording_start(args, imu: ImuCsvReader | None, gps: GpsCsvReader | None) -> float:
    """When the recording's first video frame (or first sample) was captured, before any --clock shift."""
    if args.video_start:
        return parse_time(args.video_start)
    if gps is not None and gps.video_start() is not None:
        return gps.video_start()
    if imu is not None and not imu.relative:
        if args.video and args.offset == 0.0:
            logging.warning("no --video-start or --offset: assuming the video starts with the IMU log")
        return imu.first_time() + args.offset
    if gps is not None:
        return gps.first_time()
    return time.time()


def format_summary(s: dict) -> str:
    lines = ["", f"Run summary ({s['mode']})", "-" * 60,
             f"  recording     {s['captureSeconds']:.0f} s   camera {s['cameraSeconds']:.0f} s   "
             f"distance {s['distanceKm']:.2f} km",
             f"  compute       {', '.join(f'{k} {v}' for k, v in s['compute'].items())}"]
    for model, st in s["latencyMs"].items():
        lines.append(f"  {model:<3} latency  p50 {st['p50']:7.1f} ms   p95 {st['p95']:7.1f} ms   runs {st['runs']}")
    if "deferred" in s:
        d = s["deferred"]
        lines.append(f"  deferred lane busy {d['utilisation']:.0%}   max queue {d['maxDepth']}   dropped {d['dropped'] or 0}"
                     f"   frames skipped {d['framesSkipped']}")
    lines.append("  events        " + ", ".join(f"{k} {v}" for k, v in sorted(s["events"].items())))
    u = s["uplink"]
    line = f"  uplink        {u['totalBytes'] / 1e6:.2f} MB ({u['mediaFiles']} crops)"
    if u["savedPct"] is not None:
        line += f" vs {u['rawVideoBytes'] / 1e9:.2f} GB of raw video: {u['savedPct']:.1f}% saved"
    lines.append(line)
    if "backend" in s:
        b = s["backend"]
        lines.append(f"  backend       accepted {b.get('accepted', 0)}, duplicates {b.get('duplicates', 0)}, "
                     f"rejected {b.get('rejected', 0)}, undelivered {b.get('undelivered', 0)} -> {b['endpoint']}")
        for reason, n in b.get("rejectReasons", {}).items():
            lines.append(f"                rejected x{n}: {reason}")
    lines += [f"  output        {s['outDir']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
