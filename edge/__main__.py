"""Command line: replay a recorded ride through the edge agent.

    python -m edge --imu ride.csv                                     # IMU only: S1 + S2
    python -m edge --video ride.mp4 --imu ride.csv --video-start 2026-10-05T08:02:11.000Z --hud out/ride.mp4
    python -m edge --video ride.mp4 --imu ride.csv --rate 1          # real-time pacing, as on the bus
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from edge.agent import EdgeAgent
from edge.capture.replay import ImuCsvReader, ReplaySource, VideoReader
from edge.config import load_config, repo_path
from edge.util import parse_time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m edge", description="Replay a recorded ride through the Vidur edge agent.")
    p.add_argument("--video", help="front-camera video of the ride")
    p.add_argument("--imu", help="IMU + GPS log (CSV: time,ax,ay,az,wx,wy,wz,latitude,longitude,...,speed)")
    p.add_argument("--video-start", help="UTC time of the video's first frame, ISO 8601. "
                                         "Default: the IMU log's first timestamp plus --offset")
    p.add_argument("--offset", type=float, default=0.0, help="seconds from the IMU log's start to the first video frame")
    p.add_argument("--rate", type=float, default=0.0,
                   help="replay rate: 1 = real time (frames are skipped if the agent falls behind, as on the bus), "
                        "4 = 4x, 0 = as fast as possible with no drops (default)")
    p.add_argument("--config", nargs="*", default=[], help="config overlays merged over edge/config.yaml, in order")
    p.add_argument("--device", help="override device.compute: auto | cpu | cuda:0")
    p.add_argument("--schools", help="GeoJSON of school zones (overrides school_zones.geojson)")
    p.add_argument("--out", help="output folder (default: out/run-<timestamp>)")
    p.add_argument("--hud", help="write an annotated demo video to this .mp4 path")
    p.add_argument("--show", action="store_true", help="show the annotated video in a window (q quits)")
    p.add_argument("--max-seconds", type=float, help="stop after this many seconds of recording")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not args.video and not args.imu:
        p.error("give --video, --imu, or both")

    cfg = load_config(*args.config)
    if args.device:
        cfg.device["compute"] = args.device
    if args.schools:
        cfg.school_zones["geojson"] = args.schools

    imu = ImuCsvReader(args.imu) if args.imu else None
    video = None
    if args.video:
        if args.video_start:
            start = parse_time(args.video_start)
        elif imu is not None:
            start = imu.first_time() + args.offset
            if args.offset == 0.0:
                logging.warning("no --video-start or --offset: assuming the video starts with the IMU log")
        else:
            start = time.time()
        video = VideoReader(args.video, start)

    out_dir = repo_path(args.out or f"{cfg.output.dir}/run-{time.strftime('%Y%m%d-%H%M%S')}")
    agent = EdgeAgent(cfg, use_video=video is not None, use_imu=imu is not None, realtime=args.rate > 0,
                      out_dir=out_dir, hud_path=Path(args.hud) if args.hud else None,
                      hud_fps=video.fps if video is not None else 30.0, show=args.show)
    summary = agent.run(ReplaySource(video, imu, rate=args.rate, max_seconds=args.max_seconds))
    print(format_summary(summary))
    return 0


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
    lines += [line, f"  output        {s['outDir']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
