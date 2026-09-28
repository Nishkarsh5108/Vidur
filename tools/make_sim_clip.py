"""Cut a synced clip from a BeamNG trip: dashcam video + the matching IMU rows.

    python tools/make_sim_clip.py <trip_dir> --start 3180 --end 3680

Samples are 100 Hz (sample_number = sim_time x 100). Dashcam frames are dashcam/frame_<sample>.jpg,
nominally every 5 samples (20 FPS) with some missing. The video is written at 100 / step FPS with one
video frame per `step` samples; a missing frame repeats the previous one, so video time t always
corresponds to sample start + 100 t. Output (default recordings/, not versioned):

    <name>_dashcam.mp4          H.264, plays everywhere (browser, PowerPoint)
    <name>_imu_speed_data.csv   the imu_speed_data.csv rows start..end, plus video_time_s and video_frame
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "model_lab"))

from engine import H264Writer  # noqa: E402

SAMPLE_HZ = 100


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trip_dir", type=Path)
    p.add_argument("--start", type=int, required=True, help="first sample_number (inclusive)")
    p.add_argument("--end", type=int, required=True, help="last sample_number (inclusive)")
    p.add_argument("--step", type=int, default=5, help="samples per dashcam frame (default 5 = 20 FPS)")
    p.add_argument("--out", type=Path, default=REPO / "recordings")
    p.add_argument("--name", help="output name prefix (default: <car>_<map>_<trip>_<start>-<end>)")
    args = p.parse_args()

    trip = args.trip_dir
    frames = {int(m.group(1)): f for f in (trip / "dashcam").iterdir()
              if (m := re.fullmatch(r"frame_(\d+)\.jpg", f.name))}
    name = args.name or f"{trip.parents[1].name}_{trip.parent.name}_{trip.name}_{args.start}-{args.end}"
    args.out.mkdir(parents=True, exist_ok=True)

    # Video: one frame per `step` samples; hold the last available frame over missing ones.
    ticks = range(args.start, args.end + 1, args.step)
    first = max((s for s in frames if s <= args.start), default=None)
    if first is None:
        first = min(s for s in frames if s >= args.start)
    current = cv2.imread(str(frames[first]))
    height, width = current.shape[:2]
    video_path = args.out / f"{name}_dashcam.mp4"
    writer = H264Writer(video_path, width, height, SAMPLE_HZ / args.step)
    shown, held = 0, 0
    for sample in ticks:
        if sample in frames:
            current = cv2.imread(str(frames[sample]))
            shown += 1
        else:
            held += 1
        writer.write(current)
    writer.close()

    # IMU rows for the same samples, with the video time they line up with.
    csv_path = args.out / f"{name}_imu_speed_data.csv"
    rows = 0
    with open(trip / "imu_speed_data.csv", newline="") as src, open(csv_path, "w", newline="") as dst:
        reader, out = csv.DictReader(src), None
        for row in reader:
            sample = int(row["sample_number"])
            if not args.start <= sample <= args.end:
                continue
            if out is None:
                out = csv.DictWriter(dst, fieldnames=[*reader.fieldnames, "video_time_s", "video_frame"])
                out.writeheader()
            offset = sample - args.start
            out.writerow({**row, "video_time_s": f"{offset / SAMPLE_HZ:.2f}", "video_frame": offset // args.step})
            rows += 1

    print(f"{video_path}\n  {len(ticks)} video frames at {SAMPLE_HZ / args.step:g} FPS "
          f"({len(ticks) / (SAMPLE_HZ / args.step):.2f} s): {shown} dashcam frames, {held} held over missing ones")
    print(f"{csv_path}\n  {rows} IMU rows (samples {args.start}-{args.end}, {rows / SAMPLE_HZ:.2f} s at {SAMPLE_HZ} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
