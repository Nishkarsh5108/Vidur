"""Build the demo fleet: a per-frame GPS trail for every video, and the real school zones along the routes.

    python tools/sim/build_scenario.py                      # uses tools/sim/demo_fleet.yaml
    python tools/sim/build_scenario.py --spec my_fleet.yaml --traces-dir D:/path/to/traces

Writes into the spec's out_dir (default recordings/fleet/, not versioned):
    <vehicle>_gps.csv        one row per video frame: frame,video_time_s,time,latitude,longitude,speed_mps,heading_deg
    <vehicle>_route.geojson  the route as a line (drop it on geojson.io to check it)
    school_zones.geojson     geofences around real OpenStreetMap schools near the routes
    scenario.json            what tools/sim/run_fleet.py runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trails import (fetch_schools, load_trace, pick_start, school_zones, trail_distance_mode,  # noqa: E402
                    trail_time_mode, write_trail, zone_features)


def video_info(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, default=Path(__file__).with_name("demo_fleet.yaml"))
    p.add_argument("--traces-dir", type=Path, help="overrides traces_dir in the spec")
    args = p.parse_args()

    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    traces_dir = args.traces_dir or Path(spec["traces_dir"])
    out_dir = REPO / spec.get("out_dir", "recordings/fleet")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "osm_cache"

    buses, bboxes = [], []
    for bus in spec["buses"]:
        vehicle = bus["vehicle"]
        video = REPO / bus["video"]
        fps, n_frames = video_info(video)
        duration = n_frames / fps
        trace = load_trace(Path(bus["trace"]) if Path(bus["trace"]).is_absolute() else traces_dir / bus["trace"])

        zones = None
        if bus.get("pass_school"):
            raw = fetch_schools(trace.bbox(300), cache / f"{Path(trace.name).stem}.json")
            zones = school_zones(zone_features(raw))
            print(f"{vehicle}: {len(raw)} OSM schools around {trace.name}")

        if bus.get("imu"):
            imu = pd.read_csv(REPO / bus["imu"])
            video_t = np.arange(n_frames) / fps
            speed = np.interp(video_t, imu["video_time_s"], imu["speed_ms"])
            distance = float(np.sum(speed * (1.0 / fps)))
            t0 = pick_start(trace, duration, distance_m=distance, min_speed=bus.get("min_speed_mps", 2.0), zones=zones)
            rows = trail_distance_mode(trace, t0, video_t, speed)
            mode = "distance (IMU speed)"
        else:
            t0 = pick_start(trace, duration, min_speed=bus.get("min_speed_mps", 2.0),
                            target_kmh=bus.get("target_kmh"), zones=zones)
            rows = trail_time_mode(trace, t0, n_frames, fps)
            mode = "time (trace speed)"

        gps_csv, route = out_dir / f"{vehicle}_gps.csv", out_dir / f"{vehicle}_route.geojson"
        write_trail(rows, gps_csv, route, {"vehicle": vehicle, "video": bus["video"], "trace": trace.name,
                                           "traceStart": rows[0]["time"], "mode": mode})
        lat = np.array([r["latitude"] for r in rows])
        lon = np.array([r["longitude"] for r in rows])
        bboxes.append((lat.min(), lon.min(), lat.max(), lon.max()))
        km = float(np.sum(np.hypot(np.diff(lat) * 111_320, np.diff(lon) * 111_320 * np.cos(np.radians(lat[:-1]))))) / 1000
        speeds = [r["speed_mps"] * 3.6 for r in rows]
        print(f"{vehicle}: {video.name} ({duration:.0f} s) on {trace.name} from {rows[0]['time']}, {mode}: "
              f"{km:.2f} km at {np.mean(speeds):.0f} km/h (max {max(speeds):.0f})")
        buses.append({"vehicle": vehicle, "video": str(video), "gps": str(gps_csv),
                      "imu": str(REPO / bus["imu"]) if bus.get("imu") else None, "note": bus.get("note", "")})

    # One school-zone file for the whole fleet, around all routes.
    s = min(b[0] for b in bboxes) - 0.005, min(b[1] for b in bboxes) - 0.005
    n = max(b[2] for b in bboxes) + 0.005, max(b[3] for b in bboxes) + 0.005
    raw = []
    for i, box in enumerate(bboxes):
        pad = 0.004
        raw += fetch_schools((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), cache / f"route_{i}.json")
    unique = {f["properties"]["osm_id"]: f for f in raw}.values()
    zones_path = out_dir / "school_zones.geojson"
    zones_path.write_text(json.dumps({"type": "FeatureCollection", "features": zone_features(list(unique))}),
                          encoding="utf-8")
    zones = school_zones(zone_features(list(unique)))
    for bus in buses:
        trail = pd.read_csv(bus["gps"])
        passed = sorted({z.name for z in zones for la, lo in zip(trail.latitude[::10], trail.longitude[::10])
                         if z.contains(la, lo)})
        bus["schoolZones"] = passed
        print(f"{bus['vehicle']}: passes school zones {passed or 'none'}")

    scenario = {"schools": str(zones_path), "bbox": [s[0], s[1], n[0], n[1]], "buses": buses}
    (out_dir / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(f"\n{len(unique)} school zones -> {zones_path}\nscenario -> {out_dir / 'scenario.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
