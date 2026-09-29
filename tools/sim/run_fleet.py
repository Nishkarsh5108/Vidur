"""Run the demo fleet: one edge agent per bus, in parallel on this laptop, streaming events to the backend.

    python tools/sim/run_fleet.py --backend http://127.0.0.1:8000             # real time, all buses
    python tools/sim/run_fleet.py --backend http://<backend-ip>:8000 --rate 2 --only BUS_TRAFFIC_03 --hud

Each bus replays its video with its per-frame GPS trail (and IMU, if it has one) through the same edge
agent a real bus would run, re-timed to start now, so the dashboard sees a live fleet. Build the
scenario first with tools/sim/build_scenario.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def backend_get(url: str, path: str, timeout: float = 5.0):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as response:
        return json.load(response)


def stream(prefix: str, proc: subprocess.Popen, keep: list[str]) -> None:
    for line in proc.stdout:
        line = line.rstrip()
        if any(noise in line for noise in ("oneDNN", "absl::", "port.cc", "XNNPACK")):
            continue
        keep.append(line)
        print(f"[{prefix}] {line}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", type=Path, default=REPO / "recordings/fleet/scenario.json")
    p.add_argument("--backend", help="backend URL, e.g. http://127.0.0.1:8000 (omit to only write envelopes locally)")
    p.add_argument("--device-key", default="demo-secret-key")
    p.add_argument("--rate", type=float, default=1.0, help="1 = real time (default), 2 = twice as fast, 0 = no pacing")
    p.add_argument("--clock", default="now", help="'now' (default) or an ISO 8601 start time, e.g. 2026-09-29T08:20:00+05:30 "
                                                   "to replay inside school hours")
    p.add_argument("--only", nargs="*", help="run only these vehicles")
    p.add_argument("--hud", action="store_true", help="also write an annotated video per bus to out/fleet/")
    p.add_argument("--max-concurrent", type=int, default=2,
                   help="how many bus processes load their models at once (default 2). Each one briefly needs "
                        "several GB of RAM while PyTorch/CUDA/Ultralytics import; too many at once can exceed the "
                        "Windows paging file even when steady-state usage is fine. Lower this if you see "
                        "'OSError: [WinError 1455] The paging file is too small'.")
    p.add_argument("--stagger", type=float, default=6.0, help="seconds to wait before starting each next process")
    args = p.parse_args()

    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    buses = [b for b in scenario["buses"] if not args.only or b["vehicle"] in args.only]
    if args.backend:
        try:
            health = backend_get(args.backend, "/api/v1/health")
            print(f"backend {args.backend}: {health}")
        except Exception as exc:
            print(f"WARNING: backend {args.backend} is not reachable ({exc}); the buses will keep retrying.")

    # One clock for the whole fleet, so the buses drive "at the same time" on the dashboard.
    clock = datetime.now(timezone.utc).isoformat() if args.clock == "now" else args.clock
    run_id = time.strftime("%Y%m%d-%H%M%S")
    started, done, slot_free = [], [], threading.Semaphore(args.max_concurrent)

    def launch(bus: dict, index: int) -> None:
        vehicle = bus["vehicle"]
        if index > 0:
            time.sleep(args.stagger)             # spreads out the heaviest part: PyTorch/CUDA/Ultralytics import
        slot_free.acquire()
        try:
            out = REPO / "out" / "fleet" / f"{run_id}_{vehicle}"
            cmd = [sys.executable, "-m", "edge", "--config", str(REPO / "edge/config.laptop.yaml"),
                   "--video", bus["video"], "--gps", bus["gps"], "--schools", scenario["schools"],
                   "--vehicle-id", vehicle, "--device-id", f"EDGE_{vehicle}", "--trip-id", f"{vehicle}-{run_id}",
                   "--clock", clock, "--rate", str(args.rate), "--out", str(out)]
            if bus.get("imu"):
                cmd += ["--imu", bus["imu"]]
            if args.backend:
                cmd += ["--backend", args.backend, "--device-key", args.device_key]
            if args.hud:
                cmd += ["--hud", str(out.with_name(out.name + "_hud.mp4"))]
            print(f"starting {vehicle}: {Path(bus['video']).name} {'+ IMU ' if bus.get('imu') else ''}+ GPS "
                  f"({bus.get('note', '')})")
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            lines: list[str] = []
            reader = threading.Thread(target=stream, args=(vehicle, proc, lines), daemon=True)
            reader.start()
            started.append((vehicle, proc, reader, lines))
            proc.wait()
            reader.join()
            done.append((vehicle, proc.returncode))
        except Exception as exc:
            print(f"[{vehicle}] failed to launch: {exc}")
            done.append((vehicle, 1))
        finally:
            slot_free.release()                  # frees a slot once this bus's own process has exited

    launchers = [threading.Thread(target=launch, args=(bus, i)) for i, bus in enumerate(buses)]
    for t in launchers:
        t.start()
    for t in launchers:
        t.join()

    failed = [vehicle for vehicle, code in done if code]

    print("\n" + "=" * 70)
    for vehicle, _, _, lines in started:
        for line in lines:
            if line.strip().startswith(("events", "backend", "rejected")):
                print(f"{vehicle:15s} {line.strip()}")
    if args.backend:
        try:
            k = backend_get(args.backend, "/api/v1/kpis")
            by_type = ", ".join(f"{r['name']} {r['count']}" for r in k["issuesByType"])
            print(f"\nbackend now holds {k['totalEvents']} events, {k['openIssues']} open issues "
                  f"({by_type}), {k['activeAlerts']} active alerts, {k['activeVehicles']} active vehicles")
            iri = k["roadQuality"]
            by_class = ", ".join(f"{r['name']} {r['count']}" for r in iri["byClass"])
            print(f"road quality: {iri['samples']} IRI windows, mean {iri['avgIri']} m/km, {by_class}")
        except Exception as exc:
            print(f"(could not read KPIs: {exc})")
    if failed:
        print(f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
