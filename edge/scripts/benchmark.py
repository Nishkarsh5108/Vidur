"""Per-model latency on this machine: p50 / p95 over N runs on a real 1080p-sized frame.

    python edge/scripts/benchmark.py                                   # the .pt models from edge/config.yaml
    python edge/scripts/benchmark.py --config edge/config.pi5.yaml     # on the Pi, after export_models.py
    python edge/scripts/benchmark.py --image frame.jpg --runs 200 --device cpu

This measures one model at a time. On the Pi, also run the full agent for 30 minutes on a recorded
ride (python -m edge ... --rate 1) and watch `vcgencmd measure_temp` and `vcgencmd get_throttled`:
the budget that matters is the whole pipeline, hot, at the same time.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.config import load_config, repo_path  # noqa: E402
from edge.governor import read_cpu_temp_c  # noqa: E402
from edge.runners.yolo import YoloModel  # noqa: E402


def timed_runs(fn, runs: int, warmup: int = 5) -> np.ndarray:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return np.array(samples)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="*", default=[])
    p.add_argument("--image", help="a real camera frame (default: Ultralytics' bus.jpg resized to 1920x1080)")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--device", help="override device.compute: auto | cpu | cuda:0")
    args = p.parse_args()

    import cv2
    from ultralytics.utils import ASSETS

    cfg = load_config(*args.config)
    device = args.device or cfg.device.compute
    frame = cv2.imread(args.image) if args.image else cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (1920, 1080))
    crop = cv2.resize(frame[100:400, 1400:1700], (160, 160))    # a sign-sized crop for V3

    print(f"{platform.node()} | {platform.system()} {platform.machine()} | Python {platform.python_version()}"
          f" | CPU temp {read_cpu_temp_c() or '-'} C")
    print(f"frame {frame.shape[1]}x{frame.shape[0]}, {args.runs} runs per model\n")
    print(f"{'model':<4} {'weights':<58} {'device':<7} {'imgsz':<9} {'p50 ms':>8} {'p95 ms':>8}")
    for key, image in (("v2", frame), ("v1", frame), ("v3", crop)):
        spec = cfg.models[key]
        model = YoloModel(repo_path(spec.weights), spec.imgsz, device, spec.get("conf"))
        ms = timed_runs(lambda: model.predict(image), args.runs)
        size = "x".join(map(str, model.imgsz))
        print(f"{key.upper():<4} {str(spec.weights)[-58:]:<58} {model.device:<7} {size:<9}"
              f" {np.percentile(ms, 50):8.1f} {np.percentile(ms, 95):8.1f}")
    print(f"\nCPU temp after: {read_cpu_temp_c() or '-'} C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
