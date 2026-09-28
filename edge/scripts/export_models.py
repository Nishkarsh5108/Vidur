"""Export V1, V2 and V3 for edge runtimes, using the (rectangular) input sizes in edge/config.yaml.

    python edge/scripts/export_models.py                    # NCNN FP16 for the Raspberry Pi 5 CPU (default)
    python edge/scripts/export_models.py --format onnx      # ONNX FP32, for inspection or other runtimes
    python edge/scripts/export_models.py --format hailo --data calib.yaml   # AI HAT+ (Hailo-8), x86-64 Linux only

Outputs go to exports/<model>_<size>_<format>_model (the paths edge/config.pi5.yaml expects).

Precision, per target (docs/edge-deployment.md, "Quantisation"):
  NCNN   FP16. Accuracy is unchanged and files are half the size. Ultralytics' NCNN export has no INT8.
  Hailo  INT8 only. Calibrate on 500-1,000 frames of your own bus footage, not on the training sets.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.config import load_config, repo_path  # noqa: E402
from edge.runners.yolo import imgsz_hw  # noqa: E402

NAMES = {"v1": "v1_roadsense", "v2": "v2_idd", "v3": "v3_signs"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "openvino", "hailo"])
    p.add_argument("--models", nargs="+", default=list(NAMES), choices=list(NAMES))
    p.add_argument("--data", help="calibration dataset YAML (required for hailo)")
    p.add_argument("--hailo-chip", default="hailo8", choices=["hailo8", "hailo8l", "hailo10h"])
    p.add_argument("--config", nargs="*", default=[])
    args = p.parse_args()
    if args.format == "hailo" and not args.data:
        p.error("--format hailo needs --data: INT8 calibration frames from your own footage")

    from ultralytics import YOLO

    cfg = load_config(*args.config)
    out_root = repo_path("exports")
    out_root.mkdir(exist_ok=True)
    for key in args.models:
        spec = cfg.models[key]
        h, w = imgsz_hw(spec.imgsz)
        kwargs = {"format": args.format, "imgsz": [h, w]}
        if args.format == "ncnn":
            kwargs["quantize"] = 16
        elif args.format == "hailo":
            kwargs.update(quantize=8, data=args.data, name=args.hailo_chip)
        exported = Path(YOLO(str(repo_path(spec.weights))).export(**kwargs))

        size = f"{h}" if h == w else f"{h}x{w}"
        if exported.is_dir():
            dest = out_root / f"{NAMES[key]}_{size}_{args.format}_model"
        else:
            dest = out_root / f"{NAMES[key]}_{size}{exported.suffix}"
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(exported), dest)
        print(f"{key}: {dest.relative_to(repo_path('.'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
