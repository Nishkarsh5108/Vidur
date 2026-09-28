"""Loading Ultralytics models from .pt checkpoints or exported folders (NCNN, ONNX, ...), with warm-up."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Box:
    xyxy: tuple[float, float, float, float]   # pixels of the image the model was given
    conf: float
    cls: int
    name: str
    track_id: int | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def resolve_device(preference: str, weights: str | Path) -> str:
    """'auto' picks CUDA for .pt checkpoints when a GPU is present. Exported formats run on the CPU here."""
    if Path(weights).suffix != ".pt":
        return "cpu"
    if preference != "auto":
        return preference
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def imgsz_hw(imgsz: int | list[int]) -> tuple[int, int]:
    return (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))


class YoloModel:
    """An Ultralytics model plus the fixed arguments it is always called with.

    Works with .pt checkpoints and with exported folders (NCNN on the Pi 5 CPU, Hailo HEF on an AI HAT+):
    Ultralytics picks the runtime from the path, so switching hardware is a config change.
    """

    def __init__(self, weights: str | Path, imgsz: int | list[int], device: str = "auto",
                 conf: float | None = None, threads: int = 0):
        from ultralytics import YOLO

        self.weights = Path(weights)
        self.imgsz = imgsz_hw(imgsz)
        self.device = resolve_device(device, self.weights)
        self.model = YOLO(str(self.weights), task="detect")
        self.names: dict[int, str] = dict(self.model.names)
        self.kwargs: dict[str, Any] = {"imgsz": list(self.imgsz), "device": self.device, "verbose": False}
        if self.device.startswith("cuda"):
            self.kwargs["quantize"] = 16      # FP16 compute on the GPU
        if conf is not None:
            self.kwargs["conf"] = conf
        self.predict(np.zeros((*self.imgsz, 3), dtype=np.uint8))   # warm-up: first call pays CUDA/NCNN setup
        if threads:
            self.set_threads(threads)

    def set_threads(self, threads: int) -> None:
        """NCNN only: threads per inference, so two lanes pinned to two cores each do not oversubscribe.

        NCNN packs convolution weights for the thread count it had at load time, so the net is
        reloaded with the new count rather than changed in place.
        """
        backend = getattr(getattr(self.model.predictor, "model", None), "backend", None)
        net = getattr(backend, "net", None)
        if net is None:
            return
        param = next(self.weights.glob("*.param")) if self.weights.is_dir() else self.weights
        net.clear()
        net.opt.num_threads = threads
        net.load_param(str(param))
        net.load_model(str(param.with_suffix(".bin")))

    def predict(self, image: np.ndarray) -> list[Box]:
        return boxes_of(self.model.predict(image, **self.kwargs)[0], self.names)

    def track(self, image: np.ndarray, tracker: str) -> list[Box]:
        """Detection + tracking. persist=True keeps one tracker per model instance, so one instance per camera."""
        result = self.model.track(image, persist=True, tracker=tracker, **self.kwargs)[0]
        return [b for b in boxes_of(result, self.names) if b.track_id is not None]


def boxes_of(result: Any, names: dict[int, str]) -> list[Box]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
    return [
        Box(tuple(float(v) for v in xyxy[i]), float(conf[i]), int(cls[i]), names.get(int(cls[i]), str(cls[i])),
            None if ids[i] is None else int(ids[i]))
        for i in range(len(cls))
    ]
