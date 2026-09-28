"""Inference engine for the model lab: runs V1, V2 and V3 on an image or a video and draws the boxes.

Models run as PyTorch checkpoints on the GPU (FP16) when CUDA is available. Decoding falls back to
the bundled ffmpeg for formats OpenCV cannot open, and videos are written as H.264 so browsers play them.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.config import load_config, repo_path  # noqa: E402
from edge.imaging import crop_with_margin  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".dib", ".tif", ".tiff", ".webp", ".heic", ".heif",
              ".avif", ".jp2", ".pbm", ".pgm", ".ppm", ".pnm", ".ico", ".sr", ".ras", ".exr", ".hdr"}
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".flv", ".f4v", ".wmv", ".asf", ".mpg", ".mpeg",
              ".m2v", ".3gp", ".3g2", ".ts", ".mts", ".m2ts", ".ogv", ".vob", ".mxf", ".gif", ".h264", ".hevc"}

MODELS = {
    "V1": "V1 RoadSense: potholes, zebra crossings (YOLOv8m)",
    "V2": "V2 IDD traffic: 13 Indian road-user classes (YOLOv8n)",
    "V3": "V3 sign condition: damaged / good (YOLOv8s)",
}
RENAME = {"missing_zebra": "zebra_crossing", "Damaged": "damaged", "Good": "good"}
DROPPED = {"ego vehicle"}

_V2_COLORS = {"person": (0, 140, 255), "rider": (0, 140, 255), "traffic sign": (255, 200, 0),
              "traffic light": (255, 200, 0), "pole": (200, 200, 0)}
_COLORS = {("V1", "pothole"): (0, 0, 255), ("V1", "zebra_crossing"): (255, 255, 255),
           ("V3", "damaged"): (255, 0, 255), ("V3", "good"): (0, 230, 120)}


@dataclass
class Detection:
    model: str
    name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


@dataclass
class Options:
    models: tuple[str, ...] = ("V1", "V2", "V3")
    conf: dict | None = None             # model -> threshold
    v3_mode: str = "crops"               # "crops": on V2's sign boxes, as on the bus; "full": whole frame at 640
    track: bool = True                   # ByteTrack IDs for V2 on videos
    stride: int = 1                      # run the models on every Nth video frame
    max_seconds: float = 0.0             # 0 = whole video
    max_width: int = 1920                # output video width cap


class ModelLab:
    def __init__(self, device: str = "auto"):
        import torch
        from ultralytics import YOLO

        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.gpu_name = torch.cuda.get_device_name(0) if self.device.startswith("cuda") else "CPU"
        cfg = load_config()
        self.models = {key: YOLO(str(repo_path(cfg.models[key.lower()].weights)), task="detect") for key in MODELS}
        self.tracker = None
        warm = np.zeros((640, 640, 3), np.uint8)
        for model in self.models.values():
            model.predict(warm, **self._kwargs(0.5))

    def _kwargs(self, conf: float, imgsz: int | None = None) -> dict:
        kw = {"conf": conf, "device": self.device, "verbose": False}
        if self.device.startswith("cuda"):
            kw["quantize"] = 16      # FP16 on the GPU
        if imgsz:
            kw["imgsz"] = imgsz
        return kw

    # ---------------------------------------------------------------- per-batch inference

    def infer(self, frames: list[np.ndarray], opts: Options, stats: dict, first: bool = False) -> list[list[Detection]]:
        """Detections for each frame of a batch. V2 tracks frame by frame when tracking is on."""
        conf = {"V1": 0.4, "V2": 0.25, "V3": 0.35, **(opts.conf or {})}
        out: list[list[Detection]] = [[] for _ in frames]
        need_v2 = "V2" in opts.models or ("V3" in opts.models and opts.v3_mode == "crops")

        v2: list[list[Detection]] = [[] for _ in frames]
        if need_v2:
            # Batched on the GPU; ByteTrack then runs per frame on the CPU. This is much faster than
            # model.track(), which sets up the whole predictor again for every single frame.
            if opts.track and (first or self.tracker is None):
                self.tracker = new_bytetrack()
            for i, r in enumerate(self.models["V2"].predict(frames, **self._kwargs(conf["V2"]))):
                v2[i] = self._tracked(r, stats) if opts.track else self._dets("V2", r, stats)
            if "V2" in opts.models:
                for i in range(len(frames)):
                    out[i] += v2[i]

        if "V3" in opts.models:
            if opts.v3_mode == "full":
                for i, r in enumerate(self.models["V3"].predict(frames, **self._kwargs(conf["V3"], 640))):
                    out[i] += self._dets("V3", r, stats)
            else:
                for i, found in enumerate(self._v3_on_signs(frames, v2, conf["V3"], stats)):
                    out[i] += found

        if "V1" in opts.models:
            for i, r in enumerate(self.models["V1"].predict(frames, **self._kwargs(conf["V1"], 800))):
                out[i] += self._dets("V1", r, stats)
        return out

    def _v3_on_signs(self, frames: list[np.ndarray], v2: list[list[Detection]], conf: float,
                     stats: dict) -> list[list[Detection]]:
        """V3 on crops of V2's sign boxes, as on the bus: one batched call for every sign in the batch."""
        signs = [(i, d) for i, dets in enumerate(v2) for d in dets
                 if d.name == "traffic sign" and min(d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]) >= 12]
        found: list[list[Detection]] = [[] for _ in frames]
        if not signs:
            return found
        crops = [crop_with_margin(frames[i], d.xyxy, 0.3)[0] for i, d in signs]
        for (i, sign), r in zip(signs, self.models["V3"].predict(crops, **self._kwargs(conf, 320))):
            best = max(self._dets("V3", r, stats), key=lambda d: d.conf, default=None)
            if best is not None:
                found[i].append(Detection("V3", best.name, best.conf, sign.xyxy, sign.track_id))
        return found

    def _tracked(self, result, stats: dict) -> list[Detection]:
        """V2 detections passed through ByteTrack: only tracked boxes, each with its track ID."""
        stats["V2"].append(result.speed.get("inference", 0.0))
        tracks = self.tracker.update(result.boxes.cpu().numpy(), result.orig_img)
        dets = []
        for x1, y1, x2, y2, tid, score, cls, _idx in tracks:
            name = result.names[int(cls)]
            if name not in DROPPED:
                dets.append(Detection("V2", name, float(score), (float(x1), float(y1), float(x2), float(y2)), int(tid)))
        return dets

    @staticmethod
    def _dets(model: str, result, stats: dict) -> list[Detection]:
        stats[model].append(result.speed.get("inference", 0.0))
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy, confs, cls = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        dets = []
        for b, c, k, tid in zip(xyxy, confs, cls, ids):
            name = result.names[int(k)]
            if name in DROPPED:
                continue
            dets.append(Detection(model, RENAME.get(name, name), float(c), tuple(float(v) for v in b),
                                  None if tid is None else int(tid)))
        return dets

    # ---------------------------------------------------------------- images and videos

    def run_image(self, path: Path, opts: Options, out_dir: Path) -> dict:
        image = read_image(path)
        stats: dict = defaultdict(list)
        start = time.perf_counter()
        dets = self.infer([image], opts, stats, first=True)[0]
        elapsed = time.perf_counter() - start
        annotated = draw(image, dets)
        out_path = out_dir / f"{path.stem}_annotated.png"
        cv2.imencode(".png", annotated)[1].tofile(str(out_path))
        json_path = write_json(out_dir / f"{path.stem}_detections.json", [{"frame": 0, **asdict(d)} for d in dets])
        return {"kind": "image", "rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "file": out_path,
                "json": json_path, "dets": [[d] for d in dets], "stats": stats, "elapsed": elapsed, "frames": 1,
                "size": (image.shape[1], image.shape[0])}

    def run_video(self, path: Path, opts: Options, out_dir: Path,
                  progress: Callable[[float, str], None] = lambda f, m: None, batch: int = 8) -> dict:
        reader = VideoInput(path, out_dir)
        fps, total = reader.fps, reader.frame_count
        if opts.max_seconds > 0:
            total = min(total or 10**9, int(opts.max_seconds * fps))
        out_path = out_dir / f"{path.stem}_annotated.mp4"
        stats: dict = defaultdict(list)
        records, per_frame = [], []
        last: list[Detection] = []
        done, analysed = 0, 0
        start = time.perf_counter()
        pending: list[tuple[int, np.ndarray]] = []
        # Decoding runs ahead and drawing + encoding run behind in their own threads, so the GPU is not
        # left waiting on CPU work between batches.
        encoder = EncoderThread(out_path, fps, opts.max_width)

        def flush():
            nonlocal last, analysed
            to_run = [(i, f) for i, f in pending if i % opts.stride == 0]
            results = dict(zip([i for i, _ in to_run],
                               self.infer([f for _, f in to_run], opts, stats, first=analysed == 0))) if to_run else {}
            for i, frame in pending:
                if i in results:
                    last = results[i]
                    analysed += 1
                    per_frame.append(last)
                    records.extend({"frame": i, "t": round(i / fps, 3), **asdict(d)} for d in last)
                encoder.put(frame, last)
            pending.clear()

        try:
            for index, frame in prefetch(reader, total):
                pending.append((index, frame))
                done = index + 1
                if len(pending) >= batch * opts.stride:
                    flush()
                    progress(done / total if total else 0.0, f"frame {done}" + (f" / {total}" if total else ""))
            if pending:
                flush()
        finally:
            reader.close()
            encoder.close()
        elapsed = time.perf_counter() - start
        json_path = write_json(out_dir / f"{path.stem}_detections.json", records)
        return {"kind": "video", "file": out_path, "json": json_path, "dets": per_frame, "stats": stats,
                "elapsed": elapsed, "frames": done, "analysed": analysed, "fps": fps,
                "size": (reader.width, reader.height)}


def new_bytetrack():
    """A fresh ByteTrack with Ultralytics' default settings (bytetrack.yaml)."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    return BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))


# -------------------------------------------------------------------- decoding and encoding

def read_image(path: Path) -> np.ndarray:
    """Any still format: OpenCV first, then Pillow (with HEIC/HEIF/AVIF plugins)."""
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def is_video(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return True
    if ext in IMAGE_EXTS:
        return False
    try:                                   # unknown extension: an image if any decoder accepts it
        read_image(path)
        return False
    except Exception:
        return True


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoInput:
    """Frames of any video ffmpeg can decode. OpenCV reads directly when it can; otherwise the file
    is first transcoded to H.264 with the bundled ffmpeg."""

    def __init__(self, path: Path, work_dir: Path):
        self.cap = cv2.VideoCapture(str(path))
        ok, first = self.cap.read() if self.cap.isOpened() else (False, None)
        if not ok:
            self.cap.release()
            converted = work_dir / f"{path.stem}_decoded.mp4"
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-an", "-c:v", "libx264",
                            "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", str(converted)], check=True)
            self.cap = cv2.VideoCapture(str(converted))
            ok, first = self.cap.read()
            if not ok:
                raise ValueError(f"could not decode {path.name}")
        self._first = first
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 240 else 25.0
        self.frame_count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.height, self.width = first.shape[:2]

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        yield 0, self._first
        index = 1
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield index, frame
            index += 1

    def close(self) -> None:
        self.cap.release()


def prefetch(reader: VideoInput, total: int, depth: int = 64) -> Iterator[tuple[int, np.ndarray]]:
    """Decodes frames in a background thread, up to `depth` ahead of inference."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def decode():
        for index, frame in reader:
            if stop.is_set() or (total and index >= total):
                break
            q.put((index, frame))
        q.put(None)

    thread = threading.Thread(target=decode, daemon=True)
    thread.start()
    try:
        while (item := q.get()) is not None:
            yield item
    finally:
        stop.set()
        while thread.is_alive():      # keep draining so a decoder blocked on a full queue can finish
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            thread.join(0.02)


class EncoderThread:
    """Draws boxes, resizes and H.264-encodes frames in a background thread, in order."""

    def __init__(self, path: Path, fps: float, max_width: int, depth: int = 64):
        self.path, self.fps, self.max_width = path, fps, max_width
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, frame: np.ndarray, dets: list[Detection]) -> None:
        if self.error is not None:
            raise self.error
        self.q.put((frame, dets))

    def _run(self) -> None:
        writer = None
        try:
            while (item := self.q.get()) is not None:
                annotated = fit_width(draw(*item), self.max_width)
                if writer is None:
                    writer = H264Writer(self.path, annotated.shape[1], annotated.shape[0], self.fps)
                writer.write(annotated)
        except Exception as exc:
            self.error = exc
            while self.q.get() is not None:   # drain so put() never blocks forever
                pass
        finally:
            if writer is not None:
                writer.close()

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error


class H264Writer:
    """Browser-playable MP4 (H.264, yuv420p) through the bundled ffmpeg."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        import imageio_ffmpeg
        self.size = (width - width % 2, height - height % 2)
        self._gen = imageio_ffmpeg.write_frames(
            str(path), self.size, fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            macro_block_size=1, ffmpeg_log_level="error",
            output_params=["-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"])
        self._gen.send(None)

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        self._gen.send(np.ascontiguousarray(frame[:h, :w]).tobytes())

    def close(self) -> None:
        self._gen.close()


# -------------------------------------------------------------------- drawing and reporting

def color_of(d: Detection) -> tuple[int, int, int]:
    if d.model == "V2":
        return _V2_COLORS.get(d.name, (60, 200, 60))
    return _COLORS.get((d.model, d.name), (0, 255, 255))


def draw(image: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 360))
    scale = max(0.45, min(h, w) / 1100)
    order = {"V2": 0, "V3": 1, "V1": 2}                      # V1 on top: it is the rarest and most important
    for d in sorted(dets, key=lambda d: order[d.model]):
        color = color_of(d)
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + (1 if d.model != "V2" else 0))
        label = f"{d.model} {d.name}{f' #{d.track_id}' if d.track_id is not None and d.model == 'V2' else ''} {d.conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick - 1))
        ty = y1 - 4 if y1 - th - 8 > 0 else y2 + th + 6
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + base), color, -1)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(out, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, max(1, thick - 1))
    return out


def fit_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max_width and w > max_width:
        return cv2.resize(image, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return image


def write_json(path: Path, records: list[dict]) -> Path:
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    return path


def summarise(per_frame: list[list[Detection]]) -> list[list]:
    """Rows of (model, class, frames with it, detections, unique tracks, mean conf, max conf)."""
    groups: dict[tuple[str, str], dict] = {}
    for dets in per_frame:
        seen = set()
        for d in dets:
            g = groups.setdefault((d.model, d.name), {"frames": 0, "n": 0, "tracks": set(), "confs": []})
            g["n"] += 1
            g["confs"].append(d.conf)
            if d.track_id is not None:
                g["tracks"].add(d.track_id)
            if (d.model, d.name) not in seen:
                g["frames"] += 1
                seen.add((d.model, d.name))
    rows = []
    for (model, name), g in sorted(groups.items()):
        rows.append([model, name, g["frames"], g["n"], len(g["tracks"]) or "",
                     round(float(np.mean(g["confs"])), 3), round(float(np.max(g["confs"])), 3)])
    return rows
