"""Vidur model lab: upload an image or a video, run V1, V2 and V3 on it, and see the boxes.

    python tools/model_lab/app.py            # opens http://127.0.0.1:7860 in the browser

Runs on the laptop GPU (CUDA, FP16) when available. Everything stays on this machine: the page
listens on 127.0.0.1 only, and outputs are written to out/model_lab/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import MODELS, ModelLab, Options, is_video, summarise  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "out" / "model_lab"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Loading V1, V2 and V3 ...")
LAB = ModelLab()
print(f"Models ready on {LAB.device} ({LAB.gpu_name})")

LEGEND = (
    "**V1** <span style='color:#e33'>■</span> pothole · <span style='color:#999'>□</span> zebra crossing &nbsp;&nbsp; "
    "**V2** <span style='color:#3c3'>■</span> vehicles · <span style='color:#f80'>■</span> person / rider · "
    "<span style='color:#0bf'>■</span> sign, light, pole &nbsp;&nbsp; "
    "**V3** <span style='color:#f0f'>■</span> damaged sign · <span style='color:#1d8'>■</span> good sign"
)
HEADERS = ["model", "class", "frames with it", "detections", "unique tracks", "mean conf", "max conf"]


def run(file, models, conf_v1, conf_v2, conf_v3, v3_mode, track, stride, max_seconds, progress=gr.Progress()):
    if not file:
        raise gr.Error("Upload an image or a video first.")
    if not models:
        raise gr.Error("Pick at least one model.")
    path = Path(file)
    out_dir = OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}_{path.stem[:40]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = Options(models=tuple(models), conf={"V1": conf_v1, "V2": conf_v2, "V3": conf_v3},
                   v3_mode="crops" if v3_mode.startswith("On V2") else "full",
                   track=bool(track), stride=int(stride), max_seconds=float(max_seconds or 0))
    try:
        if is_video(path):
            progress(0, desc="decoding")
            result = LAB.run_video(path, opts, out_dir, progress=lambda f, m: progress(f, desc=m))
        else:
            result = LAB.run_image(path, opts, out_dir)
    except Exception as exc:          # show decode or inference problems in the page, not just the console
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

    video = result["kind"] == "video"
    return (
        gr.Image(value=None if video else result["rgb"], visible=not video),
        gr.Video(value=str(result["file"]) if video else None, visible=video),
        report(path, result, opts),
        summarise(result["dets"]),
        [str(result["file"]), str(result["json"])],
    )


def report(path: Path, r: dict, opts: Options) -> str:
    w, h = r["size"]
    lines = [f"**{path.name}** · {w}×{h} · {LAB.device} ({LAB.gpu_name}, FP16)"]
    if r["kind"] == "video":
        speed = r["frames"] / r["elapsed"] if r["elapsed"] else 0
        lines.append(f"{r['frames']} frames ({r['frames'] / r['fps']:.1f} s at {r['fps']:.0f} FPS), "
                     f"{r['analysed']} analysed (every {opts.stride}) · processed in {r['elapsed']:.1f} s "
                     f"= **{speed:.1f} frames/s** including decode, drawing and encoding")
    else:
        lines.append(f"Processed in {r['elapsed'] * 1000:.0f} ms")
    timing = [f"{m} {np.mean(v):.1f} ms" for m, v in sorted(r["stats"].items()) if v]
    if timing:
        lines.append("Mean inference per image (per crop for V3 on signs): " + " · ".join(timing))
    lines.append("<sub>Laptop GPU timings, not Raspberry Pi 5 measurements.</sub>")
    return "\n\n".join(lines)


with gr.Blocks(title="Vidur model lab") as demo:
    gr.Markdown(f"# Vidur model lab\nUpload an image or a video to run the vision models on every frame. "
                f"Running on **{LAB.gpu_name}** ({LAB.device}).")
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            file = gr.File(label="Image or video (any format ffmpeg or Pillow can read)", type="filepath")
            models = gr.CheckboxGroup([(label, key) for key, label in MODELS.items()], value=list(MODELS),
                                      label="Models")
            with gr.Accordion("Settings", open=False):
                conf_v1 = gr.Slider(0.05, 0.95, 0.40, step=0.05, label="V1 confidence")
                conf_v2 = gr.Slider(0.05, 0.95, 0.25, step=0.05, label="V2 confidence")
                conf_v3 = gr.Slider(0.05, 0.95, 0.35, step=0.05, label="V3 confidence")
                v3_mode = gr.Radio(["On V2's sign boxes (as on the bus)", "On the whole frame"],
                                   value="On V2's sign boxes (as on the bus)", label="V3 runs")
                track = gr.Checkbox(value=True, label="Track V2 objects across video frames (ByteTrack IDs)")
                stride = gr.Slider(1, 10, 1, step=1, label="Video: analyse every Nth frame (boxes carry over)")
                max_seconds = gr.Number(value=0, label="Video: stop after N seconds (0 = whole video)")
            run_btn = gr.Button("Run models", variant="primary")
            gr.Markdown(LEGEND)
        with gr.Column(scale=2):
            out_image = gr.Image(label="Result", type="numpy", visible=True)
            out_video = gr.Video(label="Result", visible=False, autoplay=True)
            info = gr.Markdown()
            table = gr.Dataframe(headers=HEADERS, label="Detections", interactive=False, wrap=True)
            downloads = gr.File(label="Downloads: annotated file and detections JSON", file_count="multiple")

    run_btn.click(run, [file, models, conf_v1, conf_v2, conf_v3, v3_mode, track, stride, max_seconds],
                  [out_image, out_video, info, table, downloads])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", inbrowser=True, allowed_paths=[str(OUT_ROOT)], max_file_size="4gb",
        theme=gr.themes.Soft())
