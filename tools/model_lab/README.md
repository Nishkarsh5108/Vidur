# Model lab

A local page for trying the vision models on any image or video: upload a file, and V1, V2 and V3 run on every frame and draw their boxes. You get back the annotated image or video, a per-class table, and a JSON file with every detection.

```bash
pip install -r edge/requirements.txt -r tools/model_lab/requirements.txt
python tools/model_lab/app.py          # opens http://127.0.0.1:7860
```

- **GPU.** The models run as PyTorch checkpoints in FP16 on CUDA when a GPU is present. With the CUDA build of PyTorch installed, the page header names the GPU in use. On an RTX 4050 laptop, all three models plus tracking process about 25 frames/s end to end, including decoding, drawing and encoding.
- **Formats.**
  - Images: anything OpenCV or Pillow reads, including HEIC and AVIF.
  - Videos: anything ffmpeg decodes (MP4, MOV, MKV, WEBM, AVI, WMV, FLV, MPEG, 3GP, TS, animated GIF, ...).
  - Output videos are H.264 MP4, so they play in the browser.
- **Settings:**
  - pick the models;
  - set each model's confidence threshold;
  - run V3 on V2's sign boxes (as the bus does) or on the whole frame;
  - turn ByteTrack IDs on or off;
  - analyse every Nth frame (boxes carry over to the skipped frames), and stop after N seconds.
- **Outputs** go to `out/model_lab/<time>_<file>/` (not versioned).
- **Privacy.** The page listens on 127.0.0.1 only, and nothing leaves the machine.
- **Timings are the laptop's GPU, not a Raspberry Pi 5.** For Pi numbers see [docs/edge-deployment.md](../../docs/edge-deployment.md).
- **IMU models.** S1 and S2 take IMU data, not images, so they are not part of this page. To run them on a recording, use `python -m edge --imu ride.csv`.

| Colour | Meaning |
|---|---|
| Red | V1 pothole |
| White | V1 zebra crossing |
| Green | V2 vehicles |
| Orange | V2 person or rider |
| Cyan | V2 sign, light or pole |
| Magenta | V3 damaged sign |
| Teal | V3 good sign |
