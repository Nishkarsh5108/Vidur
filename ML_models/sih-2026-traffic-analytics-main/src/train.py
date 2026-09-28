"""
YOLOv8 Training Script for IDD Detection (13 Classes).
Optimized for resource-constrained hardware (e.g., RTX 2050 4GB VRAM) and scalable for Cloud GPUs.
"""

import os
import sys
import argparse
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("[!] Ultralytics not installed. Please run: pip install ultralytics")
    sys.exit(1)


def inspect_hardware():
    """Detects GPU capabilities and recommends optimal training hyperparameters."""
    has_cuda = torch.cuda.is_available()
    print("=" * 60)
    print("                HARDWARE DIAGNOSTICS")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {has_cuda}")

    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Device:      {gpu_name}")
        print(f"GPU VRAM:        {vram_gb:.2f} GB")

        # Recommendation logic
        if vram_gb <= 4.5:
            rec_model = "yolov8n.pt"
            rec_batch = 8
            rec_workers = 2
            note = "4GB VRAM detected (RTX 2050 class). Using batch=8 with FP16 AMP."
        elif vram_gb <= 8.5:
            rec_model = "yolov8s.pt"
            rec_batch = 16
            rec_workers = 4
            note = "8GB VRAM detected. Using batch=16."
        else:
            rec_model = "yolov8m.pt"
            rec_batch = 32
            rec_workers = 8
            note = "High VRAM detected (>8GB). Using batch=32."
        device = 0
    else:
        print("[!] No CUDA GPU detected by PyTorch (CPU mode).")
        print("    If you have an NVIDIA GPU, install PyTorch with CUDA via:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        rec_model = "yolov8n.pt"
        rec_batch = 4
        rec_workers = 0
        device = "cpu"
        note = "Training on CPU will be slow. Consider Google Colab for dataset training."

    print(f"Recommendation:  {note}")
    print("=" * 60 + "\n")
    return device, rec_model, rec_batch, rec_workers


def train_idd_yolo(
    data_yaml: str = "configs/data.yaml",
    model_name: str = "yolov8n.pt",
    imgsz: int = 640,
    batch_size: int = None,
    epochs: int = 30,
    device: str = None,
    workers: int = None,
    project_dir: str = "models",
    experiment_name: str = "idd_yolov8",
    resume: bool = False
):
    """Executes YOLOv8 fine-tuning on the subsampled IDD dataset."""
    dev, rec_model, rec_batch, rec_workers = inspect_hardware()

    if device is None:
        device = dev
    if batch_size is None:
        batch_size = rec_batch
    if workers is None:
        workers = rec_workers

    # Ensure config exists
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset config not found at: {yaml_path}")

    # Load baseline model
    print(f"[*] Initializing YOLO model: {model_name}...")
    model = YOLO(model_name)

    print(f"[*] Starting training on IDD 13-class dataset...")
    print(f"    - Data Config:   {yaml_path}")
    print(f"    - Image Size:    {imgsz}")
    print(f"    - Batch Size:    {batch_size}")
    print(f"    - Epochs:        {epochs}")
    print(f"    - Device:        {device}")
    print(f"    - Mixed Precision (AMP): True")
    print(f"    - Output:        {project_dir}/{experiment_name}")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers,
        project=project_dir,
        name=experiment_name,
        resume=resume,
        amp=True,                # Crucial for 4GB VRAM
        patience=10,             # Early stopping if mAP doesn't improve
        save_period=5,           # Save checkpoint every 5 epochs
        # Indian Traffic Augmentation tuning
        mosaic=1.0,              # Mosaic helps complex multi-vehicle scenes
        mixup=0.1,               # Subtle mixup for occlusions
        degrees=5.0,             # Minor rotation tolerance
        hsv_h=0.015,             # Hue adjustments for sunlight variations
        hsv_s=0.6,               # Saturation
        hsv_v=0.4,               # Value/Exposure
        fliplr=0.5,              # Horizontal flip
        plots=True               # Save training curves & confusion matrix
    )

    print("\n[+] Training complete!")
    print(f"    Best weights saved at: {project_dir}/{experiment_name}/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on IDD Detection Dataset")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model checkpoint (yolov8n.pt / yolov8s.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (auto-tuned if omitted)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('0' or 'cpu')")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader worker processes")
    parser.add_argument("--project", type=str, default="models", help="Output directory for weights")
    parser.add_argument("--name", type=str, default="idd_yolov8", help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    train_idd_yolo(
        data_yaml=args.data,
        model_name=args.model,
        imgsz=args.imgsz,
        batch_size=args.batch,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        project_dir=args.project,
        experiment_name=args.name,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
