"""
Training pipeline for RoadSense YOLOv8 model.
"""
from ultralytics import YOLO

def train():
    model = YOLO("yolov8m.pt")
    model.train(
        data="configs/data.yaml",
        epochs=100,
        imgsz=800,
        batch=8,
        cos_lr=True,
        box=8.5,
        cls=1.5,
        mosaic=1.0,
        mixup=0.15,
        project="runs",
        name="roadsense_yolov8"
    )

if __name__ == "__main__":
    train()
