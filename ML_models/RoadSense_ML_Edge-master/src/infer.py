from ultralytics import YOLO
from src.modules.severity import calculate_pothole_severity
from src.utils.formatter import format_event_payload

class RoadSenseDetector:
    def __init__(self, model_path="models/roadsense_yolov8/weights/best.pt", conf_thresh=0.40):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.class_names = {0: "pothole", 1: "missing_zebra"}

    def predict(self, frame, current_lat=28.6142, current_lon=77.2110, vehicle_id="BUS_DEMO_01"):
        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                event_name = self.class_names.get(cls_id)
                if not event_name:
                    continue

                metadata = {}
                if event_name == "pothole":
                    coords = box.xyxy[0].tolist()
                    metadata["severity"] = calculate_pothole_severity(coords)

                event = format_event_payload(
                    event_type=event_name,
                    confidence=conf,
                    lat=current_lat,
                    lon=current_lon,
                    vehicle_id=vehicle_id,
                    metadata=metadata
                )
                detections.append(event)

        return detections
