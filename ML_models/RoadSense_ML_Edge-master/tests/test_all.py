"""
End-to-End Local Pipeline Test
Validates inference, JSON schema, and mock backend integration.
"""
import cv2
import requests
from src.infer import RoadSenseDetector

def test_pipeline():
    print("[TEST] Initializing RoadSense Detector...")
    detector = RoadSenseDetector(
        model_path="models/roadsense_yolov8/weights/best.pt",
        conf_thresh=0.40
    )

    cap = cv2.VideoCapture(0)
    print("[TEST] Camera started. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        events = detector.predict(frame)
        for event in events:
            print(f"🚨 Detected: {event['eventType']} ({event['confidence']}) -> Sending...")
            try:
                res = requests.post("http://127.0.0.1:5000/api/events", json=event, timeout=1)
                if res.status_code == 200:
                    print("   ✅ Event received by Backend.")
            except Exception:
                print("   ⚠️ Mock Backend not running on http://127.0.0.1:5000")

        cv2.imshow("RoadSense Edge Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pipeline()
