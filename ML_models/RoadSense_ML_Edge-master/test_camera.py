import cv2
import requests
from road_detector import RoadSenseDetector

print("Loading AI Model...")
# Make sure your best.pt file is in the exact same folder as this script!
detector = RoadSenseDetector(model_path="best.pt", conf_thresh=0.40)

cap = cv2.VideoCapture(0)
print("Webcam started! Show a picture of a pothole to the camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Ask the AI to look at the frame
    events = detector.predict(frame)

    # If it sees a pothole/zebra crossing, send it to the backend
    for event in events:
        print(f"🚨 Detected: {event['eventType']}! Sending to backend...")
        try:
            requests.post("http://127.0.0.1:5000/api/events", json=event, timeout=2)
        except:
            print("⚠️ Backend offline. Start mock_backend.py in another terminal!")

    # Show the video feed on screen
    cv2.imshow("RoadSense AI Test", frame)
    
    # Press 'q' on your keyboard to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()