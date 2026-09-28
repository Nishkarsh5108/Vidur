import os
import cv2
import torch
import pandas as pd
import numpy as np
import subprocess
import joblib
import gc
import multiprocessing  # <--- Added this back to fix the NameError
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
import torch.nn as nn

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_W, VIDEO_H = 1280, 720
GRAPH_W = 1280
TOTAL_W = VIDEO_W + GRAPH_W
TOTAL_H = VIDEO_H
PLAYBACK_FPS = 10
MAX_PARALLEL_TRIPS = 1  

# ACCURACY & SYNC SETTINGS
SYNC_LOOK_AHEAD = 25    # Compensates for dashcam-to-wheel distance
SMOOTHING_WINDOW = 15   # Stability: Majority vote over 1.5 seconds
CONF_THRESHOLD = 0.82   # Accuracy: Ignore low-confidence anomalies

# LOCAL SSD PATHS
BASE_DIR = r"C:\IRI"
LABEL_FILE = os.path.join(BASE_DIR, "master_label.txt") 
MODEL_PATH = os.path.join(BASE_DIR, "road_vision_final_85plus.pt")
SCALER_PATH = os.path.join(BASE_DIR, "context_scaler.pkl")

CLASS_NAMES  = ['Excellent', 'Patches', 'Med Pothole', 'Big Pothole']
CLASS_COLORS = [(60, 200, 80), (255, 159, 28), (40, 200, 220), (60, 60, 230)]

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class RoadFinalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout1d(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout1d(0.2))
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(260, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 4))

    def forward(self, x, ctx):
        x = self.conv1(x); x = self.conv2(x)
        avg_p = self.pool(x).squeeze(-1)
        max_p = self.max_pool(x).squeeze(-1)
        return self.fc(torch.cat([avg_p, max_p, ctx], dim=1))

# ==========================================
# UTILITIES
# ==========================================
def parse_master_labels(filepath):
    label_ranges = []
    if not os.path.exists(filepath): return []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line: continue
            try:
                parts = line.split(':'); score = int(parts[0].strip())
                if score == -1: continue
                samples = parts[1].split(',')
                label_ranges.append({'score': score, 'start': int(samples[0].strip()), 'end': int(samples[1].strip())})
            except: continue
    return label_ranges

def add_classification_overlay(cam_frame, pred_idx, truth_idx, conf):
    h, w = cam_frame.shape[:2]
    overlay = cam_frame.copy()
    cv2.rectangle(overlay, (0, h - 100), (w, h), (15, 15, 26), -1)
    cv2.addWeighted(overlay, 0.7, cam_frame, 0.3, 0, cam_frame)
    
    truth_text = f"ACTUAL: {CLASS_NAMES[truth_idx]}" if truth_idx != -1 else "ACTUAL: N/A"
    cv2.putText(cam_frame, truth_text, (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (224, 224, 255), 2)
    cv2.putText(cam_frame, f"PREDICTED: {CLASS_NAMES[pred_idx]} ({conf:.1%})", 
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLASS_COLORS[pred_idx], 2)
    
    match_color = (0, 255, 0) if pred_idx == truth_idx else (0, 0, 255)
    cv2.circle(cam_frame, (w - 60, h - 50), 20, match_color, -1)
    return cam_frame

def predict_road_grade(model, device, scaler, window_df, context_basics):
    with torch.no_grad():
        az = window_df['az'].values
        sig = np.diff(az, prepend=az[0])
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
        rms = np.sqrt(np.mean(sig**2)) + 1e-6
        crest = np.max(np.abs(sig)) / rms
        
        full_context = np.array([context_basics[0], context_basics[1], rms, crest])
        ctx_scaled = scaler.transform(full_context.reshape(1, -1))
        x_input = np.stack([np.abs(sig), np.gradient(sig)], axis=0)
        
        x_t = torch.FloatTensor(x_input).unsqueeze(0).to(device)
        ctx_t = torch.FloatTensor(ctx_scaled).to(device)

        logits = model(x_t, ctx_t)
        return torch.argmax(logits, dim=1).cpu().item(), torch.softmax(logits, dim=1).max().cpu().item()

# ==========================================
# LOCAL RENDERING WORKER
# ==========================================
def create_video_for_trip(trip_dir):
    try:
        torch.cuda.empty_cache()
        gc.collect()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        inf_model = RoadFinalNet().to(device)
        sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        inf_model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items() if k != 'n_averaged'})
        inf_model.eval()
        
        scaler = joblib.load(SCALER_PATH)
        global_labels = parse_master_labels(LABEL_FILE)
        
        df = pd.read_csv(os.path.join(trip_dir, 'readings.csv')).set_index('sample_number')
        
        cam_dir = os.path.join(trip_dir, 'dashcam')
        video_out_path = os.path.join(trip_dir, f'ROAD_ANALYSIS_{os.path.basename(trip_dir)}.mp4')
        frame_files = sorted([f for f in os.listdir(cam_dir) if f.lower().endswith('.jpg')], 
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{TOTAL_W}x{TOTAL_H}', '-pix_fmt', 'bgr24', '-r', f'{PLAYBACK_FPS}',
            '-i', '-', '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '22', '-pix_fmt', 'yuv420p',
            video_out_path
        ]
        pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        prediction_buffer = deque(maxlen=SMOOTHING_WINDOW) 
        rendered = 0

        for filename in frame_files:
            sample_num = int(filename.split('_')[1].split('.')[0])
            cam_frame = cv2.imread(os.path.join(cam_dir, filename))
            if cam_frame is None: continue

            try:
                target_sample = sample_num + SYNC_LOOK_AHEAD
                base_idx = df.index.get_loc(target_sample)
                window_data = df.iloc[max(0, base_idx - 64):min(len(df), base_idx + 64)]
                if len(window_data) < 128: continue
            except: continue

            speed = window_data['speed_ms'].mean()
            p_idx, conf = predict_road_grade(inf_model, device, scaler, window_data, [0, speed])
            
            if p_idx >= 2 and conf < CONF_THRESHOLD: p_idx = 0
            
            prediction_buffer.append(p_idx)
            smoothed_p_idx = max(set(prediction_buffer), key=prediction_buffer.count)

            t_idx = -1
            for r in global_labels:
                if r['start'] <= sample_num <= r['end']:
                    t_idx = r['score']; break

            cam_frame = cv2.resize(cam_frame, (VIDEO_W, VIDEO_H))
            gui_frame = add_classification_overlay(cam_frame, smoothed_p_idx, t_idx, conf)
            full_frame = np.hstack((gui_frame, np.zeros((TOTAL_H, GRAPH_W, 3), dtype=np.uint8)))
            
            pipe.stdin.write(full_frame.tobytes())
            rendered += 1

        pipe.stdin.close(); pipe.wait()
        return True, trip_dir, f"Rendered {rendered} frames"

    except Exception:
        import traceback
        return False, trip_dir, traceback.format_exc()

# ==========================================
# DYNAMIC TRIP COLLECTION
# ==========================================
def collect_trips(base_dir):
    trips = []
    if not os.path.exists(base_dir): return []
    for vehicle in os.listdir(base_dir):
        v_path = os.path.join(base_dir, vehicle)
        if not os.path.isdir(v_path): continue
        for road_map in os.listdir(v_path):
            m_path = os.path.join(v_path, road_map)
            if not os.path.isdir(m_path): continue
            for trip in os.listdir(m_path):
                t_path = os.path.join(m_path, trip)
                if os.path.isdir(t_path) and trip.startswith('trip_'):
                    trips.append(t_path)
    return trips

def main():
    trips = collect_trips(BASE_DIR)
    if not trips: 
        print(f"❌ Error: No trips detected in {BASE_DIR}")
        return
    
    print('=' * 65)
    print(f'🚀 LOCAL SSD RENDERING | RTX 3050 | ACCURACY: MAX')
    print('=' * 65)

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_TRIPS) as pool:
        futures = {pool.submit(create_video_for_trip, t): t for t in trips}
        for fut in as_completed(futures):
            ok, path, info = fut.result()
            print(f"{'✅' if ok else '⚠️'} {os.path.basename(path)}: {info}")

if __name__ == '__main__':
    multiprocessing.freeze_support() # Now multiprocessing is defined
    main()