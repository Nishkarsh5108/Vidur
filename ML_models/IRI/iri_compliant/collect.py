import os
import math
import csv
import time
import threading
import queue
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import AdvancedIMU, Camera, Lidar

# ==========================================

# CONFIGURATION
# ==========================================
SELECTED_MAP = 'automation_test_track'
SELECTED_CAR = 'vivace'      

MAPS = {
    'east_coast_usa': (-871.022, 579.281, 25.152), #hilly area with parts of dirt road
    'automation_test_track': (930.476, -474.506, 122.294), #all over package
    'west_coast_usa': (-715.628, 542.540, 120) #sanfrasisco, higways
}

CARS = {
    'hopper': 'vehicles/hopper/xt6_A.pc',
    'sunburst2': 'vehicles/sunburst2/base_EU_M.pc',
    'vivace': 'vehicles/vivace/tograc_110_m.pc'
}

HZ_IMU = 100
HZ_IRI = 100
HZ_CAM = 20

# Threading Architecture
bng_lock = threading.Lock() 
data_lock = threading.Lock() 

shared_data = {
    'ax': 0.0, 'ay': 0.0, 'az': 0.0,
    'wx': 0.0, 'wy': 0.0, 'wz': 0.0,
    'speed': 0.0,
    'pos_x': 0.0, 'pos_y': 0.0, 'pos_z': 0.0,
    'lidar_z_left': 0.0, 'lidar_z_right': 0.0,
    'latest_image': None,  # The Drop-Box for perfect frame syncing
    'run_threads': True
}

def setup_directories(car_name, map_name):
    script_dir = os.getcwd()
    base_dir = os.path.abspath(os.path.join(script_dir, '../../data/IRI_bs', car_name, map_name))
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    
    existing_trips = [d for d in os.listdir(base_dir) if d.startswith('trip_')]
    trip_numbers = [int(d.split('_')[1]) for d in existing_trips if d.split('_')[1].isdigit()]
    next_trip_num = max(trip_numbers) + 1 if trip_numbers else 1
    
    trip_dir = os.path.join(base_dir, f'trip_{next_trip_num}')
    cam_dir = os.path.join(trip_dir, 'dashcam')
    os.makedirs(cam_dir)
    return trip_dir, cam_dir

def extract_lidar_z(lidar_poll_data):
    """ Safely extracts the absolute Z-elevation from the BeamNG Shared Memory Point Cloud """
    if not lidar_poll_data: return 0.0
    pc = lidar_poll_data.get('pointCloud')
    if pc is None or len(pc) == 0: return 0.0
    
    try:
        # Check if it is a numpy array (Shared Memory default)
        if hasattr(pc, 'ndim'):
            if pc.ndim == 2 and pc.shape[1] >= 3:
                return float(np.mean(pc[:, 2]))
            elif pc.ndim == 1 and len(pc) >= 3:
                return float(np.mean(pc[2::3]))
        # Fallback to standard lists
        elif isinstance(pc, list):
            if isinstance(pc[0], (list, tuple)):
                return float(pc[0][2])
            else:
                return float(pc[2])
    except Exception:
        pass
    return 0.0

# ==========================================
# MULTITHREADED WORKER CLASSES
# ==========================================
class ImuStateWorker(threading.Thread):
    def __init__(self, imu, vehicle):
        super().__init__(daemon=True)
        self.imu = imu
        self.vehicle = vehicle

    def run(self):
        while shared_data['run_threads']:
            with bng_lock:
                imu_data = self.imu.poll()
                self.vehicle.sensors.poll()
                state = self.vehicle.state

            vel = state.get('vel', [0.0, 0.0, 0.0]) if isinstance(state, dict) else [0.0, 0.0, 0.0]
            pos = state.get('pos', [0.0, 0.0, 0.0]) if isinstance(state, dict) else [0.0, 0.0, 0.0]
            speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

            readings = imu_data if isinstance(imu_data, list) else [imu_data]
            for raw_reading in readings:
                if not raw_reading: continue
                r = raw_reading.get('dashboard_imu', raw_reading) if isinstance(raw_reading, dict) else raw_reading
                if isinstance(r, dict):
                    acc = r.get('accSmooth', r.get('accRaw', r.get('acc', [0.0, 0.0, 0.0])))
                    gyro = r.get('angVelSmooth', r.get('angVelRaw', r.get('angVel', [0.0, 0.0, 0.0])))
                    
                    with data_lock:
                        shared_data['speed'] = speed
                        shared_data['pos_x'], shared_data['pos_y'], shared_data['pos_z'] = pos[0], pos[1], pos[2]
                        
                        # Re-route the IMU axes to match standard smartphone orientation!
                        if len(acc) >= 3: 
                            # Raw BeamNG: acc[0]=Forward, acc[1]=Up, acc[2]=Lateral
                            shared_data['ax'] = float(acc[2])  # Lateral (Left/Right)
                            shared_data['ay'] = float(acc[0])  # Longitudinal (Forward/Backward)
                            shared_data['az'] = float(acc[1])  # Vertical (Up/Down)
                            
                        if len(gyro) >= 3: 
                            # Raw BeamNG: gyro[0]=Roll, gyro[1]=Yaw, gyro[2]=Pitch
                            shared_data['wx'] = float(gyro[2]) # Pitch (Rotation around X / Lateral)
                            shared_data['wy'] = float(gyro[0]) # Roll (Rotation around Y / Longitudinal)
                            shared_data['wz'] = float(gyro[1]) # Yaw (Rotation around Z / Vertical)
            time.sleep(1/HZ_IMU)

class LidarWorker(threading.Thread):
    def __init__(self, lidar_l, lidar_r):
        super().__init__(daemon=True)
        self.lidar_l = lidar_l
        self.lidar_r = lidar_r

    def run(self):
        first_success = False
        while shared_data['run_threads']:
            with bng_lock:
                dl = self.lidar_l.poll()
                dr = self.lidar_r.poll()
            
            z_left = extract_lidar_z(dl)
            z_right = extract_lidar_z(dr)
            
            if not first_success and (z_left != 0.0 or z_right != 0.0):
                print(f"\n[DEBUG] LIDAR ACTIVE! Sensors hitting road perfectly -> Left Z: {z_left:.2f}, Right Z: {z_right:.2f}")
                first_success = True

            with data_lock:
                if z_left != 0.0: shared_data['lidar_z_left'] = z_left
                if z_right != 0.0: shared_data['lidar_z_right'] = z_right
            
            time.sleep(1/HZ_IRI)

class CameraWorker(threading.Thread):
    def __init__(self, camera):
        super().__init__(daemon=True)
        self.camera = camera

    def run(self):
        while shared_data['run_threads']:
            with bng_lock:
                cam_data = self.camera.poll()
            
            if cam_data and isinstance(cam_data, dict) and 'colour' in cam_data:
                with data_lock:
                    shared_data['latest_image'] = cam_data['colour']
            
            time.sleep(1/HZ_CAM)

# ==========================================
# ASYNC IMAGE SAVING THREAD
# ==========================================
image_queue = queue.Queue()

def image_writer():
    while True:
        task = image_queue.get()
        if task is None: break
        filepath, image = task
        try:
            image.convert('RGB').save(filepath)
        except Exception: pass
        image_queue.task_done()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"Initializing Session...")
    trip_dir, cam_dir = setup_directories(SELECTED_CAR, SELECTED_MAP)
    
    bng = BeamNGpy(
        host='localhost', port=64256,
        home=r"D:\softwares\BeamNG\BeamNG.tech.v0.38.3.0",
        user=r"C:\Users\nishk\AppData\Local\BeamNG\BeamNG.tech"
    )
    bng.open()
    
    scenario = Scenario(SELECTED_MAP, 'iri_data_collection')
    vehicle = Vehicle('ego_vehicle', model=SELECTED_CAR, partConfig=CARS[SELECTED_CAR])
    scenario.add_vehicle(vehicle, pos=MAPS[SELECTED_MAP], rot_quat=(0, 0, 0, 1))
    
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()

    print("Spawning vehicle and configuring Shared Memory sensors for GPU acceleration...")
    
    imu = AdvancedIMU('dashboard_imu', bng, vehicle, 
                      pos=(0, -0.5, 1.0), dir=(0, -1, 0), up=(0, 0, 1), 
                      physics_update_time=1/HZ_IMU,
                    #   requested_update_time=1/HZ_IMU, # <--- PUT THIS HERE
                      is_using_gravity=True, is_send_immediately=True)
    
    dashcam = Camera('dashcam', bng, vehicle, 
                     pos=(0.0, -1.8, 1.2), dir=(0, -1, 0), up=(0, 0, 1), 
                     resolution=(640, 360), field_of_view_y=70, 
                     is_using_shared_memory=True, requested_update_time=1/HZ_CAM)
    
    # FIX: Removed horizontal_resolution. Kept Z=0.5 to clear the undercarriage
    lidar_left = Lidar('lidar_left', bng, vehicle, 
                       pos=(.95, 0, 0.5), dir=(0, 0, -1), up=(0, 1, 0), 
                       vertical_resolution=2, 
                       vertical_angle=1.0, horizontal_angle=1.0,
                       max_distance=5.0, is_360_mode=False,
                       is_using_shared_memory=True, requested_update_time=1/HZ_IRI)
                       
    lidar_right = Lidar('lidar_right', bng, vehicle, 
                        pos=(-.95, 0, 0.5), dir=(0, 0, -1), up=(0, 1, 0), 
                        vertical_resolution=2, 
                        vertical_angle=1.0, horizontal_angle=1.0,
                        max_distance=5.0, is_360_mode=False,
                        is_using_shared_memory=True, requested_update_time=1/HZ_IRI)

    print("Waiting 3 seconds for physics and VRAM mappings to settle...")
    time.sleep(3)

    writer_thread = threading.Thread(target=image_writer, daemon=True)
    writer_thread.start()

    imu_file = open(os.path.join(trip_dir, 'imu_speed_data.csv'), 'w', newline='')
    imu_writer = csv.writer(imu_file)
    imu_writer.writerow(['sample_number', 'sim_time', 'speed_ms', 'ax', 'ay', 'az', 'wx', 'wy', 'wz'])

    iri_file = open(os.path.join(trip_dir, 'iri_sensor_data.csv'), 'w', newline='')
    iri_writer = csv.writer(iri_file)
    iri_writer.writerow(['sample_number', 'sim_time', 'pos_x', 'pos_y', 'pos_z', 'road_elevation_left', 'road_elevation_right'])

    print("\n========================================================")
    print("ALL THREADS ACTIVE! Drive around.")
    print("When finished, click the ⏹️ 'Interrupt Kernel' button at the top of Jupyter.")
    print("========================================================")

    w_imu = ImuStateWorker(imu, vehicle)
    w_lidar = LidarWorker(lidar_left, lidar_right)
    w_cam = CameraWorker(dashcam)
    
    w_imu.start()
    w_lidar.start()
    w_cam.start()

    dt_100hz = 1.0 / HZ_IMU
    cam_interval = int(HZ_IMU / HZ_CAM)
    sample_number = 0
    next_wake = time.perf_counter() + dt_100hz

    try:
        while True:
            current_sim_time = round(sample_number * dt_100hz, 4)
            
            with data_lock:
                # 1. Write Data
                imu_writer.writerow([
                    sample_number, current_sim_time, round(shared_data['speed'], 4),
                    round(shared_data['ax'], 4), round(shared_data['ay'], 4), round(shared_data['az'], 4), 
                    round(shared_data['wx'], 4), round(shared_data['wy'], 4), round(shared_data['wz'], 4)
                ])

                iri_writer.writerow([
                    sample_number, current_sim_time, 
                    round(shared_data['pos_x'], 4), round(shared_data['pos_y'], 4), round(shared_data['pos_z'], 4), 
                    round(shared_data['lidar_z_left'], 4), round(shared_data['lidar_z_right'], 4)
                ])

                # 2. Grab Image for this exact Sample Number
                if sample_number % cam_interval == 0:
                    img = shared_data['latest_image']
                    if img:
                        image_queue.put((os.path.join(cam_dir, f'frame_{sample_number}.jpg'), img))
                        shared_data['latest_image'] = None # Clear drop-box to prevent duplicates

            sample_number += 1

            # Strict Metronome Pacing
            sleep_time = next_wake - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_wake += dt_100hz

    except KeyboardInterrupt:
        print("\n[!] Stop signal received. Shutting down threads safely...")
    finally:
        shared_data['run_threads'] = False
        w_imu.join(timeout=2.0)
        w_lidar.join(timeout=2.0)
        w_cam.join(timeout=2.0)
        
        imu_file.close()
        iri_file.close()
        
        print("Saving remaining Dashcam frames from memory...")
        image_queue.put(None)
        writer_thread.join(timeout=10.0)
        
        bng.close()
        print(f"Data saved perfectly. Have a great day!")

if __name__ == '__main__':
    main()