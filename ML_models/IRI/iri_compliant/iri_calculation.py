"""
IRI Computation Pipeline — With Mesh Tessellation Correction
for Simulator Data

This module adds a curvature-adaptive profile correction step that
removes systematic mesh tessellation artifacts from driving simulator
road surfaces while preserving real roughness features (potholes,
cracks, surface deterioration).

The correction is applied BEFORE the standard IRI computation,
treating it as a data-quality correction step — analogous to how
real-world profiler data is cleaned before IRI calculation.

References:
  - WB TP45/46 (Sayers, Gillespie, Queiroz/Paterson, 1986)
  - Sayers (1995), TRR 1501
  - ASTM E1926-08
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import StateSpace, lsim
from scipy.interpolate import interp1d

# ==========================================
# 1. CONSTANTS
# ==========================================
DX = 0.25  # meters — standard IRI sample interval
SEGMENT_LENGTH_M = 100.0
LEAD_IN_M = 20.0
V_SIM_KMH = 80.0
V_SIM_MS = V_SIM_KMH / 3.6

# --- Mesh Tessellation Correction Parameters ---
# These control the curvature-adaptive smoothing that removes
# simulator mesh artifacts from the road profile.

# Window for estimating "design grade" (road geometry)
# Must be much longer than mesh polygon spacing but shorter
# than major road features.
DESIGN_GRADE_WINDOW_M = 10.0  # meters

# Anti-tessellation smoothing window
# Should be >= the mesh polygon spacing to smooth out faceting.
# From PSD analysis: mesh artifacts have wavelength ~1-3m.
ANTI_TESS_WINDOW_M = 5.0  # meters

# Curvature threshold for activating smoothing
# Below this, the road is "flat enough" that mesh artifacts are negligible.
# Units: 1/meter (second derivative of elevation profile)
CURVATURE_THRESHOLD = 0.00015  # ≈ R_v > 5000m → effectively flat

# Grade threshold for flagging
GRADE_FLAG_THRESHOLD = 0.03

# ==========================================
# 2. GOLDEN CAR
# ==========================================
c_s, k_s, k_t, mu = 6.0, 63.3, 653.0, 0.15

A_mat = np.array([
    [0,       1,     0,                0      ],
    [-k_s,   -c_s,   k_s,             c_s     ],
    [0,       0,     0,                1      ],
    [k_s/mu,  c_s/mu, -(k_s + k_t)/mu, -c_s/mu]
])
B_mat = np.array([[0], [0], [0], [k_t / mu]])
C_out = np.array([[1, 0, -1, 0]])
D_out = np.array([[0]])

golden_car = StateSpace(A_mat, B_mat, C_out, D_out)


# ==========================================
# 3. CURVATURE-ADAPTIVE PROFILE CORRECTION
# ==========================================
def correct_mesh_tessellation(elevation, dx):
    """
    Remove simulator mesh tessellation artifacts from the elevation profile
    using curvature-adaptive smoothing.

    On road sections with significant vertical curvature (flyovers, ramps,
    grade transitions), the simulator's polygon mesh creates a systematic
    high-frequency oscillation in the profile. This function estimates the
    local curvature and applies targeted smoothing only where needed.

    On flat or constant-grade sections, the profile is left untouched,
    preserving real roughness features like potholes.

    Args:
        elevation: Uniformly-sampled elevation profile (after linear interp)
        dx: Sample interval in meters

    Returns:
        Corrected elevation profile (same length)
    """
    n = len(elevation)
    if n < 20:
        return elevation.copy()

    # Step 1: Estimate "design grade" profile using long moving average
    # This captures the intended road geometry without mesh artifacts
    design_k = max(3, int(np.round(DESIGN_GRADE_WINDOW_M / dx)))
    if design_k % 2 == 0:
        design_k += 1  # Ensure odd for symmetric window

    pad_d = design_k // 2
    padded_d = np.pad(elevation, (pad_d, pad_d), mode='edge')
    design_grade = np.convolve(padded_d, np.ones(design_k) / design_k,
                               mode='valid')[:n]

    # Step 2: Estimate local curvature (second derivative of design grade)
    # Use a smoothed second derivative to avoid noise
    curvature = np.zeros(n)
    if n > 4:
        # Second derivative: (Y[i+1] - 2Y[i] + Y[i-1]) / dx^2
        curvature[1:-1] = np.abs(
            design_grade[2:] - 2 * design_grade[1:-1] + design_grade[:-2]
        ) / (dx ** 2)
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

        # Smooth the curvature estimate to avoid sharp transitions
        curv_k = max(3, int(np.round(DESIGN_GRADE_WINDOW_M / dx)))
        if curv_k % 2 == 0:
            curv_k += 1
        pad_c = curv_k // 2
        padded_c = np.pad(curvature, (pad_c, pad_c), mode='edge')
        curvature = np.convolve(padded_c, np.ones(curv_k) / curv_k,
                                mode='valid')[:n]

    # Step 3: Compute the anti-tessellation smoothed profile
    tess_k = max(3, int(np.round(ANTI_TESS_WINDOW_M / dx)))
    if tess_k % 2 == 0:
        tess_k += 1

    pad_t = tess_k // 2
    padded_t = np.pad(elevation, (pad_t, pad_t), mode='edge')
    smoothed = np.convolve(padded_t, np.ones(tess_k) / tess_k,
                           mode='valid')[:n]

    # Step 4: Blend between original and smoothed based on curvature
    # alpha = 0: use original profile (flat sections, potholes preserved)
    # alpha = 1: use smoothed profile (curved sections, mesh artifact removed)
    alpha = np.clip(
        (curvature - CURVATURE_THRESHOLD) /
        (CURVATURE_THRESHOLD * 5),  # Ramp over 5× the threshold
        0.0, 1.0
    )

    corrected = (1.0 - alpha) * elevation + alpha * smoothed

    return corrected


# ==========================================
# 4. STANDARD 250 mm MOVING AVERAGE
# ==========================================
def apply_250mm_moving_average(elevation, dx):
    """Apply standard 250 mm tire enveloping filter (ASTM E1926)."""
    ma_base = 0.25
    if dx >= ma_base:
        return elevation.copy()
    k = max(1, int(np.round(ma_base / dx)))
    if k <= 1:
        return elevation.copy()
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    padded = np.pad(elevation, (pad_left, pad_right), mode='edge')
    kernel = np.ones(k) / k
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(elevation)]


# ==========================================
# 5. IRI CALCULATION
# ==========================================
def compute_iri(elevation_smoothed, dx, lead_in_samples=0):
    """
    Compute IRI from a preprocessed elevation profile.
    Returns (IRI in m/km, mean absolute grade).
    """
    n = len(elevation_smoothed)
    if n < 3:
        return 0.0, 0.0

    slope = np.diff(elevation_smoothed) / dx
    dt = dx / V_SIM_MS
    n_slope = len(slope)
    t = np.arange(n_slope) * dt

    _, y_out, _ = lsim(golden_car, U=slope, T=t)

    if lead_in_samples > 0 and lead_in_samples < n_slope:
        y_eval = y_out[lead_in_samples:]
        slope_eval = slope[lead_in_samples:]
    else:
        y_eval = y_out
        slope_eval = slope

    if len(y_eval) == 0:
        return 0.0, 0.0

    L_eval_m = len(y_eval) * dx
    iri = (np.sum(np.abs(y_eval)) * dx / L_eval_m) * 1000.0
    mean_abs_grade = np.mean(np.abs(slope_eval))

    return iri, mean_abs_grade


# ==========================================
# 6. TRIP PROCESSING PIPELINE
# ==========================================
def process_trip(trip_dir):
    """
    Full pipeline:
      1. Load and merge data
      2. Compute 2D cumulative distance
      3. Resample to uniform grid (linear interp)
      4. Correct mesh tessellation artifacts (curvature-adaptive)
      5. Apply 250 mm moving average
      6. Compute IRI per 100 m segment with 20 m lead-in
      7. Save results
    """
    imu_path = os.path.join(trip_dir, 'imu_speed_data.csv')
    iri_path = os.path.join(trip_dir, 'iri_sensor_data.csv')

    if not os.path.exists(imu_path) or not os.path.exists(iri_path):
        return False

    df = pd.merge(
        pd.read_csv(imu_path),
        pd.read_csv(iri_path),
        on=['sample_number', 'sim_time']
    )
    valid = (df['pos_x'] != 0.0) & (df['road_elevation_left'] != 0.0)
    df = df[valid].copy().reset_index(drop=True)
    if len(df) < 50:
        return False

    # Cumulative horizontal distance (2D)
    dx_pos = np.diff(df['pos_x'].values, prepend=df['pos_x'].values[0])
    dy_pos = np.diff(df['pos_y'].values, prepend=df['pos_y'].values[0])
    df['cumulative_distance'] = np.cumsum(np.sqrt(dx_pos**2 + dy_pos**2))

    max_dist = df['cumulative_distance'].max()
    if max_dist < SEGMENT_LENGTH_M:
        print(f"    [!] Trip too short ({max_dist:.1f} m)")
        return False

    spatial_grid = np.arange(0, max_dist, DX)
    dist_vals = df['cumulative_distance'].values
    mask = np.concatenate([[True], np.diff(dist_vals) > 0])
    dist_unique = dist_vals[mask]

    z_iri = {}
    for side in ['left', 'right']:
        elev_unique = df[f'road_elevation_{side}'].values[mask]
        f_interp = interp1d(
            dist_unique, elev_unique,
            kind='linear', bounds_error=False,
            fill_value=(elev_unique[0], elev_unique[-1])
        )
        z_raw = f_interp(spatial_grid)

        # DATA CORRECTION: Remove mesh tessellation artifacts
        z_corrected = correct_mesh_tessellation(z_raw, DX)

        # STANDARD: Apply 250 mm moving average (tire enveloping)
        z_iri[side] = apply_250mm_moving_average(z_corrected, DX)

    # Compute IRI per segment
    seg_samples = int(SEGMENT_LENGTH_M / DX)
    lead_samples = int(LEAD_IN_M / DX)

    records = []
    for seg_start in range(0, len(spatial_grid) - seg_samples + 1, seg_samples):
        start_with_lead = max(0, seg_start - lead_samples)
        seg_end = seg_start + seg_samples
        if seg_end > len(spatial_grid):
            break

        actual_lead = seg_start - start_with_lead

        iri_sides = {}
        grades = {}
        for side in ['left', 'right']:
            segment = z_iri[side][start_with_lead:seg_end]
            iri_val, grade = compute_iri(
                segment, DX,
                lead_in_samples=actual_lead if actual_lead > 0 else 0
            )
            iri_sides[side] = iri_val
            grades[side] = grade

        mri = (iri_sides['left'] + iri_sides['right']) / 2.0
        mean_grade = (grades['left'] + grades['right']) / 2.0
        grade_flag = mean_grade > GRADE_FLAG_THRESHOLD

        records.append({
            'start_dist_m': spatial_grid[seg_start],
            'end_dist_m': spatial_grid[min(seg_end - 1, len(spatial_grid) - 1)],
            'IRI_left': iri_sides['left'],
            'IRI_right': iri_sides['right'],
            'IRI': mri,
            'mean_abs_grade': mean_grade,
            'grade_flag': grade_flag,
        })

    seg_df = pd.DataFrame(records)
    if seg_df.empty:
        return False

    # Map back to original samples
    bin_edges = np.concatenate([
        seg_df['start_dist_m'].values,
        [seg_df['end_dist_m'].iloc[-1] + 0.001]
    ])
    idx = np.clip(
        np.searchsorted(bin_edges, df['cumulative_distance'].values,
                         side='right') - 1,
        0, len(seg_df) - 1
    )
    df['IRI'] = seg_df['IRI'].values[idx]
    df['grade_flag'] = seg_df['grade_flag'].values[idx]

    out_cols = ['sample_number', 'ax', 'ay', 'az', 'wx', 'wy', 'wz',
                'speed_ms', 'IRI', 'grade_flag']
    out_path = os.path.join(trip_dir, 'readings_2.csv')
    df[out_cols].to_csv(out_path, index=False)

    n_flagged = seg_df['grade_flag'].sum()
    print(f"    [+] Saved readings_2.csv | "
          f"Segments: {len(seg_df)} | "
          f"Max IRI: {seg_df['IRI'].max():.1f} m/km | "
          f"Mean IRI: {seg_df['IRI'].mean():.1f} m/km | "
          f"Grade-flagged: {n_flagged}/{len(seg_df)}")
    return True


def main():
    base_dir = os.path.abspath('../../data/IRI_new_experiments')
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    for car in sorted(os.listdir(base_dir)):
        car_p = os.path.join(base_dir, car)
        if not os.path.isdir(car_p):
            continue
        for route in sorted(os.listdir(car_p)):
            route_p = os.path.join(car_p, route)
            if not os.path.isdir(route_p):
                continue
            for trip in sorted(os.listdir(route_p)):
                trip_dir = os.path.join(route_p, trip)
                if os.path.isdir(trip_dir) and trip.startswith('trip_'):
                    print(f"Processing: {trip_dir}")
                    process_trip(trip_dir)


if __name__ == '__main__':
    main()