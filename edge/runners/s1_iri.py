"""S1: International Roughness Index for each 100 m of travel, from a 6-axis IMU plus GPS speed.

Pre-processing reproduces `process_trip_data` / `extract_context_features` in
ML_models/IRI/iri_compliant/model.ipynb, minus the training-time sample-rate jitter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from edge.runners.tflite import find_input, load_interpreter

MODEL_NAME = "S1-iri"
MODEL_VERSION = "dual-cnn-v2-tflite-drq"
GRID_POINTS = 400            # 0.25 m resolution over a 100 m window
REF_SPEED_MPS = 22.22        # 80 km/h: the ASTM "Golden Car" reference speed used for the labels
MIN_NORM_SPEED_MPS = 5.0

IRI_CLASSES = ((4.0, "good"), (8.0, "fair"), (14.0, "poor"), (float("inf"), "very_poor"))


def context_features(raw: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """The 13 context features, in training order, computed on the normalised spatial grid."""
    ax, ay, az, _wx, wy, wz = (raw[:, i] for i in range(6))
    n = len(az)
    speed_mean = float(np.mean(speed))
    rms_az = np.sqrt(np.mean(az ** 2))
    psd = np.abs(np.fft.rfft(az)) ** 2
    if speed_mean > 0:
        freqs = np.fft.rfftfreq(n, d=0.25) * speed_mean     # spatial frequency -> temporal, at the mean speed
        total = np.sum(psd) + 1e-6
        ratio_1_4 = np.sum(psd[(freqs >= 1.0) & (freqs <= 4.0)]) / total      # body bounce
        ratio_4_15 = np.sum(psd[(freqs > 4.0) & (freqs <= 15.0)]) / total     # wheel hop
    else:
        ratio_1_4 = ratio_4_15 = 0.0
    return np.array([
        speed_mean,
        np.std(speed),
        rms_az,
        np.sqrt(np.mean(ay ** 2)),
        np.var(az),
        np.max(np.abs(az)) / (rms_az + 1e-6),
        len(np.where(np.diff(np.sign(az)))[0]) / n,
        np.max(az) - np.min(az),
        np.sqrt(np.mean(wz ** 2)),
        np.sqrt(np.mean(wy ** 2)),
        np.mean(np.abs(ax)),
        ratio_1_4,
        ratio_4_15,
    ])


def window_tensors(dist: np.ndarray, imu6: np.ndarray, speed: np.ndarray,
                   start_m: float, window_m: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Resamples one window of travel onto the 400-point grid and builds both model inputs.

    dist: metres travelled at each sample; imu6: [n, 6] as ax, ay, az, wx, wy, wz; speed: m/s.
    Returns raw_imu [400, 6] and context_stats [13], both float32.
    """
    grid = np.linspace(start_m, start_m + window_m, GRID_POINTS)
    # np.interp holds the end values outside the samples, the same as training's edge-padded interp1d.
    raw = np.column_stack([np.interp(grid, dist, imu6[:, i]) for i in range(6)])
    v = np.interp(grid, dist, speed)
    raw[:, 2] *= (REF_SPEED_MPS / np.maximum(v, MIN_NORM_SPEED_MPS)) ** 2   # always applied, as in training
    return raw.astype(np.float32), context_features(raw, v).astype(np.float32)


def iri_class(iri: float) -> str:
    return next(label for bound, label in IRI_CLASSES if iri < bound)


class IriModel:
    def __init__(self, tflite_path: str | Path, lut_path: str | Path):
        self.interpreter = load_interpreter(tflite_path)
        self._raw = find_input(self.interpreter, "raw_imu", rank=3)
        self._ctx = find_input(self.interpreter, "context_stats", rank=2)
        self._out = self.interpreter.get_output_details()[0]["index"]
        with open(lut_path, encoding="utf-8") as fh:
            lut = json.load(fh)
        self._lut_x = np.asarray(lut["x_raw"], dtype=np.float64)
        self._lut_y = np.asarray(lut["y_calibrated"], dtype=np.float64)

    def predict(self, raw_imu: np.ndarray, context: np.ndarray) -> tuple[float, float]:
        """Returns (calibrated IRI, raw IRI) in m/km."""
        self.interpreter.set_tensor(self._raw, raw_imu[None])
        self.interpreter.set_tensor(self._ctx, context[None])
        self.interpreter.invoke()
        iri_raw = float(np.expm1(self.interpreter.get_tensor(self._out)[0][0]))
        # The isotonic table clamps to its calibrated range, 1.21-15.57 m/km.
        return float(np.interp(iri_raw, self._lut_x, self._lut_y)), iri_raw
