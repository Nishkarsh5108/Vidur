"""S2: grades a 1.28 s window of vertical acceleration into four shock levels.

Its classes are quartiles of window jerk energy from simulator driving, not human road labels
(docs/ml-models-spec.md §7), so the agent uses it as a relative shock trigger, not a pothole detector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edge.runners.tflite import find_input, load_interpreter

MODEL_NAME = "S2-road-shock"
MODEL_VERSION = "roadfinalnet-swa-tflite-fp32"
WINDOW = 128                 # samples at 100 Hz
LABELS = ("Excellent", "Patches", "Med Pothole", "Big Pothole")

# StandardScaler from context_scaler.pkl. The usage doc's constants are placeholders and break the rms feature.
CTX_MEAN = np.array([0.0, 12.464910954632682, 0.9999906116945316, 4.2572717833764555])
CTX_SCALE = np.array([1.0, 6.405128570025835, 6.002955638863807e-06, 0.8918762876362021])


def preprocess(az: np.ndarray, mean_speed_mps: float) -> tuple[np.ndarray, np.ndarray]:
    """128 vertical-acceleration samples (m/s2, 100 Hz) -> (vibration [1, 128, 2], context [1, 4]) float32."""
    az = np.asarray(az, dtype=np.float64)          # float64 matters: rms's signal is in the 6th-7th decimal
    sig = np.diff(az, prepend=az[0])               # gravity cancels here
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    rms = np.sqrt(np.mean(sig ** 2))               # no epsilon, as in training (vid.py adds one and skews it)
    crest = np.max(np.abs(sig)) / (rms + 1e-6)
    context = (np.array([0.0, mean_speed_mps, rms, crest]) - CTX_MEAN) / CTX_SCALE
    # Channels-last [1, 128, 2]: the buffer interleaves |sig| and its gradient per time step.
    # Writing the two channels one after the other (planar), as the usage doc does, gives wrong logits.
    vibration = np.stack([np.abs(sig), np.gradient(sig)], axis=-1)[None]
    return vibration.astype(np.float32), context[None].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def gate(probs: np.ndarray, min_confidence: float = 0.82) -> int:
    """argmax, except that an unsure pothole-grade call (class 2 or 3 below min_confidence) counts as 0."""
    cls = int(np.argmax(probs))
    return 0 if cls >= 2 and probs[cls] < min_confidence else cls


class ShockModel:
    def __init__(self, tflite_path: str | Path):
        self.interpreter = load_interpreter(tflite_path)
        self._vib = find_input(self.interpreter, "vibration", rank=3)
        self._ctx = find_input(self.interpreter, "context", rank=2)
        self._out = self.interpreter.get_output_details()[0]["index"]

    def predict(self, az: np.ndarray, mean_speed_mps: float) -> np.ndarray:
        """Class probabilities [4] for one window."""
        vibration, context = preprocess(az, mean_speed_mps)
        self.interpreter.set_tensor(self._vib, vibration)
        self.interpreter.set_tensor(self._ctx, context)
        self.interpreter.invoke()
        return softmax(self.interpreter.get_tensor(self._out)[0].astype(np.float64))
