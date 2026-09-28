"""TFLite interpreter loading and input binding, shared by the S1 and S2 runners."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any


def load_interpreter(path: str | Path, num_threads: int | None = 1) -> Any:
    """LiteRT on the Pi and Linux; full TensorFlow as the fallback (Windows has no LiteRT wheel)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            warnings.filterwarnings("ignore", message=r".*tf\.lite\.Interpreter is deprecated.*")
    interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def find_input(interpreter: Any, name_hint: str, rank: int) -> int:
    """Tensor index of an input, bound by name and falling back to rank.

    Never bind by position: S1's inputs are in the opposite order to its usage doc.
    """
    details = interpreter.get_input_details()
    for d in details:
        if name_hint in d["name"]:
            return d["index"]
    matches = [d["index"] for d in details if len(d["shape"]) == rank]
    if len(matches) != 1:
        raise ValueError(f"cannot bind input {name_hint!r}: inputs are {[(d['name'], list(d['shape'])) for d in details]}")
    return matches[0]
