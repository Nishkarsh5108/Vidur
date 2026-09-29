"""Launches tools/sim/run_fleet.py as a subprocess and tracks it: one simulation at a time.

This is a control plane for a local demo, not a job queue: state lives in this process's memory
(restarting the backend forgets a running simulation, though the OS process keeps running until it
finishes or is stopped by PID). That's the right amount of engineering for a laptop demo.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings

log = logging.getLogger("fleet_sense.simulate")


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.running = False
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self.return_code: int | None = None
        self.params: dict[str, Any] = {}
        self.log_lines: deque[str] = deque(maxlen=500)
        self.error: str | None = None


_state = _State()


def _pump_output(proc: subprocess.Popen) -> None:
    """Reads the child's combined stdout/stderr until it exits, then records the outcome."""
    try:
        for line in proc.stdout:
            _state.log_lines.append(line.rstrip())
    finally:
        proc.wait()
        with _state.lock:
            _state.running = False
            _state.return_code = proc.returncode
            _state.ended_at = datetime.now(timezone.utc).isoformat()
            if proc.returncode:
                _state.error = f"exited with code {proc.returncode} — see the log for details"


def start(backend_url: str, *, rate: float = 1.0, clock: str = "now", only: list[str] | None = None,
         max_concurrent: int = 2, stagger: float = 6.0, hud: bool = False) -> dict:
    with _state.lock:
        if _state.running:
            raise RuntimeError("a simulation is already running; stop it first or wait for it to finish")
        script = Path(settings.SIMULATE_SCRIPT)
        if not script.is_file():
            raise FileNotFoundError(f"{script} does not exist — run tools/sim/build_scenario.py once first")

        cmd = [settings.SIMULATE_PYTHON, str(script), "--backend", backend_url, "--rate", str(rate),
              "--clock", clock, "--max-concurrent", str(max_concurrent), "--stagger", str(stagger)]
        if only:
            cmd += ["--only", *only]
        if hud:
            cmd.append("--hud")

        log.info("starting simulation: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=str(script.parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        _state.proc = proc
        _state.running = True
        _state.started_at = datetime.now(timezone.utc).isoformat()
        _state.ended_at = None
        _state.return_code = None
        _state.error = None
        _state.params = {"rate": rate, "clock": clock, "only": only, "maxConcurrent": max_concurrent,
                         "stagger": stagger, "hud": hud, "backend": backend_url}
        _state.log_lines.clear()
        threading.Thread(target=_pump_output, args=(proc,), daemon=True, name="simulate-reader").start()
    return status()


def status() -> dict:
    with _state.lock:
        return {
            "running": _state.running, "pid": _state.proc.pid if _state.proc else None,
            "startedAt": _state.started_at, "endedAt": _state.ended_at, "returnCode": _state.return_code,
            "error": _state.error, "params": _state.params, "log": list(_state.log_lines)[-200:],
        }


def stop() -> dict:
    with _state.lock:
        proc = _state.proc
    if proc is None or proc.poll() is not None:
        return status()
    log.info("stopping simulation (pid %d)", proc.pid)
    if os.name == "nt":
        # A plain terminate() only kills run_fleet.py itself; /T also kills the bus processes it spawned.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    return status()
