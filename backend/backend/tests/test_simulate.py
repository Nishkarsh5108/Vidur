"""POST /api/v1/simulate/start|stop, GET /status. subprocess.Popen is mocked throughout — these tests
check the state machine (one simulation at a time, status reflects reality), not tools/sim/run_fleet.py
itself, which would mean actually launching the edge's model stack."""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import app.services.simulate_service as simulate_service


@pytest.fixture(autouse=True)
def _reset_simulate_state():
    """The module keeps one global _State singleton; without this, a simulation "started" by one test
    would still look running to the next one."""
    yield
    with simulate_service._state.lock:
        simulate_service._state.proc = None
        simulate_service._state.running = False
        simulate_service._state.started_at = simulate_service._state.ended_at = None
        simulate_service._state.return_code = simulate_service._state.error = None
        simulate_service._state.params = {}
        simulate_service._state.log_lines.clear()


def _finishing_popen(lines):
    """A fake Popen whose stdout yields `lines` and then ends immediately (a process that finishes fast)."""
    def factory(*args, **kwargs):
        proc = MagicMock()
        proc.pid = 4242
        proc.stdout = iter(lines)
        proc.wait.return_value = 0
        proc.returncode = 0
        return proc
    return factory


def _blocking_popen(gate: threading.Event, pid: int = 99):
    """A fake Popen whose stdout blocks until the test releases `gate` (a process still running)."""
    def stdout_gen():
        gate.wait(5)      # once released (or after 5s as a safety net), fall through and end the "stream"
        return
        yield             # unreachable; makes this a generator function so `return` raises StopIteration

    def factory(*args, **kwargs):
        proc = MagicMock()
        proc.pid = pid
        proc.stdout = stdout_gen()
        proc.wait.return_value = 0
        proc.returncode = 0
        proc.poll.return_value = None
        return proc
    return factory


def _wait_until_idle(client, timeout=2.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/v1/simulate/status").json()
        if not status["running"]:
            return status
        time.sleep(0.05)
    raise AssertionError("simulation never finished")


def test_status_when_idle(client):
    status = client.get("/api/v1/simulate/status").json()
    assert status == {"running": False, "pid": None, "startedAt": None, "endedAt": None,
                      "returnCode": None, "error": None, "params": {}, "log": []}


def test_start_fails_cleanly_when_the_scenario_has_not_been_built(client):
    with patch("app.services.simulate_service.settings.SIMULATE_SCRIPT", "/no/such/file.py"):
        r = client.post("/api/v1/simulate/start", json={})
    assert r.status_code == 422


def test_start_runs_to_completion_and_status_reflects_it(client):
    with patch("app.services.simulate_service.subprocess.Popen", side_effect=_finishing_popen(["[BUS_1] hello\n"])):
        r = client.post("/api/v1/simulate/start", json={"only": ["BUS_1"], "rate": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is True and body["pid"] == 4242 and body["params"]["only"] == ["BUS_1"]

    status = _wait_until_idle(client)
    assert status["returnCode"] == 0 and any("hello" in line for line in status["log"])


def test_a_second_simulation_cannot_start_while_one_is_running(client):
    gate = threading.Event()
    try:
        with patch("app.services.simulate_service.subprocess.Popen", side_effect=_blocking_popen(gate)):
            first = client.post("/api/v1/simulate/start", json={})
            assert first.status_code == 200
            second = client.post("/api/v1/simulate/start", json={})
        assert second.status_code == 409
    finally:
        gate.set()
        _wait_until_idle(client)


def test_stop_asks_the_os_to_kill_the_whole_process_tree(client):
    # subprocess.run's own implementation calls subprocess.Popen internally, so patching Popen alone
    # would also swallow the taskkill/killpg call inside stop() — mock subprocess.run explicitly instead
    # and assert on how it was invoked, which is more precise than letting a real OS kill command run.
    gate = threading.Event()
    try:
        with patch("app.services.simulate_service.subprocess.Popen", side_effect=_blocking_popen(gate)), \
             patch("app.services.simulate_service.subprocess.run") as mock_run:
            client.post("/api/v1/simulate/start", json={})
            assert client.get("/api/v1/simulate/status").json()["running"] is True
            client.post("/api/v1/simulate/stop")
        assert mock_run.call_count == 1
        args = mock_run.call_args.args[0]
        assert args[0] == "taskkill" and "/T" in args and "99" in args   # /T: the whole tree, not just run_fleet.py
    finally:
        gate.set()
