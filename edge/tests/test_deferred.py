"""Deferred-lane policies: priority, eviction when full, pausing, back-pressure and draining."""

import threading
import time

from edge.deferred import DeferredLane, DeferredQueue, Job


def job(kind, priority, t=0.0):
    return Job(kind, priority, t)


def test_highest_priority_first_fifo_among_equals():
    q = DeferredQueue(10)
    for j in (job("a", 1.0, 1), job("b", 3.0, 2), job("c", 1.0, 3), job("d", 2.0, 4)):
        q.put(j)
    assert [q.get().kind for _ in range(4)] == ["b", "d", "a", "c"]


def test_full_queue_evicts_lowest_or_drops_newcomer():
    q = DeferredQueue(2)
    q.put(job("v1_sweep", 0.5))
    q.put(job("v1_trigger", 2.4))
    assert q.put(job("v3_sign", 1.5))            # evicts the sweep
    assert q.dropped == {"v1_sweep": 1}
    assert not q.put(job("v1_sweep", 0.5))       # outranked by everything queued: dropped
    assert q.dropped == {"v1_sweep": 2}
    assert sorted(j.kind for j in (q.get(), q.get())) == ["v1_trigger", "v3_sign"]


def test_pause_blocks_get():
    q = DeferredQueue(4)
    q.put(job("v1_trigger", 2.0))
    q.set_paused(True)
    assert q.get(timeout=0.01) is None
    q.set_paused(False)
    assert q.get(timeout=0.01).kind == "v1_trigger"


def test_blocking_put_waits_for_space():
    q = DeferredQueue(1)
    q.put(job("a", 1.0))
    done = threading.Event()

    def producer():
        q.put(job("b", 1.0), block=True)
        done.set()

    threading.Thread(target=producer, daemon=True).start()
    time.sleep(0.05)
    assert not done.is_set()                     # waiting, not dropping
    q.get()
    assert done.wait(1.0)
    assert q.dropped == {}


def test_lane_runs_jobs_and_drains():
    q = DeferredQueue(8)
    seen = []
    lane = DeferredLane(q, {"x": lambda j: seen.append(j.t) or j.t * 2})
    lane.start()
    for i in range(5):
        q.put(job("x", 1.0, float(i)))
    assert lane.drain(timeout=5.0)
    results = sorted(lane.results.get()[1] for _ in range(5))
    lane.stop()
    assert sorted(seen) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert results == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_failing_job_does_not_kill_the_lane():
    q = DeferredQueue(8)

    def handler(j):
        if j.t == 0:
            raise RuntimeError("boom")
        return "ok"

    lane = DeferredLane(q, {"x": handler})
    lane.start()
    q.put(job("x", 1.0, 0.0))
    q.put(job("x", 1.0, 1.0))
    assert lane.drain(timeout=5.0)
    outcomes = [lane.results.get()[1] for _ in range(2)]
    lane.stop()
    assert sorted(map(str, outcomes)) == ["None", "ok"]
