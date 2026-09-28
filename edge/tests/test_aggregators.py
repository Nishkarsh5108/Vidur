import numpy as np

from edge.aggregators.potholes import RouteDeduper
from edge.aggregators.school_zone import SchoolZoneMonitor
from edge.aggregators.signs import vote
from edge.aggregators.traffic import TrafficAggregator
from edge.capture import ImuSample
from edge.config import load_config
from edge.geo import Fix, _make_zone, square_around
from edge.imu import ShockStream
from edge.runners.v1_roadsense import severity
from edge.runners.yolo import Box
from edge.util import parse_time

THRESHOLDS = (0.0087, 0.0217)


def test_severity_is_resolution_independent():
    # The data-contract example: bbox [812, 604, 1011, 699] on 1920x1080 -> 0.91% of the frame -> severity 2.
    area = (1011 - 812) * (699 - 604)
    assert severity(area, 1920 * 1080, THRESHOLDS) == 2
    # The same pothole seen by a 1280x720 camera covers the same fraction, so the same severity.
    assert severity(area * (1280 / 1920) ** 2, 1280 * 720, THRESHOLDS) == 2
    assert severity(10_000, 1920 * 1080, THRESHOLDS) == 1
    assert severity(60_000, 1920 * 1080, THRESHOLDS) == 3


def _box(name, track_id, x=100.0):
    return Box((x, 100.0, x + 50, 150.0), 0.8, 0, name, track_id)


def test_traffic_window_closes_on_time_or_distance():
    agg = TrafficAggregator(window_s=10.0, window_m=100.0)
    assert agg.add(0.0, [_box("car", 1), _box("car", 2), _box("person", 3)], 0.0, 5.0) is None
    assert agg.add(5.0, [_box("car", 1)], 20.0, 5.0) is None
    sample = agg.add(10.0, [_box("bus", 4)], 40.0, 5.0)
    m = sample["metadata"]
    assert m["meanCounts"]["car"] == 1.0 and m["maxCounts"]["car"] == 2 and m["uniqueTracks"] == 4
    assert agg.add(11.0, [], 40.0, 10.0) is None
    assert agg.add(12.0, [], 145.0, 10.0) is not None          # 105 m travelled: closes early


def test_sign_vote_majority_and_tie_break():
    assert vote([("damaged", 0.6), ("good", 0.9), ("damaged", 0.7)])[0] == "damaged"
    condition, conf, votes = vote([("damaged", 0.5), ("good", 0.9)])
    assert condition == "good" and votes == {"damaged": 1, "good": 1} and conf == 0.9
    assert vote([]) is None


def test_route_deduper():
    d = RouteDeduper()
    assert d.accept("pothole", 100.0, 15.0)
    assert not d.accept("pothole", 110.0, 15.0)
    assert d.accept("pothole", 116.0, 15.0)
    assert d.accept("zebra", 110.0, 30.0)                      # kinds are independent


def test_school_zone_levels_cooldown_and_escalation():
    zone = _make_zone("42", "Test School", square_around(28.6, 77.2, 120.0))
    monitor = SchoolZoneMonitor([zone], [["07:30", "09:30"]], [1, 2, 3, 4, 5, 6], 300.0, [0.15, 0.55, 0.85, 0.85])
    fix = Fix(0, 28.6, 77.2, 5.0, 0.0, 0.0)
    person = Box((600.0, 400.0, 660.0, 600.0), 0.7, 0, "person", 1)   # bottom-centre (630, 600) inside the ROI
    rider = Box((300.0, 400.0, 360.0, 600.0), 0.7, 1, "rider", 2)
    after_school = parse_time("2026-12-10T06:30:00Z")     # Thursday 12:00 IST
    in_school_hours = parse_time("2026-12-10T02:45:00Z")  # Thursday 08:15 IST

    alert = monitor.update(after_school, fix, [rider], (1280, 720))
    assert alert["level"] == "ADVISORY" and alert["riderCount"] == 1
    assert monitor.update(after_school + 10, fix, [rider], (1280, 720)) is None      # cooldown
    monitor2 = SchoolZoneMonitor([zone], [["07:30", "09:30"]], [1, 2, 3, 4, 5, 6], 300.0, [0.15, 0.55, 0.85, 0.85])
    assert monitor2.update(in_school_hours, fix, [rider], (1280, 720))["level"] == "ADVISORY"   # riders don't escalate
    assert monitor2.update(in_school_hours + 5, fix, [person], (1280, 720))["level"] == "CRITICAL"  # escalation
    assert monitor.update(after_school, Fix(0, 28.7, 77.2, 5.0, 0.0, 0.0), [person], (1280, 720)) is None


def _imu(t, az, speed=10.0):
    return ImuSample(t, 0.0, 0.0, az, 0.0, 0.0, 0.0, 28.0, 77.0, speed)


def test_jerk_trigger_fires_once_per_jolt_and_not_when_stopped():
    cfg = load_config().triggers
    stream = ShockStream(cfg, model=None, min_speed_mps=1.0)
    rng = np.random.default_rng(0)
    triggers = []
    for i in range(1000):                                      # 10 s at 100 Hz, jolt at t = 5 s
        az = rng.normal(0, 0.3) + (12.0 if i == 500 else -12.0 if i == 501 else 0.0)
        _, trig = stream.add(_imu(i * 0.01, az), odometer=i * 0.1)
        triggers += trig
    assert len(triggers) == 1 and abs(triggers[0].t - 5.0) < 0.05 and triggers[0].cls == 3

    stopped = ShockStream(cfg, model=None, min_speed_mps=1.0)
    fired = []
    for i in range(1000):
        az = 12.0 if i == 500 else -12.0 if i == 501 else 0.0
        fired += stopped.add(_imu(i * 0.01, az, speed=0.0), odometer=0.0)[1]
    assert fired == []                                          # door slams at a stop are ignored
