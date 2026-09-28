import json
import uuid

import pytest

from edge.geo import GpsTrack, bearing_deg, haversine_m, load_school_zones, point_in_ring, square_around
from edge.util import in_school_hours, iso_utc, parse_time, uuid7


def test_haversine_and_bearing():
    assert haversine_m(28.0, 77.0, 28.001, 77.0) == pytest.approx(111.2, abs=0.2)
    assert bearing_deg(28.0, 77.0, 28.01, 77.0) == pytest.approx(0.0, abs=0.01)
    assert bearing_deg(28.0, 77.0, 28.0, 77.01) == pytest.approx(90.0, abs=0.01)


def test_point_in_ring_and_square():
    ring = square_around(28.6, 77.2, 120.0)
    assert point_in_ring(28.6, 77.2, ring)
    assert not point_in_ring(28.6 + 0.002, 77.2, ring)        # ~220 m north


def test_gps_track_odometer_heading_and_lookup():
    track = GpsTrack()
    for i in range(101):                                       # 10 s due north at 10 m/s, 10 Hz
        track.update(1000.0 + i * 0.1, 28.0 + i * 1e-5 * 0.8993, 77.0, 10.0)
    assert track.odometer == pytest.approx(100.0, rel=1e-6)
    assert track.heading == pytest.approx(0.0, abs=0.5)
    mid = track.at(1005.0)
    assert mid.odometer == pytest.approx(50.0, abs=1.0)
    assert mid.lat == pytest.approx(28.0 + 50 * 1e-5 * 0.8993, abs=2e-6)


def test_load_school_zones(tmp_path):
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Point School", "osm_id": 1},
         "geometry": {"type": "Point", "coordinates": [77.2, 28.6]}},
        {"type": "Feature", "properties": {"name": "Polygon School", "osm_id": 2},
         "geometry": {"type": "Polygon", "coordinates": [[[77.0, 28.0], [77.01, 28.0], [77.01, 28.01], [77.0, 28.01], [77.0, 28.0]]]}},
    ]}
    path = tmp_path / "schools.geojson"
    path.write_text(json.dumps(geojson))
    zones = load_school_zones(path, 120.0)
    assert [z.name for z in zones] == ["Point School", "Polygon School"]
    assert zones[0].contains(28.6, 77.2) and zones[1].contains(28.005, 77.005)
    assert not zones[1].contains(28.02, 77.005)


def test_uuid7_is_version_7_and_time_ordered():
    a, b = uuid7(1_790_000_000.0), uuid7(1_790_000_001.0)
    assert uuid.UUID(a).version == 7 and uuid.UUID(a).variant == uuid.RFC_4122
    assert a < b


def test_time_round_trip_and_school_hours():
    t = parse_time("2026-12-10T08:14:00.640Z")
    assert iso_utc(t) == "2026-12-10T08:14:00.640Z"
    hours, days = [["07:30", "09:30"], ["13:30", "15:30"]], [1, 2, 3, 4, 5, 6]
    assert in_school_hours(parse_time("2026-12-10T02:45:00Z"), hours, days)       # Thu 08:15 IST
    assert not in_school_hours(parse_time("2026-12-10T06:30:00Z"), hours, days)   # Thu 12:00 IST
    assert not in_school_hours(parse_time("2026-12-13T02:45:00Z"), hours, days)   # Sunday
