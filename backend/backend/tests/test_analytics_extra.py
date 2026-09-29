"""traffic/bottlenecks and infrastructure/missing-signs: the two analytics endpoints computed on
demand rather than stored, and the two correctness properties that matter most for each."""
from datetime import datetime, timedelta, timezone

from conftest import BASE_LAT, BASE_LON, make_event, offset, post_batch


def _traffic_event(lat, lon, speed_mps, vehicle_count):
    return make_event("traffic_density", lat=lat, lon=lon,
                      metadata={"vehicleCount": vehicle_count, "speedMps": speed_mps})


def test_bottlenecks_needs_at_least_two_distinct_locations(client):
    post_batch(client, [_traffic_event(BASE_LAT, BASE_LON, 5.0, 3)])
    b = client.get("/api/v1/traffic/bottlenecks").json()
    assert b["freeFlowSpeedMps"] is None and b["bottlenecks"] == []


def test_a_slow_dense_cell_is_flagged_against_a_fast_light_one(client):
    jammed = BASE_LAT, BASE_LON
    free_flowing = offset(BASE_LAT, BASE_LON, north_m=1000)     # far enough to land in a different grid cell
    post_batch(client, [_traffic_event(*jammed, 2.0, 10) for _ in range(3)])
    post_batch(client, [_traffic_event(*free_flowing, 15.0, 2) for _ in range(3)])

    b = client.get("/api/v1/traffic/bottlenecks").json()
    assert b["cellsObserved"] == 2
    assert len(b["bottlenecks"]) == 1
    hotspot = b["bottlenecks"][0]
    assert hotspot["meanSpeedMps"] == 2.0 and hotspot["meanDensity"] == 10.0 and hotspot["sampleCount"] == 3
    assert abs(hotspot["location"]["latitude"] - jammed[0]) < 0.001


def test_a_sign_revisited_by_a_different_trip_without_reconfirmation_is_flagged(client):
    now = datetime.now(timezone.utc)
    post_batch(client, [make_event("traffic_sign", capturedAt=(now - timedelta(minutes=20)).isoformat(),
                                   metadata={"condition": "damaged"})], trip_id="TRIP_A")
    # A different trip drives past the same spot later, but with no matching traffic_sign event.
    post_batch(client, [make_event("pothole", capturedAt=(now - timedelta(minutes=10)).isoformat())],
              trip_id="TRIP_B", vehicle_id="BUS_T2")

    missing = client.get("/api/v1/infrastructure/missing-signs").json()
    assert missing["count"] == 1
    assert missing["signs"][0]["revisitedBy"] == "BUS_T2"


def test_the_same_trip_driving_further_does_not_count_as_a_revisit(client):
    now = datetime.now(timezone.utc)
    post_batch(client, [make_event("traffic_sign", capturedAt=(now - timedelta(minutes=20)).isoformat(),
                                   metadata={"condition": "damaged"})], trip_id="TRIP_A")
    # Same trip, a few seconds later and a few metres on — exactly what a bus does while still driving past.
    post_batch(client, [make_event("pothole", lat=offset(BASE_LAT, BASE_LON, north_m=5)[0], lon=BASE_LON,
                                   capturedAt=(now - timedelta(minutes=19, seconds=55)).isoformat())],
              trip_id="TRIP_A")

    missing = client.get("/api/v1/infrastructure/missing-signs").json()
    assert missing["count"] == 0


def test_missing_signs_on_an_empty_database_is_a_clean_empty_result(client):
    missing = client.get("/api/v1/infrastructure/missing-signs").json()
    assert missing == {"count": 0, "signs": [], "note": missing["note"]}
