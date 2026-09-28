from datetime import datetime, timedelta, timezone

import pytest

from conftest import BASE_LAT, BASE_LON, HEADERS, make_event, post_batch


def test_four_sample_event_types_are_accepted(client, clean_db):
    events = [
        make_event("pothole", metadata={"severity": 2}),
        make_event("traffic_density", model="traffic_model",
                   metadata={"vehicleCount": 23, "carCount": 12, "busCount": 2, "truckCount": 3, "twoWheelerCount": 6}),
        make_event("traffic_sign", model="sign_model", metadata={"condition": "damaged"}),
        make_event("pedestrian_alert", model="traffic_model", metadata={"level": "CRITICAL", "pedestrianCount": 4}),
    ]
    r = post_batch(client, events)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["accepted"], body["duplicates"], body["rejected"]) == (4, 0, 0)
    assert clean_db.events.count_documents({}) == 4

    stored = clean_db.events.find_one({"eventId": events[0]["eventId"]})
    assert stored["location"] == {"type": "Point", "coordinates": [BASE_LON, BASE_LAT]}
    assert stored["receivedAt"] is not None
    assert stored["vehicleId"] == "BUS_T1" and stored["deviceId"] == "EDGE_T1"


def test_duplicate_event_is_stored_once(client, clean_db):
    event = make_event()
    r1 = post_batch(client, [event])
    r2 = post_batch(client, [event])                      # edge retry
    r3 = post_batch(client, [event, event])               # duplicate inside one batch
    assert r1.json()["accepted"] == 1
    assert (r2.json()["accepted"], r2.json()["duplicates"]) == (0, 1)
    assert (r3.json()["accepted"], r3.json()["duplicates"]) == (0, 2)
    assert clean_db.events.count_documents({"eventId": event["eventId"]}) == 1
    assert clean_db.issues.find_one()["eventCount"] == 1  # duplicates never touch the issue


@pytest.mark.parametrize("bad_event, field", [
    (make_event(location={"latitude": 95.0, "longitude": BASE_LON}), "latitude"),
    (make_event(location={"latitude": BASE_LAT, "longitude": -181.0}), "longitude"),
    (make_event(confidence=1.5), "confidence"),
    ({k: v for k, v in make_event().items() if k != "eventType"}, "eventType"),
    (make_event(capturedAt="not-a-date"), "capturedAt"),
    (make_event(capturedAt=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()), "capturedAt"),
])
def test_invalid_event_is_rejected_without_failing_the_batch(client, clean_db, bad_event, field):
    good = make_event(lat=BASE_LAT + 0.01)
    r = post_batch(client, [bad_event, good])
    assert r.status_code == 200
    body = r.json()
    assert (body["accepted"], body["rejected"]) == (1, 1)
    rejected = body["results"][0]
    assert rejected["status"] == "rejected" and field in rejected["reason"]
    assert clean_db.events.count_documents({}) == 1


def test_missing_or_wrong_device_key_is_401(client):
    assert post_batch(client, [make_event()], headers={}).status_code == 401
    assert post_batch(client, [make_event()], headers={"X-Device-Key": "wrong"}).status_code == 401


def test_bad_envelope_is_422(client):
    r = client.post("/api/v1/ingest", headers=HEADERS, json={"vehicleId": "BUS", "events": [make_event()]})
    assert r.status_code == 422      # deviceId missing
    r = client.post("/api/v1/ingest", headers=HEADERS, json={"deviceId": "E", "vehicleId": "BUS", "events": []})
    assert r.status_code == 422      # empty batch


def test_vehicle_and_trip_are_tracked(client, clean_db):
    post_batch(client, [make_event(), make_event("traffic_density", metadata={"vehicleCount": 5})])
    vehicle = clean_db.vehicles.find_one({"_id": "BUS_T1"})
    assert vehicle["eventCount"] == 2 and vehicle["lastTripId"] == "TRIP_T1"
    trip = clean_db.trips.find_one({"_id": "TRIP_T1"})
    assert trip["vehicleId"] == "BUS_T1" and trip["eventCount"] == 2
