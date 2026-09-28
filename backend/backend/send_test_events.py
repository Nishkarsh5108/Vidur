"""Send sample events to a running backend and check the results — no ML models needed.

Usage:
    python send_test_events.py                       # http://localhost:8000, key demo-secret-key
    python send_test_events.py --url http://192.168.1.20:8000 --key demo-secret-key

Each run uses a slightly different base location, so repeated runs don't interfere.
Exit code is 0 only if every check passes.
"""
import argparse
import math
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

import httpx

RESULTS: list[bool] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{'  -> ' + detail if detail and not condition else ''}")


def moved(lat: float, lon: float, north_m: float = 0, east_m: float = 0) -> tuple[float, float]:
    return lat + north_m / 111_320, lon + east_m / (111_320 * math.cos(math.radians(lat)))


def event(event_type: str, lat: float, lon: float, model: str, metadata: dict, confidence: float = 0.9) -> dict:
    return {
        "eventId": str(uuid.uuid4()),
        "eventType": event_type,
        "capturedAt": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat().replace("+00:00", "Z"),
        "location": {"latitude": lat, "longitude": lon},
        "confidence": confidence,
        "model": model,
        "modelVersion": "1.0",
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--key", default="demo-secret-key")
    args = parser.parse_args()

    api = httpx.Client(base_url=args.url, timeout=10)
    headers = {"X-Device-Key": args.key}

    def send(events, vehicle="BUS_DEMO_01", trip="TRIP_001", key_headers=headers):
        body = {"schemaVersion": "1.0", "deviceId": "EDGE_DEMO_01", "vehicleId": vehicle,
                "tripId": trip, "events": events}
        return api.post("/api/v1/ingest", json=body, headers=key_headers)

    # Random base point in central Delhi so each run creates its own issues.
    lat0, lon0 = moved(28.6139, 77.2090, random.uniform(-3000, 3000), random.uniform(-3000, 3000))

    print("1. Health")
    try:
        health = api.get("/api/v1/health")
    except httpx.ConnectError:
        print(f"  Cannot connect to {args.url}. Is uvicorn running?")
        return 1
    check("backend and database are up", health.status_code == 200, health.text)

    print("2. Four sample event types")
    samples = [
        event("pothole", lat0, lon0, "road_model", {"severity": 2}),
        event("traffic_density", *moved(lat0, lon0, 40), "traffic_model",
              {"vehicleCount": 23, "carCount": 12, "busCount": 2, "truckCount": 3, "twoWheelerCount": 6}),
        event("traffic_sign", *moved(lat0, lon0, 80), "sign_model", {"condition": "damaged"}, 0.84),
        event("pedestrian_alert", *moved(lat0, lon0, 120), "traffic_model",
              {"level": "CRITICAL", "pedestrianCount": 4, "riderCount": 1, "schoolName": "Demo School"}, 0.77),
    ]
    r = send(samples)
    body = r.json()
    check("4 accepted", r.status_code == 200 and body["accepted"] == 4, r.text)

    print("3. Duplicate event")
    r = send([samples[0]])
    check("re-sent event reported as duplicate", r.json().get("duplicates") == 1, r.text)

    print("4. Invalid events (each batch also has one valid event)")
    invalid = {
        "invalid latitude": {**event("pothole", 95.0, lon0, "road_model", {}), },
        "invalid longitude": {**event("pothole", lat0, 200.0, "road_model", {})},
        "invalid confidence": {**event("pothole", lat0, lon0, "road_model", {}), "confidence": 1.7},
        "missing eventType": {k: v for k, v in event("pothole", lat0, lon0, "road_model", {}).items()
                              if k != "eventType"},
    }
    for name, bad in invalid.items():
        good = event("traffic_density", *moved(lat0, lon0, 500), "traffic_model", {"vehicleCount": 5})
        r = send([bad, good]).json()
        check(f"{name}: rejected=1 accepted=1", r.get("rejected") == 1 and r.get("accepted") == 1,
              str(r.get("results")))
        print(f"        reason: {r['results'][0].get('reason')}")

    print("5. Wrong API key")
    r = send([event("pothole", lat0, lon0, "road_model", {})], key_headers={"X-Device-Key": "wrong"})
    check("401 Unauthorized", r.status_code == 401, str(r.status_code))

    print("6. Nearby potholes (~6 m apart) -> same issue")
    base_lat, base_lon = moved(lat0, lon0, 1000)
    r1 = send([event("pothole", base_lat, base_lon, "road_model", {"severity": 1}, 0.72)]).json()
    r2 = send([event("pothole", *moved(base_lat, base_lon, 5, 3), "road_model", {"severity": 3}, 0.88)]).json()
    issue_a = r1["results"][0]["issueId"]
    check("same issueId", issue_a is not None and issue_a == r2["results"][0]["issueId"])

    print("7. Distant potholes (~200 m apart) -> separate issues")
    r3 = send([event("pothole", *moved(base_lat, base_lon, 200), "road_model", {"severity": 2})]).json()
    check("different issueId", r3["results"][0]["issueId"] not in (None, issue_a))

    print("8. Same pothole seen by a different vehicle -> same issue, status probable")
    r4 = send([event("pothole", *moved(base_lat, base_lon, -4), "road_model", {"severity": 2}, 0.8)],
              vehicle="BUS_DEMO_02", trip="TRIP_101").json()
    check("same issueId", r4["results"][0]["issueId"] == issue_a)
    issue = api.get(f"/api/v1/issues/{issue_a}").json()
    check("vehicleCount=2, eventCount=3, status=probable",
          issue["vehicleCount"] == 2 and issue["eventCount"] == 3 and issue["status"] == "probable",
          f"{issue['vehicleCount']=}, {issue['eventCount']=}, {issue['status']=}")
    print(f"        issue {issue_a}: confidence={issue['confidence']} severity={issue['severity']}")

    print("9. KPIs")
    kpis = api.get("/api/v1/kpis").json()
    for key in ("totalEvents", "openIssues", "activeVehicles", "activeAlerts"):
        print(f"        {key}: {kpis[key]}")
    print(f"        issuesByType: {kpis['issuesByType']}")
    check("KPIs reflect data", kpis["totalEvents"] > 0 and kpis["openIssues"] > 0)

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
