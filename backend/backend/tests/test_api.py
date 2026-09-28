from conftest import BASE_LAT, BASE_LON, make_event, offset, post_batch


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["database"] == "ok"


def test_kpis_on_empty_database_are_zero(client):
    k = client.get("/api/v1/kpis").json()
    assert k["totalEvents"] == 0 and k["openIssues"] == 0 and k["activeVehicles"] == 0
    assert k["activeAlerts"] == 0 and k["issuesByType"] == [] and k["eventsByModel"] == []
    assert k["roadQuality"]["samples"] == 0 and k["roadQuality"]["avgIri"] is None
    assert k["traffic"]["samples"] == 0


def _seed(client):
    return post_batch(client, [
        make_event("pothole", metadata={"severity": 3}),
        make_event("road_quality", model="iri_model", metadata={"iri": 8.4, "iriClass": "poor"}),
        make_event("road_quality", model="iri_model", metadata={"iri": 3.6, "iriClass": "good"}),
        make_event("traffic_density", model="traffic_model",
                   metadata={"vehicleCount": 20, "carCount": 10, "busCount": 2, "truckCount": 2, "twoWheelerCount": 6}),
        make_event("traffic_density", model="traffic_model",
                   metadata={"vehicleCount": 10, "carCount": 6, "busCount": 0, "truckCount": 1, "twoWheelerCount": 3}),
        make_event("pedestrian_alert", model="traffic_model", metadata={"level": "CRITICAL", "pedestrianCount": 3}),
    ])


def test_kpis_road_health_and_traffic_reflect_data(client):
    assert _seed(client).json()["accepted"] == 6
    k = client.get("/api/v1/kpis").json()
    assert k["totalEvents"] == 6 and k["openIssues"] == 1
    assert k["activeVehicles"] == 1 and k["activeAlerts"] == 1
    assert {"name": "iri_model", "count": 2} in k["eventsByModel"]
    assert k["issuesByType"] == [{"name": "pothole", "count": 1}]

    rh = client.get("/api/v1/road-health").json()
    assert rh["iri"]["samples"] == 2 and rh["iri"]["avgIri"] == 6.0 and rh["iri"]["maxIri"] == 8.4
    assert rh["openRoadIssues"] == 1 and rh["openIssuesBySeverity"] == [{"name": 3, "count": 1}]

    t = client.get("/api/v1/traffic").json()
    assert t["summary"]["samples"] == 2 and t["summary"]["avgVehicleCount"] == 15.0
    assert t["summary"]["maxVehicleCount"] == 20 and t["summary"]["avgByClass"]["carCount"] == 8.0
    assert len(t["recentSamples"]) == 2


def test_events_list_filters_and_detail(client):
    _seed(client)
    r = client.get("/api/v1/events", params={"eventType": "road_quality"}).json()
    assert r["total"] == 2 and all(e["eventType"] == "road_quality" for e in r["items"])
    event_id = r["items"][0]["eventId"]
    assert client.get(f"/api/v1/events/{event_id}").json()["eventId"] == event_id
    assert client.get("/api/v1/events/does-not-exist").status_code == 404
    assert client.get("/api/v1/events", params={"limit": 2}).json()["total"] == 6


def test_vehicles(client):
    _seed(client)
    vehicles = client.get("/api/v1/vehicles").json()
    assert vehicles["total"] == 1 and vehicles["items"][0]["id"] == "BUS_T1"
    detail = client.get("/api/v1/vehicles/BUS_T1").json()
    assert detail["trips"][0]["id"] == "TRIP_T1" and len(detail["recentEvents"]) == 6
    assert client.get("/api/v1/vehicles/NOPE").status_code == 404


def test_map_endpoints_return_geojson(client):
    far_lat, far_lon = offset(BASE_LAT, BASE_LON, north_m=5000)
    post_batch(client, [make_event(), make_event(lat=far_lat, lon=far_lon)])

    fc = client.get("/api/v1/map/events").json()
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    feature = fc["features"][0]
    assert feature["type"] == "Feature" and feature["geometry"]["type"] == "Point"
    assert "eventType" in feature["properties"] and "location" not in feature["properties"]

    issues = client.get("/api/v1/map/issues", params={"issueType": "pothole"}).json()
    assert len(issues["features"]) == 2

    bbox = f"{BASE_LON - 0.01},{BASE_LAT - 0.01},{BASE_LON + 0.01},{BASE_LAT + 0.01}"
    assert len(client.get("/api/v1/map/events", params={"bbox": bbox}).json()["features"]) == 1
    assert client.get("/api/v1/map/events", params={"bbox": "1,2,3"}).status_code == 400


def test_websocket_receives_live_updates(client):
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json()["type"] == "connected"
        post_batch(client, [make_event()])
        types = [ws.receive_json()["type"] for _ in range(3)]
        assert types == ["event", "issue_update", "vehicle_update"]
