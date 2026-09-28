from conftest import BASE_LAT, BASE_LON, make_event, offset, post_batch


def test_nearby_potholes_join_the_same_issue(client, clean_db):
    lat2, lon2 = offset(BASE_LAT, BASE_LON, north_m=5, east_m=3)      # ~6 m away
    r1 = post_batch(client, [make_event(confidence=0.7, metadata={"severity": 1})])
    r2 = post_batch(client, [make_event(lat=lat2, lon=lon2, confidence=0.9, metadata={"severity": 3})])
    issue_id = r1.json()["results"][0]["issueId"]
    assert issue_id and r2.json()["results"][0]["issueId"] == issue_id

    issues = list(clean_db.issues.find())
    assert len(issues) == 1
    issue = issues[0]
    assert issue["eventCount"] == 2
    assert issue["severity"] == 3                      # max severity
    assert issue["status"] == "candidate"              # same vehicle, same trip -> still one pass
    assert issue["confidence"] == round(0.9 * 0.6, 3)  # best confidence x 1-pass factor


def test_distant_potholes_create_separate_issues(client, clean_db):
    lat2, lon2 = offset(BASE_LAT, BASE_LON, north_m=200)
    post_batch(client, [make_event(), make_event(lat=lat2, lon=lon2)])
    assert clean_db.issues.count_documents({}) == 2


def test_different_types_at_same_place_are_separate_issues(client, clean_db):
    post_batch(client, [make_event("pothole"), make_event("zebra_crossing", metadata={})])
    assert clean_db.issues.count_documents({}) == 2


def test_same_pothole_seen_by_two_vehicles(client, clean_db):
    lat2, lon2 = offset(BASE_LAT, BASE_LON, east_m=8)
    post_batch(client, [make_event(confidence=0.8)], vehicle_id="BUS_A", trip_id="TRIP_A1")
    post_batch(client, [make_event(lat=lat2, lon=lon2, confidence=0.8)], vehicle_id="BUS_B", trip_id="TRIP_B1")
    issue = clean_db.issues.find_one()
    assert clean_db.issues.count_documents({}) == 1
    assert (issue["vehicleCount"], issue["tripCount"], issue["eventCount"]) == (2, 2, 2)
    assert issue["status"] == "probable"
    assert issue["confidence"] == round(0.8 * 0.85, 3)


def test_non_issue_types_do_not_create_issues(client, clean_db):
    r = post_batch(client, [make_event("traffic_density", metadata={"vehicleCount": 10}),
                            make_event("road_quality", confidence=None,   # measurements may omit confidence
                                       metadata={"iri": 6.1, "iriClass": "fair"})])
    assert r.json()["accepted"] == 2
    assert all(res["issueId"] is None for res in r.json()["results"])
    assert clean_db.issues.count_documents({}) == 0


def test_traffic_sign_condition_votes(client, clean_db):
    post_batch(client, [make_event("traffic_sign", metadata={"condition": "damaged"}),
                        make_event("traffic_sign", metadata={"condition": "damaged"}),
                        make_event("traffic_sign", metadata={"condition": "good"})])
    issue = clean_db.issues.find_one()
    assert issue["metadata"]["conditionVotes"] == {"damaged": 2, "good": 1}


def test_patch_issue_status(client, clean_db):
    issue_id = post_batch(client, [make_event()]).json()["results"][0]["issueId"]

    r = client.patch(f"/api/v1/issues/{issue_id}", json={"status": "verified", "note": "checked on site"})
    assert r.status_code == 200 and r.json()["status"] == "verified"

    # A verified issue keeps collecting evidence but keeps its human-set status.
    post_batch(client, [make_event()], vehicle_id="BUS_X", trip_id="TRIP_X")
    issue = client.get(f"/api/v1/issues/{issue_id}").json()
    assert issue["status"] == "verified" and issue["eventCount"] == 2 and len(issue["events"]) == 2

    # Once resolved, a new detection at the same place opens a NEW issue.
    client.patch(f"/api/v1/issues/{issue_id}", json={"status": "resolved"})
    new_id = post_batch(client, [make_event()]).json()["results"][0]["issueId"]
    assert new_id != issue_id

    assert client.patch(f"/api/v1/issues/{issue_id}", json={"status": "fixed"}).status_code == 422
    assert client.patch("/api/v1/issues/000000000000000000000000", json={"status": "verified"}).status_code == 404
    assert client.get("/api/v1/issues/not-an-id").status_code == 404
