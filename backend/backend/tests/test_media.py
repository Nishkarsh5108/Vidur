"""POST /api/v1/media (edge upload, auth) and GET /api/v1/media/{id} (dashboard display, no auth)."""
from conftest import HEADERS


def test_upload_then_fetch_a_photo(client):
    r = client.post("/api/v1/media", headers=HEADERS, files={"file": ("crop.jpg", b"\xff\xd8fake-jpeg", "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["mediaId"].endswith(".jpg") and body["url"] == f"/api/v1/media/{body['mediaId']}"

    got = client.get(body["url"])
    assert got.status_code == 200 and got.content == b"\xff\xd8fake-jpeg"
    assert got.headers["content-type"] == "image/jpeg"


def test_upload_requires_the_device_key(client):
    r = client.post("/api/v1/media", files={"file": ("crop.jpg", b"data", "image/jpeg")})
    assert r.status_code == 401


def test_upload_rejects_an_unsupported_content_type(client):
    r = client.post("/api/v1/media", headers=HEADERS, files={"file": ("crop.txt", b"data", "text/plain")})
    assert r.status_code == 415


def test_reuploading_the_same_bytes_reuses_the_same_id(client):
    a = client.post("/api/v1/media", headers=HEADERS, files={"file": ("a.jpg", b"same-bytes", "image/jpeg")}).json()
    b = client.post("/api/v1/media", headers=HEADERS, files={"file": ("b.jpg", b"same-bytes", "image/jpeg")}).json()
    assert a["mediaId"] == b["mediaId"]           # content-addressed: identical crops are one file


def test_missing_and_path_traversal_are_both_404(client):
    assert client.get("/api/v1/media/does-not-exist.jpg").status_code == 404
    assert client.get("/api/v1/media/..%2f..%2fapp%2fmain.py").status_code == 404
