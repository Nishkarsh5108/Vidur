"""edge/backend.py: event translation to the backend's contract (docs/hello.md), and that a crop is
uploaded to the backend's media store exactly once per local id, even across a retried batch."""

import io
import json
from unittest.mock import patch

from edge.backend import BackendUploader, to_backend


def _ok(data: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps(data).encode())


def _pothole_event(**overrides) -> dict:
    event = {
        "eventId": "e1", "eventType": "pothole", "capturedAt": "2026-01-01T00:00:00Z",
        "location": {"latitude": 28.6, "longitude": 77.2}, "confidence": 0.8,
        "source": {"model": "V1-roadsense", "modelVersion": "v1", "sensor": "cam_front"}, "metadata": {},
    }
    event.update(overrides)
    return event


def test_to_backend_drops_events_with_no_location():
    assert to_backend(_pothole_event(location=None)) is None


def test_to_backend_translates_type_model_and_metadata():
    event = _pothole_event(eventType="iri_window", confidence=None, speedMps=3.0, headingDeg=90.0,
                           source={"model": "S1-iri", "modelVersion": "v2", "sensor": "imu"},
                           metadata={"iri": 5.0, "iriClass": "fair"})
    out = to_backend(event)
    assert out["eventType"] == "road_quality" and out["model"] == "iri_cnn" and out["modelVersion"] == "v2"
    assert out["metadata"]["speedMps"] == 3.0 and out["metadata"]["headingDeg"] == 90.0 and out["metadata"]["sensor"] == "imu"
    assert out["metadata"]["iri"] == 5.0
    assert "confidence" not in out          # None confidence is omitted, not sent as a literal null
    assert "mediaId" not in out             # a local media id means nothing to the backend on its own


def test_to_backend_expands_traffic_density_class_counts():
    event = _pothole_event(eventType="traffic_sample", source={"model": "V2-idd-traffic"},
                           metadata={"meanCounts": {"car": 2.0, "bus": 1.0, "person": 0.5, "rider": 0.25}})
    out = to_backend(event)
    assert out["eventType"] == "traffic_density"
    assert out["metadata"]["carCount"] == 2.0 and out["metadata"]["busCount"] == 1.0
    assert out["metadata"]["vehicleCount"] == 3.0
    assert out["metadata"]["pedestrianCount"] == 0.5 and out["metadata"]["riderCount"] == 0.25


def test_prepare_uploads_a_local_crop_once_and_attaches_the_backend_id(tmp_path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "m_abc.jpg").write_bytes(b"fake-jpeg-bytes")
    uploader = BackendUploader("http://x", "key", "dev", "veh", "trip", tmp_path, media_dir=media_dir)

    with patch("edge.backend.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = _ok({"mediaId": "abc123.jpg"})
        events = [_pothole_event(eventId="e1", mediaId="m_abc"), _pothole_event(eventId="e2", mediaId="m_abc")]
        prepared = uploader._prepare(events)

    assert mock_urlopen.call_count == 1                    # same local id, uploaded only once
    assert prepared[0]["metadata"]["mediaId"] == "abc123.jpg"
    assert prepared[0]["metadata"]["mediaUrl"] == "http://x/api/v1/media/abc123.jpg"
    assert prepared[1]["metadata"]["mediaId"] == "abc123.jpg"


def test_prepare_omits_media_id_when_the_file_is_missing(tmp_path):
    uploader = BackendUploader("http://x", "key", "dev", "veh", "trip", tmp_path, media_dir=tmp_path / "empty")
    prepared = uploader._prepare([_pothole_event(mediaId="does-not-exist")])
    assert "mediaId" not in prepared[0]["metadata"] and "mediaUrl" not in prepared[0]["metadata"]


def test_prepare_skips_events_with_no_location_and_counts_them(tmp_path):
    uploader = BackendUploader("http://x", "key", "dev", "veh", "trip", tmp_path)
    prepared = uploader._prepare([_pothole_event(location=None), _pothole_event(eventId="e2")])
    assert len(prepared) == 1 and uploader.skipped_no_location == 1
