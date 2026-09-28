The server takes one envelope format, and the events inside it change shape depending on which model produced them.

**Request:** `POST http://<your-IP>:8000/api/v1/ingest`, with the header `X-Device-Key: demo-secret-key`.

## The envelope (every request has this shape)
```json
{
  "schemaVersion": "1.0",
  "deviceId": "EDGE_DEMO_01",
  "vehicleId": "BUS_DEMO_01",
  "tripId": "TRIP_001",
  "events": [ ...one or more events from below... ]
}
```
One request can carry up to 500 events, and they can come from different models.

## Model 1 — RoadSense (potholes, zebra crossings)
```json
{
  "eventId": "c1a7e3f0-1b2c-4d5e-8f90-111111111111",
  "eventType": "pothole",
  "capturedAt": "2026-09-28T10:30:15Z",
  "location": {"latitude": 28.6141, "longitude": 77.2093},
  "confidence": 0.82,
  "model": "roadsense_yolov8m",
  "modelVersion": "1.0",
  "metadata": {"severity": 2}
}
```
For a zebra crossing, send `"eventType": "zebra_crossing"` (not `missing_zebra`) with `"metadata": {}`.

## Model 2 — Traffic (vehicle counts + pedestrian alerts)
Send a vehicle-count event every ~5 seconds, not every frame:
```json
{
  "eventId": "c1a7e3f0-1b2c-4d5e-8f90-222222222222",
  "eventType": "traffic_density",
  "capturedAt": "2026-09-28T10:30:20Z",
  "location": {"latitude": 28.6143, "longitude": 77.2096},
  "confidence": 0.71,
  "model": "idd_yolov8n",
  "modelVersion": "1.0",
  "metadata": {
    "vehicleCount": 23,
    "carCount": 12,
    "busCount": 2,
    "truckCount": 3,
    "twoWheelerCount": 6,
    "autorickshawCount": 0,
    "bicycleCount": 0
  }
}
```
Send a pedestrian alert only when the level is not `NORMAL`:
```json
{
  "eventId": "c1a7e3f0-1b2c-4d5e-8f90-333333333333",
  "eventType": "pedestrian_alert",
  "capturedAt": "2026-09-28T10:30:25Z",
  "location": {"latitude": 28.6145, "longitude": 77.2099},
  "model": "idd_yolov8n",
  "metadata": {
    "level": "CRITICAL",
    "pedestrianCount": 4,
    "riderCount": 1,
    "schoolName": "Demo School"
  }
}
```

## Model 3 — Traffic sign condition
```json
{
  "eventId": "c1a7e3f0-1b2c-4d5e-8f90-444444444444",
  "eventType": "traffic_sign",
  "capturedAt": "2026-09-28T10:30:30Z",
  "location": {"latitude": 28.6147, "longitude": 77.2102},
  "confidence": 0.86,
  "model": "sign_yolov8s",
  "modelVersion": "1.0",
  "metadata": {"condition": "damaged"}
}
```
`condition` is `"damaged"` for class 0 and `"good"` for class 1.

## Model 4 — IRI (road roughness, one event per 100 m)
```json
{
  "eventId": "c1a7e3f0-1b2c-4d5e-8f90-555555555555",
  "eventType": "road_quality",
  "capturedAt": "2026-09-28T10:30:40Z",
  "location": {"latitude": 28.6150, "longitude": 77.2108},
  "model": "iri_cnn",
  "modelVersion": "1.0",
  "metadata": {"iri": 8.4, "iriClass": "poor"}
}
```
`iriClass` is `good` below 4, `fair` from 4 to 8, `poor` from 8 to 14, and `very_poor` at 14 or above.

## Field rules
| Field | Required | Rule |
|---|---|---|
| `eventId` | yes | Unique per detection. Use `str(uuid.uuid4())`. Resending the same id is ignored as a duplicate. |
| `eventType` | yes | One of the types above. |
| `capturedAt` | yes | Video frame time in UTC, ending in `Z`. It can't be more than 5 minutes in the future. |
| `location` | yes | GPS synced to that frame. Latitude −90 to 90, longitude −180 to 180. |
| `model` | yes | Any name. |
| `confidence` | no | 0 to 1. Leave it out when there isn't one (IRI, pedestrian alert). |
| `modelVersion`, `metadata` | no | |

The server replies with:
```json
{"accepted": 5, "duplicates": 0, "rejected": 0, "results": [...]}
```
If an event is rejected, its entry in `results` says why, for example "latitude must be ≤ 90".