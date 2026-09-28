import datetime

def format_event_payload(event_type, confidence, lat=28.6142, lon=77.2110, vehicle_id="BUS_DEMO_01", metadata=None):
    """
    Formats the JSON event according to the backend team's schema.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    payload = {
        "eventType": event_type,
        "confidence": round(float(confidence), 2),
        "timestamp": timestamp,
        "vehicleId": vehicle_id,
        "location": {
            "latitude": float(lat),
            "longitude": float(lon)
        },
        "metadata": metadata if metadata is not None else {}
    }
    return payload
