"""Application settings (read from environment / .env) and issue-aggregation rules."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MONGODB_URI: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "fleet_sense"
    DEVICE_API_KEY: str = "demo-secret-key"

    MAX_BATCH_SIZE: int = 500
    MAX_FUTURE_SECONDS: int = 300          # reject events captured > 5 min in the future (clock skew guard)
    ACTIVE_VEHICLE_MINUTES: int = 10       # vehicle is "active" if it sent data this recently
    ALERT_WINDOW_MINUTES: int = 30         # pedestrian_alert / incident events this recent count as "active alerts"


settings = Settings()


# Which event types become persistent issues, and how close two detections must be
# (in metres and hours) to count as the same real-world problem.
# Event types not listed here (traffic_density, road_quality, pedestrian_alert,
# device_health, ...) are stored as events only and feed the analytics endpoints.
ISSUE_RULES: dict[str, dict[str, float]] = {
    "pothole":              {"radius_m": 15,  "window_hours": 24 * 30},
    "road_damage":          {"radius_m": 15,  "window_hours": 24 * 30},
    "zebra_crossing":       {"radius_m": 25,  "window_hours": 24 * 30},
    "traffic_sign":         {"radius_m": 20,  "window_hours": 24 * 30},
    "infrastructure_issue": {"radius_m": 20,  "window_hours": 24 * 30},
    "traffic_bottleneck":   {"radius_m": 100, "window_hours": 2},
    "incident":             {"radius_m": 50,  "window_hours": 6},
}

# Issue confidence = best single-event confidence x factor for the number of distinct trips.
# Frames from one pass are not independent, so repeated passes (not frames) raise confidence.
PASS_FACTOR = {1: 0.6, 2: 0.85}   # 3 or more trips -> 1.0

OPEN_STATUSES = ["candidate", "probable", "verified"]
ALERT_EVENT_TYPES = ["pedestrian_alert", "incident"]
