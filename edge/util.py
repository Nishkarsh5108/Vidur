"""Small shared helpers: timestamps, UUIDv7 event IDs and school-hours checks."""

from __future__ import annotations

import datetime as dt
import os
import time
import uuid

# India has no daylight saving, so a fixed offset avoids needing tzdata on Windows.
IST = dt.timezone(dt.timedelta(hours=5, minutes=30), "IST")


def iso_utc(t: float) -> str:
    """Epoch seconds -> '2026-12-10T08:14:00.640Z'."""
    stamp = dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def parse_time(value: str) -> float:
    """ISO 8601 ('...Z' or with an offset; naive means UTC) or epoch seconds -> epoch seconds."""
    try:
        return float(value)
    except ValueError:
        pass
    stamp = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp.timestamp()


def uuid7(t: float | None = None) -> str:
    """RFC 9562 UUIDv7: a 48-bit Unix-millisecond timestamp followed by random bits.

    Time-ordered IDs keep the backend's primary-key index append-mostly, and because the edge
    generates them, a retried upload carries the same ID and the backend can drop the duplicate.
    """
    ms = int((time.time() if t is None else t) * 1000) & 0xFFFF_FFFF_FFFF
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 68) & 0xFFF
    rand_b = rand & ((1 << 62) - 1)
    value = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(uuid.UUID(int=value))


def in_school_hours(t: float, windows: list[list[str]], iso_weekdays: list[int]) -> bool:
    """True if epoch time t falls inside any ("HH:MM", "HH:MM") window, in IST, on one of the given weekdays."""
    local = dt.datetime.fromtimestamp(t, IST)
    if local.isoweekday() not in iso_weekdays:
        return False
    hhmm = local.strftime("%H:%M")
    return any(start <= hhmm < end for start, end in windows)
