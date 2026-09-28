"""Per-sign bookkeeping for the V2 -> V3 cascade: V3 runs once per physical sign, on its best crops."""

from __future__ import annotations

import heapq
import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from edge.imaging import XYXY, crop_with_margin, shift, sharpness
from edge.runners.v2_idd import SIGN_CLASS
from edge.runners.yolo import Box

_PEOPLE = {"person", "rider"}


@dataclass
class SignCrop:
    score: float                   # box area x sharpness: big and in focus wins
    t: float
    crop: np.ndarray
    bbox: XYXY                     # in the full frame
    frame_size: tuple[int, int]    # width, height
    blur_boxes: list[XYXY]         # people inside the crop, crop coordinates


@dataclass
class SignTrack:
    track_id: int
    first_t: float
    last_t: float
    frames_seen: int = 0
    crops: list[tuple[float, int, SignCrop]] = field(default_factory=list)   # min-heap on score

    def best(self) -> SignCrop:
        return max(self.crops)[2]


class SignTracker:
    """Keeps the best `crops_per_track` crops of every tracked sign until its track ends.

    Without this, one sign seen at 4 FPS becomes dozens of events; with it, one V3 job per sign.
    """

    def __init__(self, crops_per_track: int, margin: float, timeout_s: float, min_box_px: int):
        self.k, self.margin = crops_per_track, margin
        self.timeout_s, self.min_box_px = timeout_s, min_box_px
        self.tracks: dict[int, SignTrack] = {}
        self._order = itertools.count()

    def update(self, t: float, image: np.ndarray, boxes: list[Box]) -> None:
        people = [b.xyxy for b in boxes if b.name in _PEOPLE]
        height, width = image.shape[:2]
        for box in boxes:
            if box.name != SIGN_CLASS or box.track_id is None:
                continue
            track = self.tracks.setdefault(box.track_id, SignTrack(box.track_id, t, t))
            track.last_t = t
            track.frames_seen += 1
            x1, y1, x2, y2 = box.xyxy
            if min(x2 - x1, y2 - y1) < self.min_box_px:
                continue
            crop, (left, top) = crop_with_margin(image, box.xyxy, self.margin)
            if crop.size == 0:
                continue
            score = box.area * sharpness(crop)
            if len(track.crops) == self.k and score <= track.crops[0][0]:
                continue
            candidate = SignCrop(score, t, crop.copy(), box.xyxy, (width, height), shift(people, left, top))
            entry = (score, next(self._order), candidate)
            if len(track.crops) < self.k:
                heapq.heappush(track.crops, entry)
            else:
                heapq.heapreplace(track.crops, entry)

    def pop_ended(self, t: float) -> list[SignTrack]:
        """Tracks not seen for timeout_s, which have at least one usable crop."""
        ended = [tid for tid, tr in self.tracks.items() if t - tr.last_t >= self.timeout_s]
        return [tr for tr in (self.tracks.pop(tid) for tid in ended) if tr.crops]

    def pop_all(self) -> list[SignTrack]:
        tracks = [tr for tr in self.tracks.values() if tr.crops]
        self.tracks.clear()
        return tracks


def vote(results: list[tuple[str, float]]) -> tuple[str, float, dict[str, int]] | None:
    """Majority condition over a track's crops (ties go to the higher summed confidence).
    Returns (condition, mean confidence of the winning votes, vote counts)."""
    if not results:
        return None
    counts = Counter(condition for condition, _ in results)
    condition = max(counts, key=lambda c: (counts[c], sum(conf for cc, conf in results if cc == c)))
    confidence = float(np.mean([conf for c, conf in results if c == condition]))
    return condition, confidence, {"damaged": counts.get("damaged", 0), "good": counts.get("good", 0)}


def sign_metadata(track: SignTrack, condition: str, votes: dict[str, int]) -> dict[str, Any]:
    best = track.best()
    return {
        "condition": condition,
        "votes": votes,
        "framesSeen": track.frames_seen,
        "trackId": track.track_id,
        "bbox": [round(v, 1) for v in best.bbox],
        "frameSize": list(best.frame_size),
    }
