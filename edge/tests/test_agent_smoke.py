"""End to end on a synthetic ride: a short video plus a slice of the team's real IMU log with a jolt injected.

Checks the plumbing (lanes, triggers, events, envelopes), not model accuracy. Skipped without Ultralytics.
"""

import csv
import json

import cv2
import numpy as np
import pytest

from edge.config import load_config, repo_path

ULTRALYTICS = pytest.importorskip("ultralytics")
pytest.importorskip("tensorflow")

IMU_LOG = repo_path("ML_models/IRI/iri_compliant/delhi_mumbai_expressway.csv")


def _make_video(path, seconds=12, fps=10):
    from ultralytics.utils import ASSETS
    base = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (1280, 720))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for i in range(seconds * fps):
        writer.write(np.roll(base, i * 3, axis=1))             # slow pan so the tracker has motion
    writer.release()


def _make_imu(path, seconds=12, jolt_at=6.0):
    """The first `seconds` of the real log, with a +/-16 m/s2 jolt at the row nearest jolt_at."""
    from edge.util import parse_time
    with open(IMU_LOG, newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        header = next(reader)
        rows = []
        for row in reader:
            if rows and parse_time(row[0]) - parse_time(rows[0][0]) > seconds:
                break
            rows.append(row)
    t0 = parse_time(rows[0][0])
    az = header.index("az")
    i = int(np.argmin([abs(parse_time(r[0]) - t0 - jolt_at) for r in rows]))
    rows[i][az] = str(float(rows[i][az]) + 16.0)
    rows[i + 1][az] = str(float(rows[i + 1][az]) - 16.0)
    with open(path, "w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerow(header)
        writer.writerows(rows)
    return t0


def test_replay_produces_contract_events(tmp_path):
    from edge.agent import EdgeAgent
    from edge.capture.replay import ImuCsvReader, ReplaySource, VideoReader

    video_path, imu_path, out = tmp_path / "ride.mp4", tmp_path / "ride.csv", tmp_path / "out"
    _make_video(video_path)
    t0 = _make_imu(imu_path)
    cfg = load_config()
    cfg.traffic["window_s"] = 4.0
    agent = EdgeAgent(cfg, use_video=True, use_imu=True, realtime=False, out_dir=out)
    summary = agent.run(ReplaySource(VideoReader(video_path, t0), ImuCsvReader(imu_path), rate=0.0))

    assert summary["latencyMs"]["V2"]["runs"] >= 30                    # ~4 FPS over 12 s
    assert summary["latencyMs"]["V1"]["runs"] >= 1                     # the injected jolt ran V1
    assert summary["deferred"]["dropped"] == {}                        # offline replay never drops
    assert summary["events"].get("traffic_sample", 0) >= 2
    assert summary["events"].get("road_shock", 0) >= 1

    envelopes = [json.loads(line) for line in (out / "envelopes.jsonl").read_text().splitlines()]
    events = [e for env in envelopes for e in env["events"]]
    for env in envelopes:
        assert {"schemaVersion", "deviceId", "vehicleId", "tripId", "sentAt", "telemetry", "events"} <= env.keys()
    for e in events:
        assert {"eventId", "eventType", "capturedAt", "location", "confidence", "source", "metadata"} <= e.keys()
    traffic = next(e for e in events if e["eventType"] == "traffic_sample")
    assert traffic["metadata"]["meanCounts"]["person"] > 0             # bus.jpg has people in it
    assert traffic["location"] is not None
