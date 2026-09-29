# Decision log

*Decisions that shape the edge system, with the reasoning behind them. Newest last. Each links to the document with the detail.*

---

### D1. Edge hardware: a Raspberry Pi 5 CPU as the baseline, the AI HAT+ as an option (28 Sep 2026)

**Decision.**
- Design for one Raspberry Pi 5 (8 GB, Active Cooler) per bus, CPU only.
- Treat the AI HAT+ 26 TOPS (Hailo-8) as an optional upgrade for more cameras and a continuous V1 scan.
- Buy hardware only after the idea round.

**Why.**
- Gating keeps the estimated load at about 40–60% of the CPU.
- "Runs on a bare Pi 5" is the stronger feasibility claim.
- The CPU pipeline is the fallback if the Hailo compile toolchain causes trouble.

**Detail:** [edge-deployment.md §4–5](edge-deployment.md).

### D2. The models are combined as a gated two-lane cascade, not merged into one network (28 Sep 2026)

**Decision.**
- **Real-time lane:** V2, S1 and S2.
- **Deferred lane:** V1 on the frames before each IMU jolt, and V3 once per sign on crops. It runs from a bounded priority queue on its own cores.

**Why.**
- The training sets label disjoint classes, so a merged model would learn unlabelled objects as background.
- The models need different inputs at different rates.
- A cascade can skip its expensive stages; a merged network cannot.

**Detail:** [edge-deployment.md §2](edge-deployment.md); code in [edge/](../edge/).

### D3. Quantisation: NCNN FP16 with rectangular inputs on the CPU; INT8 only on Hailo (28 Sep 2026)

**Decision.**
- Export V1, V2 and V3 to NCNN FP16 at 480×800, 384×640 and 320×320.
- Leave S1 (already weight-quantised) and S2 as they are.
- Use INT8 only for a Hailo accelerator, calibrated on our own footage.

**Why.**
- FP16 costs nothing in accuracy on the Pi 5.
- The rectangular export cut V1's time by 36% (measured).
- Ultralytics' NCNN export has no INT8, and INT8 on the CPU is an unproven gain on already-weak models.

**Detail:** [edge-deployment.md §3](edge-deployment.md).

### D4. The demo is a laptop replay of recorded rides, clearly labelled (28 Sep 2026)

**Decision.**
- The demo runs the edge agent on a laptop, replaying recorded rides.
- Pi 5 figures are presented as estimates, with their derivation, until measured on hardware.

**Why.**
- The problem statement is software-first. A replay is reproducible on stage.
- Every claim stays true as stated. The HUD video is labelled "not a Pi 5 measurement".

**Detail:** [edge-deployment.md §6–8](edge-deployment.md).

### D5. V1 stays for the demo; a nano successor is built in October (28 Sep 2026)

**Decision.**
- The current V1 remains the pothole and zebra-crossing model.
- A nano road-surface model (with cracks added) is trained from it in October.
- The successor is adopted only if it beats V1 on a frozen test set and on event recall.

**Why.**
- V1 is too heavy to run continuously on a Pi 5.
- V1 cannot be reproduced from the repository.
- A nano model makes a continuous scan affordable.

**Detail:** [v1-successor-plan.md](v1-successor-plan.md).

### D6. Until S2's files are added, V1 is triggered by a jerk threshold (28 Sep 2026)

**Decision.**
- When S2's TFLite file is missing, the agent triggers V1 on peak-to-peak vertical acceleration ≥ 9 m/s² with a crest factor ≥ 4 in a 1.28 s window.
- S2 is used automatically once its file is present.

**Why.**
- S2's model files are not in the repository.
- S2's classes are relative jerk quartiles anyway (see [ml-models-spec.md §7](ml-models-spec.md)).
- The thresholds were calibrated on the team's field logs: 0.26 triggers per km on 110 km of the KMP expressway.

### D7. Repository hygiene (28 Sep 2026)

**Decision.**
- The three demo checkpoints are versioned in git (each under GitHub's 100 MB limit), so a clone runs as it is.
- `ML_models/` is kept as delivered, apart from two fixes: V1's `severity.py` import path, and its `ultralytics` version pin.
- Scratch copies, duplicate outputs, bytecode and superseded plans are ignored by git.

### D8. Demo videos without their own GPS are paired with real GPS traces, not invented ones (29 Sep 2026)

**Decision.**
- For demo videos with no GPS of their own, a per-frame GPS trail is built by pairing the video with a stretch of one of the team's own recorded drives (`tools/sim`), using that stretch's real road geometry and real speed profile.
- School zones for the pedestrian-alert demo come from real OpenStreetMap `amenity=school` polygons along those routes, not hand-picked coordinates.
- Every generated trail records which trace it came from and that the pairing is simulated, and [tools/sim/README.md](../tools/sim/README.md) says so up front.

**Why.**
- A synthetic trail (e.g. a straight line at constant speed) would be an obviously fake track on the dashboard's map and would defeat the point of demoing real map/GIS behaviour.
- Real speed profiles and real school locations are the parts of the demo that are honest to present as real; only the video-to-road *pairing* is a simulation, and that is exactly what should be labelled.
- One BeamNG clip (`BUS_IRI_01`) already has synced IMU from [tools/make_sim_clip.py](../tools/make_sim_clip.py); its trail moves at that IMU log's own speed profile ("distance mode") so the two stay consistent, instead of at the borrowed trace's speed.

**Detail:** [tools/sim/README.md](../tools/sim/README.md).

### D9. The edge agent posts directly to the backend's event contract, not just its own (29 Sep 2026)

**Decision.**
- `edge/backend.py` translates every edge event into the backend team's simpler contract ([docs/hello.md](hello.md)) and uploads it to `POST /api/v1/ingest` from a background thread, with retries and the same event IDs on every retry (so a retried batch is reported as duplicates, never double-stored).
- This runs alongside, not instead of, the edge's own richer `envelopes.jsonl` log.

**Why.**
- The backend's contract is intentionally simpler than the edge's internal one (flat `metadata`, different event-type names), and it's the backend team's contract to evolve — the edge adapts to it, not the other way round.
- Keeping the retry queue on a background thread means a slow or unreachable backend never stalls the cameras or the IMU stream.
- Verified end-to-end (29 Sep, laptop): all four demo buses run through `tools/sim/run_fleet.py` against a live backend + MongoDB, producing pothole/zebra/sign/road-shock/IRI/pedestrian-alert events, aggregated issues, and populated KPIs — not just unit-tested in isolation.

**Update (same day):** media upload was added to both sides (`POST /api/v1/media` on the backend, `edge/backend.py` uploading the local crop and rewriting `metadata.mediaId`/`mediaUrl` to the backend's own id) — see D11.

### D10. The ops dashboard is a no-build-step static page, not the Vite/React stack originally sketched (29 Sep 2026)

**Decision.**
- Built as plain HTML/CSS/ES modules with MapLibre GL JS vendored locally, served by the FastAPI backend itself (`StaticFiles` mounted at `/`) — see [dashboard/README.md](../dashboard/README.md).
- `docs/dashboard-backend-design.md §5` originally specified `dashboard/ (Vite React TS: pages/, layers/, api/, ws/)`; this replaces that plan, not extends it.

**Why.**
- Node/npm were available in this environment, so this wasn't a fallback for lack of tooling — it was a direct trade for a hackathon demo: zero build step means no dev server to keep alive on stage, no build/commit drift, and the whole app is inspectable by opening the files.
- MapLibre's native `heatmap`/`circle`/`line`/`fill` layer types cover every visualisation the dashboard needed (density heatmap, severity-coloured pins, school-zone polygons, bus trails), so deck.gl was never actually necessary.
- Raster basemap tiles (CARTO Positron/Dark Matter) need no glyph/sprite server, unlike a full vector style — one less moving part for an offline-leaning demo.
- The six-view single-page layout (Overview/Road/Traffic/Infrastructure/Safety/Fleet, one map, one context panel) consolidates the original multi-page design (Command Center, Road Health, Infrastructure, Traffic, Safety, Review Queue, Fleet & Edge) into what a judge actually clicks through in one sitting; per-view layer whitelists are what keep it from being a cluttered wall of pins.

### D11. Backend-supervised "Simulate" button, not a separate orchestration service (29 Sep 2026)

**Decision.**
- `POST /api/v1/simulate/start` on the FastAPI backend launches `tools/sim/run_fleet.py` as a subprocess (`--backend` always forced to `127.0.0.1`, since the bus processes always run on the same machine as the backend), tracks it in an in-memory singleton, and exposes `/status` (with a live log tail) and `/stop` (kills the whole process tree).
- Media upload was wired at the same time: `edge/backend.py`'s `BackendUploader` now uploads each event's local crop to `POST /api/v1/media` in its own background thread and rewrites the event's `mediaId`/`mediaUrl` to the backend's own content-addressed id before sending the event — closing the loop so the dashboard's issue-detail photos are real, not a placeholder.

**Why.**
- The dashboard's browser JS cannot launch a local OS process; something server-side has to, and the backend already runs on the same laptop as the edge simulation, so it's the natural place rather than a third service.
- State lives in memory, not a database: this is a demo control surface for one laptop, not a job queue, and the README says so (`backend/backend/README.md §7`).
- Verified: `/simulate/start|status|stop` end-to-end against a live 4-bus run (not just the mocked pytest suite) — 273 events, 4 vehicles, 20 issues, 25 photos uploaded and fetched back correctly.

### D12. Bottleneck and missing-signboard detection are computed on demand, with their approximation stated in the response (29 Sep 2026)

**Decision.**
- `GET /api/v1/traffic/bottlenecks`: buckets `traffic_density` events into a plain lat/lon grid (not H3) and compares each cell's mean bus speed against *this session's own* 85th-percentile cell speed (not a long-run free-flow baseline) — both approximations spelled out in the response's `note` field.
- `GET /api/v1/infrastructure/missing-signs`: flags an open sign issue only once a *different trip* (`tripId` not already on the issue) has passed within its radius without reconfirming it — not just a later timestamp, which a first pass of this logic got wrong (see below).

**Why.**
- The full design (`docs/dashboard-backend-design.md §4`) needs an H3 dependency and days of historical data this MVP has neither of; approximating both from the current session, and saying so in every response, is more honest than silently returning nothing or dressing up a rough number as the real thing.
- **Bug caught in testing:** the first version of the missing-sign check used "any later nearby event," which flags almost every sign as missing, because the *same* bus keeps logging other events (traffic samples, IRI windows) a few seconds and metres further on while still driving past the sign it just saw. Requiring the nearby event's `tripId` to be one that never contributed to that issue fixes it — proven by `test_the_same_trip_driving_further_does_not_count_as_a_revisit` in `backend/backend/tests/test_analytics_extra.py`, and by the live 4-bus run correctly returning zero flagged signs (every bus in the demo drives its route once).

**Detail:** `backend/backend/README.md §4a–§4b, §7`.
