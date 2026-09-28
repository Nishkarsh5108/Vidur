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
