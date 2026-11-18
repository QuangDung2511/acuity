# 0002 — EAR validity guards

**Status:** accepted (carried over from the predecessor codebase).

**Constants:** `EAR_PITCH_GATE_DEG`, `EAR_PITCH_GATE_RELEASE_DEG`, `EAR_YAW_RATE_GATE_DPS`, `EAR_YAW_RATE_SUSTAIN_S`, `EAR_PLAUSIBLE_MAX`, `EAR_GATE_SETTLE_S` (in `acuity/utils/constants.py`) and `_PITCH_GATE_WARN_S` (in `acuity/core/fatigue_engine.py`).

Every figure below comes from one subject on one camera (private corpus, n=1).

## Context

EAR (eye aspect ratio) is only meaningful while the face mesh is actually describing an eye. Two things break that without the face disappearing:

- **Foreshortening.** Looking down narrows the eyelids geometrically, so EAR falls with no change in alertness.
- **Mesh failure.** A fast head turn leaves the landmarks unsolved; they read garbage and take time to re-converge.

All four guards below route through the same pause branch as face loss: EAR, blink and microsleep state stop accruing while any of them is engaged.

## Decisions

### Pitch gate: engage 6° below baseline, release at 4°

Measured share of face-tracked frames under the adaptive blink threshold, per degree head-down from the calibrated baseline (25 sessions, 295 728 frames):

| ° down | 0 | 2 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| below threshold | 4.9 % | 8.1 % | 15.4 % | 17.5 % | 20.4 % | 17.7 % | 19.9 % |

The rate climbs to 6° and then sits on a flat plateau at ~4× neutral. History: 25° (sat above the whole degradation band; 32 of 81 microsleep-length runs occurred between 3° and 15° down) → 12° (left the plateau live) → 6°. Gating at 6° suppresses 5.8 % of face-tracked frames. Going lower is not worth the lost observation: 4° costs 14.2 %, 3° costs 25.3 %.

The 2° hysteresis band: frame-to-frame pitch jitter is p50 0.06° / p90 0.38° / p95 0.66°, so 2° is 3× p95. Without it a head dwelling near the line flaps the gate, and every flap clears and rebuilds the EAR, blink and microsleep state.

Rejected:

- **Correcting EAR by cos(pitch).** The EAR/pitch relationship is not clean foreshortening (EAR dips near 5–10° then recovers at steeper angles), so dividing over-corrects. When the head is that far down, EAR is simply not trustworthy.
- **A body-relative gate.** The gate answers a camera-geometry question — can the camera resolve the eye? — so it is relative to the camera, not the torso. Leaning over paper foreshortens the eye as much as bending the neck.

The gate is in degrees below the user's *own* baseline, so camera height and habitual posture normalise out. The knee angle itself depends on eyelid geometry (deep-set eyes, epicanthic fold, lash occlusion) and camera offset, and needs multi-subject re-derivation. The debug CSV already logs per-frame pitch and EAR, so no adaptive machinery is built on n=1.

**Stuck-gate warning (`_PITCH_GATE_WARN_S` = 120 s).** A baseline that no longer describes how the user sits would park them permanently gated, with EAR-based detection silently off. Under the 6°/4° gate, real gated stretches run p50 1.5 s · p95 11 s · p99 27 s · max 44 s (713 stretches), so 120 s only trips when the baseline has drifted.

### Yaw-rate gate: 60°/s sustained for 0.10 s

Traced from a labelled false microsleep: the user turned ~60° and swung back. Mid-turn one eye read EAR 0.97 (physically impossible, a degenerate mesh); on the way back the landmarks had not re-converged, so both eyes sat well under the adaptive threshold for 2.5 s. Pitch normalised first, releasing the pitch gate, and the very next frame was trusted. The composite score was 0.31 — the scored model was right; the microsleep override fired anyway.

Share of frames under the adaptive threshold (62 sessions, 351 794 frames):

| Condition | Share below threshold |
|---|---|
| static \|yaw\| 0–5° / 15–20° / 40–45° | 9.3 % / 5.5 % / 5.5 % |
| yaw rate at rest / 25°/s / 100°/s / 150°/s | 7.2 % / 16.5 % / 24.8 % / 28.7 % |
| time since gate release < 0.5 s / 0.5–1 s / 1–2 s / 2–4 s / > 4 s | 15.0 % / 9.5 % / 9.2 % / 8.3 % / 7.5 % |

Static yaw does **not** degrade EAR, so a static yaw gate was measured and rejected. The transient breaks the mesh, so the guard is on angular velocity. 60°/s sits at the foot of the rate curve's climb and well above ordinary fidgeting.

**Sustain (`EAR_YAW_RATE_SUSTAIN_S`).** 42 % of the 1 784 excursions above 60°/s are one frame long — the mesh re-solving, not the head moving. A real turn sustains it (51 % run 3+ frames). Charging a settle window to a one-frame blip is not free: it removes observation time from the independent-PERCLOS denominator and pushed a scripted genuine closure below the 10 s floor.

0.05 s was sized for 30 FPS (two intervals). The pipeline actually runs at ~15 FPS (deciles 7–23 FPS), where one interval clears 0.05 s alone. Guard onsets over 24.2 h of logs, by fast intervals at engagement:

| Sustain | Onsets/h | 1 interval | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| 0.05 s | 282 | 5 232 | 1 600 | 0 | 0 |
| 0.10 s | 163 | 81 | 2 807 | 1 005 | 40 |
| 0.15 s | 90 | 18 | 196 | 1 313 | 641 |
| 0.20 s | 55 | 0 | 67 | 498 | 772 |

0.10 s is the smallest value that removes single-interval onsets. Two-interval runs are real movement: 1 867 of 1 984 travel one way (median net 9.2°). Replay: EAR suppression 17.7 % → 16.1 % of face time; the traced false microsleep stays suppressed at every value in the table. At 7 FPS one interval (143 ms) trips it alone — correctly, since 60°/s over 143 ms is 8.6° of real travel.

The frame-rate finding is not n=1: any user below ~20 FPS was getting single-interval behaviour.

### Plausibility cap: EAR > 0.45 per eye

An EAR above this is a collapsed mesh, not a wide eye: the artefact frames above reached 0.97 against a resting median near 0.25. Applied per eye, since a turn degrades the far eye first.

### Settle window: 0.75 s after every guard clears

The release-recency row above puts the elevated period at under ~1 s. Costs 1.3 % of observation time alone; 5.1 % together with the yaw-rate and plausibility guards. Replaying 62 sessions through the microsleep path: fires 11 → 9, removing the traced event and one sibling, and leaving all 8 fires in quiet conditions (no gating, peak yaw rate 7–50°/s in the preceding 5 s) untouched.

## Consequences

- EAR-based signals are paused 5–6 % of face time for the pitch gate and ~5 % for the motion guards.
- The pitch knee and the yaw-rate thresholds are n=1 and need multi-subject data. The debug CSV logs everything needed to redo the measurements.
