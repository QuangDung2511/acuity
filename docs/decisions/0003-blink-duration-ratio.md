# 0003 — Blink duration: baseline-relative ratio

**Status:** accepted (carried over from the predecessor codebase).

**Constants:** `_BLINK_DURATION_FATIGUE_RATIO` = 1.8 and `_BLINK_DURATION_BASELINE_MAX_MS` = 600 (in `acuity/core/fatigue_engine.py`).

Every figure below comes from one subject on one camera (private corpus, n=1).

## Context

The blink-duration signal used absolute anchors from the literature: 150 ms → 0.0, 500 ms → 1.0. Those figures describe physiological full closure. The engine measures something else: the full width of a sub-threshold stretch at 0.70 × baseline EAR on a 0.10 s median-filtered signal. Across 16 sessions the per-session resting median ran 195–743 ms — above the "normal" anchor in every session — so the signal carried a permanent positive floor and pinned at 1.00 on 13 of 20 logged fatigue events.

## Decision

Blink duration is judged against the user's own baseline, captured early in the session, like every other threshold in the engine. The signal reaches 1.0 when the rolling duration rises to **1.8×** baseline.

**Median on both sides.** The ratio was first derived with a *mean* on the signal side against a *median* baseline. Durations are right-skewed, so that pairing started each session above zero; in one session the signal sat at 1.76× (near full scale) from two minutes in, and three ordinary blink-rate dips each raised a warning. Baseline capture, refinement and the signal now all use the median of the duration buffer.

**Re-derivation.** Replaying 144 sessions with ≥ 5 min after capture, anchoring on the captured baseline and measuring how far the rolling median later rose:

| Ratio to anchor | median | p75 | p90 | max |
|---|---|---|---|---|
| session p90 | 1.19 | 1.38 | 1.72 | 2.32 |
| session p99 | 1.28 | 1.51 | 1.93 | 2.46 |
| session max | 1.32 | 1.54 | 1.93 | 2.63 |

At 1.8 a typical session peaks at signal 0.40, the p75 session at 0.68, and 19 of 144 saturate; 1.5 would saturate 39. The literature-derived 3.3× (150 → 500 ms) would leave the signal dead.

**Evaluation.** On a labelled evaluation set (12 sessions, 12.5 h), switching mean → median:

- labelled false warnings 7 → 0 (all 7 still fired under the mean);
- labelled correct alerts 4 → 4 (3 sleep alerts on time; one microsleep is withheld for lack of context, and the same doze fires as a 6.3 s sleep 31 s later).

**Baseline ceiling: 600 ms.** A user who is already drowsy, or whose tracking is degraded, during baseline capture would bake in a long reference and suppress the signal for the session. 3 of 16 sessions captured anchors above 690 ms and then *improved* (later/anchor 0.90–1.19), which is that signature; other anchors sat near 430 ms. This is the weakest-evidenced of the blink-duration values; revisit with data.

## Consequences

- With the false floor gone, the score never reached the warning tier in the 12.5 h evaluation set, so no warning and no in-context microsleep fired — only sleep. **Warning-tier recall is unmeasured.**
- All four positives in the evaluation set come from a single day. n=1, one subject and one camera.
