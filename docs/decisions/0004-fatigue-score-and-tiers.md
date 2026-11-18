# 0004 — Fatigue score inputs and tier timing

**Status:** accepted (carried over from the predecessor codebase).

**Constants:** `FATIGUE_W_*`, `PERCLOS_SIGNAL_SATURATION`, `FATIGUE_SUSTAIN_WINDOW_S`, `FATIGUE_WARNING_SUSTAIN_S`, `FATIGUE_ALERT_SUSTAIN_S` (in `acuity/utils/constants.py`) and `_BLINK_WINDOW_MIN_OBSERVED_S` (in `acuity/core/fatigue_engine.py`).

Every figure below comes from one subject on one camera (private corpus, n=1).

## Context

The composite fatigue score is a weighted sum of four signals in [0, 1]: PERCLOS, blink duration, yawning and blink frequency. The warning tier fires when the score holds ≥ 0.6, the alert tier when it holds ≥ 0.8. Four problems kept the score from reaching those tiers on genuinely drowsy stretches, or pushed it there for the wrong reasons.

## Decisions

### Head nod removed from the score

Head nod was a fifth signal at weight 0.10. Recomputed over 24 955 s of logged pitch it fires 292–654 times (up to 94/hour) in sessions where the user was mostly alert, saturating its 3-per-60 s signal 7–15 % of the time. Sweeping does not help: even a 15° drop with a 1.0 s recovery yields 444 events (64/hour). It also became unreachable once the EAR pitch gate (6°) dropped below the nod threshold (8°). The remaining weights were rescaled by 1/0.90 so they still sum to 1.0 and the 0.6/0.8 tiers keep their meaning. Nods are still detected for debug and CSV output.

### PERCLOS ramp tops out at 0.40 (`PERCLOS_SIGNAL_SATURATION`)

The PERCLOS signal ramped from the 0.15 gate to an implicit 1.0. PERCLOS 1.0 is eyes shut for a whole 30 s window — sleep, not drowsiness — so the heaviest-weighted signal spent its life in the bottom third of its range. Over 66 sessions (15.9 h, 770 686 frames):

| | max | p99 | p95 | p50 |
|---|---|---|---|---|
| PERCLOS | 0.816 | 0.407 | 0.217 | 0.079 |
| signal (old ramp) | 0.783 | 0.302 | 0.079 | 0.000 |

A worked example at PERCLOS 0.404 (drowsy by any published criterion): old signal 0.299, contributing 0.116 of a possible 0.389; new signal 1.000. 0.40 is the FHWA/FMCSA severe-drowsiness region — the same literature as the 0.15 gate — so the ramp spans gate → severe instead of gate → asleep.

The structural finding transfers (the old anchor is unreachable for anyone); the exact 0.40 is literature-anchored, not corpus-fitted. Sweeping 0.30–0.45 moves the warning rate only 2.37 → 2.56/hour. Corpus effect: warnings 23 (1.45/h) → 31 (1.95/h).

### Tier sustain is a duty cycle over 30 s (`FATIGUE_SUSTAIN_WINDOW_S`)

`*_SUSTAIN_S` used to mean an uninterrupted run: one frame below the threshold reset the timer. The composite score does not plateau, because its inputs are near-independent (pairwise |r| ≤ 0.10 except blink duration × yawn at +0.37, over 770 686 frames) and take turns rather than stack: 2 of 4 signals are ≥ 0.5 together on 9.3 % of frames, 3 of 4 on 1.9 %, all 4 on 0.00 %. One session spent 31 s above 0.30 in three separate runs, longest 26 s — evidence the old rule discarded.

Sustain is now *accumulated* time above threshold inside a 30 s rolling window. 30 s matches PERCLOS, itself a 30 s duty cycle; with a 15 s sustain the warning tier is "at least half of the last 30 s". The window width is not a sensitive dial (30/45/60 s give identical counts). The change buys robustness to one-frame dropouts: warnings 31 → 38 (1.95 → 2.39/h).

**Banked time across guard pauses.** The bank survives an EAR-guard or face-loss pause shorter than `_SHORT_PAUSE_THRESHOLD_S` (15 s) and is cleared only by a longer one. Clearing it on every paused frame reintroduced the reset the window removes: the guards pause EAR ~250 times an hour (median 1.1 s), and a stretch held at 0.66–0.78 for 27 s lost its bank five times and never warned. Paused frames bank nothing, and evidence still ages out after 30 s. In a tier-only simulation (16.3 h), clearing after 3, 5 or 15 s, or never, gave identical counts.

### Blink frequency divides by observed time (`_BLINK_WINDOW_MIN_OBSERVED_S` = 30 s)

Blink rate divided blinks in the 60 s window by the whole window, so every paused second read as a second of not blinking. Over 24.2 h of replayed logs, rate/baseline fell with the paused share, not with fatigue:

| Paused share | 0–10 % | 10–20 % | 20–30 % | 30–40 % | 40–50 % | 50–60 % | 60 %+ |
|---|---|---|---|---|---|---|---|
| rate/baseline before | 0.97 | 0.95 | 0.82 | 0.75 | 0.61 | 0.44 | 0.28 |
| after | 0.95 | 1.00 | 0.98 | 0.97 | 0.95 | 0.84 | 0.82 |

It now divides by observed time, and renders no verdict until at least half the window was observed. Together with the banked-time change, replay warnings went 5 → 9 (0.21 → 0.37/h); one of the additions is a stretch where the user reported being fatigued, one followed a live microsleep alert, and one is a scripted-clip weakness (PERCLOS inflated by preceding gaze-down segments) that the old reset had been masking by accident.

## Consequences

- The score can now reach the tiers on broken-up evidence, as drowsiness actually presents.
- The weights themselves are a literature ranking, not a fit. Tune from labels once multi-subject data exists.
