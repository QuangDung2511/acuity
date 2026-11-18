# 0001 — Microsleep and sleep gates

**Status:** accepted (carried over from the predecessor codebase).

**Constants:** `MICROSLEEP_THRESHOLD_S`, `MICROSLEEP_SHUT_RATIO`, `MICROSLEEP_BILATERAL_SHARE`, `MICROSLEEP_BLINK_CONFIRM`, `MICROSLEEP_CONTEXT_S`, `SLEEP_CLOSURE_S`, `PERCLOS_GATE`, `PERCLOS_WINDOW_S` (in `acuity/utils/constants.py`) and `_PERCLOS_MIN_INDEPENDENT_S` (in `acuity/core/fatigue_engine.py`).

Every figure below comes from one subject on one camera (private corpus, n=1). The structural arguments transfer; the exact values need re-measuring on other people.

## Context

Acuity's contract is: **warn** on signs of fatigue, **alert** when the user sleeps or microsleeps. A microsleep alert bypasses the warning tier, so a false one costs a broken focus block and a rest prompt. A desk is not a car: a missed 2 s microsleep costs a re-read paragraph. The gates below therefore favour specificity, and each one was added after a specific false positive was traced in the logs.

A candidate episode is a stretch where the smoothed `ear_avg` is under the blink threshold. To alert, it must pass, in order:

1. **Duration** of genuinely *shut* time ≥ `MICROSLEEP_THRESHOLD_S`.
2. **Bilateral** — both eyes closed for enough of the episode.
3. **Blink confirmation** — the blendshape says the lids are shut, not narrowed.
4. **PERCLOS gate** — sustained closure in the surrounding window, excluding the episode itself.
5. **Fatigue context** — or, failing that, the episode is long enough to be sleep.

## Decisions

### Duration: 1.5 s (`MICROSLEEP_THRESHOLD_S`)

The academic blink/microsleep boundary is 0.4 s, but natural extended blinks reach 0.3–0.8 s and 0.4 s is a dozen frames of landmark jitter. 1.0 s was tried first. Of 81 closed-eye runs longer than 1.0 s, 23 fell in the 1.00–1.25 s band and were slow blinks and downward glances, not drowsiness. 1.5 s is still inside the clinical microsleep range (1.5–3 s).

### What counts as shut: 0.50 × blink threshold (`MICROSLEEP_SHUT_RATIO`)

The timer used to count every frame under the *blink* threshold, so a slow partial reopening or a held squint accrued microsleep time at 30–40 % aperture with the eyes plainly open. One labelled false alert had 0.2 s of real closure followed by 1.4 s at roughly half the threshold.

Longest contiguous genuinely-shut stretch in the four alerts the user labelled false: 0.00, 0.00, 0.00 and 0.15 s. Genuine long closures carry 1.4–5.6 s of shut time.

| Shut ratio | Episodes qualifying (1.5 s contiguous) | Labelled false positives |
|---|---|---|
| 0.40 | 14 | 0 |
| 0.50 | 19 | 0 |
| 0.60 | 24 | 0 |
| 0.70 | 32 | 1 |

0.50 sits mid-band. End-to-end replay takes fires 16 → 5, with none of the four labelled false positives surviving.

Rejected:

- **Raising the duration threshold.** Survivors always cluster just above the cut (1.75 s → all 1.8–1.9 s; 2.0 s → all 2.0–2.1 s). It moves the cut instead of measuring the right quantity.
- **A composite-score floor.** All surviving fires scored 0.03–0.19, so any meaningful floor disables the override outright.
- **Requiring nearby corroborating closures.** The false positives had 0, 0, 2 and 4 other ≥ 1.0 s closures within 5 minutes: no split.

Caveat: this cuts microsleep fires by about 69 %, and the corpus has no user-confirmed genuine microsleep, so the recall cost is unmeasured. It is justified on specificity and on physiology: a microsleep is the eyes shut, not narrowed.

### Bilateral share: 0.60 (`MICROSLEEP_BILATERAL_SHARE`)

The eyes-closed test is on `ear_avg`, so one eye is enough to clear it: a wink, or one eye's landmarks collapsing, drags the mean under while the other eye is open. A deliberate one-eye-at-a-time test fired an alert on the fourth wink, and a labelled false alert had one eye well under the threshold and the other just above it.

Real closure is bilateral. Across 4 243 closure episodes the two eyes cross the threshold within 100 ms of each other 94.8 % of the time (p50 onset lag: same frame; p99 0.215 s). Of 79 microsleep-length episodes, genuine ones have a median both-eyes share of 1.00 (p05 0.72); the five labelled artefacts sit at 0.00, 0.18, 0.33, 0.47 and 0.57.

| Share ≥ | Artefacts firing | Genuine lost |
|---|---|---|
| 0.50 | 1/5 | 1/74 |
| 0.60 | 0/5 | 2/74 |
| 0.70 | 0/5 | 3/74 |

End-to-end: fires 16 → 15, artefacts 2 → 0, genuine detections 14 → 15. Genuine detections go *up*, because suppressing an artefact frees the 60 s cooldown that a real closure was being swallowed by.

Rejected:

- **Bilateral PERCLOS.** Costs 3 genuine detections for no further artefact benefit, so PERCLOS stays on `ear_avg`.
- **A per-frame both-eyes test on `below`.** It re-measures each episode from the second eye closing to the first reopening and fragments episodes on single-frame de-synchronisation. It dropped 14 genuine closures, all 1.50–1.65 s, pushed under the duration threshold. Qualifying the whole episode costs none of them. Do not "simplify" to this.

Caveat: the populations overlap slightly (highest artefact 0.57 against the 0.60 line), and three of the five artefacts come from the one deliberate wink test.

### Blink confirmation: median eyeBlink ≥ 0.72 (`MICROSLEEP_BLINK_CONFIRM`)

EAR is a 2-D distance ratio, so a lid lowered by *gaze* looks identical to a lid lowered by *closure*. In a scripted clip with the head held level and the eyes on the keyboard, aperture fell to 13 % of open (inside the 5–10 % band genuine closures occupy) while head pitch stayed 2–11°, never reaching the pitch gate. That episode ran 7.2 s of qualifying microsleep time with PERCLOS 0.69 and no guard engaged.

Two geometric discriminators were measured and **rejected**:

| Feature | Gaze-down | Genuine closure |
|---|---|---|
| iris vertical position / eye width | +0.037 | +0.020 … +0.033 |
| iris diameter / eye width | 0.376 | 0.360 … 0.373 |

Both fail for the same reason: MediaPipe drags the iris landmarks down under a closing lid and fits a fixed-size iris regardless of occlusion. The `eyeBlinkLeft/Right` blendshape works because it is a learned appearance signal, not more of the same geometry.

Per-episode median eyeBlink over 21 shut episodes ≥ 0.8 s in the scripted clip: genuine closures (12) 0.738–0.794; awake (9, squints and gaze-down) 0.588–0.700. No overlap; 0.72 sits mid-gap. Cost of enabling blendshapes: median detection time 9.08 → 9.27 ms per frame.

The gate **fails open**: an episode with no blendshape samples is let through, so a model that reports none cannot silently disable microsleep detection.

Caveat: one recording session, a 0.038 margin, and the threshold is an absolute blendshape score — unlike the EAR thresholds it is not normalised per user. Re-measure before trusting it on anyone else.

### Fatigue context: 120 s (`MICROSLEEP_CONTEXT_S`)

The eyes cannot separate a microsleep from drinking. A labelled false alert was the user drinking: blendshape 0.75–0.79 (inside the genuine band), aperture ~8 % of open, bilateral share 1.00, no guard engaged — the eyes really were shut for 2.3 s. What disagreed was everything else: fatigue score 0.41, and every earlier microsleep fire had scored ≤ 0.50. Real microsleeps arrive in the middle of drowsiness; drinking arrives while alert.

So a microsleep that passes every gate above still alerts only if the score is at the warning tier, or a warning fired within the last 120 s. Otherwise it is logged (`microsleep_withheld = "context"`) and nothing is shown. The uncorroborated case is dropped rather than downgraded to a warning toast.

120 s is a product choice, not a fit: "a warning in the last two minutes means the user is in a fatigued stretch". It deliberately exceeds the 30 s warning cooldown so a microsleep between two warnings qualifies.

Caveat: no user-confirmed genuine microsleep exists in the corpus, so the recall cost is unmeasured. A genuine closure in an alert-looking user is caught only if it reaches the sleep threshold.

### Sleep: 5.0 s shut, no context (`SLEEP_CLOSURE_S`)

Someone who dozes off without prior warning signs has a low score and low prior PERCLOS, so neither can be asked to corroborate. Only the bilateral and blink gates apply; the blink gate keeps a long lowered gaze out (the 7.2 s gaze-down episode above is refused at a blendshape median of ~0.69 and fires nothing).

Shut-episode distribution over the replayed logs (188 sessions, 30.7 h): in sessions logging the blendshape (29 sessions, 6.5 h) the longest gate-passing episode is 3.79 s, and the drinking episode 2.60 s. Older sessions without the blendshape (gate fails open) hold 13 episodes of 5.0–9.5 s of unknown nature. 5.0 s clears everything seen with blink evidence by 1.2 s. Replay fires 8 sleep alerts (0.26/h), all in the older sessions, so that is an upper bound; one is very likely genuine (a 9.5 s closure while the user reported being fatigued).

Caveat: there is no recorded, labelled sleep episode, so recall is unmeasured. A deliberate long eye-rest (eyes closed 5 s+ on purpose) is indistinguishable and **will** alert as sleep. Record a scripted clip (eyes shut 10 s, eye-rest, head droop) before moving this value.

### PERCLOS gate: 0.15 over 30 s, excluding the episode (`PERCLOS_GATE`)

A microsleep fires only if the eyes were under the blink threshold for at least 15 % of the last 30 s (the FMCSA figure). The gate is evaluated **excluding the candidate episode's own closed time** (`_perclos_excluding`).

Measured against the full window it is partly self-confirming: a 1.5 s closure contributes 1.5/30 = 0.05 of the 0.15 it must clear, and contributes more the longer it runs. 10 of 13 microsleep fires in the older logs landed in a 0.150–0.174 band, four at exactly 0.150: the signature of a tripwire tripped by its own trigger. Of the 7 whose window could be reconstructed, removing the episode's own contribution drops 4 below the gate. The fatigue score at fire never exceeded 0.50 in any of the 13.

The 0.15 value is on notice: the per-session resting closed fraction ranges 0.053–0.305, tracking the blink threshold, so an absolute gate is a poor fit for a per-user quantity. Revisit as a relative gate if false positives persist.

### Independent observation floor: 10 s (`_PERCLOS_MIN_INDEPENDENT_S`)

Because the gate excludes the episode, a window holding little else than the closure supports no verdict either way. A logged session fired at t = 1.06 s with PERCLOS 1.000: a blink as calibration ended, with the whole window inside it. The generic 1.0 s observation floor is far too weak for that. 10 s keeps microsleep available early in a focus block (each block re-calibrates and resets the engine) while making "sustained over the last 30 s" mean something.

## Consequences

- Microsleep fires dropped sharply in replay; every labelled false positive in the corpus is suppressed.
- Recall for genuine microsleep and sleep is **unmeasured**. The next evidence to collect is a labelled genuine episode.
- The blink-confirmation threshold is the least portable value here: absolute, n=1, one session.
