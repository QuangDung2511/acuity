# Global constants: thresholds, bounds and defaults.
# ─────────────────────────────────────────────────────────────────
# Every threshold carries its evidence: a summarised measurement, or
# "unmeasured default, tune from labels".  Measurements marked
# "(private corpus, n=1)" come from one subject on one camera.  The long
# tuning histories behind the fatigue thresholds live in docs/decisions/.
#
# Thread: any (values only, no logic).
# ─────────────────────────────────────────────────────────────────

# ── Application Metadata ────────────────────────────────────────
APP_NAME = "Acuity"
APP_ORG = "Acuity"
APP_VERSION = "1.0.0"

# ── Calibration ─────────────────────────────────────────────────
# Every focus block starts with a mandatory 10-second silent
# calibration that captures the per-user baselines.
CALIBRATION_DURATION_S = 10  # seconds

# ── Focus / Distraction — Head Pose (Trigger A) ────────────────
# "Bounds: Min 5s | Max 300s  |  Initial Setting: 15s"
HEAD_POSE_THRESHOLD_MIN_S = 5  # seconds
HEAD_POSE_THRESHOLD_MAX_S = 300  # seconds
HEAD_POSE_THRESHOLD_DEFAULT_S = 15  # seconds

# ── Desk Writing Mode ──────────────────────────────────────────
# When enabled, suppresses head-pose distraction (Trigger A) and
# PRESENT_NO_FACE distraction timing so notebook / desk work
# does not generate false alerts.
DESK_WRITING_MODE_DEFAULT = False

# ── Fatigue — Microsleep ────────────────────────────────────────
# The gates below are applied in order to a candidate eyes-closed
# episode.  Full measurements, sweeps and rejected alternatives:
# docs/decisions/0001-microsleep-and-sleep-gates.md.

# Microsleep = at least this much continuous *shut* time (see
# MICROSLEEP_SHUT_RATIO).  The academic 0.4 s boundary is hit by extended
# blinks and landmark jitter; at 1.0 s, 23 of 81 long closures were slow
# blinks and downward glances in the 1.00–1.25 s band (private corpus,
# n=1).  1.5 s is still inside the clinical microsleep range (1.5–3 s).
MICROSLEEP_THRESHOLD_S = 1.5  # seconds — eyes-closed duration

# Fraction of the blink threshold below which the eye counts as SHUT.
# MICROSLEEP_THRESHOLD_S is measured against *this*, not against the blink
# threshold — read the two together.  Counting every frame under the blink
# threshold let held squints at 30-40% aperture accrue microsleep time.
# The four alerts the user labelled false had at most 0.15 s of contiguous
# shut time; genuine long closures carry 1.4-5.6 s.  0.50 sits mid-band of
# a 0.40-0.60 sweep with no false positives, and cuts replayed fires 16 -> 5
# (private corpus, n=1).
#
# CAVEAT — recall is unmeasured: the corpus holds no user-confirmed genuine
# microsleep.  Justified on specificity and physiology: a microsleep is the
# eyes shut, not narrowed.
MICROSLEEP_SHUT_RATIO = 0.50  # x blink_threshold — below this the eye is shut

# Fraction of a candidate microsleep episode that must have BOTH eyes below
# the blink threshold.  The eyes-closed test is on ``ear_avg``, so one eye
# (a wink, or one eye's landmarks collapsing) is enough to clear it.  Real
# closure is bilateral: across 4 243 closures the eyes cross within 100 ms
# of each other 94.8% of the time; genuine microsleep-length episodes have
# a median share of 1.00 (p05 0.72) and the five labelled artefacts
# 0.00-0.57.  At 0.60: 0/5 artefacts fire, 2/74 genuine are lost (private
# corpus, n=1).
#
# DO NOT "simplify" this into a per-frame both-eyes test on ``below``: that
# variant fragments episodes and dropped 14 genuine closures.  See the ADR.
#
# CAVEAT — the populations overlap slightly (top artefact 0.57), and three
# of the five artefacts come from one deliberate wink test.
MICROSLEEP_BILATERAL_SHARE = 0.60  # fraction of episode frames with both eyes closed

# ── Fatigue — Microsleep Blink Confirmation ─────────────────────
# Median of MediaPipe's eyeBlinkLeft/Right blendshapes across a candidate
# microsleep episode.  Below this the episode is refused: the lids are
# narrowed, not shut.
#
# EAR cannot make this distinction: a lid lowered by *gaze* looks identical
# to a lid lowered by *closure*.  Two iris-geometry discriminators were
# measured and REJECTED — do not re-try them (see the ADR).  Over 21 shut
# episodes >= 0.8 s in a scripted clip, genuine closures median 0.738-0.794
# and awake episodes (squints, gaze down) 0.588-0.700: no overlap, 0.72
# sits mid-gap (private corpus, n=1).  Enabling blendshapes costs ~0.2 ms
# per frame.
#
# The gate FAILS OPEN.  An episode with no blendshape samples is allowed
# through, because a model that reports none must not silently disable
# microsleep detection altogether.
#
# CAVEAT — one recording session, a 0.038 margin, and an absolute score NOT
# normalised per user like the EAR thresholds.  Re-measure before trusting
# it on anyone else.
MICROSLEEP_BLINK_CONFIRM = 0.72  # median eyeBlink blendshape over the episode

# ── Fatigue — Microsleep Context and Sleep ──────────────────────
# Acuity's contract: WARN on signs of fatigue, ALERT when the user sleeps
# or microsleeps.  A microsleep that passes every gate above still only
# alerts in fatigue context: the score is at the warning tier, or a warning
# fired within MICROSLEEP_CONTEXT_S.  Otherwise it is logged
# (microsleep_withheld = "context") and nothing is shown.
#
# The eyes cannot separate a microsleep from drinking: a labelled false
# alert was the user drinking, with eyes genuinely shut 2.3 s and every eye
# gate passed, but a fatigue score of 0.41 (private corpus, n=1).  Real
# microsleeps arrive in the middle of drowsiness; drinking arrives while
# alert.
#
# 120 s is a product choice, not a fit: "a warning in the last two minutes
# means the user is in a fatigued stretch".  It deliberately exceeds the
# 30 s warning cooldown so a microsleep between two warnings qualifies.
#
# CAVEAT — recall is unmeasured.  A genuine closure in an alert-looking
# user is caught only if it reaches SLEEP_CLOSURE_S.
MICROSLEEP_CONTEXT_S = 120.0  # seconds — a warning this recent is fatigue context

# Eyes shut this long is sleep, and alerts with NO context requirement —
# someone who dozes off without prior warning signs has a low score and low
# prior PERCLOS, so neither can be asked to corroborate.  Only the bilateral
# and blink gates apply; the blink gate keeps a long lowered gaze out.  The
# longest gate-passing episode in sessions logging the blendshape is 3.79 s;
# 5.0 s clears it by 1.2 s (private corpus, n=1).
#
# CAVEAT — recall is unmeasured (no labelled sleep episode), and a
# deliberate long eye-rest WILL alert as sleep.  Record a scripted clip
# (eyes shut 10 s, eye-rest, head droop) before moving this value.
SLEEP_CLOSURE_S = 5.0  # seconds — continuous shut time that alerts as sleep

# Standard EAR threshold for blink detection (MediaPipe 478 landmarks).
# Used as a population-wide fallback floor; the engine derives a
# per-user adaptive threshold from the calibrated baseline_ear.
EAR_BLINK_THRESHOLD = 0.21

# PERCLOS (Percentage of Eye Closure) gate for microsleep confirmation.
# A microsleep event only fires when the eyes have been below the blink
# threshold for at least PERCLOS_GATE fraction of the last PERCLOS_WINDOW_S
# seconds.  This filters isolated episodes (likely false positives) from
# genuinely sustained drowsiness.
# Source: FMCSA research uses 15% as the clinically validated boundary.
#
# The gate is evaluated EXCLUDING the candidate episode's own closed time
# (fatigue_engine._perclos_excluding), and the value below assumes that.
# Against the full window it is partly self-confirming — a 1.5 s closure
# supplies a third of the bar for free — and 10 of 13 older fires landed
# just over 0.15 (private corpus, n=1).
#
# The 0.15 value is on notice: the per-session resting closed fraction
# ranges 0.053-0.305 (private corpus, n=1), so an absolute gate is a poor
# fit for a per-user quantity.  Revisit as a relative gate if false
# positives persist.  See docs/decisions/0001-microsleep-and-sleep-gates.md.
PERCLOS_WINDOW_S = 30.0  # seconds — rolling observation window
PERCLOS_GATE = 0.15  # fraction — minimum closed-eye time to confirm

# Upper anchor of the PERCLOS→signal ramp: PERCLOS_GATE maps to signal 0.0
# and this value maps to 1.0 (see FatigueEngine._perclos_signal).
#
# It used to be an implicit 1.0 — eyes shut for a whole 30 s window, which
# is sleep, not drowsiness — so the heaviest-weighted signal sat at a
# median of 0.000 and delivered ~30% of its weight even at severe closure
# (PERCLOS p99 0.407, private corpus, n=1).  0.40 is the FHWA/FMCSA
# severe-drowsiness region, the same literature as the 0.15 gate.
#
# The *structural* finding transfers (the old anchor is unreachable for
# anyone); the exact 0.40 is literature-anchored, not corpus-fitted, and
# sweeping 0.30–0.45 barely moves the warning rate.  Do not re-tune it off
# a single session.  See docs/decisions/0004-fatigue-score-and-tiers.md.
PERCLOS_SIGNAL_SATURATION = 0.40  # fraction — PERCLOS mapping to signal 1.0

# Minimum cooldown (seconds) between any two fatigue alerts
# (alert tier, contextual microsleep, or sleep).  After any fatigue alert
# fires, none of them can fire again until this period has elapsed.
# Prevents notification spam while keeping the system responsive.
FATIGUE_COOLDOWN_S = 60.0  # seconds — alert + microsleep + sleep cooldown

# ── Fatigue Scoring Model (DMS-Grade) ────────────────────────────
# Four signals are normalized to [0, 1] and combined into a composite
# fatigue score.  Weights must sum to 1.0.
# Source: relative ranking from PERCLOS/blink-duration/MAR literature;
# tune via CSV benchmark data after deployment.
#
# Head nod was REMOVED from the model (was 0.10) and the remaining four
# weights rescaled by 1/0.90 so they still sum to 1.0 — the score stays a
# true [0, 1] and the 0.6/0.8 tiers keep their meaning.  The detector has
# no usable specificity: up to 94 nods/hour in mostly-alert sessions
# (private corpus, n=1), and it became unreachable once EAR_PITCH_GATE_DEG
# dropped to 6°, below HEAD_NOD_PITCH_DROP_DEG.  Nods are still detected
# and reported for debug/CSV purposes; they simply no longer feed the
# score.  See docs/decisions/0004-fatigue-score-and-tiers.md.
FATIGUE_W_PERCLOS = 0.389  # most validated single indicator (FHWA)
FATIGUE_W_BLINK_DUR = 0.278  # strong individual predictor
FATIGUE_W_YAWN = 0.222  # reliable behavioural signal, low FP rate
FATIGUE_W_BLINK_FREQ = 0.111  # supplementary; sensitive to individual variation

# Two-tier alert thresholds.  The tier values, sustain times and warning
# cooldown are unmeasured defaults, tune from labels; the sustain window
# below them is measured.
# Warning (score ≥ threshold, sustained) → gentle nudge, no rest trigger.
# Alert   (score ≥ threshold, sustained) → full alert + Fluid-mode rest.
FATIGUE_WARNING_THRESHOLD = 0.6
FATIGUE_ALERT_THRESHOLD = 0.8
FATIGUE_WARNING_SUSTAIN_S = 15.0  # seconds score must be ≥ WARNING_THRESHOLD
FATIGUE_ALERT_SUSTAIN_S = 10.0  # seconds score must be ≥ ALERT_THRESHOLD
FATIGUE_WARNING_COOLDOWN_S = 30.0  # warning-specific cooldown (shorter = more responsive)
# Note: alert, microsleep and sleep share the existing FATIGUE_COOLDOWN_S (60 s).

# The SUSTAIN_S figures above are *accumulated* time above the threshold
# inside this rolling window — a duty cycle — not one uninterrupted run.
#
# An uninterrupted run demands a plateau the composite score does not
# produce: its four inputs are near-independent (pairwise |r| <= 0.10 bar
# one pair at +0.37) and take turns rather than stack — all four are >= 0.5
# together on 0.00% of frames (private corpus, n=1).
#
# 30 s matches PERCLOS_WINDOW_S, and deliberately so: PERCLOS is itself a
# duty-cycle statistic over 30 s, so the tier test now asks the same
# question of the composite score that PERCLOS asks of eye closure.  With
# SUSTAIN_S 15 s the warning tier is "at least half of the last 30 s".
# The window width is not a sensitive dial: 30/45/60 s gave identical
# replayed counts.
#
# The banked time survives an EAR-guard / face-loss pause shorter than the
# engine's _SHORT_PAUSE_THRESHOLD_S (15 s) and is cleared only by a longer
# one.  Clearing it on every paused frame reintroduced the reset this
# window exists to remove: the guards pause EAR ~250 times an hour (median
# 1.1 s, private corpus, n=1).  Paused frames bank nothing, and evidence
# still ages out after 30 s.  See docs/decisions/0004-fatigue-score-and-tiers.md.
FATIGUE_SUSTAIN_WINDOW_S = 30.0  # seconds — rolling window for the duty cycle

# Fatigue score decay rate while the face is not visible.
# The score is not reset to zero on face-loss; it decays slowly so
# pre-existing fatigue remains reflected after brief look-aways.
FATIGUE_SCORE_DECAY_RATE = 0.02  # score units per second

# ── Fatigue — Yawn (MAR) Detection ──────────────────────────────
# A yawn is detected when the smoothed MAR exceeds the per-user
# threshold continuously for at least YAWN_MIN_DURATION_S seconds:
#
#     threshold = max(baseline_mar + YAWN_MAR_DELTA, YAWN_MAR_MIN)
#
# The offset is ADDITIVE, not multiplicative.  MAR is measured from
# inner-lip landmarks that are nearly coincident when the lips touch,
# so baseline_mar (resting, closed mouth) is a near-zero number —
# median 0.010 across 13 logged sessions, with a per-session std that
# often exceeds the mean.  A multiplier on that baseline produces a
# threshold of ~0.015, which sits at the median of closed-mouth
# landmark noise: ~40 % of all recorded frames cleared it, yielding
# 207 spurious yawn events over 35 739 frames (private corpus, n=1).
#
# Measured MAR distribution (13 sessions, mouth closed / working):
#   p50 0.013 · p90 0.041 · p95 0.063 · p99 0.247 · p99.9 1.153
# Genuine yawns reach 0.5–1.33, leaving a clean valley at 0.3–0.5.
# A 0.40 threshold recovers 6 sustained events across the same data.
YAWN_MIN_DURATION_S = 1.5  # seconds of sustained mouth-open to count as yawn
YAWN_WINDOW_S = 300.0  # rolling window for yawn-count signal (5 min)
YAWN_MAR_DELTA = 0.35  # MAR rise above resting baseline = mouth open
YAWN_MAR_MIN = 0.40  # absolute floor — guards bad/absent calibration

# ── Fatigue — Blink Duration ─────────────────────────────────────
# Blink duration is baseline-relative like every other threshold in this
# engine; the former absolute anchors (150 ms → 0.0, 500 ms → 1.0) measured
# a different quantity from the one the engine times.  The tuning values
# live beside their blink-frequency siblings in acuity/core/fatigue_engine.py:
# _BLINK_DURATION_FATIGUE_RATIO and _BLINK_DURATION_BASELINE_MAX_MS.  See
# docs/decisions/0003-blink-duration-ratio.md.

# ── Fatigue — Head Nod Detection ─────────────────────────────────
# A "nod" is a pitch drop ≥ HEAD_NOD_PITCH_DROP_DEG followed by
# recovery within HEAD_NOD_RECOVERY_S.  Nods are counted in a
# HEAD_NOD_WINDOW_S rolling window.
HEAD_NOD_PITCH_DROP_DEG = 8.0  # degrees of pitch drop to qualify
HEAD_NOD_RECOVERY_S = 3.0  # seconds: max recovery time for a valid nod
HEAD_NOD_WINDOW_S = 60.0  # rolling window for nod-count signal

# ── Fatigue — Pitch-Gated EAR Suppression ───────────────────────
# When the head pitch deviates more than this many degrees downward
# from the calibrated baseline, the fatigue engine pauses EAR
# tracking.  Downward gaze narrows the eyelids geometrically,
# producing false low-EAR readings.
#
# The share of frames under the adaptive blink threshold climbs from 4.9%
# at baseline pitch to ~20% at 6° down and then plateaus, so 6° is the
# knee.  It suppresses 5.8% of face-tracked frames; 4° would cost 14.2%
# (private corpus, n=1).  History: 25° → 12° → 6°.
#
# EAR_PITCH_GATE_RELEASE_DEG adds hysteresis: a 2° band is 3x the p95
# frame-to-frame pitch jitter.  Without it a head near the line flaps the
# gate, and every flap clears and rebuilds the EAR, blink and microsleep
# state.
#
# CAVEAT — n=1.  The gate is relative to the user's *own* calibrated
# baseline, so camera height and posture normalise out, but the knee angle
# depends on eyelid geometry and camera offset and needs multi-subject
# re-derivation.  The debug CSV logs everything required.
#
# This is a *suppression* gate, not a cos(pitch) correction (which
# over-corrects), and deliberately *not* body-relative: it asks whether
# the camera can resolve the eye.  See docs/decisions/0002-ear-validity-guards.md.
EAR_PITCH_GATE_DEG = 6.0  # degrees below baseline — engage gate
EAR_PITCH_GATE_RELEASE_DEG = 4.0  # degrees below baseline — release gate

# ── Fatigue — Head-Motion EAR Suppression ───────────────────────
# The pitch gate above answers "is the eye foreshortened?".  It does not
# answer "did the mesh solve the eye at all?", and a fast head turn breaks
# the second question while leaving the first one happy.
#
# Traced from a labelled false microsleep: the user turned ~60° and swung
# back; mid-turn one eye read an impossible EAR of 0.97, and on the way
# back the unconverged landmarks read both eyes shut for 2.5 s.  *Static*
# yaw does not degrade EAR (measured and rejected); the transient does —
# frames under the threshold rise from 7.2% at rest to 24.8% at 100°/s
# (private corpus, n=1).  So the guard is on angular velocity, not angle.
# See docs/decisions/0002-ear-validity-guards.md.

# Frame-to-frame |Δyaw|/dt above which EAR is not trustworthy.  60°/s sits at
# the foot of the rate curve's climb while staying far above ordinary desk
# fidgeting (private corpus, n=1).
EAR_YAW_RATE_GATE_DPS = 60.0  # degrees/second — engage EAR suppression

# How long the rate must stay above the gate before EAR is suppressed.
# 42% of excursions above the gate are one frame long — the mesh
# re-solving, not the head moving (private corpus, n=1).  Charging a
# settling window to such a blip costs independent-PERCLOS observation
# time and can push a *real* microsleep below _PERCLOS_MIN_INDEPENDENT_S.
#
# Expressed in seconds, not frames.  0.10 s is the smallest value that
# removes single-interval onsets at the pipeline's real ~15 FPS (0.05 s
# was sized for 30 FPS and did nothing at 15).  At 7 FPS one interval
# already trips it — correctly, since 60°/s over 143 ms is 8.6° of travel.
#
# CAVEAT — n=1, one subject and one camera.  The frame-rate finding is not:
# any user whose loop runs below ~20 FPS was on single-interval behaviour.
EAR_YAW_RATE_SUSTAIN_S = 0.10  # seconds above the rate gate before it engages

# An eye aspect ratio above this is not a wide-open eye, it is a collapsed
# mesh solution: the artifact frames above reached 0.97 against a resting
# median near 0.25.  Applied per-eye, since a turn degrades the far eye first.
EAR_PLAUSIBLE_MAX = 0.45  # EAR units — above this the landmarks are garbage

# EAR judgement stays suppressed this long after *every* suppression
# condition clears, so the mesh has time to re-converge before its output is
# believed.  The elevated period after a gate releases lasts under ~1 s.
# Costs 1.3% of observation time on its own, and in replay removed the
# traced false microsleep while leaving every fire in quiet conditions
# untouched (private corpus, n=1).
#
# CAVEAT — n=1, one subject and one camera, same as EAR_PITCH_GATE_DEG.
EAR_GATE_SETTLE_S = 0.75  # seconds — post-suppression settling window

# ── Face Lost / Away State (Auto-Pause) ─────────────────────────
# "Bounds: Min 5s | Max 120s  |  Initial Setting: 15s"
FACE_LOST_THRESHOLD_MIN_S = 5  # seconds
FACE_LOST_THRESHOLD_MAX_S = 120  # seconds
FACE_LOST_THRESHOLD_DEFAULT_S = 15  # seconds

# ── Manual Pause ────────────────────────────────────────────────
# How long a manual pause may run before it is treated like an AWAY
# period: the camera is released (no camera held through a long break)
# and resuming re-runs the 10 s calibration instead of reusing the
# pre-pause baselines.
#
# Below this, a pause is an interruption the user takes *in the chair*
# (a question from someone nearby, reading something off-screen) — the
# calibrated EAR / pitch / shoulder baselines still describe them, and
# charging a 10 s calibration for a 20 s pause costs more than the
# staleness it avoids.  Past a minute the user has usually left the
# desk, and returning means a different seating position, so the
# baselines must be re-measured — the same reasoning AWAY already
# applies, here on a manual trigger rather than a detected absence.
PAUSE_RECALIBRATE_S = 60.0  # seconds

# ── Timer Modes ─────────────────────────────────────────────────
# Mode A: Structured (Pomodoro) — "25m work / 5m rest"
STRUCTURED_WORK_DEFAULT_M = 25  # minutes
STRUCTURED_REST_DEFAULT_M = 5  # minutes (short rest between blocks)

# Structured Mode — Long Rest / Pomodoro Cycle
# After every N focus blocks, a long rest replaces the short rest.
STRUCTURED_LONG_REST_DEFAULT_M = 15  # minutes
STRUCTURED_LONG_REST_MIN_M = 1  # minutes (bounds for Settings slider)
STRUCTURED_LONG_REST_MAX_M = 60  # minutes

BLOCKS_BEFORE_LONG_REST_DEFAULT = 4  # focus blocks per cycle
BLOCKS_BEFORE_LONG_REST_MIN = 2  # bounds
BLOCKS_BEFORE_LONG_REST_MAX = 10  # bounds

# Mode B: Fluid — "Default 40m focus | Rest Time = Elapsed Focus Time / 4"
FLUID_WORK_DEFAULT_M = 40  # minutes (initial stopwatch target)
FLUID_REST_DIVISOR = 4

# ── Manual Rest Button — Unlock Gate (both modes) ───────────────
# Minimum focus time before the "Take a Rest" button becomes clickable.
# The gate earns its keep differently per mode:
#   Fluid      — rest is elapsed / FLUID_REST_DIVISOR, so an early click
#                buys a rest too short to be one (a click 90 s in yields
#                22 s).  At the 10 min default the shortest rest is 2:30.
#   Structured — rest length is fixed by config, so nothing scales down;
#                the gate is the only thing stopping a full Short Rest
#                being claimed a minute into a 25 min block.
# 10 min is a judgement call, not a measurement: it is short enough to
# stay usable mid-block and long enough that the Fluid floor is a real
# break.  Users who disagree have the setting.
REST_BUTTON_UNLOCK_DEFAULT_M = 10  # minutes
REST_BUTTON_UNLOCK_MIN_M = 0  # 0 = no gate, button live immediately
REST_BUTTON_UNLOCK_MAX_M = 120

# ── Fluid Mode — Fatigue Rest Behaviour ─────────────────────────
# Controls what happens when fatigue is detected in Fluid mode.
#   "prompt"  — show a dialog letting the user choose (default)
#   "dynamic" — auto-transition to rest with a notification
FATIGUE_REST_PROMPT = "prompt"
FATIGUE_REST_DYNAMIC = "dynamic"
FATIGUE_REST_DEFAULT = FATIGUE_REST_PROMPT

# ── Alert Modes ─────────────────────────────────────────────────
# Each alert type maps to one of these behaviour flags.
ALERT_LOG_ONLY = "log_only"
ALERT_NOTIFICATION_ONLY = "notification_only"
ALERT_SOUND_ONLY = "sound_only"
ALERT_NOTIFICATION_AND_SOUND = "notification_and_sound"

ALERT_MODES = (
    ALERT_LOG_ONLY,
    ALERT_NOTIFICATION_ONLY,
    ALERT_SOUND_ONLY,
    ALERT_NOTIFICATION_AND_SOUND,
)

# Default alert behaviours (the Settings tab's initial settings)
ALERT_DEFAULT_DISTRACTION = ALERT_LOG_ONLY  # "Log Only"
ALERT_DEFAULT_FATIGUE_WARNING = ALERT_NOTIFICATION_ONLY  # "Notification Only" — gentle nudge
ALERT_DEFAULT_FATIGUE = ALERT_NOTIFICATION_AND_SOUND  # "Notification + Sound" — full alert

# ── Database ────────────────────────────────────────────────────
DB_FILENAME = "acuity.db"

# Event type identifiers (Table Events → event_type column)
EVENT_DISTRACTION_POSE = "Distraction_Pose"
# Foreground window off the user's focus rules.  Fired by the window
# engine (ticket 12); alerts and Insights already handle it.
EVENT_DISTRACTION_WINDOW = "Distraction_Window"
EVENT_FATIGUE = "Fatigue"  # alert tier + contextual microsleep + sleep
EVENT_FATIGUE_WARNING = "Fatigue_Warning"  # warning tier (no rest trigger)
EVENT_AWAY = "Away"

EVENT_TYPES = (
    EVENT_DISTRACTION_POSE,
    EVENT_DISTRACTION_WINDOW,
    EVENT_FATIGUE,
    EVENT_FATIGUE_WARNING,
    EVENT_AWAY,
)

# Session mode identifiers
SESSION_MODE_STRUCTURED = "Structured"
SESSION_MODE_FLUID = "Fluid"
SESSION_MODE_DEFAULT = SESSION_MODE_STRUCTURED

# Rest type identifiers (Rests table → rest_type column)
REST_TYPE_SHORT = "Short"
REST_TYPE_LONG = "Long"
REST_TYPE_FLUID = "Fluid"

REST_TYPES = (REST_TYPE_SHORT, REST_TYPE_LONG, REST_TYPE_FLUID)

# ── Insights Tab ────────────────────────────────────────────────
# Display hygiene and takeaway evidence gates.  These are not detection
# thresholds — they decide what the Insights tab is willing to *claim*.
# CAVEAT — n=1: every figure below comes from one user's history
# (105 sessions over about three months, private corpus, n=1).  They are
# deliberately conservative so they transfer; revisit once other users'
# data exists.

# Range selector keys.  The comparison period for takeaways is the
# equal-length window immediately before; "all" has none.
INSIGHTS_RANGE_TODAY = "today"
INSIGHTS_RANGE_7D = "7d"
INSIGHTS_RANGE_30D = "30d"
INSIGHTS_RANGE_ALL = "all"
INSIGHTS_RANGE_DEFAULT = INSIGHTS_RANGE_7D

# Blocks shorter than this are start-and-stop noise, not focus: 46 of
# 105 sessions had < 1 min of focus.  Filtered out of every range query.
INSIGHTS_MIN_BLOCK_S = 60

# A time bucket needs this much focus before its events-per-focus-hour
# rate is drawn; below it the bucket is a gap, not a spike.  Measured on
# 38 active days / 78 active hours (private corpus, n=1):
#   no floor   day max 744.7/h (19 events in 1.5 min), p90 94.3
#              hour max 744.7/h, p90 72.1
#   600 s      day: 25 of 38 kept, max 48.6/h, p90 27.6
#              hour: 41 of 78 kept, max 72.1/h
#   1800 s     day: only 14 of 38 kept — hides most of the history
# 300 s gives the same day-level max as 600 s but keeps 30–60 s-of-focus
# hours at ~100/h; 600 s is the smallest floor that removes every
# double-digit-minute artefact without discarding most days.
INSIGHTS_MIN_RATE_FOCUS_S = 600

# Histogram geometry (block length and minutes-into-block charts).
# Values ≥ the max land in one overflow bin: the longest block is
# 104 min, and fatigue onsets past 60 min are rare (4 of 85).
INSIGHTS_HIST_BIN_MIN = 5
INSIGHTS_HIST_MAX_MIN = 60

# Takeaways (sidebar).  A rule states nothing unless its gate passes.
INSIGHTS_TAKEAWAY_MAX = 4
INSIGHTS_TAKEAWAY_MAX_RATE_ITEMS = 2  # rate-change sentences shown at most
# Both periods need ≥ 1 h of focus before any comparison is made.
INSIGHTS_TAKEAWAY_MIN_FOCUS_S = 3600
# Combined events across both periods before a rate change is claimed.
# Head pose over the last 7 days vs the 7 before was 5 vs 1 — "5× worse"
# on a single event.  Posture over the same windows was 16 vs 23.
INSIGHTS_TAKEAWAY_MIN_EVENTS = 8
INSIGHTS_TAKEAWAY_MIN_RATE_CHANGE = 0.25  # relative, per focus hour
INSIGHTS_TAKEAWAY_MIN_FOCUS_CHANGE = 0.15  # relative, total focus
# Fatigue alerts needed before a typical onset is stated (median).
INSIGHTS_TAKEAWAY_MIN_ONSETS = 5
# Active days needed before a peak focus window is stated.
INSIGHTS_TAKEAWAY_MIN_ACTIVE_DAYS = 3
# Labelled events of one type needed before its precision is shown.
# Unmeasured default, tune from labels: below this one label moves the
# figure by more than 20 points.
INSIGHTS_PRECISION_MIN_LABELS = 5

# ── Event labels ────────────────────────────────────────────────
# The user's verdict on an event (Events.label), set in the review
# dialog.  Labels let a user check each detector's accuracy on themselves.
EVENT_LABEL_CORRECT = "correct"
EVENT_LABEL_FALSE_ALARM = "false_alarm"

# ── Logging ─────────────────────────────────────────────────────
LOG_FILENAME = "acuity.log"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
LOG_BACKUP_COUNT = 3  # number of rotated backup files

# ── Camera ───────────────────────────────────────────────────────
# Mirror the camera image horizontally (selfie view).  Applied to the
# raw frame right after capture, so calibration, the CV engines and
# the debug overlay all see the same orientation.
CAMERA_FLIP_DEFAULT = False

# ── Debug Overlay ────────────────────────────────────────────────
# Off by default; toggled via Settings checkbox or Ctrl+D shortcut.
DEBUG_OVERLAY_DEFAULT = False

# How often to redraw the overlay (1 = every frame; 2 = every other, etc.)
# Drawing is cheap (~1-2 ms) so 1 is the right default.  The option
# exists to allow future heavier rendering without rework.
DEBUG_OVERLAY_FRAME_SKIP_DEFAULT = 1  # 1 = draw every frame
