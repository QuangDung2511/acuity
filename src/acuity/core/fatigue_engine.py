# DMS-grade fatigue detection: 4-signal weighted scoring model.
# ─────────────────────────────────────────────────────────────────
# Replaces the old binary dual-detector (microsleep + blink frequency)
# with a continuous composite fatigue score fused from four signals:
#
#   Signal              Weight  Source
#   ──────────────────  ──────  ──────────────────────────────────
#   PERCLOS             0.389   % eye-closed time in 30 s window
#   Blink Duration      0.278   rolling median vs the user's own baseline
#   MAR / Yawning       0.222   yawn count in 5-min rolling window
#   Blink Frequency     0.111   rolling blinks/min vs calibrated baseline
#
# Head Nod was a fifth signal at weight 0.10.  It is still detected and
# reported for debug/CSV purposes but no longer feeds the score: it has
# no usable specificity (private corpus, n=1), and it became unreachable
# once the EAR pitch gate moved below the nod drop threshold.  See
# docs/decisions/0004-fatigue-score-and-tiers.md.
#
# Blink duration is judged against a per-session baseline rather than an
# absolute millisecond anchor — the engine times sub-threshold width, not
# physiological closure, and resting medians span 195–743 ms per session
# (private corpus, n=1).  See docs/decisions/0003-blink-duration-ratio.md.
#
# Microsleep requires a BILATERAL episode.  The eyes-closed test is on
# ``ear_avg``, so one eye is enough to clear it — a wink, or one eye's
# landmarks collapsing, drags the mean under while the other eye is open.
# The episode is still delimited by ``ear_avg`` (so its duration is
# unchanged) but must have had both eyes below threshold for at least
# MICROSLEEP_BILATERAL_SHARE of its frames.  See that constant for why a
# per-frame both-eyes test on ``below`` was measured and rejected.
#
# Two-tier alert system:
#   Warning  — score ≥ 0.6 sustained for 15 s → EVENT_FATIGUE_WARNING
#              (30 s cooldown; no Fluid-mode rest trigger)
#   Alert    — score ≥ 0.8 sustained for 10 s → EVENT_FATIGUE
#              (60 s cooldown; triggers Fluid-mode rest prompt)
#   Microsleep  — ≥ 1.5 s shut, *in fatigue context* (score ≥ warning
#                 threshold or a warning within MICROSLEEP_CONTEXT_S)
#                 → EVENT_FATIGUE; without context it is logged only
#   Sleep       — ≥ SLEEP_CLOSURE_S shut, no context needed → EVENT_FATIGUE
#   (both share the 60 s alert cooldown)
#
# Warning means signs of fatigue; alert means the user is asleep or
# microsleeping.  A lone microsleep in an otherwise alert user is not an
# alert: eyes shut for ~2 s is also what drinking looks like (one labelled
# false alert was exactly that), and the eye data cannot tell the two apart.
# See docs/decisions/0001-microsleep-and-sleep-gates.md.
#
# Face-loss handling:
#   Score decays at FATIGUE_SCORE_DECAY_RATE per second instead of
#   resetting.  PERCLOS denominator excludes face-loss frames (already
#   ratio-based).  Blink/yawn/nod buffers are preserved across pauses.
#
# EAR validity:
#   EAR is only believed while the mesh is actually describing an eye.
#   Three guards suppress it, all routed through the same pause branch as
#   face loss: the pitch gate (head down, eye foreshortened), a yaw-rate
#   gate (fast head turn, landmarks not solved), and a per-eye plausibility
#   cap (a collapsed solution reading far above any real aperture).  A
#   settling window follows the last offending frame, because pitch
#   normalises before the landmarks re-converge.  See
#   docs/decisions/0002-ear-validity-guards.md for the measurements.
#
# Frame-rate independence:
#   All smoothing and hysteresis windows are measured in seconds and
#   PERCLOS is time-weighted, so behaviour is identical at 7 FPS and
#   30 FPS.  See the "Frame-rate independence" constants block below.
#
# All weights and thresholds sourced from constants.py.
#
# Thread: pure (no Qt, no I/O); updated per frame on the CV worker thread.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from acuity.core.face_mesh import FaceResult
from acuity.core.presence import PresenceState
from acuity.utils.angles import wrap_deg
from acuity.utils.constants import (
    EAR_BLINK_THRESHOLD,
    EAR_GATE_SETTLE_S,
    EAR_PITCH_GATE_DEG,
    EAR_PITCH_GATE_RELEASE_DEG,
    EAR_PLAUSIBLE_MAX,
    EAR_YAW_RATE_GATE_DPS,
    EAR_YAW_RATE_SUSTAIN_S,
    EVENT_FATIGUE,
    EVENT_FATIGUE_WARNING,
    FATIGUE_ALERT_SUSTAIN_S,
    FATIGUE_ALERT_THRESHOLD,
    FATIGUE_COOLDOWN_S,
    FATIGUE_SCORE_DECAY_RATE,
    FATIGUE_SUSTAIN_WINDOW_S,
    FATIGUE_W_BLINK_DUR,
    FATIGUE_W_BLINK_FREQ,
    FATIGUE_W_PERCLOS,
    FATIGUE_W_YAWN,
    FATIGUE_WARNING_COOLDOWN_S,
    FATIGUE_WARNING_SUSTAIN_S,
    FATIGUE_WARNING_THRESHOLD,
    HEAD_NOD_PITCH_DROP_DEG,
    HEAD_NOD_RECOVERY_S,
    HEAD_NOD_WINDOW_S,
    MICROSLEEP_BILATERAL_SHARE,
    MICROSLEEP_BLINK_CONFIRM,
    MICROSLEEP_CONTEXT_S,
    MICROSLEEP_SHUT_RATIO,
    MICROSLEEP_THRESHOLD_S,
    PERCLOS_GATE,
    PERCLOS_SIGNAL_SATURATION,
    PERCLOS_WINDOW_S,
    SLEEP_CLOSURE_S,
    YAWN_MAR_DELTA,
    YAWN_MAR_MIN,
    YAWN_MIN_DURATION_S,
    YAWN_WINDOW_S,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# ── Module-level constants ───────────────────────────────────────

# Rolling window for blink-frequency signal (seconds).
_BLINK_WINDOW_S = 60.0

# Blink-frequency signal: drop to this fraction of baseline → signal = 1.0.
_BLINK_FREQ_DROP_RATIO = 0.50

# EMA alpha for baseline blink rate refinement.
_BASELINE_EMA_ALPHA = 0.3

# Minimum baseline blink rate (blinks/min); below this the frequency
# signal is disabled (avoids false positives for naturally low blinkers).
_MIN_BASELINE_BLINK_RATE = 5.0

# Require 1 full extra window before frequency detection starts.
_STABILIZATION_WINDOWS = 1

# Minimum *observed* seconds inside the trailing _BLINK_WINDOW_S before the
# blink-frequency signal renders a verdict.
#
# blink_rate divides the blinks counted in the window by the time the eyes
# were actually observed in it.  It used to divide by the whole window once
# the session was a minute old, so every paused second — each EAR guard
# blip, each face loss — silently read as a second of no blinking: in
# replay, rate / baseline fell from 0.97 to 0.28 as the paused share of the
# minute rose, independent of fatigue (private corpus, n=1).  Dividing by
# observed time removes the bias; this floor keeps a minute that was mostly
# unobservable from delivering a verdict off a handful of seconds.  Half
# the window.  See docs/decisions/0004-fatigue-score-and-tiers.md.
_BLINK_WINDOW_MIN_OBSERVED_S = 30.0

# Blink-duration signal: rise to this multiple of the user's own baseline
# blink duration → signal = 1.0.
#
# Both sides of the comparison are the MEDIAN of the duration buffer
# (capture, refinement and signal).  A mean against a median baseline
# started right-skewed sessions near full scale and raised false warnings.
# Re-derived median-against-median over 144 replayed sessions: a typical
# session's rolling median peaks at 1.32x its baseline and the p90 session
# at 1.93x, so at 1.8 a typical session peaks at signal 0.40 and 19 of 144
# saturate (private corpus, n=1).  The literature-derived 3.3x (150→500 ms)
# would leave this signal dead.
#
# On a labelled 12.5 h evaluation set, mean → median took labelled false
# warnings 7 → 0 and kept all 4 correct alerts; with the false floor gone
# no warning fired at all, so warning-tier recall is unmeasured (private
# corpus, n=1).
#
# CAVEAT — n=1, one subject and one camera, and all four positives come
# from a single day.  See docs/decisions/0003-blink-duration-ratio.md.
_BLINK_DURATION_FATIGUE_RATIO = 1.8

# Ceiling on the stored baseline blink duration (ms).  A user who is
# already drowsy — or whose tracking is degraded — during the baseline
# window would otherwise bake in a long reference and suppress the signal
# for the rest of the session.  3 of 16 logged sessions captured anchors
# above 690 ms and then *improved* (later/anchor 0.90–1.19), which is the
# signature of exactly that.  Session anchors otherwise sat near 430 ms
# (private corpus, n=1).
# Weakest-evidenced of the three tuning values here — revisit with data.
_BLINK_DURATION_BASELINE_MAX_MS = 600.0

# ── Frame-rate independence ──────────────────────────────────────
# Every window below is expressed in SECONDS, never in frames.  An
# earlier revision sized these in frames against an assumed 30 FPS
# capture loop; when the real pipeline ran at ~7 FPS each window
# silently stretched 4x, the EAR median filter merged adjacent blinks
# into a single 1.2 s "closure", and the engine reported false
# microsleeps.  Frame counts are not a unit of time — do not
# reintroduce one here.

# Continuous eyes-open time required to confirm an episode ended.
_EYES_OPEN_CONFIRM_S = 0.10

# Tolerance for time comparisons that a frame interval can land exactly on.
# At 30 FPS three frame intervals sum to _EYES_OPEN_CONFIRM_S precisely, so
# accumulated float error — not the eye — would otherwise decide whether an
# eyes-closed episode ends.  A missed end leaves _microsleep_fired latched
# and suppresses the *next* genuine microsleep.
_TIME_EPS_S = 1e-6

# PERCLOS sample cap — a safety bound only (supports up to 120 FPS).
# The real window is enforced by timestamp eviction, not by capacity.
_PERCLOS_MAX_SAMPLES = int(PERCLOS_WINDOW_S * 120)

# Minimum observed time before PERCLOS is treated as meaningful.
_PERCLOS_MIN_OBSERVED_S = 1.0

# Minimum *independent* observation (seconds) required before the microsleep
# PERCLOS gate will render a verdict at all.  The gate is evaluated over the
# window minus the candidate episode itself (see _perclos_excluding), so a
# window holding little else than the closure supports no verdict either way.
# Guards the startup case: a logged session fired at t=1.06 s with PERCLOS
# 1.000 — a blink as calibration ended, with the whole window inside it
# (private corpus, n=1).  _PERCLOS_MIN_OBSERVED_S (1.0 s) is far too weak to catch that.  10 s
# keeps microsleep available early in a focus block (each block re-calibrates
# and resets the engine) while making the gate's "sustained over the last 30 s"
# claim mean something.
_PERCLOS_MIN_INDEPENDENT_S = 10.0

# Per-frame dt is clamped to this before it is credited to PERCLOS, so
# one long stall cannot inject seconds of "eyes closed" from a single
# sampled frame.
_MAX_FRAME_DT_S = 0.5

# Hard floor for the adaptive blink threshold.
_BLINK_THRESHOLD_FLOOR = 0.15

# EAR smoothing window (seconds) — sliding median over a time window.
# 0.10 s holds 3–4 samples at 30 FPS (enough for a median to reject a
# single-frame landmark outlier) and collapses to 1 sample below ~10 FPS.
# Collapsing is the correct degradation: at 7 FPS a frame already carries
# 143 ms of information, and any smoothing wide enough to hold two samples
# would lag longer than the gap between two consecutive blinks.
_EAR_SMOOTH_WINDOW_S = 0.10

# MAR smoothing window (seconds) — sliding median over a time window.
_MAR_SMOOTH_WINDOW_S = 0.15

# Smoothing factor for the measured frame-interval EMA.
_FRAME_DT_EMA_ALPHA = 0.1

# Maximum blink-duration history.
_BLINK_DUR_MAXLEN = 50

# Brief face-loss still credited to the blink window (seconds).
_SHORT_PAUSE_THRESHOLD_S = 15.0

# Maximum nods counted in the head-nod signal window.
# At 3 nods/min the signal saturates: score = 1.0.
_NOD_SIGNAL_SATURATE = 3

# Warn once per episode when the pitch gate has held this long (seconds).
# A calibrated baseline that no longer describes how the user actually
# sits would park them permanently gated with EAR-based fatigue detection
# silently off.  Under a simulated 6°/4° gate the 713 real gated stretches
# in the logs run p50 1.5 s · p95 11 s · p99 27.3 s · max 43.5 s — nothing
# above 60 s (private corpus, n=1) — so 120 s cannot fire on genuine desk
# work and only trips when the baseline has drifted.
_PITCH_GATE_WARN_S = 120.0


# ── Time-windowed median filter ──────────────────────────────────


class _TimedMedian:
    """Sliding median over a fixed *time* window, not a frame count.

    The number of samples held varies with the frame rate, so the
    amount of noise rejection degrades as capture slows — but the
    filter's *lag* stays bounded by ``window_s`` at any rate.  That is
    the property that matters here: lag longer than the gap between two
    consecutive blinks is what merges them into a phantom microsleep.

    At least one sample is always retained, so the filter still has a
    value at arbitrarily low frame rates.
    """

    __slots__ = ("_window_s", "_buf")

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._buf: deque[tuple[float, float]] = deque()

    def add(self, now: float, value: float) -> float:
        """Append *value* at time *now* and return the current median."""
        self._buf.append((now, value))
        cutoff = now - self._window_s
        while len(self._buf) > 1 and self._buf[0][0] < cutoff:
            self._buf.popleft()
        return self.value

    @property
    def value(self) -> float:
        """Current median, or 0.0 when no samples have been added."""
        if not self._buf:
            return 0.0
        return statistics.median(v for _, v in self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __bool__(self) -> bool:
        return bool(self._buf)


# ── Adaptive threshold helper ────────────────────────────────────


def _compute_blink_threshold(baseline_ear: float, baseline_ear_std: float) -> float:
    """Compute a fully adaptive per-user EAR blink threshold.

    Uses two independent formulas and picks the more conservative result,
    then clamps to ``_BLINK_THRESHOLD_FLOOR``.

    1. **Ratio-based**: ``baseline_ear × 0.70``
    2. **Stats-based**: ``baseline_ear − 2 × baseline_ear_std``
    """
    if baseline_ear <= 0.0:
        return EAR_BLINK_THRESHOLD

    ratio_based = baseline_ear * 0.70

    if baseline_ear_std > 0.0:
        stats_based = baseline_ear - 2.0 * baseline_ear_std
        threshold = min(ratio_based, stats_based)
    else:
        threshold = ratio_based

    return max(threshold, _BLINK_THRESHOLD_FLOOR)


# ── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class FatigueEvent:
    """Emitted when a fatigue warning or alert fires.

    Attributes
    ----------
    event_type : str
        ``EVENT_FATIGUE_WARNING`` or ``EVENT_FATIGUE``.
    reason : str
        Identifies the trigger: ``"warning_threshold"``,
        ``"alert_threshold"``, or ``"microsleep"``.
    duration : float
        For microsleep: eye-closed duration (seconds).
        For threshold events: sustain duration (seconds).
    fatigue_score : float
        Composite score [0.0 – 1.0] at the moment of firing.
    perclos : float
        PERCLOS value [0.0 – 1.0] at the moment of firing.
    avg_blink_duration_ms : float
        Average blink duration (ms) from the last 50 blinks.
    yawn_count : int
        Number of yawns detected in the last 5 minutes.
    blink_rate : float
        Current rolling blink rate (blinks/min).
    timestamp : float
        ``time.time()`` when the event was created.
    """

    event_type: str
    reason: str
    duration: float
    fatigue_score: float
    perclos: float
    avg_blink_duration_ms: float
    yawn_count: int
    blink_rate: float
    timestamp: float


# Type alias for the event callback.
FatigueCallback = Callable[[FatigueEvent], None]


# ── FatigueEngine ────────────────────────────────────────────────


class FatigueEngine:
    """DMS-grade fatigue detector: 5-signal weighted scoring model.

    Parameters
    ----------
    baseline_ear : float
        Calibrated resting EAR (from :class:`CalibrationResult`).
    baseline_ear_std : float
        Standard deviation of EAR samples during calibration.
    baseline_mar : float
        Calibrated resting MAR (from :class:`CalibrationResult`).
        Used to derive the per-user yawn threshold.
    baseline_mar_std : float
        Standard deviation of MAR samples during calibration.
        Informational; not used in threshold logic.
    callback : FatigueCallback or None
        Called once per fatigue episode.  If ``None``, events are
        only logged.

    Example
    -------
    ::

        engine = FatigueEngine(
            baseline_ear=cal.baseline_ear,
            baseline_ear_std=cal.baseline_ear_std,
            baseline_mar=cal.baseline_mar,
            baseline_mar_std=cal.baseline_mar_std,
            callback=my_handler,
        )

        # Per-frame loop:
        presence = determine_presence(face_result, pose_result)
        engine.update(face_result, presence, head_pitch=face.pitch,
                      baseline_pitch=cal.baseline_pitch, mar=face.mar)
    """

    def __init__(
        self,
        baseline_ear: float,
        baseline_ear_std: float = 0.0,
        baseline_mar: float = 0.0,
        baseline_mar_std: float = 0.0,
        callback: FatigueCallback | None = None,
    ) -> None:
        self._baseline_ear = baseline_ear
        self._baseline_ear_std = baseline_ear_std
        self._baseline_mar = baseline_mar
        self._baseline_mar_std = baseline_mar_std
        self._callback = callback

        # Adaptive per-user blink threshold.
        self._blink_threshold = _compute_blink_threshold(baseline_ear, baseline_ear_std)

        # Per-user yawn threshold (MAR units) — additive offset above the
        # resting baseline, clamped to an absolute floor.  See the
        # YAWN_MAR_DELTA / YAWN_MAR_MIN block in constants.py for why this
        # is not a multiplier: baseline_mar is near-zero for a closed mouth,
        # so a multiplicative threshold lands inside landmark noise.
        #
        # The floor also makes the threshold valid when baseline_mar is 0.0
        # (no face detected during calibration), so yawn detection no longer
        # has to be disabled in that case — unlike EAR, an absolute MAR
        # threshold is meaningful without a per-user baseline.
        self._yawn_threshold: float = max(baseline_mar + YAWN_MAR_DELTA, YAWN_MAR_MIN)

        # ── EAR smoothing ────────────────────────────────────────
        self._ear_buf = _TimedMedian(_EAR_SMOOTH_WINDOW_S)
        # Per-eye filters used ONLY to classify a frame as bilateral for the
        # microsleep share test.  Everything else — `below`, PERCLOS, blink
        # detection, debug_smoothed_ear — still runs off `_ear_buf` on
        # `ear_avg`, deliberately: see MICROSLEEP_BILATERAL_SHARE for why a
        # per-frame both-eyes test on `below` was measured and rejected.
        self._ear_buf_left = _TimedMedian(_EAR_SMOOTH_WINDOW_S)
        self._ear_buf_right = _TimedMedian(_EAR_SMOOTH_WINDOW_S)

        # ── PERCLOS buffer ───────────────────────────────────────
        # (timestamp, eyes_closed, dt) — PERCLOS is time-weighted, so a
        # slow frame counts for the time it actually represents rather
        # than as one equal vote alongside a fast one.
        self._perclos_buf: deque[tuple[float, bool, float]] = deque(maxlen=_PERCLOS_MAX_SAMPLES)
        # Running sums kept in step with the buffer so `perclos` is O(1).
        self._perclos_closed_s: float = 0.0
        self._perclos_total_s: float = 0.0

        # ── Microsleep state ─────────────────────────────────────
        self._eyes_closed_start: float | None = None
        self._microsleep_fired: bool = False
        # Latched once the episode has produced (or been refused by cooldown
        # for) a sleep alert — one per episode.
        self._sleep_fired: bool = False
        # Time the eyes were first seen open after a closure (None while closed).
        self._eyes_open_since: float | None = None
        # Frames in the current eyes-closed episode, and how many of them had
        # BOTH eyes below threshold.  A wink never accumulates the second.
        self._episode_frames: int = 0
        self._episode_bilateral_frames: int = 0
        # Per-frame blink blendshape samples for the current episode.
        # Empty means no evidence -> the gate fails open.
        self._episode_blink_scores: list[float] = []
        # Why a qualifying microsleep did not fire this frame, for the
        # debug CSV.  Empty when nothing was withheld.  Observation only:
        # the cooldown in particular hides whole false-positive paths
        # (a 7.23 s qualifying episode produced no event and no trace).
        self._microsleep_withheld: str = ""

        # ── Blink detection ──────────────────────────────────────
        self._prev_below_threshold: bool = False

        # ── Blink duration tracking ──────────────────────────────
        # Stores duration of each blink in milliseconds (last 50).
        self._blink_durations: deque[float] = deque(maxlen=_BLINK_DUR_MAXLEN)
        # Monotonic time when the current blink started (eyes went below threshold).
        self._current_blink_close_start: float | None = None

        # ── Blink frequency window ───────────────────────────────
        self._blink_times: deque[float] = deque()
        # (timestamp, dt) of observed frames inside the trailing window, and
        # their running sum — the denominator of blink_rate.  Paused frames
        # add nothing, exactly as they add no blinks.
        self._blink_obs_buf: deque[tuple[float, float]] = deque()
        self._blink_obs_s: float = 0.0
        self._window_active_s_observed: float = 0.0
        self._window_active_s_total: float = 0.0
        self._last_update_time: float | None = None
        self._baseline_blink_rate: float | None = None
        self._baseline_blink_duration_ms: float | None = None
        self._baseline_captured: bool = False
        self._windows_since_baseline: int = 0

        # ── MAR / Yawning ────────────────────────────────────────
        self._mar_buf = _TimedMedian(_MAR_SMOOTH_WINDOW_S)
        self._yawn_start: float | None = None  # start of current yawn
        self._prev_yawning: bool = False  # was yawning last frame
        self._yawn_times: deque[float] = deque()  # timestamps of completed yawns

        # ── Head nod detection ───────────────────────────────────
        self._nod_times: deque[float] = deque()  # timestamps of detected nods
        self._nod_state: str = "neutral"  # "neutral" | "dropping" | "recovering"
        self._nod_drop_start: float | None = None  # time pitch started dropping
        self._nod_baseline_pitch: float | None = None  # pitch at drop start

        # ── Pitch gate (EAR suppression while head is down) ──────
        self._pitch_gated: bool = False  # gate active this frame
        self._pitch_gated_since: float | None = None  # start of current gated episode
        self._pitch_gate_warned: bool = False  # over-gating warning already emitted

        # ── Head-motion EAR suppression ──────────────────────────
        # Widens the pitch gate to the two other ways the mesh stops
        # describing an eye: a fast head turn, and a collapsed solution.
        self._ear_suppressed: bool = False  # any EAR guard active this frame
        self._prev_yaw: float | None = None  # previous frame's yaw (deg)
        self._prev_yaw_time: float | None = None  # timestamp of _prev_yaw
        self._yaw_rate_high_since: float | None = None  # start of the fast-turn stretch
        self._ear_suppressed_until: float = 0.0  # settling window deadline

        # ── Scoring model ────────────────────────────────────────
        self._fatigue_score: float = 0.0

        # ── Tier duty-cycle buffers ──────────────────────────────
        # (timestamp, above_threshold, dt) over FATIGUE_SUSTAIN_WINDOW_S,
        # one per tier.  Time-weighted for the same reason as
        # _perclos_buf: a slow frame represents the time it actually
        # covers rather than casting one equal vote beside a fast one.
        # A tier fires on *accumulated* above-threshold time reaching its
        # SUSTAIN_S, so a one-frame dip no longer discards the evidence
        # collected so far — see FATIGUE_SUSTAIN_WINDOW_S in constants.py.
        self._warning_buf: deque[tuple[float, bool, float]] = deque()
        self._alert_buf: deque[tuple[float, bool, float]] = deque()
        # Running sums kept in step with the buffers so the check is O(1).
        self._warning_above_s: float = 0.0
        self._alert_above_s: float = 0.0

        # ── Cooldown timestamps ──────────────────────────────────
        self._last_warning_time: float = 0.0  # last EVENT_FATIGUE_WARNING fire
        self._last_alert_time: float = 0.0  # last EVENT_FATIGUE fire

        # ── Pause tracking ───────────────────────────────────────
        self._pause_start: float | None = None

        # ── Observed frame rate (diagnostic) ─────────────────────
        self._frame_dt_ema: float | None = None

        # ── Session pause/resume ───────────────────────────────
        self._session_paused: bool = False
        self._score_at_pause: float = 0.0
        self._pause_wall_time: float = 0.0

        log.info(
            "FatigueEngine (v2) initialised "
            "(baseline_ear=%.3f±%.3f, blink_thresh=%.3f, "
            "baseline_mar=%.3f, yawn_thresh=%.3f).",
            self._baseline_ear,
            self._baseline_ear_std,
            self._blink_threshold,
            self._baseline_mar,
            self._yawn_threshold,
        )

    # ── Properties ──────────────────────────────────────────────

    @property
    def blink_threshold(self) -> float:
        """EAR value below which the eyes are considered closed."""
        return self._blink_threshold

    @property
    def shut_threshold(self) -> float:
        """EAR below which the eye counts as genuinely shut.

        Stricter than :attr:`blink_threshold`: the microsleep timer runs on
        this, while PERCLOS and blink counting stay on the blink threshold.
        Exposed so ``cv_pipeline`` reads it rather than re-deriving the
        ratio — the same rule the pitch gate follows.
        """
        return self._blink_threshold * MICROSLEEP_SHUT_RATIO

    @property
    def yawn_threshold(self) -> float:
        """MAR value above which the mouth is considered open (yawning)."""
        return self._yawn_threshold

    @property
    def blink_rate(self) -> float:
        """Current rolling blink rate (blinks/min).

        Blinks in the trailing ``_BLINK_WINDOW_S`` divided by the time the
        eyes were actually observed in that window, so a paused stretch is
        neither counted as blinks nor as blink-free time.  See
        ``_BLINK_WINDOW_MIN_OBSERVED_S``.

        Returns 0.0 if insufficient observation time has elapsed.
        """
        if self._blink_obs_s < 1.0:
            return 0.0
        # Capped at the window length: a sample is kept by its end timestamp,
        # so one whose interval straddles the cutoff can push the sum past
        # _BLINK_WINDOW_S.  With nothing paused this reproduces the old
        # min(observed, window) denominator exactly.
        effective_s = min(self._blink_obs_s, _BLINK_WINDOW_S)
        return len(self._blink_times) / (effective_s / 60.0)

    @property
    def baseline_blink_rate(self) -> float | None:
        """Baseline blink rate captured from the first full window.

        ``None`` until the window fills for the first time.
        """
        return self._baseline_blink_rate

    @property
    def perclos(self) -> float:
        """Percentage of Eye Closure over the last ``PERCLOS_WINDOW_S`` seconds.

        Time-weighted: each sample contributes the wall-clock interval it
        represents, so the result is the same at 7 FPS and at 30 FPS.

        Returns a value in ``[0.0, 1.0]``, or ``0.0`` when less than
        ``_PERCLOS_MIN_OBSERVED_S`` of eye-visible time has accumulated.
        """
        if self._perclos_total_s < _PERCLOS_MIN_OBSERVED_S:
            return 0.0
        return self._perclos_closed_s / self._perclos_total_s

    @property
    def fatigue_score(self) -> float:
        """Current composite fatigue score in ``[0.0, 1.0]``."""
        return self._fatigue_score

    @property
    def cooldown_remaining(self) -> float:
        """Seconds until the next alert event may fire (0.0 if ready)."""
        elapsed = time.monotonic() - self._last_alert_time
        return max(0.0, FATIGUE_COOLDOWN_S - elapsed)

    @property
    def microsleep_active(self) -> bool:
        """True if the eyes are currently in a confirmed microsleep episode.

        "Closed" here means genuinely *shut* — below
        ``blink_threshold * MICROSLEEP_SHUT_RATIO`` — not merely below the
        blink threshold, which a squint or a slow reopening also satisfies.
        """
        if self._eyes_closed_start is None:
            return False
        return (time.monotonic() - self._eyes_closed_start) > MICROSLEEP_THRESHOLD_S

    # ── Debug diagnostic properties ─────────────────────────────

    @property
    def debug_smoothed_ear(self) -> float:
        """Most recent median-filtered EAR value (0.15 s window)."""
        return self._ear_buf.value

    @property
    def debug_ear_observed(self) -> bool:
        """True when the EAR filter holds samples (i.e. eyes are trackable).

        Callers deriving an ``eyes_closed`` flag from
        :attr:`debug_smoothed_ear` must check this first: the filter is
        cleared while the engine is paused, and 0.0 would otherwise read
        as a closed eye.
        """
        return bool(self._ear_buf)

    @property
    def debug_measured_fps(self) -> float:
        """Observed frame rate of the update loop (0.0 until measured).

        The engine's windows are all time-based and do not depend on this
        value; it is exposed so a degraded capture rate is visible rather
        than silent.
        """
        if not self._frame_dt_ema:
            return 0.0
        return 1.0 / self._frame_dt_ema

    @property
    def debug_microsleep_elapsed_s(self) -> float:
        """Duration of the current eyes-*shut* episode in seconds.

        Measured against ``blink_threshold * MICROSLEEP_SHUT_RATIO``, so it
        is shorter than the sub-threshold stretch reported by
        ``eyes_closed`` and can be 0.0 while the latter is True.
        """
        if self._eyes_closed_start is None:
            return 0.0
        return time.monotonic() - self._eyes_closed_start

    @property
    def debug_episode_bilateral_share(self) -> float:
        """Fraction of the current eyes-closed episode with both eyes closed.

        1.0 when no episode is in progress, so callers reading it outside an
        episode see "nothing disqualifying" rather than a spurious zero.
        Read-only observation point; the gate itself lives in
        :meth:`_update_microsleep`.
        """
        if self._episode_frames <= 0:
            return 1.0
        return self._episode_bilateral_frames / self._episode_frames

    @property
    def debug_episode_blink_confirm(self) -> float:
        """Median eyeBlink blendshape across the current episode.

        1.0 when there is nothing to disqualify — no episode running, or
        no blendshape samples collected — matching
        :attr:`debug_episode_bilateral_share`.  Callers must not read 1.0
        as "the model confirmed a closure"; the gate in
        :meth:`_update_microsleep` checks the sample list itself so an
        absent model fails open rather than silently suppressing.

        Read-only observation point; see MICROSLEEP_BLINK_CONFIRM.
        """
        if not self._episode_blink_scores:
            return 1.0
        return statistics.median(self._episode_blink_scores)

    @property
    def debug_microsleep_withheld(self) -> str:
        """Why a qualifying microsleep did not fire this frame.

        One of ``"bilateral"``, ``"blink"``, ``"perclos"``,
        ``"observation"``, ``"context"``, ``"cooldown"``, or ``""`` when
        nothing was withheld.  Read-only observation point for the debug CSV: without
        it a suppressed episode leaves no trace in the logged corpus, and
        scoring runs undercount false positives.
        """
        return self._microsleep_withheld

    @property
    def debug_avg_blink_duration_ms(self) -> float:
        """Average blink duration from the last 50 blinks (ms)."""
        if not self._blink_durations:
            return 0.0
        return statistics.mean(self._blink_durations)

    @property
    def debug_yawn_count(self) -> int:
        """Number of yawns detected in the last ``YAWN_WINDOW_S`` seconds."""
        now = time.monotonic()
        cutoff = now - YAWN_WINDOW_S
        return sum(1 for t in self._yawn_times if t >= cutoff)

    @property
    def debug_nod_count(self) -> int:
        """Number of head nods detected in the last ``HEAD_NOD_WINDOW_S`` seconds."""
        now = time.monotonic()
        cutoff = now - HEAD_NOD_WINDOW_S
        return sum(1 for t in self._nod_times if t >= cutoff)

    @property
    def debug_fatigue_score(self) -> float:
        """Current composite fatigue score (alias for fatigue_score)."""
        return self._fatigue_score

    @property
    def debug_signal_perclos(self) -> float:
        """Normalized PERCLOS sub-signal [0.0, 1.0]."""
        return self._perclos_signal()

    @property
    def debug_signal_blink_dur(self) -> float:
        """Normalized blink-duration sub-signal [0.0, 1.0]."""
        return self._blink_duration_signal()

    @property
    def debug_signal_yawn(self) -> float:
        """Normalized yawn-count sub-signal [0.0, 1.0]."""
        return self._yawn_signal()

    @property
    def debug_signal_blink_freq(self) -> float:
        """Normalized blink-frequency-drop sub-signal [0.0, 1.0]."""
        return self._blink_freq_signal()

    @property
    def debug_signal_head_nod(self) -> float:
        """Normalized head-nod sub-signal [0.0, 1.0]."""
        return self._head_nod_signal()

    @property
    def debug_mar_smoothed(self) -> float:
        """Most recent median-filtered MAR value (0.15 s window)."""
        return self._mar_buf.value

    @property
    def debug_yawn_active(self) -> bool:
        """True if the mouth is currently open past the yawn threshold."""
        return self._prev_yawning

    @property
    def debug_warning_active(self) -> bool:
        """True while warning-tier above-threshold time is accumulating.

        Reads the duty-cycle window rather than a run timer: True once any
        above-threshold time is banked in the trailing
        FATIGUE_SUSTAIN_WINDOW_S, so it stays True across the brief dips
        that used to zero the old timer.  Read-only observation point.
        """
        return self._warning_above_s > 0.0

    @property
    def debug_alert_active(self) -> bool:
        """True while alert-tier above-threshold time is accumulating.

        See :attr:`debug_warning_active`.
        """
        return self._alert_above_s > 0.0

    @property
    def debug_pitch_gated(self) -> bool:
        """True while the pitch gate is suppressing EAR tracking.

        Reflects the *pitch* condition alone — a pause caused by presence
        loss reads False — which is what the ``pitch_gated`` overlay
        indicator and CSV column mean.  Read-only observation point; the
        gate decision itself lives in :meth:`update`.
        """
        return self._pitch_gated

    @property
    def debug_ear_suppressed(self) -> bool:
        """True while *any* guard is suppressing EAR tracking.

        Superset of :attr:`debug_pitch_gated`: also covers the fast-head-turn
        and degenerate-landmark guards and the settling window that follows
        them.  Kept separate from ``pitch_gated`` so the overlay and CSV can
        still tell "head is down" from "the mesh is not describing an eye" —
        they have different fixes.  Like every ``debug_*`` member this is a
        read-only observation point; the decision lives in :meth:`update`.
        """
        return self._ear_suppressed

    # ── Public API ──────────────────────────────────────────────

    def update(
        self,
        face: FaceResult,
        presence: PresenceState = PresenceState.PRESENT,
        head_pitch: float = 0.0,
        baseline_pitch: float = 0.0,
        mar: float = 0.0,
    ) -> None:
        """Process one frame and advance fatigue tracking.

        Parameters
        ----------
        face : FaceResult
            Output from :class:`FaceMeshAnalyzer.process`.
        presence : PresenceState
            Fused presence signal.
        head_pitch : float
            Current head pitch from face mesh (degrees).
        baseline_pitch : float
            Calibrated resting head pitch (degrees).
        mar : float
            Current Mouth Aspect Ratio from face mesh.

        Call this once per frame during a focus block.
        """
        now = time.monotonic()

        # Per-frame observation flag: cleared here so it describes only
        # what happened on this frame, and re-set by whichever gate in
        # _update_microsleep refuses the episode.
        self._microsleep_withheld = ""

        # ── Pitch gate (hysteresis) ───────────────────────────────
        # EAR is a 2-D pixel ratio with no pose correction, so a head
        # pitched down foreshortens the eye opening and reads as closed
        # with the eyes wide open.  Suppress EAR tracking past the gate.
        # The release threshold is looser than the engage threshold so a
        # head dwelling near the line does not flap in and out, clearing
        # and rebuilding the EAR/blink/microsleep state on every flap.
        pitch_down = 0.0
        if face.face_detected:
            pitch_down = wrap_deg(baseline_pitch - head_pitch)
            threshold = EAR_PITCH_GATE_RELEASE_DEG if self._pitch_gated else EAR_PITCH_GATE_DEG
            self._pitch_gated = pitch_down > threshold
        else:
            self._pitch_gated = False

        # Over-gating watchdog: a baseline that no longer describes how
        # the user actually sits would keep the gate held indefinitely,
        # turning EAR-based fatigue detection off without saying so.
        if self._pitch_gated:
            if self._pitch_gated_since is None:
                self._pitch_gated_since = now
            elif (
                not self._pitch_gate_warned and now - self._pitch_gated_since >= _PITCH_GATE_WARN_S
            ):
                self._pitch_gate_warned = True
                log.warning(
                    "Pitch gate held %.0fs (head %.1f° below baseline) — "
                    "EAR-based fatigue detection is suppressed. The "
                    "calibrated baseline may no longer match seating posture.",
                    now - self._pitch_gated_since,
                    pitch_down,
                )
        else:
            self._pitch_gated_since = None
            self._pitch_gate_warned = False

        # ── Head-motion EAR suppression ───────────────────────────
        # The pitch gate asks whether the eye is foreshortened.  These two
        # guards ask whether the mesh solved the eye at all — a question a
        # fast head turn answers "no" while pitch stays perfectly neutral.
        # See the EAR_YAW_RATE_GATE_DPS block in constants.py.
        yaw_turning = False
        if face.face_detected:
            yaw_rate = 0.0
            interval_start = self._prev_yaw_time
            if self._prev_yaw is not None and interval_start is not None:
                dt_yaw = now - interval_start
                # Ignore a stalled or non-advancing interval: dividing the
                # yaw travelled across a long gap by that gap is not an
                # angular velocity the camera ever observed.
                if _TIME_EPS_S < dt_yaw <= _MAX_FRAME_DT_S:
                    yaw_rate = abs(wrap_deg(face.yaw - self._prev_yaw)) / dt_yaw
            if yaw_rate > EAR_YAW_RATE_GATE_DPS:
                # Anchor on the START of the interval that measured high, so
                # the sustain test spans measured travel rather than waiting
                # an extra frame for the clock to catch up.
                if self._yaw_rate_high_since is None:
                    self._yaw_rate_high_since = interval_start
                yaw_turning = (
                    self._yaw_rate_high_since is not None
                    and now - self._yaw_rate_high_since >= EAR_YAW_RATE_SUSTAIN_S
                )
            else:
                self._yaw_rate_high_since = None
            self._prev_yaw = face.yaw
            self._prev_yaw_time = now
        else:
            # Drop the anchors on face loss, so the first frame back does not
            # measure a rate across the whole absence.
            self._prev_yaw = None
            self._prev_yaw_time = None
            self._yaw_rate_high_since = None

        ear_degenerate = (
            face.face_detected and max(face.ear_left, face.ear_right) > EAR_PLAUSIBLE_MAX
        )

        suppress_now = self._pitch_gated or yaw_turning or ear_degenerate
        if suppress_now:
            self._ear_suppressed_until = now + EAR_GATE_SETTLE_S
        # The settling window keeps EAR suppressed after the last offending
        # frame: pitch normalises before the landmarks re-converge, and the
        # frames in between read as a closed eye on a wide-open one.
        self._ear_suppressed = suppress_now or now < self._ear_suppressed_until

        # ── Determine pause ───────────────────────────────────────
        # _ear_suppressed subsumes _pitch_gated (itself False whenever
        # face_detected is False) and adds the head-motion guards; it
        # deliberately reuses this same pause branch so PERCLOS
        # accrual, microsleep clearing and score decay behave identically
        # for every reason EAR is untrustworthy.  That is also what keeps
        # the artifact out of the PERCLOS buffer the microsleep gate is
        # tested against — suppressed frames contribute no samples.
        should_pause = presence != PresenceState.PRESENT or self._ear_suppressed

        if should_pause:
            if self._pause_start is None:
                self._pause_start = now
            # Clear microsleep state — EAR not observable.
            self._eyes_closed_start = None
            self._microsleep_fired = False
            self._sleep_fired = False
            self._eyes_open_since = None
            self._episode_frames = 0
            self._episode_bilateral_frames = 0
            self._episode_blink_scores.clear()
            self._prev_below_threshold = False
            self._current_blink_close_start = None
            self._ear_buf.clear()
            self._ear_buf_left.clear()
            self._ear_buf_right.clear()
            # Tier evidence HOLDS through a brief pause and is dropped only
            # once the pause is a genuine absence.  Paused frames bank no
            # above-threshold time (this branch returns before
            # _check_fatigue_level), and the 30 s timestamp eviction still
            # ages old evidence out, so holding cannot fabricate a sustain.
            # Clearing on every paused frame used to wipe the bank on each
            # ~1 s guard blip — 252 of them an hour — so a score held above
            # 0.6 for 27 s never banked the 15 s the warning needs.  See
            # FATIGUE_SUSTAIN_WINDOW_S.
            if now - self._pause_start >= _SHORT_PAUSE_THRESHOLD_S:
                self._clear_tier_windows()
            # Decay score during face-loss.
            if self._last_update_time is not None:
                dt = now - self._last_update_time
                self._fatigue_score = max(0.0, self._fatigue_score - FATIGUE_SCORE_DECAY_RATE * dt)
            self._last_update_time = now
            return

        # ── Resume from pause ─────────────────────────────────────
        resumed_from_pause = self._pause_start is not None
        if self._pause_start is not None:
            pause_duration = now - self._pause_start
            self._pause_start = None

            if pause_duration < _SHORT_PAUSE_THRESHOLD_S:
                self._window_active_s_total += pause_duration
                log.debug(
                    "Fatigue: short pause of %.1fs credited to blink window.",
                    pause_duration,
                )
            else:
                log.debug(
                    "Fatigue: long pause of %.1fs NOT credited (genuine absence).",
                    pause_duration,
                )

        # ── Advance active-time accumulator ───────────────────────
        # On a resume frame the elapsed interval IS the pause, which the
        # block above has already accounted for.  Adding dt as well would
        # double-count a short pause and silently credit a long one — the
        # "NOT credited" branch above would be dead code — while also
        # letting pause time leak into _window_active_s_observed, which
        # must stay pure observation time.
        #
        # ``frame_dt`` is the observation time this frame represents.  It is
        # 0.0 on the first frame and on a resume frame (the elapsed interval
        # there is pause time, not observation time) so neither can be
        # credited to PERCLOS.
        frame_dt = 0.0
        if self._last_update_time is not None and not resumed_from_pause:
            frame_dt = now - self._last_update_time
            self._window_active_s_observed += frame_dt
            self._window_active_s_total += frame_dt
            self._frame_dt_ema = (
                frame_dt
                if self._frame_dt_ema is None
                else _FRAME_DT_EMA_ALPHA * frame_dt
                + (1.0 - _FRAME_DT_EMA_ALPHA) * self._frame_dt_ema
            )
        self._last_update_time = now

        # ── Evict stale blinks ────────────────────────────────────
        self._evict_old_blinks(now)
        self._add_blink_observation(now, frame_dt)

        # ── EAR analysis ─────────────────────────────────────────
        ear = self._ear_buf.add(now, face.ear_avg)
        below = ear < self._blink_threshold

        # Per-eye classification for the microsleep bilateral test.  Smoothed
        # on the same window as the average so a single-frame landmark blip
        # on one eye cannot mark an otherwise bilateral frame as one-eyed.
        ear_left = self._ear_buf_left.add(now, face.ear_left)
        ear_right = self._ear_buf_right.add(now, face.ear_right)
        bilateral = ear_left < self._blink_threshold and ear_right < self._blink_threshold

        # Genuinely-shut test for the microsleep timer.  `below` is the
        # *blink* boundary; timing a microsleep against it counts a slow
        # partial reopening or a held squint as closure.  See
        # MICROSLEEP_SHUT_RATIO for the measurements.
        shut = ear < self._blink_threshold * MICROSLEEP_SHUT_RATIO

        # Feed PERCLOS buffer — deliberately from `below`, not `shut`.
        # PERCLOS is a duty-cycle statistic whose PERCLOS_GATE is calibrated
        # against sub-threshold time; switching it would silently move that
        # gate too.
        self._add_perclos_sample(now, below, frame_dt)

        # Blink duration tracking — track close start.
        if below and self._current_blink_close_start is None:
            self._current_blink_close_start = now

        # Microsleep detection — on shut time, not sub-threshold time.
        self._update_microsleep(shut, bilateral, now, face.blink_score)

        # Blink registration on closed→open transition.
        if self._prev_below_threshold and not below:
            # Record blink duration.
            if self._current_blink_close_start is not None:
                dur_ms = (now - self._current_blink_close_start) * 1000.0
                self._blink_durations.append(dur_ms)
                self._current_blink_close_start = None
            self._register_blink(now)

        if not below:
            self._current_blink_close_start = None  # reset if eyes open without transition

        self._prev_below_threshold = below

        # ── MAR / Yawning ─────────────────────────────────────────
        self._mar_buf.add(now, mar)
        self._update_yawning(now)

        # ── Head nod ─────────────────────────────────────────────
        # Deliberately INSIDE the non-paused path.  Moving it outside so
        # that it survives the pitch gate was measured against the logged
        # pitch streams and made things worse: nod count over 24 955 s of
        # tracking went 292 -> 654, and the time signal_head_nod sat pinned
        # at 1.0 went 7.3% -> 14.9%.  The old 12° gate had been suppressing
        # the detector during head-down stretches by accident, and that
        # accident was load-bearing.  Sweeping the nod parameters does not
        # rescue it — even at a 15° drop with a 1.0 s recovery window the
        # detector still fires 444 times (64/hour), saturating a 3-per-60 s
        # signal 10% of the time.  See the note on FATIGUE_W_HEAD_NOD.
        self._update_head_nod(head_pitch, baseline_pitch, now)

        # ── Blink frequency baseline ──────────────────────────────
        self._update_blink_frequency_baseline(now)

        # ── Compute composite score ───────────────────────────────
        self._fatigue_score = self._compute_fatigue_score()

        # ── Two-tier threshold checks ────────────────────────────
        self._check_fatigue_level(now, frame_dt)

    def reset(self) -> None:
        """Reset all internal state.

        Use when the session enters a rest period or re-calibrates
        after an Away state.
        """
        self._eyes_closed_start = None
        self._microsleep_fired = False
        self._sleep_fired = False
        self._eyes_open_since = None
        self._episode_frames = 0
        self._episode_bilateral_frames = 0
        self._episode_blink_scores.clear()
        self._prev_below_threshold = False
        self._current_blink_close_start = None
        self._ear_buf.clear()
        self._ear_buf_left.clear()
        self._ear_buf_right.clear()
        self._perclos_buf.clear()
        self._perclos_closed_s = 0.0
        self._perclos_total_s = 0.0
        self._mar_buf.clear()
        self._blink_times.clear()
        self._blink_durations.clear()
        self._yawn_times.clear()
        self._nod_times.clear()
        self._frame_dt_ema = None
        self._blink_obs_buf.clear()
        self._blink_obs_s = 0.0
        self._window_active_s_observed = 0.0
        self._window_active_s_total = 0.0
        self._last_update_time = None
        self._baseline_blink_rate = None
        self._baseline_blink_duration_ms = None
        self._baseline_captured = False
        self._windows_since_baseline = 0
        self._fatigue_score = 0.0
        self._clear_tier_windows()
        self._last_warning_time = 0.0
        self._last_alert_time = 0.0
        self._pause_start = None
        self._session_paused = False
        self._score_at_pause = 0.0
        self._yawn_start = None
        self._prev_yawning = False
        self._nod_state = "neutral"
        self._nod_drop_start = None
        self._nod_baseline_pitch = None
        self._pitch_gated = False
        self._pitch_gated_since = None
        self._pitch_gate_warned = False
        self._ear_suppressed = False
        self._prev_yaw = None
        self._prev_yaw_time = None
        self._yaw_rate_high_since = None
        self._ear_suppressed_until = 0.0
        log.debug("FatigueEngine state reset.")

    def on_session_pause(self) -> None:
        """Called when the user manually pauses the session.

        Snapshots the current fatigue score and wall-clock time so that
        ``on_session_resume`` can apply the correct decay or full reset
        depending on how long the break lasted.

        Also clears the in-flight eye state — the same fields the
        face-loss branch of :meth:`update` clears.  Those anchors are
        absolute timestamps, so a pause taken between an eye-close and
        the matching open would otherwise be charged to the closure:
        the first frame back registers a pause-long blink and can trip
        the microsleep override outright.
        """
        self._session_paused = True
        self._score_at_pause = self._fatigue_score
        self._pause_wall_time = time.monotonic()
        self._eyes_closed_start = None
        self._microsleep_fired = False
        self._sleep_fired = False
        self._eyes_open_since = None
        self._episode_frames = 0
        self._episode_bilateral_frames = 0
        self._episode_blink_scores.clear()
        self._prev_below_threshold = False
        self._current_blink_close_start = None
        self._ear_buf.clear()
        self._ear_buf_left.clear()
        self._ear_buf_right.clear()
        # Same reasoning as the eye anchors above: _prev_yaw_time is an
        # absolute timestamp, so a pause taken mid-turn would otherwise have
        # the first frame back divide by the whole pause.
        self._prev_yaw = None
        self._prev_yaw_time = None
        self._yaw_rate_high_since = None
        # Tier windows are timestamped the same way, and the score they
        # were judging is about to be decayed or reset by on_session_resume.
        self._clear_tier_windows()
        log.debug(
            "FatigueEngine: session paused (fatigue_score=%.3f).",
            self._score_at_pause,
        )

    def on_session_resume(self) -> None:
        """Called when the user resumes after a manual pause.

        If the break was 5+ minutes, the fatigue score is reset to 0.0
        and all rolling buffers are cleared (the user is considered
        rested).  For shorter breaks, the score decays proportionally
        to the pause duration.
        """
        self._session_paused = False
        pause_duration = time.monotonic() - self._pause_wall_time
        if pause_duration > 300.0:  # 5+ min break = rested
            self._fatigue_score = 0.0
            self._blink_durations.clear()
            self._yawn_times.clear()
            self._nod_times.clear()
            log.info(
                "FatigueEngine: long pause (%.0fs) — fatigue score reset to 0.",
                pause_duration,
            )
        else:
            decay = FATIGUE_SCORE_DECAY_RATE * pause_duration
            self._fatigue_score = max(0.0, self._score_at_pause - decay)
            log.debug(
                "FatigueEngine: resumed after %.0fs pause (score %.3f → %.3f, decay=%.3f).",
                pause_duration,
                self._score_at_pause,
                self._fatigue_score,
                decay,
            )
        self._last_update_time = None  # fresh dt on resume

    def set_baseline_ear(
        self,
        baseline_ear: float,
        baseline_ear_std: float = 0.0,
    ) -> None:
        """Update the calibrated EAR baseline (e.g. after re-calibration)."""
        self._baseline_ear = baseline_ear
        self._baseline_ear_std = baseline_ear_std
        self._blink_threshold = _compute_blink_threshold(baseline_ear, baseline_ear_std)
        self.reset()
        log.info(
            "Baseline EAR updated (ear=%.3f±%.3f, blink_thresh=%.3f).",
            baseline_ear,
            baseline_ear_std,
            self._blink_threshold,
        )

    # ── Five signal methods ──────────────────────────────────────

    def _perclos_signal(self) -> float:
        """Normalize PERCLOS to [0.0, 1.0].

        PERCLOS is already a fraction in [0, 1].  We scale so that
        PERCLOS_GATE (0.15) maps to 0.0 and PERCLOS_SIGNAL_SATURATION
        (0.40) maps to 1.0.  Below the gate: signal = 0.

        The upper anchor was an implicit 1.0 until it was measured: that
        asks for the eyes to be shut for a *whole* 30 s window before the
        heaviest-weighted signal in the model reads full scale, which is
        sleep rather than drowsiness.  Observed PERCLOS tops out near 0.8
        (p99 ~0.40, private corpus, n=1), so the ramp never left its bottom
        third and this signal's median contribution was 0.000.  See
        docs/decisions/0004-fatigue-score-and-tiers.md for the
        distributions and the FHWA/FMCSA anchoring.
        """
        p = self.perclos
        if p <= PERCLOS_GATE:
            return 0.0
        span = PERCLOS_SIGNAL_SATURATION - PERCLOS_GATE
        if span <= 0.0:
            # Degenerate configuration — treat any closure past the gate
            # as full scale rather than dividing by zero.
            return 1.0
        return min(1.0, (p - PERCLOS_GATE) / span)

    def _blink_duration_signal(self) -> float:
        """Normalize median blink duration against the user's own baseline.

        0.0 at the baseline duration captured for this session.
        1.0 at baseline × _BLINK_DURATION_FATIGUE_RATIO (clamped).

        Baseline-relative rather than absolute because the engine does not
        measure physiological full closure: it times the full width of a
        sub-threshold stretch at 0.70×baseline EAR on a median-filtered
        signal.  Measured resting medians ran 195–743 ms across 16 sessions
        of the *same* user, so any fixed "normal blink" anchor is below some
        users' resting level and pins this 0.25-weight signal at 1.0.
        Suppressed until the baseline is captured and has stabilized, in
        the same way as :meth:`_blink_freq_signal`.
        """
        if (
            not self._baseline_captured
            or self._baseline_blink_duration_ms is None
            or self._windows_since_baseline < _STABILIZATION_WINDOWS
            or not self._blink_durations
        ):
            return 0.0

        baseline = self._baseline_blink_duration_ms
        span = baseline * (_BLINK_DURATION_FATIGUE_RATIO - 1.0)
        if span <= 0.0:
            return 0.0

        # Median, as in _capture_baseline_blink_duration: durations are
        # right-skewed, so a mean read against a median baseline starts
        # above zero at rest (near full scale in one logged session,
        # private corpus, n=1), and one long stretch moves a mean on its
        # own.
        median_ms = statistics.median(self._blink_durations)
        return max(0.0, min(1.0, (median_ms - baseline) / span))

    def _yawn_signal(self) -> float:
        """Normalize yawn count in YAWN_WINDOW_S to [0.0, 1.0].

        Saturates at 3 yawns per window (score = 1.0).
        """
        now = time.monotonic()
        cutoff = now - YAWN_WINDOW_S
        count = sum(1 for t in self._yawn_times if t >= cutoff)
        return min(1.0, count / 3.0)  # saturate at 3 yawns

    def _blink_freq_signal(self) -> float:
        """Normalize blink-frequency drop to [0.0, 1.0].

        0.0 when rate is at or above baseline.
        1.0 when rate drops to _BLINK_FREQ_DROP_RATIO (50%) or lower.
        """
        if (
            not self._baseline_captured
            or self._baseline_blink_rate is None
            or self._baseline_blink_rate < _MIN_BASELINE_BLINK_RATE
            or self._windows_since_baseline < _STABILIZATION_WINDOWS
            # Too little of the window observed to judge a rate drop — and
            # a near-empty window would otherwise read as a full drop.
            or self._blink_obs_s < _BLINK_WINDOW_MIN_OBSERVED_S
        ):
            return 0.0

        current_rate = self.blink_rate
        baseline = self._baseline_blink_rate
        drop_threshold = baseline * _BLINK_FREQ_DROP_RATIO

        if current_rate >= baseline:
            return 0.0
        if current_rate <= drop_threshold:
            return 1.0
        # Linear interpolation between baseline and drop_threshold.
        return (baseline - current_rate) / (baseline - drop_threshold)

    def _head_nod_signal(self) -> float:
        """Normalize head nod count in HEAD_NOD_WINDOW_S to [0.0, 1.0].

        Saturates at _NOD_SIGNAL_SATURATE nods per window.
        """
        now = time.monotonic()
        cutoff = now - HEAD_NOD_WINDOW_S
        count = sum(1 for t in self._nod_times if t >= cutoff)
        return min(1.0, count / _NOD_SIGNAL_SATURATE)

    # ── Composite score ──────────────────────────────────────────

    def _compute_fatigue_score(self) -> float:
        """Weighted sum of the four scored signals → [0.0, 1.0].

        Head nod is deliberately absent: it is still detected and exposed
        through debug_nod_count / debug_signal_head_nod for the overlay and
        CSV, but it does not feed the score.  See FATIGUE_W_PERCLOS in
        constants.py for the measurements behind that removal.
        """
        score = (
            FATIGUE_W_PERCLOS * self._perclos_signal()
            + FATIGUE_W_BLINK_DUR * self._blink_duration_signal()
            + FATIGUE_W_YAWN * self._yawn_signal()
            + FATIGUE_W_BLINK_FREQ * self._blink_freq_signal()
        )
        return min(1.0, max(0.0, score))

    # ── Two-tier threshold check ─────────────────────────────────

    def _check_fatigue_level(self, now: float, frame_dt: float) -> None:
        """Check both alert tiers and fire events when sustained.

        "Sustained" is a duty cycle, not an unbroken run: each tier fires
        once its accumulated above-threshold time inside the trailing
        FATIGUE_SUSTAIN_WINDOW_S reaches its SUSTAIN_S.  The composite
        score is built from four near-independent signals that take turns
        rather than stack, so it crosses a threshold in bursts; the old
        unbroken-run rule reset to zero on any single frame below and
        discarded that evidence.  See FATIGUE_SUSTAIN_WINDOW_S.

        *frame_dt* is the observation time this frame represents — 0.0 on
        the first frame and on a resume frame, matching PERCLOS, so pause
        time is never credited as time spent above a threshold.
        """
        score = self._fatigue_score

        # ── Warning tier (score ≥ 0.6, 15 s of the last 30 s) ────
        above_s = self._accumulate_tier(
            self._warning_buf,
            now,
            frame_dt,
            score >= FATIGUE_WARNING_THRESHOLD,
            "_warning_above_s",
        )
        if above_s >= FATIGUE_WARNING_SUSTAIN_S:
            if (now - self._last_warning_time) >= FATIGUE_WARNING_COOLDOWN_S:
                self._fire_warning(above_s)
                # Clear rather than decay: the window that earned this fire
                # must not also earn the next one the moment the cooldown
                # lapses.  The cooldown bounds the rate; this bounds the
                # evidence.
                self._reset_tier(self._warning_buf, "_warning_above_s")

        # ── Alert tier (score ≥ 0.8, 10 s of the last 30 s) ──────
        above_s = self._accumulate_tier(
            self._alert_buf,
            now,
            frame_dt,
            score >= FATIGUE_ALERT_THRESHOLD,
            "_alert_above_s",
        )
        if above_s >= FATIGUE_ALERT_SUSTAIN_S:
            if (now - self._last_alert_time) >= FATIGUE_COOLDOWN_S:
                self._fire_alert("alert_threshold", above_s)
                self._reset_tier(self._alert_buf, "_alert_above_s")

    def _accumulate_tier(
        self,
        buf: deque[tuple[float, bool, float]],
        now: float,
        frame_dt: float,
        above: bool,
        sum_attr: str,
    ) -> float:
        """Add this frame to a tier's duty-cycle window; return above-time.

        Mirrors :meth:`_update_perclos`: evict before appending so the
        running sum in *sum_attr* stays in step with *buf*, and clamp
        *frame_dt* to _MAX_FRAME_DT_S so one stalled frame cannot inject
        seconds of above-threshold time it never measured.
        """
        dt = min(max(frame_dt, 0.0), _MAX_FRAME_DT_S)
        total = getattr(self, sum_attr)

        cutoff = now - FATIGUE_SUSTAIN_WINDOW_S
        while buf and buf[0][0] < cutoff:
            _, was_above, old_dt = buf.popleft()
            if was_above:
                total -= old_dt

        if dt > 0.0:
            buf.append((now, above, dt))
            if above:
                total += dt

        if not buf:
            total = 0.0  # window empty — clear accumulated float drift

        total = max(0.0, total)
        setattr(self, sum_attr, total)
        return total

    def _reset_tier(
        self,
        buf: deque[tuple[float, bool, float]],
        sum_attr: str,
    ) -> None:
        """Empty a tier's duty-cycle window and zero its running sum."""
        buf.clear()
        setattr(self, sum_attr, 0.0)

    def _clear_tier_windows(self) -> None:
        """Drop both tier duty-cycle windows.

        Called wherever the score itself stops being a live observation for
        long enough to matter — reset, manual pause, and an EAR-suppression /
        face-loss pause that has reached _SHORT_PAUSE_THRESHOLD_S — so stale
        above-threshold time cannot survive a genuine absence and combine
        with fresh evidence on the other side of it.  A *brief* pause
        deliberately does not call this; see the pause branch of update().
        """
        self._warning_buf.clear()
        self._alert_buf.clear()
        self._warning_above_s = 0.0
        self._alert_above_s = 0.0

    # ── Microsleep detection ─────────────────────────────────────

    def _update_microsleep(
        self,
        eyes_shut: bool,
        bilateral: bool,
        now: float,
        blink_score: float | None = None,
    ) -> None:
        """Track continuous eyes-*shut* duration for microsleep detection.

        *eyes_shut* is the strict test (``blink_threshold *
        MICROSLEEP_SHUT_RATIO``), not the blink-threshold test that drives
        PERCLOS and blink counting.  Timing the episode against the blink
        boundary counted partial aperture as closure — see
        MICROSLEEP_SHUT_RATIO.

        Two EVENT_FATIGUE triggers share this episode, both bypassing the
        sustain windows but respecting the shared 60 s cooldown:

        - *sleep* — the closure reaches SLEEP_CLOSURE_S.  Bilateral and
          blink gates only; no context required.
        - *microsleep* — the closure passes MICROSLEEP_THRESHOLD_S, every
          gate, and the user is already in fatigue context (see
          MICROSLEEP_CONTEXT_S).  Otherwise withheld as ``"context"``.

        *bilateral* says whether BOTH eyes were below threshold on this frame.
        The episode is still delimited by ``eyes_closed`` (the ``ear_avg``
        test) so its duration is unchanged; the bilateral frames are merely
        counted, and a mostly one-eyed episode is refused at the gate below.

        *blink_score* is the frame's eyeBlink blendshape, or None when the
        model reported none.  Samples are collected the same way and used
        the same way: the episode's duration is untouched, and a narrowed
        but open eye is refused at the gate.  See MICROSLEEP_BLINK_CONFIRM
        for why no EAR threshold can make this call.
        """
        if eyes_shut:
            self._eyes_open_since = None
            if self._eyes_closed_start is None:
                self._eyes_closed_start = now
                self._microsleep_fired = False
                self._sleep_fired = False
                self._episode_frames = 0
                self._episode_bilateral_frames = 0
                self._episode_blink_scores.clear()

            self._episode_frames += 1
            if bilateral:
                self._episode_bilateral_frames += 1
            if blink_score is not None:
                self._episode_blink_scores.append(blink_score)

            duration = now - self._eyes_closed_start

            # Sleep: a closure that has not ended.  No context and no PERCLOS
            # gate — someone dropping off suddenly has low prior PERCLOS and
            # a low score, and the length of the closure is itself the
            # evidence.  Only the bilateral and blink gates apply, which
            # refuse winks, collapsed solves and a lowered gaze (the 7.23 s
            # gaze-down episode in MICROSLEEP_BLINK_CONFIRM is that case).
            # Checked before the microsleep gates so their early returns
            # cannot withhold it.
            if duration >= SLEEP_CLOSURE_S and not self._sleep_fired:
                if self.debug_episode_bilateral_share >= MICROSLEEP_BILATERAL_SHARE and (
                    not self._episode_blink_scores
                    or self.debug_episode_blink_confirm >= MICROSLEEP_BLINK_CONFIRM
                ):
                    self._sleep_fired = True
                    self._microsleep_fired = True
                    self._fire_alert("sleep", duration)
                    return

            if duration > MICROSLEEP_THRESHOLD_S and not self._microsleep_fired:
                # The gate must exclude this episode's own closed time, or it
                # confirms itself: any closure past MICROSLEEP_THRESHOLD_S
                # donates >= 1.5/30 = 0.05 of the 0.15 it needs, and donates
                # more the longer it runs.  See _perclos_excluding.
                # Real eye closure is bilateral — the lids are yoked.  An
                # episode carried by one eye is a wink or a collapsed solve
                # on that side, not drowsiness.  Checked before the PERCLOS
                # gate because it is the cheaper and more specific test.
                share = self.debug_episode_bilateral_share
                if share < MICROSLEEP_BILATERAL_SHARE:
                    log.debug(
                        "Microsleep suppressed — episode not bilateral "
                        "(share=%.2f < %.2f, duration=%.2fs).",
                        share,
                        MICROSLEEP_BILATERAL_SHARE,
                        duration,
                    )
                    self._microsleep_withheld = "bilateral"
                    return

                # A lowered lid is not a shut lid.  EAR cannot tell the two
                # apart — it is a distance ratio, so a downward gaze and a
                # closure look the same — and the blendshape is a learned
                # appearance signal that can.  Median over the episode,
                # because per-frame scores overlap far more than
                # per-episode ones.  Empty means the model reported no
                # blendshapes: fail open rather than disable microsleep.
                confirm = self.debug_episode_blink_confirm
                if self._episode_blink_scores and confirm < MICROSLEEP_BLINK_CONFIRM:
                    log.debug(
                        "Microsleep suppressed — eyes narrowed, not shut "
                        "(blink=%.3f < %.2f, duration=%.2fs).",
                        confirm,
                        MICROSLEEP_BLINK_CONFIRM,
                        duration,
                    )
                    self._microsleep_withheld = "blink"
                    return

                independent = self._perclos_excluding(self._eyes_closed_start)
                if independent is None:
                    log.debug(
                        "Microsleep suppressed — insufficient independent "
                        "observation (duration=%.2fs).",
                        duration,
                    )
                    self._microsleep_withheld = "observation"
                elif independent < PERCLOS_GATE:
                    log.debug(
                        "Microsleep suppressed by PERCLOS gate "
                        "(independent=%.3f < gate=%.2f, raw=%.3f, "
                        "duration=%.2fs).",
                        independent,
                        PERCLOS_GATE,
                        self.perclos,
                        duration,
                    )
                    self._microsleep_withheld = "perclos"
                elif not (
                    self._fatigue_score >= FATIGUE_WARNING_THRESHOLD
                    or (
                        self._last_warning_time > 0.0
                        and now - self._last_warning_time <= MICROSLEEP_CONTEXT_S
                    )
                ):
                    # A lone microsleep in someone who otherwise looks alert
                    # is logged, not alerted — drinking is exactly this shape
                    # (a labelled false alert was the user drinking, private
                    # corpus, n=1).  Not latched: if the score reaches the
                    # warning tier during this closure it still fires.  See
                    # MICROSLEEP_CONTEXT_S.
                    log.debug(
                        "Microsleep withheld — no fatigue context (score=%.2f, duration=%.2fs).",
                        self._fatigue_score,
                        duration,
                    )
                    self._microsleep_withheld = "context"
                else:
                    self._microsleep_fired = True
                    self._fire_alert("microsleep", duration)
        else:
            if self._eyes_open_since is None:
                self._eyes_open_since = now
            open_for = now - self._eyes_open_since
            if (
                self._eyes_closed_start is not None
                and open_for >= _EYES_OPEN_CONFIRM_S - _TIME_EPS_S
            ):
                log.debug(
                    "Eyes-closed episode ended after %.2fs (confirmed after %.2fs open).",
                    self._eyes_open_since - self._eyes_closed_start,
                    open_for,
                )
                self._eyes_closed_start = None
                self._microsleep_fired = False
                self._sleep_fired = False
                self._episode_frames = 0
                self._episode_bilateral_frames = 0
                self._episode_blink_scores.clear()

    # ── PERCLOS helpers ──────────────────────────────────────────

    def _add_perclos_sample(self, now: float, closed: bool, dt: float) -> None:
        """Credit *dt* seconds of eye state to the PERCLOS window.

        Samples older than ``PERCLOS_WINDOW_S`` are evicted by timestamp,
        so the window is a true 30 s of observation at any frame rate.
        ``dt`` is clamped to ``_MAX_FRAME_DT_S`` so a single stalled frame
        cannot inject seconds of closure it never actually measured.
        """
        dt = min(max(dt, 0.0), _MAX_FRAME_DT_S)

        # Evict before appending: a maxlen-triggered drop would silently
        # desynchronise the running sums from the buffer.
        cutoff = now - PERCLOS_WINDOW_S
        while self._perclos_buf and self._perclos_buf[0][0] < cutoff:
            self._pop_perclos_sample()
        while len(self._perclos_buf) >= _PERCLOS_MAX_SAMPLES:
            self._pop_perclos_sample()

        if dt > 0.0:
            self._perclos_buf.append((now, closed, dt))
            self._perclos_total_s += dt
            if closed:
                self._perclos_closed_s += dt

        if not self._perclos_buf:
            # Nothing left in the window — clear accumulated float drift.
            self._perclos_total_s = 0.0
            self._perclos_closed_s = 0.0

    def _pop_perclos_sample(self) -> None:
        """Drop the oldest PERCLOS sample, keeping the running sums in step."""
        _, closed, dt = self._perclos_buf.popleft()
        self._perclos_total_s -= dt
        if closed:
            self._perclos_closed_s -= dt

    def _perclos_excluding(self, since: float) -> float | None:
        """PERCLOS over the window, ignoring samples recorded after *since*.

        Used to gate microsleep on evidence *independent* of the episode
        being judged.  ``self.perclos`` cannot do this: the in-progress
        closure feeds the same buffer the gate is tested against, so a
        long-enough closure helps clear the gate on its own — 10 of the 13
        recorded microsleep events fired in a 0.150–0.174 band, four of
        them at exactly the 0.150 gate value.

        Returns ``None`` when less than ``_PERCLOS_MIN_INDEPENDENT_S`` of
        observation remains outside the episode: too little to support a
        verdict either way, so the caller suppresses rather than guesses.

        Samples are appended in ``time.monotonic()`` order, so walking back
        from the tail and stopping at the first sample on or before *since*
        visits only the episode's own samples.  Boundary is ``<=``: a sample
        stamped exactly at the episode start is kept.
        """
        closed_s = self._perclos_closed_s
        total_s = self._perclos_total_s

        for ts, closed, dt in reversed(self._perclos_buf):
            if ts <= since:
                break
            total_s -= dt
            if closed:
                closed_s -= dt

        if total_s < _PERCLOS_MIN_INDEPENDENT_S:
            return None
        return max(0.0, closed_s) / total_s

    # ── Yawning ──────────────────────────────────────────────────

    def _update_yawning(self, now: float) -> None:
        """Detect yawn events from the MAR smoothing buffer."""
        if not self._mar_buf:
            return

        smoothed_mar = self._mar_buf.value
        yawning_now = smoothed_mar >= self._yawn_threshold

        if yawning_now and not self._prev_yawning:
            # Yawn opened.
            self._yawn_start = now
        elif not yawning_now and self._prev_yawning:
            # Yawn closed — check duration.
            if self._yawn_start is not None:
                dur = now - self._yawn_start
                if dur >= YAWN_MIN_DURATION_S:
                    self._yawn_times.append(now)
                    log.debug(
                        "Yawn detected (duration=%.2fs, total_5min=%d).", dur, self.debug_yawn_count
                    )
                self._yawn_start = None

        self._prev_yawning = yawning_now

    # ── Head nod ─────────────────────────────────────────────────

    def _update_head_nod(self, head_pitch: float, baseline_pitch: float, now: float) -> None:
        """Detect drowsiness-related head nods from pitch trajectory.

        A nod is: pitch drops ≥ HEAD_NOD_PITCH_DROP_DEG below the
        calibrated baseline, then recovers within HEAD_NOD_RECOVERY_S.
        """
        # Pitch below baseline = looking down (positive = head dropped).
        # Wrapped to the shortest arc so a head crossing the ±180°
        # boundary cannot register as a phantom ~360° nod.
        pitch_drop = wrap_deg(baseline_pitch - head_pitch)

        if self._nod_state == "neutral":
            if pitch_drop >= HEAD_NOD_PITCH_DROP_DEG:
                self._nod_state = "dropping"
                self._nod_drop_start = now
                self._nod_baseline_pitch = head_pitch

        elif self._nod_state == "dropping":
            if pitch_drop < HEAD_NOD_PITCH_DROP_DEG:
                # Head started recovering.
                self._nod_state = "recovering"
            elif self._nod_drop_start is not None:
                if (now - self._nod_drop_start) > HEAD_NOD_RECOVERY_S:
                    # Stayed down too long — not a drowsy nod, maybe intentional.
                    # Park in "recovering", NOT "neutral": the head is still
                    # down, so "neutral" re-arms "dropping" on the very next
                    # frame with a fresh anchor, and it re-arms again every
                    # HEAD_NOD_RECOVERY_S for as long as the user writes.  The
                    # eventual head-lift then measures elapsed from the last
                    # re-arm (≤3 s), passes the recovery check, and scores a
                    # phantom nod — every single time the user looks up from
                    # the desk.  Logged sessions show signal_head_nod = 1.00 on
                    # 5 of 20 fatigue events, all at near-neutral pitch.
                    # "recovering" with a null anchor lets the lift pass through
                    # counting nothing, and only then returns to "neutral" —
                    # where the head is up and cannot re-arm.
                    self._nod_state = "recovering"
                    self._nod_drop_start = None

        elif self._nod_state == "recovering":
            if pitch_drop < 2.0:
                # Recovered to near-neutral — count as nod.
                if self._nod_drop_start is not None:
                    elapsed = now - self._nod_drop_start
                    if elapsed <= HEAD_NOD_RECOVERY_S:
                        self._nod_times.append(now)
                        log.debug(
                            "Head nod detected (elapsed=%.2fs, count_1min=%d).",
                            elapsed,
                            self.debug_nod_count,
                        )
                self._nod_state = "neutral"
                self._nod_drop_start = None

    # ── Blink frequency helpers ──────────────────────────────────

    def _register_blink(self, now: float) -> None:
        """Record a blink timestamp in the rolling window."""
        self._blink_times.append(now)
        log.debug(
            "Blink registered (window count=%d, active=%.1fs).",
            len(self._blink_times),
            self._window_active_s_observed,
        )

    def _evict_old_blinks(self, now: float) -> None:
        """Remove blink timestamps older than the rolling window."""
        cutoff = now - _BLINK_WINDOW_S
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

    def _add_blink_observation(self, now: float, dt: float) -> None:
        """Credit *dt* seconds of observation to the blink-rate window.

        Mirrors :meth:`_add_perclos_sample`'s evict-before-append so the
        running sum stays in step with the buffer.  Unlike PERCLOS, *dt* is
        NOT clamped to _MAX_FRAME_DT_S: this replaces the unclamped
        ``_window_active_s_observed`` as the rate denominator, and a long
        frame interval is still time in which a blink would have been seen
        on either side of it.
        """
        dt = max(dt, 0.0)
        cutoff = now - _BLINK_WINDOW_S
        while self._blink_obs_buf and self._blink_obs_buf[0][0] < cutoff:
            _, old_dt = self._blink_obs_buf.popleft()
            self._blink_obs_s -= old_dt
        if dt > 0.0:
            self._blink_obs_buf.append((now, dt))
            self._blink_obs_s += dt
        if not self._blink_obs_buf:
            self._blink_obs_s = 0.0  # window empty — clear accumulated float drift

    def _capture_baseline_blink_duration(self) -> float | None:
        """Median of the durations currently in the rolling buffer, capped.

        Returns None when no blink has been recorded yet, so the caller can
        try again on a later window rather than locking in a missing value.
        """
        if not self._blink_durations:
            return None
        return min(
            statistics.median(self._blink_durations),
            _BLINK_DURATION_BASELINE_MAX_MS,
        )

    def _update_blink_frequency_baseline(self, now: float) -> None:
        """Capture and refine the baseline blink rate and blink duration."""
        if self._window_active_s_total < _BLINK_WINDOW_S:
            return

        current_rate = self.blink_rate

        if not self._baseline_captured:
            self._baseline_blink_rate = current_rate
            self._baseline_blink_duration_ms = self._capture_baseline_blink_duration()
            self._baseline_captured = True
            log.info(
                "Baseline blink rate captured: %.1f blinks/min (duration %s).",
                current_rate,
                "n/a"
                if self._baseline_blink_duration_ms is None
                else f"{self._baseline_blink_duration_ms:.0f} ms",
            )
            return

        # Refine baseline via EMA — only when rate is above the drop threshold.
        current_window = int(self._window_active_s_total / _BLINK_WINDOW_S) - 1
        if current_window > self._windows_since_baseline:
            if self._baseline_blink_rate is not None:
                drop_threshold = self._baseline_blink_rate * _BLINK_FREQ_DROP_RATIO
                if current_rate > drop_threshold:
                    self._baseline_blink_rate = (
                        _BASELINE_EMA_ALPHA * current_rate
                        + (1.0 - _BASELINE_EMA_ALPHA) * self._baseline_blink_rate
                    )

            if self._baseline_blink_duration_ms is None:
                # No blink had been recorded when the rate baseline was
                # taken — capture on the first window that has one.
                self._baseline_blink_duration_ms = self._capture_baseline_blink_duration()
            elif self._blink_durations:
                # Mirror the rate baseline's guard: refine only while the
                # current median is not itself in fatigue territory, so a
                # long drowsy stretch cannot drag the reference up behind
                # it and mask the very rise the signal exists to detect.
                # Median, matching capture and _blink_duration_signal.
                current_median = statistics.median(self._blink_durations)
                rise_threshold = self._baseline_blink_duration_ms * _BLINK_DURATION_FATIGUE_RATIO
                if current_median < rise_threshold:
                    self._baseline_blink_duration_ms = min(
                        _BASELINE_EMA_ALPHA * current_median
                        + (1.0 - _BASELINE_EMA_ALPHA) * self._baseline_blink_duration_ms,
                        _BLINK_DURATION_BASELINE_MAX_MS,
                    )
        self._windows_since_baseline = current_window

    # ── Fire helpers ─────────────────────────────────────────────

    def _fire_warning(self, sustain_duration: float) -> None:
        """Emit EVENT_FATIGUE_WARNING (warning tier)."""
        now_mono = time.monotonic()
        self._last_warning_time = now_mono

        event = FatigueEvent(
            event_type=EVENT_FATIGUE_WARNING,
            reason="warning_threshold",
            duration=sustain_duration,
            fatigue_score=self._fatigue_score,
            perclos=self.perclos,
            avg_blink_duration_ms=self.debug_avg_blink_duration_ms,
            yawn_count=self.debug_yawn_count,
            blink_rate=self.blink_rate,
            timestamp=time.time(),
        )
        log.info(
            "Fatigue WARNING: score=%.2f perclos=%.2f sustain=%.1fs.",
            event.fatigue_score,
            event.perclos,
            sustain_duration,
        )
        if self._callback is not None:
            self._callback(event)

    def _fire_alert(self, reason: str, duration: float) -> None:
        """Emit EVENT_FATIGUE (alert tier, contextual microsleep, or sleep).

        Subject to the shared FATIGUE_COOLDOWN_S (60 s) cooldown.
        """
        now_mono = time.monotonic()
        elapsed = now_mono - self._last_alert_time
        if elapsed < FATIGUE_COOLDOWN_S:
            log.debug(
                "Fatigue alert '%s' suppressed by cooldown (%.1fs remaining).",
                reason,
                FATIGUE_COOLDOWN_S - elapsed,
            )
            if reason == "microsleep":
                # Recorded so the CSV shows a qualifying episode was
                # swallowed.  This is the case that hid an entire
                # false-positive path in the labelled corpus.
                self._microsleep_withheld = "cooldown"
            return

        self._last_alert_time = now_mono

        event = FatigueEvent(
            event_type=EVENT_FATIGUE,
            reason=reason,
            duration=duration,
            fatigue_score=self._fatigue_score,
            perclos=self.perclos,
            avg_blink_duration_ms=self.debug_avg_blink_duration_ms,
            yawn_count=self.debug_yawn_count,
            blink_rate=self.blink_rate,
            timestamp=time.time(),
        )
        log.info(
            "Fatigue ALERT: reason=%s score=%.2f perclos=%.2f duration=%.2fs.",
            reason,
            event.fatigue_score,
            event.perclos,
            duration,
        )
        if self._callback is not None:
            self._callback(event)

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<FatigueEngine v2 "
            f"score={self._fatigue_score:.2f} "
            f"perclos={self.perclos:.2f} "
            f"blink_rate={self.blink_rate:.1f}/min>"
        )
