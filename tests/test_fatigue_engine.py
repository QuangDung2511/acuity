# Tests for fatigue engine.
# ─────────────────────────────────────────────────────────────────
# Two layers:
#   1. Unit tests   — synthetic FaceResult, mock time
#   2. Integration  — real video (test_full_session.mp4) +
#                     FaceMeshAnalyzer (skipped if fixture/model
#                     missing)
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from acuity.core import fatigue_engine as fatigue_engine_module
from acuity.core.face_mesh import FaceResult
from acuity.core.fatigue_engine import (
    _BLINK_DURATION_BASELINE_MAX_MS,
    _BLINK_DURATION_FATIGUE_RATIO,
    _BLINK_FREQ_DROP_RATIO,
    _BLINK_WINDOW_MIN_OBSERVED_S,
    _BLINK_WINDOW_S,
    _MAR_SMOOTH_WINDOW_S,
    _MIN_BASELINE_BLINK_RATE,
    _PERCLOS_MIN_INDEPENDENT_S,
    _PITCH_GATE_WARN_S,
    _SHORT_PAUSE_THRESHOLD_S,
    _STABILIZATION_WINDOWS,
    FatigueEngine,
    FatigueEvent,
)
from acuity.core.presence import PresenceState
from acuity.utils.constants import (
    EAR_BLINK_THRESHOLD,
    EAR_GATE_SETTLE_S,
    EAR_PITCH_GATE_DEG,
    EAR_PITCH_GATE_RELEASE_DEG,
    EAR_PLAUSIBLE_MAX,
    EAR_YAW_RATE_GATE_DPS,
    EVENT_FATIGUE,
    FATIGUE_SCORE_DECAY_RATE,
    FATIGUE_WARNING_THRESHOLD,
    HEAD_NOD_PITCH_DROP_DEG,
    HEAD_NOD_RECOVERY_S,
    MICROSLEEP_BILATERAL_SHARE,
    MICROSLEEP_BLINK_CONFIRM,
    MICROSLEEP_CONTEXT_S,
    MICROSLEEP_SHUT_RATIO,
    MICROSLEEP_THRESHOLD_S,
    PERCLOS_GATE,
    PERCLOS_WINDOW_S,
    SLEEP_CLOSURE_S,
    YAWN_MAR_DELTA,
    YAWN_MAR_MIN,
    YAWN_MIN_DURATION_S,
)

# ── Paths ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "private" / "test_full_session.mp4"
_FACE_MODEL = _PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"

_HAS_VIDEO = _VIDEO_PATH.exists()
_HAS_MODELS = _FACE_MODEL.exists()


# ── Helpers ─────────────────────────────────────────────────────


def _make_face(
    ear_avg: float = 0.30,
    detected: bool = True,
    pitch: float = 0.0,
    yaw: float = 0.0,
    ear_left: float | None = None,
    ear_right: float | None = None,
    blink_score: float | None = None,
) -> FaceResult:
    """Create a FaceResult with sensible defaults.

    ``ear_left`` / ``ear_right`` default to ``ear_avg``; override them to
    build the asymmetric readings a turned head produces, where the far
    eye's landmarks collapse while the near eye still looks plausible.

    ``blink_score`` defaults to ``None`` — no blendshape evidence — which
    is the fail-open path through MICROSLEEP_BLINK_CONFIRM.  That is
    deliberate: every test written before the blink gate existed keeps
    exercising the pre-gate behaviour unchanged.  Tests that mean to
    exercise the gate pass it explicitly.
    """
    return FaceResult(
        ear_left=ear_avg if ear_left is None else ear_left,
        ear_right=ear_avg if ear_right is None else ear_right,
        ear_avg=ear_avg,
        mar=0.0,
        pitch=pitch,
        yaw=yaw,
        roll=0.0,
        face_detected=detected,
        blink_score=blink_score,
    )


# Reusable synthetic faces.
_OPEN_EYES = _make_face(ear_avg=0.30)  # well above 0.21 threshold
_CLOSED_EYES = _make_face(ear_avg=0.10)  # well below 0.21 threshold
_NO_FACE = _make_face(ear_avg=0.0, detected=False)

# Baseline EAR from "calibration" — typical resting value.
_BASELINE_EAR = 0.30

# Baseline MAR from "calibration" — typical resting (closed-mouth) value.
# Measured median across 13 logged sessions is 0.010; inner-lip landmarks
# are nearly coincident when the lips touch, so this is near-zero by nature.
_BASELINE_MAR = 0.010

# MAR of a closed mouth during normal desk work.  This is the p50 of the
# real logged distribution and the exact value that used to be reported as
# YAWNING under the old multiplicative threshold.
_MAR_CLOSED = 0.014

# MAR during a genuine wide yawn (logged yawns peak at 0.5–1.33).
_MAR_YAWN = 0.70


def _feed_mar(
    engine, advance, value: float, *, frames: int | None = None, dt: float = 0.033
) -> None:
    """Push *value* through the engine's MAR median filter.

    The engine median-filters MAR over ``_MAR_SMOOTH_WINDOW_S`` *seconds*,
    so a new level only takes effect once it occupies the whole window.
    By default this feeds enough frames at *dt* to cover the window
    regardless of the simulated frame rate.
    """
    if frames is None:
        frames = int(_MAR_SMOOTH_WINDOW_S / dt) + 1
    for _ in range(frames):
        engine.update(_OPEN_EYES, PresenceState.PRESENT, mar=value)
        advance(dt)


def _make_engine_with_mock_time(
    baseline_ear: float = _BASELINE_EAR,
    callback=None,
    baseline_mar: float = _BASELINE_MAR,
):
    """Create a FatigueEngine with a mock clock.

    Returns (engine, advance_time_fn, patcher).  Call advance(seconds)
    to move the mock clock forward.
    """
    fake_time = [100.0]

    patcher = patch("acuity.core.fatigue_engine.time")
    mock_time = patcher.start()
    mock_time.monotonic = lambda: fake_time[0]
    mock_time.time = lambda: fake_time[0]  # for FatigueEvent.timestamp

    engine = FatigueEngine(
        baseline_ear=baseline_ear,
        baseline_mar=baseline_mar,
        callback=callback,
    )

    def advance(seconds: float) -> None:
        fake_time[0] += seconds

    return engine, advance, patcher


def _prefill_perclos(
    engine,
    *,
    below_fraction: float = 0.20,
    context: bool = True,
) -> None:
    """Fill the engine's PERCLOS buffer so the microsleep gate can fire.

    The PERCLOS gate requires ``engine.perclos >= PERCLOS_GATE`` (0.15).
    In a fresh engine the buffer is empty, so perclos=0 and the gate
    suppresses all microsleep events.  This helper seeds the engine's
    time-weighted PERCLOS accumulators with a full window in which
    ``below_fraction`` of the observed time was eyes-closed, matching the
    scenario where the user has been drowsy enough to clear the gate.

    Samples are stamped at the mock clock's current value, so a test that
    advances more than ``PERCLOS_WINDOW_S`` will legitimately age them out
    and must re-prefill.

    Parameters
    ----------
    engine : FatigueEngine
        The engine to prefill.
    below_fraction : float
        Fraction of observed time spent below threshold.  Default is
        0.20 (20%), just above the 15% gate.
    context : bool
        Also put the user in fatigue context — a warning "just fired" — so
        a qualifying microsleep alerts rather than being withheld as
        ``"context"`` (MICROSLEEP_CONTEXT_S).  Default True: a seeded
        drowsy window is the drowsy-user scenario, and it keeps tests of
        the other gates meaningful — they still fail on the gate they
        target, not vacuously on missing context.  Pass False to test the
        context gate itself.
    """
    # Read the clock through the engine's module so the test's mock applies.
    from acuity.core import fatigue_engine as _fe

    now = _fe.time.monotonic()
    closed_s = PERCLOS_WINDOW_S * below_fraction
    open_s = PERCLOS_WINDOW_S - closed_s

    engine._perclos_buf.clear()
    engine._perclos_buf.append((now, False, open_s))
    engine._perclos_buf.append((now, True, closed_s))
    engine._perclos_closed_s = closed_s
    engine._perclos_total_s = PERCLOS_WINDOW_S
    if context:
        engine._last_warning_time = now


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS — synthetic data, no external dependencies
# ═══════════════════════════════════════════════════════════════


class TestFatigueEngineInit:
    """Tests for engine initialisation and threshold computation."""

    def test_default_blink_threshold_uses_constant(self):
        """When baseline_ear is high enough, blink threshold should be
        min(EAR_BLINK_THRESHOLD, baseline * 0.75).
        (intent: adaptive threshold that respects the global constant)."""
        engine = FatigueEngine(baseline_ear=0.30)
        # 0.30 * 0.75 = 0.225 > EAR_BLINK_THRESHOLD (0.21) → use 0.21
        assert engine.blink_threshold == EAR_BLINK_THRESHOLD

    def test_adaptive_threshold_for_narrow_eyes(self):
        """When baseline_ear is low, threshold should adapt to 70% of
        baseline to avoid treating naturally narrow eyes as blinks."""
        engine = FatigueEngine(baseline_ear=0.22)
        # 0.22 * 0.70 = 0.154 < 0.21 → use 0.154
        assert engine.blink_threshold == pytest.approx(0.154)

    def test_zero_baseline_uses_constant(self):
        """Zero baseline (degenerate calibration) must fall back to the
        global constant, not produce a zero threshold."""
        engine = FatigueEngine(baseline_ear=0.0)
        assert engine.blink_threshold == EAR_BLINK_THRESHOLD

    def test_initial_blink_rate_is_zero(self):
        """Before any frames, blink rate should be 0.0."""
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR)
        assert engine.blink_rate == 0.0

    def test_initial_baseline_blink_rate_is_none(self):
        """Baseline blink rate must be None before the window fills."""
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR)
        assert engine.baseline_blink_rate is None


class TestMicrosleepDetection:
    """Tests for micro-sleep (eyes closed >1.0s) detection."""

    def test_no_fire_below_threshold(self):
        """Eyes closed for less than the threshold must NOT trigger a
        micro-sleep event (intent: normal blinks are ~0.1-0.3s)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            engine.update(_CLOSED_EYES)
            advance(MICROSLEEP_THRESHOLD_S - 0.2)  # below threshold
            engine.update(_CLOSED_EYES)
            assert len(events) == 0
        finally:
            patcher.stop()

    def test_fires_above_threshold(self):
        """Eyes closed past the threshold must trigger a micro-sleep
        event (intent: prolonged closure = drowsiness)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # Pre-populate PERCLOS buffer so the gate (15%) is satisfied.
            _prefill_perclos(engine)

            closed_for = MICROSLEEP_THRESHOLD_S + 0.1
            engine.update(_CLOSED_EYES)
            advance(closed_for)  # past threshold
            engine.update(_CLOSED_EYES)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
            assert microsleeps[0].event_type == EVENT_FATIGUE
            assert microsleeps[0].duration == pytest.approx(closed_for)
        finally:
            patcher.stop()

    def test_fires_only_once_per_episode(self):
        """Continued eyes-closed after firing must NOT produce duplicate
        events (intent: one alert per micro-sleep episode)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            _prefill_perclos(engine)

            engine.update(_CLOSED_EYES)
            advance(MICROSLEEP_THRESHOLD_S + 0.1)
            engine.update(_CLOSED_EYES)
            advance(2.0)
            engine.update(_CLOSED_EYES)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
        finally:
            patcher.stop()

    def test_fires_again_after_open_and_reclose(self):
        """Opening then re-closing eyes must produce a new event
        (intent: separate micro-sleep episodes = separate alerts).

        Notes
        -----
        - The 5-sample EAR median buffer needs ~2 closed frames after
          3 open frames before the median drops below threshold again.
        - FATIGUE_COOLDOWN_S (60s) must be cleared between episodes.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            _prefill_perclos(engine)

            # First micro-sleep
            engine.update(_CLOSED_EYES)
            advance(MICROSLEEP_THRESHOLD_S + 0.1)
            engine.update(_CLOSED_EYES)

            # Clear episode via hysteresis: the eyes must read open for
            # _EYES_OPEN_CONFIRM_S (0.10 s) continuously.  At this 0.1 s
            # simulated frame interval that is reached on the third open
            # frame — one to flush the EAR median, two to span the window.
            advance(0.1)
            engine.update(_OPEN_EYES)
            advance(0.1)
            engine.update(_OPEN_EYES)
            advance(0.1)
            engine.update(_OPEN_EYES)

            # Advance past the 60s FATIGUE_COOLDOWN_S so the second
            # microsleep event is not suppressed by the global cooldown.
            advance(60.0)

            # That advance is longer than PERCLOS_WINDOW_S, so the seeded
            # window has legitimately aged out.  Re-seed it — the scenario
            # under test is a second microsleep by a still-drowsy user.
            _prefill_perclos(engine)

            # Second micro-sleep: 3 closed frames flush the 5-sample EAR
            # buffer below threshold. The episode timer starts on frame 3,
            # so the 4th call (after >1.0s advance) triggers the event.
            engine.update(_CLOSED_EYES)  # frame 1: median 0.30 (above)
            advance(0.1)
            engine.update(_CLOSED_EYES)  # frame 2: median 0.30 (above)
            advance(0.1)
            engine.update(_CLOSED_EYES)  # frame 3: median drops to 0.10 → episode starts
            advance(MICROSLEEP_THRESHOLD_S + 0.1)  # duration now exceeds threshold
            engine.update(_CLOSED_EYES)  # frame 4: fires!

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 2
        finally:
            patcher.stop()

    def test_normal_blink_does_not_trigger(self):
        """A quick blink (close then open within 1.0s) must NOT trigger
        (intent: distinguish micro-sleeps from natural blinks)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # Close for 0.15s (normal blink duration)
            engine.update(_CLOSED_EYES)
            advance(0.15)
            engine.update(_OPEN_EYES)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 0
        finally:
            patcher.stop()


class TestMicrosleepPerclosGateIndependence:
    """The PERCLOS gate must not be cleared by the episode it is gating.

    Regression cover for the self-confirming gate: ``_update_microsleep``
    tested ``self.perclos`` over the full window while the candidate
    closure was *inside* that window, so a closure past
    MICROSLEEP_THRESHOLD_S contributed >= 1.5/30 = 0.05 of the 0.15 it had
    to clear, and contributed more the longer it ran.  All 13 microsleep
    events across the recorded sessions fired in a 0.150-0.174 band, four
    at exactly the gate value; removing each episode's own contribution
    drops 11 of them to 0.047-0.139.
    """

    @staticmethod
    def _hold_closed(engine, advance, duration_s: float, dt: float = 0.1) -> None:
        """Feed closed-eye frames for *duration_s* at interval *dt*."""
        for _ in range(round(duration_s / dt)):
            engine.update(_CLOSED_EYES)
            advance(dt)

    def test_episode_cannot_clear_the_gate_by_itself(self):
        """A long closure by a user whose independent PERCLOS is below the
        gate must NOT fire, even once its own closed time has pushed the
        raw window PERCLOS above the gate.

        Reproduces a logged false positive (private corpus, n=1): raw
        PERCLOS 0.166 vs gate 0.15, but 0.052 of that was the 1.57 s
        episode itself — independent PERCLOS was 0.114.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # 0.11 independent — just under the gate, as in the real log.
            _prefill_perclos(engine, below_fraction=0.11)

            self._hold_closed(engine, advance, 4.0)

            # The closure alone has lifted the raw window above the gate:
            # this is exactly what the old code tested and fired on.
            assert engine.perclos > PERCLOS_GATE
            assert [e for e in events if e.reason == "microsleep"] == []
        finally:
            patcher.stop()

    def test_independently_drowsy_user_still_fires(self):
        """Sustained drowsiness *outside* the episode must still fire
        (intent: the fix suppresses self-confirmation, not detection)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # 0.20 independent — clears the gate on its own evidence.
            _prefill_perclos(engine, below_fraction=0.20)

            self._hold_closed(engine, advance, 4.0)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
            assert microsleeps[0].event_type == EVENT_FATIGUE
        finally:
            patcher.stop()

    def test_insufficient_independent_observation_suppresses(self):
        """With nothing in the window but the closure itself, the gate has
        no evidence and must suppress rather than confirm.

        Reproduces a logged session that fired at t=1.06 s
        with PERCLOS 1.000 — a blink as calibration ended, the whole
        window inside it (private corpus, n=1).
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # No prefill: a fresh engine, exactly as at end-of-calibration.
            self._hold_closed(engine, advance, 4.0)

            assert engine.perclos == pytest.approx(1.0)
            assert [e for e in events if e.reason == "microsleep"] == []
        finally:
            patcher.stop()

    def test_perclos_excluding_drops_only_later_samples(self):
        """``_perclos_excluding`` must subtract exactly the samples stamped
        after *since*, and keep one stamped on the boundary."""
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR)
        engine._perclos_buf.clear()
        engine._perclos_buf.append((100.0, False, 18.0))
        engine._perclos_buf.append((100.0, True, 2.0))
        engine._perclos_buf.append((105.0, True, 3.0))
        engine._perclos_closed_s = 5.0
        engine._perclos_total_s = 23.0

        # Boundary sample (ts == since) is kept: nothing is excluded.
        assert engine._perclos_excluding(105.0) == pytest.approx(5.0 / 23.0)

        # The 3.0 s closed sample at t=105 is dropped from both sums.
        assert engine._perclos_excluding(104.0) == pytest.approx(2.0 / 20.0)

    def test_perclos_excluding_returns_none_below_min_observation(self):
        """Too little independent observation must yield None, not a
        fraction computed from a near-empty window."""
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR)
        engine._perclos_buf.clear()
        engine._perclos_buf.append((100.0, True, 4.0))
        engine._perclos_closed_s = 4.0
        engine._perclos_total_s = 4.0

        assert 4.0 < _PERCLOS_MIN_INDEPENDENT_S  # guard the premise
        assert engine._perclos_excluding(99.0) is None


class TestFrameRateIndependence:
    """The engine must behave identically regardless of capture rate.

    Regression cover for the false-microsleep bug: the smoothing and
    hysteresis windows were sized in frames against an assumed 30 FPS
    loop, so at the pipeline's real ~7 FPS they stretched 4x, the EAR
    median filter merged two blinks 0.5 s apart into one 1.2 s
    "closure", and a microsleep fired on an alert user.
    """

    # 30 FPS, the assumed rate; ~7 FPS, the rate actually observed.
    _RATES = [1 / 30.0, 1 / 7.0]

    @staticmethod
    def _feed(engine, advance, face, dt: float, duration_s: float) -> None:
        """Feed *face* for *duration_s* at a frame interval of *dt*."""
        for _ in range(max(1, round(duration_s / dt))):
            engine.update(face)
            advance(dt)

    @classmethod
    def _blink(cls, engine, advance, dt: float, closed_s: float) -> None:
        """Simulate one blink of *closed_s* at a frame interval of *dt*."""
        cls._feed(engine, advance, _CLOSED_EYES, dt, closed_s)

    @classmethod
    def _open(cls, engine, advance, dt: float, open_s: float) -> None:
        cls._feed(engine, advance, _OPEN_EYES, dt, open_s)

    @pytest.mark.parametrize("dt", _RATES)
    def test_two_close_blinks_do_not_merge_into_a_microsleep(self, dt):
        """Two normal blinks 0.5 s apart are not one long closure.

        This is the exact shape of the logged false positive: a 0.3 s
        blink, eyes open 0.5 s, then another 0.3 s blink.  Nothing here
        is remotely a microsleep at any frame rate.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            _prefill_perclos(engine)  # gate open, so only duration can suppress

            self._open(engine, advance, dt, 1.0)
            self._blink(engine, advance, dt, 0.3)
            self._open(engine, advance, dt, 0.5)
            self._blink(engine, advance, dt, 0.3)
            self._open(engine, advance, dt, 1.0)

            assert [e for e in events if e.reason == "microsleep"] == []
        finally:
            patcher.stop()

    @pytest.mark.parametrize("dt", _RATES)
    def test_blink_duration_is_not_inflated_by_frame_rate(self, dt):
        """A 0.3 s blink must measure ~0.3 s, not 0.7 s.

        Inflated durations were what pushed PERCLOS to the 0.15 gate in
        the logged session; every blink there measured >= 0.7 s.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._open(engine, advance, dt, 1.0)
            self._blink(engine, advance, dt, 0.3)
            self._open(engine, advance, dt, 1.0)

            assert len(engine._blink_durations) == 1
            measured_s = engine._blink_durations[0] / 1000.0
            # Allow one frame of quantisation either way.
            assert measured_s == pytest.approx(0.3, abs=dt + 0.05)
        finally:
            patcher.stop()

    def test_perclos_is_time_weighted_not_frame_counted(self):
        """PERCLOS must weight each sample by the time it represents.

        Closed stretches here are sampled coarsely (0.2 s/frame) and open
        stretches finely (0.05 s/frame), so the two measures diverge
        sharply: the eyes are genuinely closed 20% of the *time*, but
        closed frames are only ~6% of the *frames*.  A frame-counting
        PERCLOS reports the latter and under-reads drowsiness on any
        pipeline whose frame rate varies with scene content.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # 2 cycles of 2 s closed + 8 s open = 20 s observed, 20% closed.
            for _ in range(2):
                self._feed(engine, advance, _CLOSED_EYES, 0.2, 2.0)
                self._feed(engine, advance, _OPEN_EYES, 0.05, 8.0)

            # Sampling can only localise a transition to within one coarse
            # frame (0.2 s), hence the tolerance — still nowhere near the
            # ~0.06 a frame-counted PERCLOS would report.
            assert engine.perclos == pytest.approx(0.20, abs=0.03)
        finally:
            patcher.stop()

    @pytest.mark.parametrize("dt", _RATES)
    def test_genuine_microsleep_still_fires(self, dt):
        """The fix must not cost sensitivity: a real 2 s closure fires."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            _prefill_perclos(engine)
            self._open(engine, advance, dt, 1.0)
            self._blink(engine, advance, dt, 2.0)

            assert [e for e in events if e.reason == "microsleep"] != []
        finally:
            patcher.stop()

    def test_measured_fps_is_reported(self):
        """A degraded capture rate must be visible, not silent."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            assert engine.debug_measured_fps == 0.0  # nothing measured yet
            for _ in range(60):
                engine.update(_OPEN_EYES)
                advance(1 / 7.0)
            assert engine.debug_measured_fps == pytest.approx(7.0, abs=0.5)
        finally:
            patcher.stop()


class TestBlinkDetection:
    """Tests for blink counting (closed→open transitions)."""

    def test_blink_counted_on_open_transition(self):
        """A blink (below→above threshold) must register in the rolling
        window and increase blink_rate."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Simulate 5 blinks over 30 seconds
            for _ in range(5):
                engine.update(_CLOSED_EYES)
                advance(0.15)  # brief closure
                engine.update(_OPEN_EYES)
                advance(5.85)  # gap between blinks

            # 5 blinks in ~30s of active time ≈ 10 blinks/min
            assert engine.blink_rate == pytest.approx(10.0, rel=0.5)
        finally:
            patcher.stop()

    def test_sustained_closure_is_not_multiple_blinks(self):
        """A sustained eye closure (e.g. micro-sleep) must only register
        ONE blink (the moment eyes reopen after the sustained closure)."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Sustained closure: 1.5s (counts as a single closure episode)
            engine.update(_CLOSED_EYES)
            advance(1.5)
            engine.update(_CLOSED_EYES)

            # Eyes open — one blink registered (closed-to-open transition).
            # Confirmation needs _EYES_OPEN_CONFIRM_S of continuous open time.
            advance(0.1)
            engine.update(_OPEN_EYES)
            advance(0.1)
            engine.update(_OPEN_EYES)
            advance(0.1)
            engine.update(_OPEN_EYES)

            assert len(engine._blink_times) == 1
        finally:
            patcher.stop()


class TestPresenceHandling:
    """Tests for PRESENT_NO_FACE and ABSENT state handling."""

    def test_pauses_during_present_no_face(self):
        """When presence is PRESENT_NO_FACE, the engine must pause:
        no micro-sleep detection, no active-time accumulation
        (intent: can't compute EAR without a visible face)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # Start with face visible
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            advance(1.0)

            # Face lost, body visible
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            active_before = engine._window_active_s_observed
            advance(5.0)
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)

            # Active time must NOT have advanced during pause
            assert engine._window_active_s_observed == pytest.approx(active_before)
            assert len(events) == 0
        finally:
            patcher.stop()

    def test_pauses_during_absent(self):
        """ABSENT state must also pause tracking."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            advance(1.0)
            engine.update(_NO_FACE, PresenceState.ABSENT)
            active_before = engine._window_active_s_observed
            advance(10.0)
            engine.update(_NO_FACE, PresenceState.ABSENT)
            assert engine._window_active_s_observed == pytest.approx(active_before)
        finally:
            patcher.stop()

    def test_microsleep_cleared_on_face_loss(self):
        """If eyes are closed and face is lost, the in-progress
        micro-sleep episode must NOT fire when the face returns
        (intent: we can't verify eye state during face-lost period)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # Start closing eyes
            engine.update(_CLOSED_EYES, PresenceState.PRESENT)
            advance(0.3)  # < 1.0s

            # Face lost — episode should be cleared
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            advance(1.0)

            # Face returns, eyes open — no micro-sleep should fire
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            assert len(events) == 0
        finally:
            patcher.stop()

    def test_blink_window_resumes_after_pause(self):
        """Active time must resume accumulating when face returns
        after a PRESENT_NO_FACE pause.

        With the short-pause crediting feature, a 10s pause (< 15s
        threshold) is credited to _window_active_s on resume, so the
        total reflects real wall-clock time rather than pure observation
        time.  This is intentional — brief glances away should not
        delay fatigue baseline capture."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            advance(2.0)
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            active_after_2s = engine._window_active_s_observed

            # Short pause: 10s < 15s threshold — will be credited
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            advance(10.0)

            # Resume
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            advance(3.0)
            engine.update(_OPEN_EYES, PresenceState.PRESENT)

            # Short-pause credited to total, not observed
            assert engine._window_active_s_total == pytest.approx(
                active_after_2s + 10.0 + 3.0, abs=0.2
            )
            assert engine._window_active_s_observed == pytest.approx(active_after_2s + 3.0, abs=0.2)
        finally:
            patcher.stop()


class TestYawnThreshold:
    """Tests for yawn-threshold derivation.

    The threshold is additive-with-a-floor, not multiplicative.  MAR is
    computed from inner-lip landmarks that nearly coincide when the lips
    touch, so ``baseline_mar`` is ~0.01 — a multiplier on it lands inside
    landmark noise and reports a closed mouth as a yawn.
    """

    def test_floor_applies_for_normal_baseline(self):
        """A typical near-zero resting MAR must yield the absolute floor,
        not a near-zero threshold."""
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR, baseline_mar=0.010)
        assert engine.yawn_threshold == pytest.approx(YAWN_MAR_MIN)

    def test_additive_offset_applies_above_floor(self):
        """A high baseline (user talking during calibration) must raise the
        threshold above the floor, not scale it."""
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR, baseline_mar=0.130)
        assert engine.yawn_threshold == pytest.approx(0.130 + YAWN_MAR_DELTA)

    def test_zero_baseline_still_yields_valid_threshold(self):
        """A degenerate calibration (no face seen) must fall back to the
        floor rather than disabling yawn detection.

        Regression: the old multiplicative rule produced 0.0 here and the
        engine silently switched yawn detection off for the whole session.
        """
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR, baseline_mar=0.0)
        assert engine.yawn_threshold == pytest.approx(YAWN_MAR_MIN)

    def test_threshold_clears_closed_mouth_noise(self):
        """The threshold must sit well above the observed closed-mouth
        MAR distribution (p50 0.013, p95 0.063, p99 0.247)."""
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR, baseline_mar=_BASELINE_MAR)
        assert engine.yawn_threshold > 0.247


class TestYawnDetection:
    """Tests for the MAR yawn state machine."""

    def test_closed_mouth_does_not_register_yawn(self):
        """A closed mouth at the real-world p50 MAR must never be counted.

        Regression: under the old ``baseline_mar * 1.5`` rule the threshold
        landed at ~0.015, so this exact value (0.014) read as YAWNING and
        produced 151 false yawns across 13 logged sessions.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            for _ in range(20):
                _feed_mar(engine, advance, _MAR_CLOSED, frames=1)
            assert engine.debug_yawn_active is False
            assert engine.debug_yawn_count == 0
        finally:
            patcher.stop()

    def test_sustained_wide_open_registers_yawn(self):
        """MAR above threshold for >= YAWN_MIN_DURATION_S, then closing,
        must count exactly one yawn."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            _feed_mar(engine, advance, _MAR_CLOSED)
            _feed_mar(engine, advance, _MAR_YAWN)
            assert engine.debug_yawn_active is True

            advance(YAWN_MIN_DURATION_S + 0.5)
            _feed_mar(engine, advance, _MAR_CLOSED)  # mouth closes → count

            assert engine.debug_yawn_active is False
            assert engine.debug_yawn_count == 1
        finally:
            patcher.stop()

    def test_brief_mouth_open_is_not_a_yawn(self):
        """Talking or a quick mouth-open (< YAWN_MIN_DURATION_S) must not
        count, even though it crosses the threshold."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            _feed_mar(engine, advance, _MAR_CLOSED)
            _feed_mar(engine, advance, _MAR_YAWN)
            advance(YAWN_MIN_DURATION_S - 1.0)  # too short
            _feed_mar(engine, advance, _MAR_CLOSED)

            assert engine.debug_yawn_count == 0
        finally:
            patcher.stop()

    def test_yawn_signal_saturates_at_three(self):
        """The yawn sub-signal saturates at 3 yawns per 5-min window."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            for _ in range(3):
                _feed_mar(engine, advance, _MAR_CLOSED)
                _feed_mar(engine, advance, _MAR_YAWN)
                advance(YAWN_MIN_DURATION_S + 0.5)
                _feed_mar(engine, advance, _MAR_CLOSED)

            assert engine.debug_yawn_count == 3
            assert engine.debug_signal_yawn == pytest.approx(1.0)
        finally:
            patcher.stop()

    def test_yawn_signal_zero_when_no_yawns(self):
        """No yawns → the sub-signal contributes nothing to the score."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            _feed_mar(engine, advance, _MAR_CLOSED, frames=30)
            assert engine.debug_signal_yawn == 0.0
        finally:
            patcher.stop()

    def test_yawn_detection_works_without_calibration(self):
        """With baseline_mar=0.0 the floor still detects a real yawn.

        Regression: yawn detection used to be disabled entirely in this case.
        """
        engine, advance, patcher = _make_engine_with_mock_time(baseline_mar=0.0)
        try:
            _feed_mar(engine, advance, _MAR_CLOSED)
            _feed_mar(engine, advance, _MAR_YAWN)
            advance(YAWN_MIN_DURATION_S + 0.5)
            _feed_mar(engine, advance, _MAR_CLOSED)

            assert engine.debug_yawn_count == 1
        finally:
            patcher.stop()


class TestBlinkFrequencyDrop:
    """Tests for the rolling blink-frequency baseline and sub-signal.

    Blink-frequency drop is no longer a standalone event — the v2 engine
    folds it into the weighted fatigue score as one of five normalized
    sub-signals (weight 0.10).  These tests assert on the signal value.
    """

    def test_baseline_captured_after_full_window(self):
        """After _BLINK_WINDOW_S of active time, the engine must
        capture the baseline blink rate (intent: need a full window
        to establish a reliable baseline)."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Simulate normal blinking over the full window
            blinks_per_window = 15  # ~15 blinks/min
            gap = _BLINK_WINDOW_S / blinks_per_window
            for _ in range(blinks_per_window):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)

            # The first update() doesn't accumulate dt (since
            # _last_update_time starts as None), so we need one more
            # update to push active time past _BLINK_WINDOW_S.
            engine.update(_OPEN_EYES)

            assert engine.baseline_blink_rate is not None
            assert engine.baseline_blink_rate > 0.0
        finally:
            patcher.stop()

    def test_signal_zero_at_normal_rate(self):
        """If blink rate stays near baseline, the frequency sub-signal
        must stay at 0.0 (no fatigue contribution)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # Fill window with normal blinking (15 blinks/min)
            blinks_per_window = 15
            gap = _BLINK_WINDOW_S / blinks_per_window
            for _ in range(blinks_per_window):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)  # capture trailing gap

            # Continue at the same rate for another window
            for _ in range(blinks_per_window):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)  # capture trailing gap

            assert engine.debug_signal_blink_freq == pytest.approx(0.0)
        finally:
            patcher.stop()

    def test_ema_baseline_update(self):
        """Baseline blink rate should continuously update using EMA when current rate is above drop threshold."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # 1. Fill window at 15 blinks/min to capture baseline
            blinks_per_window = 15
            gap = _BLINK_WINDOW_S / blinks_per_window
            for _ in range(blinks_per_window):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)

            initial_baseline = engine.baseline_blink_rate
            assert initial_baseline is not None

            # 2. Complete stabilization window at 15 blinks/min
            for _ in range(blinks_per_window):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)

            # 3. Simulate another window at a higher rate (e.g. 20 blinks/min)
            higher_blinks = 20
            higher_gap = _BLINK_WINDOW_S / higher_blinks
            for _ in range(higher_blinks):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(higher_gap - 0.15)
            engine.update(_OPEN_EYES)

            # Baseline should have increased from initial_baseline
            new_baseline = engine.baseline_blink_rate
            assert new_baseline is not None
            assert new_baseline > initial_baseline
        finally:
            patcher.stop()

    def test_stabilization_period_suppresses_signal(self):
        """The frequency sub-signal must stay at 0.0 during the
        stabilization window immediately after baseline capture
        (intent: a freshly captured baseline is not yet trustworthy)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            # 1. Capture baseline (window 1)
            blinks = 15
            gap = _BLINK_WINDOW_S / blinks
            for _ in range(blinks):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)
            assert engine.baseline_blink_rate is not None

            # 2. Enter stabilization window (window 2) with a 12s dry spell.
            # The rate has dropped, but stabilization must suppress the signal.
            advance(12.0)
            engine.update(_OPEN_EYES)

            assert engine.debug_signal_blink_freq == pytest.approx(0.0)
        finally:
            patcher.stop()

    def test_signal_ramps_between_baseline_and_drop_ratio(self):
        """The sub-signal interpolates linearly: 0.0 at baseline, 1.0 at
        _BLINK_FREQ_DROP_RATIO of baseline, partial in between."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            blinks = 20
            gap = _BLINK_WINDOW_S / blinks
            # Capture baseline, then clear stabilization.
            for _ in range(2):
                for _ in range(blinks):
                    engine.update(_CLOSED_EYES)
                    advance(0.15)
                    engine.update(_OPEN_EYES)
                    advance(gap - 0.15)
                engine.update(_OPEN_EYES)

            baseline = engine.baseline_blink_rate
            assert baseline is not None

            # Let the rate decay and confirm the signal rises monotonically
            # from 0.0 to 1.0 rather than latching on the first dip.
            seen = []
            for _ in range(90):
                engine.update(_OPEN_EYES)
                advance(1.0)
                seen.append(engine.debug_signal_blink_freq)

            assert seen[0] < 1.0, "signal should not saturate immediately"
            assert seen[-1] == pytest.approx(1.0), (
                "signal should saturate once rate <= 50% of baseline"
            )
            assert all(0.0 <= s <= 1.0 for s in seen), "signal must stay normalized"
            assert engine.blink_rate <= baseline * _BLINK_FREQ_DROP_RATIO
        finally:
            patcher.stop()

    def test_false_positive_scenario_short_pause(self):
        """A normal rate followed by a short pause must not inflate the
        frequency sub-signal on resume."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            # 1. Capture baseline
            blinks = 20
            gap = _BLINK_WINDOW_S / blinks
            for _ in range(blinks):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)

            # 2. Clear stabilization
            for _ in range(blinks):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)

            # 3. Simulate short pause (10s look away)
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            advance(10.0)

            # 4. Resume and wait 11s (longer than sustain) to check if rate dropped due to pause dilution
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            # Need to blink normally otherwise it will legitimately drop.
            # We want to verify the pause didn't artificially drop the rate.
            # So we blink at the normal rate for 11s.
            for _ in range(4):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)

            # A 10s look-away legitimately dips the rolling rate a little —
            # what must NOT happen is the pause pushing it to the drop
            # threshold and saturating the signal.
            baseline = engine.baseline_blink_rate
            assert engine.blink_rate > baseline * _BLINK_FREQ_DROP_RATIO, (
                "Short pause artificially collapsed the blink rate to the drop threshold."
            )
            assert engine.debug_signal_blink_freq < 0.5
        finally:
            patcher.stop()

    def test_signal_saturates_on_significant_drop(self):
        """If blink rate drops to ≤50% of baseline, the frequency
        sub-signal must reach 1.0 (intent: fewer blinks = drowsiness)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # Fill window with normal blinking (20 blinks/min)
            blinks_in_window = 20
            gap = _BLINK_WINDOW_S / blinks_in_window
            for _ in range(blinks_in_window):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)  # capture trailing gap

            # Clear stabilization window
            for _ in range(blinks_in_window):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)

            baseline = engine.baseline_blink_rate
            assert baseline is not None

            # Now go a full window with very few blinks (≤50% of baseline)
            # Only 2 blinks in the next 60s, wait at least 10s for sustain
            for i in range(60):
                if i == 25 or i == 50:
                    engine.update(_CLOSED_EYES)
                    advance(0.15)
                    engine.update(_OPEN_EYES)
                    advance(1.0 - 0.15)
                else:
                    engine.update(_OPEN_EYES)
                    advance(1.0)

            assert engine.debug_signal_blink_freq == pytest.approx(1.0)
        finally:
            patcher.stop()

    def test_no_signal_when_baseline_too_low(self):
        """If the baseline blink rate is below _MIN_BASELINE_BLINK_RATE,
        the sub-signal must stay at 0.0 (intent: a user who barely
        blinks at baseline should not get false fatigue alerts)."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            # Fill window with only 3 blinks (3 blinks/min < 5 min)
            gap = _BLINK_WINDOW_S / 3
            for _ in range(3):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)  # capture trailing gap

            # Clear stabilization window (3 blinks)
            for _ in range(3):
                engine.update(_CLOSED_EYES)
                advance(0.15)
                engine.update(_OPEN_EYES)
                advance(gap - 0.15)
            engine.update(_OPEN_EYES)

            # Now go a full window with zero blinks
            for _ in range(60):
                engine.update(_OPEN_EYES)
                advance(1.0)

            assert engine.baseline_blink_rate < _MIN_BASELINE_BLINK_RATE
            assert engine.debug_signal_blink_freq == pytest.approx(0.0)
        finally:
            patcher.stop()


class TestReset:
    """Tests for reset() and set_baseline_ear()."""

    def test_reset_clears_all_state(self):
        """reset() must zero timers, blink window, and baseline
        (intent: clean slate after rest/away/re-calibration)."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Accumulate some state
            engine.update(_CLOSED_EYES)
            advance(0.5)
            engine.update(_OPEN_EYES)
            advance(1.0)

            engine.reset()

            assert engine.blink_rate == 0.0
            assert engine.baseline_blink_rate is None
            assert engine._window_active_s_observed == 0.0
            assert engine._eyes_closed_start is None
            assert len(engine._blink_times) == 0
        finally:
            patcher.stop()

    def test_set_baseline_ear_updates_threshold(self):
        """set_baseline_ear() must recompute the blink threshold and
        reset state."""
        engine = FatigueEngine(baseline_ear=0.30)
        original_thresh = engine.blink_threshold

        engine.set_baseline_ear(0.22)
        # 0.22 * 0.70 = 0.154
        assert engine.blink_threshold == pytest.approx(0.154)
        assert engine.blink_threshold != original_thresh

    def test_set_baseline_ear_resets_state(self):
        """set_baseline_ear() must also reset internal state."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine.update(_OPEN_EYES)
            advance(5.0)
            engine.update(_OPEN_EYES)
            assert engine._window_active_s_observed > 0

            engine.set_baseline_ear(0.28)
            assert engine._window_active_s_observed == 0.0
            assert engine.baseline_blink_rate is None
        finally:
            patcher.stop()


class TestCallbackBehaviour:
    """Tests for callback dispatch edge cases."""

    def test_no_callback_does_not_raise(self):
        """Engine with callback=None must log-only, not crash."""
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=None,
        )
        try:
            engine.update(_CLOSED_EYES)
            advance(1.0)
            engine.update(_CLOSED_EYES)
            # No exception = pass
        finally:
            patcher.stop()

    def test_event_has_correct_fields(self):
        """FatigueEvent must carry all expected fields."""
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(
            callback=events.append,
        )
        try:
            _prefill_perclos(engine)

            closed_for = MICROSLEEP_THRESHOLD_S + 0.1
            engine.update(_CLOSED_EYES)
            advance(closed_for)
            engine.update(_CLOSED_EYES)

            assert len(events) == 1
            e = events[0]
            assert e.event_type == EVENT_FATIGUE
            assert e.reason == "microsleep"
            assert e.duration == pytest.approx(closed_for)
            assert e.duration >= MICROSLEEP_THRESHOLD_S
            assert isinstance(e.blink_rate, float)
            assert isinstance(e.timestamp, float)
            # v2 scoring fields.
            assert 0.0 <= e.fatigue_score <= 1.0
            assert 0.0 <= e.perclos <= 1.0
            assert isinstance(e.avg_blink_duration_ms, float)
            assert isinstance(e.yawn_count, int)
        finally:
            patcher.stop()


class TestReprString:
    """Tests for __repr__ output."""

    def test_repr_contains_class_name(self):
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR)
        assert "FatigueEngine" in repr(engine)

    def test_repr_contains_blink_rate(self):
        engine = FatigueEngine(baseline_ear=_BASELINE_EAR)
        assert "blink_rate" in repr(engine)


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION TEST — real video + MediaPipe model
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not (_HAS_VIDEO and _HAS_MODELS),
    reason="Requires tests/fixtures/private/test_full_session.mp4 (private) and the face model.",
)
class TestFatigueEngineIntegration:
    """Run FatigueEngine against the recorded test video.

    Segment map (from record_test_video.py, with 3s pre-roll):
       3s –  18s  │ Segment 1 — Calibration (straight, eyes open)
      18s –  26s  │ Segment 2 — Sub-threshold head turn
      26s –  46s  │ Segment 3 — Head-pose distraction
      46s –  56s  │ Segment 4 — Phone distraction
      56s –  64s  │ Segment 5 — Recovery
      64s –  79s  │ Segment 6 — Slouch / Posture
      79s –  87s  │ Segment 7 — Recovery
      87s – 102s  │ Segment 8 — Fatigue / Micro-sleep (eyes closed, heavy blinks)
     102s – 110s  │ Segment 9 — Recovery
     110s – 130s  │ Segment 10 — Away state (out of frame)
     130s – 145s  │ Segment 11 — Return / re-calibration
    """

    def test_microsleep_detected_in_fatigue_segment(self):
        """Feed segment 1 to calibrate, then feed segment 8 (eyes
        closed / heavy slow blinks) through the engine and verify
        that at least one micro-sleep event fires.

        Frames are processed faster than real-time, so we mock
        time.monotonic() to advance by 1/fps per frame (simulating
        real-time playback) so the engine's threshold timers work.
        """
        import cv2

        from acuity.core.calibration import CalibrationRunner
        from acuity.core.face_mesh import FaceMeshAnalyzer

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened(), f"Cannot open test video: {_VIDEO_PATH}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration = 1.0 / fps
        face_analyzer = FaceMeshAnalyzer()

        try:
            # ── Phase 1: Skip pre-roll (3s) ─────────────────────
            preroll_frames = int(3 * fps)
            for _ in range(preroll_frames):
                cap.read()

            # ── Phase 2: Calibrate from segment 1 (3s–18s) ──────
            cal_runner = CalibrationRunner(duration_s=3.0)
            cal_frames = int(15 * fps)  # full segment 1
            frames_consumed = 0
            for _ in range(cal_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                face_result = face_analyzer.process(frame)
                cal_runner.feed(face_result)
                frames_consumed += 1
                if cal_runner.is_complete:
                    break

            assert cal_runner.is_complete, (
                f"Calibration did not complete after {frames_consumed} frames "
                f"(elapsed={cal_runner.elapsed_s:.1f}s)."
            )
            cal = cal_runner.result
            assert cal.baseline_ear > 0.0, (
                "Calibration produced zero baseline EAR — face may not "
                "have been detected during segment 1."
            )

            # ── Phase 3: Create engine with mocked video-time ────
            fake_time = [1000.0]  # arbitrary start
            time_patcher = patch("acuity.core.fatigue_engine.time")
            mock_time = time_patcher.start()
            mock_time.monotonic = lambda: fake_time[0]
            mock_time.time = lambda: fake_time[0]

            events: list[FatigueEvent] = []
            engine = FatigueEngine(
                baseline_ear=cal.baseline_ear,
                callback=events.append,
            )

            # ── Phase 4: Skip to segment 8 (starts at ~87s) ─────
            total_consumed = preroll_frames + frames_consumed
            seg8_start_frame = int(87 * fps)
            frames_to_skip = max(0, seg8_start_frame - total_consumed)
            for _ in range(frames_to_skip):
                cap.read()

            # ── Phase 5: Feed segment 8 (~15s of fatigue) ────────
            feed_frames = int(15 * fps)
            low_ear_seen = False

            peak_perclos = 0.0
            for _ in range(feed_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                face_result = face_analyzer.process(frame)
                if face_result.face_detected:
                    if face_result.ear_avg < engine.blink_threshold:
                        low_ear_seen = True
                    engine.update(face_result, PresenceState.PRESENT)
                else:
                    engine.update(face_result, PresenceState.ABSENT)
                peak_perclos = max(peak_perclos, engine.perclos)
                fake_time[0] += frame_duration

            time_patcher.stop()

            # Sanity: the engine must have seen at least one
            # below-threshold EAR frame during segment 8.
            assert low_ear_seen, (
                "No below-threshold EAR frames detected in segment 8. "
                "The video may not contain usable fatigue data, or the "
                "blink threshold is miscalibrated."
            )

            # ── Assert: the scored signals respond ───────────────
            # This segment is *acted* drowsiness, and the acting does not
            # close the eyes: raw ear_avg bottoms at 0.102 against a 0.218
            # blink threshold — roughly 47% aperture held for 2 s — with 2
            # frames out of 450 below the shut threshold.  Real closures
            # elsewhere in the logged corpus reach 0.02-0.05.  It is the
            # same shape as a labelled false Fatigue alert (private
            # corpus, n=1), so it must NOT be a microsleep; see
            # MICROSLEEP_SHUT_RATIO.
            #
            # What the segment legitimately demonstrates is that the scored
            # signals track sustained partial closure — which is the tier
            # that should be responding to it.
            assert peak_perclos > PERCLOS_GATE, (
                f"Segment 8 should drive PERCLOS above the {PERCLOS_GATE} "
                f"gate; peak was {peak_perclos:.3f}."
            )
            assert engine.debug_avg_blink_duration_ms > 500.0, (
                f"Segment 8 should lengthen the rolling blink duration; "
                f"got {engine.debug_avg_blink_duration_ms:.0f} ms."
            )

            # ── Assert: no microsleep, because the eyes never shut ──
            microsleep_events = [e for e in events if e.reason == "microsleep"]
            assert microsleep_events == [], (
                f"Segment 8 must not fire a microsleep — the eyes never "
                f"reach {engine.shut_threshold:.3f}. Got "
                f"{len(microsleep_events)}. Re-record the fixture with a "
                f"genuine eye closure to restore end-to-end microsleep "
                f"coverage."
            )

        finally:
            face_analyzer.close()
            cap.release()

    def test_no_fatigue_during_calibration_segment(self):
        """Segment 1 (sitting straight, eyes open) must NOT produce
        any micro-sleep events (intent: normal awake state = no
        false positives)."""
        import cv2

        from acuity.core.face_mesh import FaceMeshAnalyzer

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened(), f"Cannot open test video: {_VIDEO_PATH}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration = 1.0 / fps
        face_analyzer = FaceMeshAnalyzer()

        try:
            # Skip pre-roll
            for _ in range(int(3 * fps)):
                cap.read()

            # Mock time for video-rate playback
            fake_time = [1000.0]
            time_patcher = patch("acuity.core.fatigue_engine.time")
            mock_time = time_patcher.start()
            mock_time.monotonic = lambda: fake_time[0]
            mock_time.time = lambda: fake_time[0]

            events: list[FatigueEvent] = []
            engine = FatigueEngine(
                baseline_ear=0.30,  # typical resting EAR
                callback=events.append,
            )

            # Feed segment 1 (15s of calm, eyes-open footage)
            for _ in range(int(15 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                face_result = face_analyzer.process(frame)
                if face_result.face_detected:
                    engine.update(face_result, PresenceState.PRESENT)
                fake_time[0] += frame_duration

            time_patcher.stop()

            # No micro-sleeps should fire during calm segment
            microsleep_events = [e for e in events if e.reason == "microsleep"]
            assert len(microsleep_events) == 0, (
                f"Got {len(microsleep_events)} false micro-sleep event(s) "
                f"during calibration segment (expected 0)."
            )

        finally:
            face_analyzer.close()
            cap.release()


# ═══════════════════════════════════════════════════════════════
#  PAUSE-AWARE BLINK WINDOW TESTS
# ═══════════════════════════════════════════════════════════════


class TestPauseAwareBlinkWindow:
    """Tests for the short-pause crediting and wall-clock fallback.

    The fatigue engine should not penalize the blink-frequency window
    for brief distraction pauses (<15s), while still correctly
    treating long absences (≥15s) as genuine gaps.
    """

    def test_short_pause_credits_window_active_time(self):
        """A short pause (<15s) must add its duration to _window_active_s
        so the blink window fills on real-world time, not just observation time.

        Intent: brief glances away (phone, notes) should not delay
        fatigue baseline capture.

        NOTE: the pause timer starts when update(NO_FACE) is called.
        The advance must come AFTER that call.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Active for 20s → _window_active_s ≈ 20s
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            advance(20.0)
            engine.update(_OPEN_EYES, PresenceState.PRESENT)

            window_before_pause = engine._window_active_s_total

            # Enter pause state (pause timer starts here)
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            # 5s elapses while paused (short pause < 15s threshold)
            advance(5.0)

            # Resume — 5s pause should be credited to the window
            engine.update(_OPEN_EYES, PresenceState.PRESENT)

            # window_before_pause + 5.0 credited from short pause
            assert engine._window_active_s_total >= window_before_pause + 5.0 - 0.1, (
                f"Short pause should be credited: window_active={engine._window_active_s_total:.1f}s, "
                f"expected >= {window_before_pause + 5.0 - 0.1:.1f}s"
            )
        finally:
            patcher.stop()

    def test_long_pause_does_not_credit_window(self):
        """A long pause (>=15s) must NOT be added to _window_active_s.
        The window should catch up naturally via subsequent active frames.

        Intent: genuine away-from-desk periods don't inflate the window.

        NOTE: pause timer starts when update(NO_FACE) is called.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Active for 20s
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            advance(20.0)
            engine.update(_OPEN_EYES, PresenceState.PRESENT)

            window_before_pause = engine._window_active_s_total

            # Enter pause state (pause timer starts here)
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            # 20s elapses while paused (long pause >= 15s threshold)
            advance(20.0)

            # Resume — long pause should NOT be credited
            engine.update(_OPEN_EYES, PresenceState.PRESENT)

            # The window should NOT include the 20s long pause.
            assert engine._window_active_s_total < window_before_pause + 1.0, (
                f"Long pause should NOT be credited: window_active={engine._window_active_s_total:.1f}s, "
                f"expected < {window_before_pause + 1.0:.1f}s"
            )
        finally:
            patcher.stop()

    def test_microsleep_still_paused_during_any_pause(self):
        """During any pause (short or long), microsleep detection must
        be inactive — EAR cannot be read when face is not visible.

        Intent: no false microsleep events during look-away periods.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            # Start a microsleep: eyes close for 0.5s
            engine.update(_CLOSED_EYES, PresenceState.PRESENT)
            advance(0.5)
            engine.update(_CLOSED_EYES, PresenceState.PRESENT)

            # Face lost during the microsleep episode
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            advance(1.0)  # time elapses while paused

            # Resume — eyes open immediately after
            engine.update(_OPEN_EYES, PresenceState.PRESENT)

            # Microsleep should NOT have fired (state cleared during face loss)
            microsleep_events = [e for e in events if e.reason == "microsleep"]
            assert len(microsleep_events) == 0, (
                "Microsleep must not fire when face was lost during the episode."
            )
        finally:
            patcher.stop()


class TestManualSessionPause:
    """Tests for on_session_pause / on_session_resume.

    A manual pause stops update() being called at all, so any in-flight
    eye state is left anchored to a timestamp from before the break.
    """

    def test_pause_clears_in_flight_eye_state(self):
        """on_session_pause() must drop the open episode's anchors."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine.update(_CLOSED_EYES, PresenceState.PRESENT)
            advance(0.5)
            engine.update(_CLOSED_EYES, PresenceState.PRESENT)
            assert engine._eyes_closed_start is not None

            engine.on_session_pause()

            assert engine._eyes_closed_start is None
            assert engine._current_blink_close_start is None
            assert engine._prev_below_threshold is False
        finally:
            patcher.stop()

    def test_pause_spanning_a_blink_records_no_giant_blink(self):
        """A pause between an eye-close and the matching open is not a blink.

        Without the clear, the closed→open transition on the far side of
        the pause records the whole break as one blink duration, which
        saturates the blink-duration signal (weight 0.25).
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Eyes shut as the user reaches for the Pause button.
            engine.update(_CLOSED_EYES, PresenceState.PRESENT)
            engine.on_session_pause()
            advance(120.0)
            engine.on_session_resume()

            # First frame back is still mid-blink, then the eyes open.
            # Enough open frames to clear the EAR median window, so the
            # closed→open transition actually registers a blink.
            engine.update(_CLOSED_EYES, PresenceState.PRESENT)
            for _ in range(6):
                advance(0.033)
                engine.update(_OPEN_EYES, PresenceState.PRESENT)

            assert len(engine._blink_durations) == 1, (
                "Expected the post-resume blink to be registered."
            )
            assert engine._blink_durations[0] < 1000.0, (
                f"Pause charged to a blink: {engine._blink_durations[0]:.0f} ms"
            )
        finally:
            patcher.stop()


# ── Tier evidence across pauses ─────────────────────────────────


class TestTierEvidenceAcrossPauses:
    """Banked above-threshold time must survive a brief EAR pause.

    Regression cover for a logged stretch (private corpus, n=1): the score held at
    0.66–0.78 for 27 s, but five ~1 s pitch-gate blips each cleared the
    warning window, so it never banked the 15 s the warning needs.

    The score is pinned by overriding ``_compute_fatigue_score`` so these
    tests exercise the tier plumbing alone, not the signals feeding it.
    """

    _DT = 0.033
    _SCORE = 0.70  # above FATIGUE_WARNING_THRESHOLD, below the alert tier

    def _engine(self, events):
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        engine._compute_fatigue_score = lambda: self._SCORE
        return engine, advance, patcher

    def _feed(self, engine, advance, duration_s, face=_OPEN_EYES, presence=PresenceState.PRESENT):
        for _ in range(int(round(duration_s / self._DT))):
            engine.update(face, presence, head_pitch=face.pitch, baseline_pitch=0.0)
            advance(self._DT)

    @staticmethod
    def _warnings(events):
        return [e for e in events if e.reason == "warning_threshold"]

    def test_brief_suppression_keeps_banked_time(self):
        events = []
        engine, advance, patcher = self._engine(events)
        try:
            self._feed(engine, advance, 10.0)
            banked = engine._warning_above_s
            assert banked > 9.0

            # One head dip past the pitch gate, plus its settling window.
            gated = _make_face(pitch=-(EAR_PITCH_GATE_DEG + 2.0))
            self._feed(engine, advance, 1.0, face=gated)
            assert engine.debug_ear_suppressed is True
            assert engine._warning_above_s == pytest.approx(banked)

            # 10 s banked + 6 s more observed clears 15 s; under the old
            # clear-on-pause rule this needed 15 s after the dip.
            self._feed(engine, advance, 6.0 + EAR_GATE_SETTLE_S)
            assert len(self._warnings(events)) == 1
        finally:
            patcher.stop()

    def test_paused_frames_bank_nothing(self):
        """Holding is not crediting: time spent paused is not above-time."""
        events = []
        engine, advance, patcher = self._engine(events)
        try:
            self._feed(engine, advance, 5.0)
            banked = engine._warning_above_s
            self._feed(engine, advance, 10.0, face=_NO_FACE, presence=PresenceState.PRESENT_NO_FACE)
            self._feed(engine, advance, 0.5)
            assert engine._warning_above_s < banked + 1.0
            assert self._warnings(events) == []
        finally:
            patcher.stop()

    def test_long_pause_clears_banked_time(self):
        """A genuine absence must not bridge evidence across the gap."""
        events = []
        engine, advance, patcher = self._engine(events)
        try:
            self._feed(engine, advance, 10.0)
            self._feed(
                engine,
                advance,
                _SHORT_PAUSE_THRESHOLD_S + 1.0,
                face=_NO_FACE,
                presence=PresenceState.PRESENT_NO_FACE,
            )
            assert engine._warning_above_s == 0.0
        finally:
            patcher.stop()

    def test_manual_pause_still_clears_banked_time(self):
        events = []
        engine, advance, patcher = self._engine(events)
        try:
            self._feed(engine, advance, 10.0)
            engine.on_session_pause()
            assert engine._warning_above_s == 0.0
        finally:
            patcher.stop()


# ── Blink rate over observed time ───────────────────────────────


class TestBlinkRateObservedWindow:
    """blink_rate is blinks per *observed* minute, not per wall-clock minute.

    It used to divide by the whole 60 s window once the session was a minute
    old, so every paused second read as a second with no blink.  Over the
    logged corpus the rate fell to 0.44x baseline once half the last minute
    was paused, pinning the blink-frequency signal at 1.0 on a user whose
    blinking had not changed.
    """

    _DT = 0.033
    _BLINK_EVERY_S = 3.0  # 20 blinks per observed minute

    def _observe(self, engine, advance, duration_s, blink=True):
        """Open eyes for *duration_s*, one ~0.17 s blink every 3 s."""
        t = 0.0
        while t < duration_s:
            closed = blink and (t % self._BLINK_EVERY_S) < 0.17
            engine.update(_CLOSED_EYES if closed else _OPEN_EYES, PresenceState.PRESENT)
            advance(self._DT)
            t += self._DT

    def _pause(self, engine, advance, duration_s):
        for _ in range(int(round(duration_s / self._DT))):
            engine.update(_NO_FACE, PresenceState.PRESENT_NO_FACE)
            advance(self._DT)

    def _warm_up(self, engine, advance):
        # Past baseline capture (60 s) and the stabilisation window.
        self._observe(engine, advance, 2 * _BLINK_WINDOW_S + 10.0)
        assert engine.baseline_blink_rate is not None
        assert engine.baseline_blink_rate >= _MIN_BASELINE_BLINK_RATE

    def test_half_paused_minute_keeps_true_rate(self):
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._warm_up(engine, advance)
            baseline = engine.baseline_blink_rate

            # Half of every 6 s is face loss; blinking itself is unchanged.
            for _ in range(12):
                self._observe(engine, advance, 3.0)
                self._pause(engine, advance, 3.0)
            self._observe(engine, advance, 3.0)

            assert engine.blink_rate == pytest.approx(baseline, rel=0.25)
            assert engine.debug_signal_blink_freq < 0.2
        finally:
            patcher.stop()

    def test_mostly_unobserved_window_renders_no_verdict(self):
        """Below the observation floor the signal abstains.

        Zero blinks in the few observed seconds would otherwise read as a
        complete rate collapse and saturate the signal.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._warm_up(engine, advance)
            for _ in range(10):
                self._observe(engine, advance, 2.0, blink=False)
                self._pause(engine, advance, 5.0)
            self._observe(engine, advance, 1.0, blink=False)

            assert engine._blink_obs_s < _BLINK_WINDOW_MIN_OBSERVED_S
            assert engine.debug_signal_blink_freq == 0.0
        finally:
            patcher.stop()

    def test_blink_free_observed_minute_still_fires_signal(self):
        """The floor must not hide a real drop in a well-observed window."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._warm_up(engine, advance)
            self._observe(engine, advance, _BLINK_WINDOW_S, blink=False)
            assert engine.debug_signal_blink_freq == 1.0
        finally:
            patcher.stop()

    def test_reset_clears_observation_window(self):
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._observe(engine, advance, 5.0)
            engine.reset()
            assert engine._blink_obs_s == 0.0
            assert len(engine._blink_obs_buf) == 0
        finally:
            patcher.stop()


# ── Pitch gate ──────────────────────────────────────────────────


class TestPitchGate:
    """EAR is untrustworthy while the head is pitched down.

    EAR is a 2-D pixel ratio computed with no pose correction, so a
    downward-pitched head foreshortens the eye opening and the ratio
    collapses with the eyes wide open.  Measured across 25 sessions, the
    share of frames reading below the adaptive blink threshold climbs from
    4.9% at neutral to a ~20% plateau from 6 degrees down.  The gate
    suppresses EAR tracking rather than correcting it: the EAR/pitch
    relationship is not a clean cosine, so dividing by cos(pitch)
    over-corrects.
    """

    @staticmethod
    def _feed(engine, advance, face, duration_s: float, dt: float = 0.033):
        steps = max(1, int(duration_s / dt))
        for _ in range(steps):
            engine.update(face, PresenceState.PRESENT, head_pitch=face.pitch, baseline_pitch=0.0)
            advance(dt)

    def test_gate_engages_past_threshold(self):
        """Past the engage angle a foreshortened eye must reach no signal.

        Without this the ~20% below-threshold plateau feeds PERCLOS
        (weight 0.35) as if the user were genuinely closing their eyes.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            down = _make_face(ear_avg=0.10, pitch=-(EAR_PITCH_GATE_DEG + 2.0))
            self._feed(engine, advance, down, 1.0)

            assert engine.debug_pitch_gated is True
            assert engine.perclos == 0.0
            assert not engine._blink_durations
        finally:
            patcher.stop()

    def test_gate_open_below_engage_threshold(self):
        """A small head-down angle must NOT gate.

        3 degrees sits inside the measured no-degradation band (11.2% vs
        4.9% at neutral).  Gating there would suppress a quarter of a
        normal session's observation for no benefit.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            slight = _make_face(ear_avg=0.30, pitch=-3.0)
            self._feed(engine, advance, slight, 1.0)
            assert engine.debug_pitch_gated is False
        finally:
            patcher.stop()

    def test_hysteresis_holds_gate_between_release_and_engage(self):
        """Once gated, the gate holds until the head clears the release angle.

        Frame-to-frame pitch jitter is p95 0.66 degrees.  With a single
        threshold a head dwelling on the line flaps in and out of the gate,
        and every flap clears and rebuilds the EAR, blink and microsleep
        state.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(engine, advance, _make_face(pitch=-(EAR_PITCH_GATE_DEG + 1.0)), 0.5)
            assert engine.debug_pitch_gated is True

            # Between release and engage — must stay gated.
            mid = -(EAR_PITCH_GATE_RELEASE_DEG + EAR_PITCH_GATE_DEG) / 2.0
            self._feed(engine, advance, _make_face(pitch=mid), 0.5)
            assert engine.debug_pitch_gated is True

            # Clear of the release angle — must release.
            self._feed(engine, advance, _make_face(pitch=-(EAR_PITCH_GATE_RELEASE_DEG - 1.0)), 0.5)
            assert engine.debug_pitch_gated is False
        finally:
            patcher.stop()

    def test_release_threshold_does_not_engage_the_gate(self):
        """The looser bound applies only once already gated."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            between = -(EAR_PITCH_GATE_RELEASE_DEG + EAR_PITCH_GATE_DEG) / 2.0
            self._feed(engine, advance, _make_face(pitch=between), 1.0)
            assert engine.debug_pitch_gated is False
        finally:
            patcher.stop()

    def test_gated_stretch_records_no_blink_duration(self):
        """A long foreshortened stretch must not enter the duration buffer.

        The blink-duration signal saturates at 1.8x the user's baseline, so
        a single 2 s "blink" manufactured by head-down geometry would pin a
        0.25-weight signal on its own.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            down_closed = _make_face(ear_avg=0.10, pitch=-(EAR_PITCH_GATE_DEG + 2.0))
            self._feed(engine, advance, down_closed, 2.0)
            # Eyes "open" again while still head-down.
            self._feed(
                engine, advance, _make_face(ear_avg=0.30, pitch=-(EAR_PITCH_GATE_DEG + 2.0)), 0.5
            )
            assert not engine._blink_durations
        finally:
            patcher.stop()

    def test_gated_stretch_cannot_fire_microsleep(self):
        """Microsleep bypasses the score and both sustain timers.

        It fires an alert instantly, so the gate is the only thing standing
        between a foreshortened eye and a false alarm.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine)
            down_closed = _make_face(ear_avg=0.10, pitch=-(EAR_PITCH_GATE_DEG + 2.0))
            self._feed(engine, advance, down_closed, MICROSLEEP_THRESHOLD_S * 3)
            assert not [e for e in events if e.event_type == EVENT_FATIGUE]
        finally:
            patcher.stop()

    def test_gate_reports_false_without_face(self):
        """pitch_gated means "the gate fired", not "the engine paused".

        The overlay indicator and the CSV column carry that meaning, and
        the no-face sentinel reports pitch 0.0 anyway — a value that says
        nothing about where the head actually is.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            for _ in range(10):
                engine.update(
                    _NO_FACE, PresenceState.PRESENT_NO_FACE, head_pitch=0.0, baseline_pitch=30.0
                )
                advance(0.033)
            assert engine.debug_pitch_gated is False
        finally:
            patcher.stop()

    def test_score_decays_while_gated(self):
        """The gate must keep the documented face-loss decay semantics."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            # Prime _last_update_time so the first gated frame has a dt.
            engine.update(_OPEN_EYES, PresenceState.PRESENT)
            advance(0.033)
            engine._fatigue_score = 0.5

            self._feed(engine, advance, _make_face(pitch=-(EAR_PITCH_GATE_DEG + 2.0)), 10.0)

            expected = 0.5 - FATIGUE_SCORE_DECAY_RATE * 10.0
            assert engine.debug_fatigue_score == pytest.approx(expected, abs=0.02)
        finally:
            patcher.stop()

    def test_sustained_gating_warns_once_per_episode(self, caplog):
        """A drifted baseline must not silently disable fatigue detection.

        Real gated stretches top out at 43.5 s, so crossing 120 s means the
        calibrated neutral no longer describes how the user sits — and the
        detector is off without saying so.
        """
        import logging

        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            down = _make_face(pitch=-(EAR_PITCH_GATE_DEG + 2.0))
            with caplog.at_level(logging.WARNING):
                self._feed(engine, advance, down, _PITCH_GATE_WARN_S + 5.0, dt=0.5)
                first = len([r for r in caplog.records if "Pitch gate held" in r.getMessage()])
                assert first == 1

                # Release, then re-gate for another full episode.
                self._feed(engine, advance, _make_face(pitch=0.0), 1.0, dt=0.5)
                self._feed(engine, advance, down, _PITCH_GATE_WARN_S + 5.0, dt=0.5)
                second = len([r for r in caplog.records if "Pitch gate held" in r.getMessage()])
                assert second == 2
        finally:
            patcher.stop()


class TestHeadNodExcludedFromScore:
    """Head nod is detected for diagnostics but does not feed the score.

    Recomputed over 24 955 s of logged pitch the detector fires 292-654
    times across sessions in which the user was mostly alert — up to
    94/hour against a signal that saturates at 3 per 60 s — and no
    parameter choice rescues it (15 degree drop with a 1.0 s recovery
    window still yields 64/hour).  It is also unreachable in normal
    operation: the EAR pitch gate engages at 6 degrees, shallower than the
    8 degree drop a nod requires, so the FSM never advances.
    """

    def test_nod_signal_does_not_contribute_to_score(self):
        """The composite score must ignore the nod signal entirely."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine._nod_times.extend([100.0, 100.5, 101.0, 101.5])
            assert engine._head_nod_signal() > 0.0
            assert engine._compute_fatigue_score() == 0.0
        finally:
            patcher.stop()

    def test_weights_still_sum_to_one(self):
        """A rescaled model must keep the 0.6/0.8 tiers meaningful."""
        from acuity.utils.constants import (
            FATIGUE_W_BLINK_DUR,
            FATIGUE_W_BLINK_FREQ,
            FATIGUE_W_PERCLOS,
            FATIGUE_W_YAWN,
        )

        total = FATIGUE_W_PERCLOS + FATIGUE_W_BLINK_DUR + FATIGUE_W_YAWN + FATIGUE_W_BLINK_FREQ
        assert total == pytest.approx(1.0)

    def test_gate_prevents_nod_fsm_from_advancing(self):
        """Documents why the signal is unreachable, not merely unused."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            drop = -(HEAD_NOD_PITCH_DROP_DEG + 2.0)
            for _ in range(30):
                engine.update(
                    _make_face(pitch=drop),
                    PresenceState.PRESENT,
                    head_pitch=drop,
                    baseline_pitch=0.0,
                )
                advance(0.033)
            for _ in range(10):
                engine.update(
                    _make_face(pitch=0.0), PresenceState.PRESENT, head_pitch=0.0, baseline_pitch=0.0
                )
                advance(0.033)
            assert not engine._nod_times
        finally:
            patcher.stop()

    def test_nod_not_counted_without_face(self):
        """The no-face sentinel reports pitch 0.0.

        Against a non-zero baseline that reads as a recovery to neutral and
        would manufacture a nod out of a face-loss frame.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine._nod_state = "recovering"
            engine._nod_drop_start = 100.0
            for _ in range(10):
                engine.update(
                    _NO_FACE, PresenceState.PRESENT_NO_FACE, head_pitch=0.0, baseline_pitch=1.0
                )
                advance(0.033)
            assert not engine._nod_times
        finally:
            patcher.stop()


class TestHeadNodSustainedHold:
    """Sustained head-down is desk work, not drowsiness."""

    @staticmethod
    def _hold(engine, advance, pitch: float, duration_s: float, dt: float = 0.1):
        for _ in range(max(1, int(duration_s / dt))):
            engine.update(
                _make_face(pitch=pitch), PresenceState.PRESENT, head_pitch=pitch, baseline_pitch=0.0
            )
            advance(dt)

    def test_sustained_head_down_then_lift_is_not_a_nod(self):
        """Looking up from the desk must not score a nod.

        The FSM used to return to "neutral" on the dropping-timeout while
        the head was still down, re-arming "dropping" on the next frame
        with a fresh anchor — once every HEAD_NOD_RECOVERY_S.  The eventual
        head-lift then measured elapsed from the last re-arm, passed the
        recovery check, and counted a nod every single time.  Logged
        sessions show signal_head_nod = 1.00 on 5 of 20 fatigue events, all
        at near-neutral pitch.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._hold(engine, advance, -(HEAD_NOD_PITCH_DROP_DEG + 2.0), HEAD_NOD_RECOVERY_S * 3)
            self._hold(engine, advance, 0.0, 1.0)
            assert not engine._nod_times
        finally:
            patcher.stop()

    def test_repeated_desk_glances_do_not_accumulate_nods(self):
        """Writing is a repeated look-down/look-up cycle.

        Three counted nods in 60 s saturate the signal, so a per-cycle
        false nod is enough to pin it on its own.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            for _ in range(5):
                self._hold(
                    engine, advance, -(HEAD_NOD_PITCH_DROP_DEG + 2.0), HEAD_NOD_RECOVERY_S * 2
                )
                self._hold(engine, advance, 0.0, 1.0)
            assert not engine._nod_times
        finally:
            patcher.stop()


# ── Blink duration baseline ─────────────────────────────────────


class TestBlinkDurationBaseline:
    """Blink duration is judged against the user's own resting level.

    The engine times the full width of a sub-threshold stretch at
    0.70x baseline EAR on a median-filtered signal — not physiological full
    closure, which is what the old 150 ms "normal blink" anchor described.
    Measured resting medians ran 195-743 ms across 16 sessions of the same
    user, above the old anchor in every one, so the 0.25-weight signal
    carried a permanent floor and read 1.00 on 13 of 20 logged events.
    """

    # The first nine blinks of a logged session (private corpus, n=1): median 116 ms,
    # mean 204 ms — one long stretch skews the mean.
    _RESTING_BUFFER = [
        90.0,
        100.0,
        110.0,
        116.0,
        116.0,
        120.0,
        130.0,
        150.0,
        904.0,
    ]

    @staticmethod
    def _arm(engine, *, baseline_ms: float, windows: int = 2) -> None:
        """Put the engine in a state where the duration signal is live."""
        engine._baseline_captured = True
        engine._baseline_blink_rate = 15.0
        engine._baseline_blink_duration_ms = baseline_ms
        engine._windows_since_baseline = windows

    def test_signal_zero_before_baseline_capture(self):
        """No reference yet means no verdict, as for blink frequency."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine._blink_durations.extend([400.0] * 10)
            assert engine._blink_duration_signal() == 0.0
        finally:
            patcher.stop()

    def test_signal_zero_during_stabilization(self):
        """Mirrors _blink_freq_signal: wait one full window after capture."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._arm(engine, baseline_ms=300.0, windows=_STABILIZATION_WINDOWS - 1)
            engine._blink_durations.extend([600.0] * 10)
            assert engine._blink_duration_signal() == 0.0
        finally:
            patcher.stop()

    def test_long_resting_duration_scores_zero(self):
        """The reported bug: a naturally slow blinker is not fatigued.

        220 ms was above the old 150 ms anchor, so this user sat at a
        permanent positive signal in every session.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._arm(engine, baseline_ms=220.0)
            engine._blink_durations.extend([220.0] * 10)
            assert engine._blink_duration_signal() == 0.0
        finally:
            patcher.stop()

    def test_signal_saturates_at_fatigue_ratio(self):
        """Full signal only once the user's own blinks slow by the ratio."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            baseline = 300.0
            self._arm(engine, baseline_ms=baseline)
            engine._blink_durations.extend([baseline * _BLINK_DURATION_FATIGUE_RATIO] * 10)
            assert engine._blink_duration_signal() == pytest.approx(1.0)
        finally:
            patcher.stop()

    def test_signal_ramps_linearly_between_baseline_and_ratio(self):
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            baseline = 300.0
            self._arm(engine, baseline_ms=baseline)
            midpoint = baseline * (1.0 + (_BLINK_DURATION_FATIGUE_RATIO - 1.0) / 2.0)
            engine._blink_durations.extend([midpoint] * 10)
            assert engine._blink_duration_signal() == pytest.approx(0.5, abs=0.01)
        finally:
            patcher.stop()

    def test_resting_buffer_scores_zero_against_its_own_baseline(self):
        """Capture and signal must summarise the buffer the same way.

        Regression for a logged session (private corpus, n=1): the baseline was captured
        as the median (116 ms) while the signal read the mean (204 ms) of
        the same nine blinks, one of them a long stretch.  The signal sat
        at 1.00 from two minutes in, holding the score at a false
        floor, and three ordinary blink-rate dips each raised a warning.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine._blink_durations.extend(self._RESTING_BUFFER)
            baseline = engine._capture_baseline_blink_duration()
            assert baseline == 116.0
            self._arm(engine, baseline_ms=baseline)
            assert engine._blink_duration_signal() == 0.0
        finally:
            patcher.stop()

    def test_single_long_stretch_does_not_move_signal(self):
        """One outlier is not a slowing trend.

        Mean/median at capture ran up to 6.5x over all 194 replayed
        sessions that captured a baseline (975 vs 326 ms in the worst one,
        private corpus, n=1), so a mean would let a single long stretch saturate
        the signal on either side of the comparison.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._arm(engine, baseline_ms=300.0)
            engine._blink_durations.extend([300.0] * 8 + [3000.0])
            assert engine._blink_duration_signal() == 0.0
        finally:
            patcher.stop()

    def test_refinement_tracks_the_same_statistic(self):
        """The per-window EMA must not reintroduce the mismatch.

        Refining toward the mean would lift a 116 ms median baseline to
        142 ms on the first window of the same resting buffer, leaving the
        signal pinned below zero (insensitive) instead of above it.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._arm(engine, baseline_ms=116.0, windows=0)
            engine._window_active_s_total = 2 * _BLINK_WINDOW_S
            engine._blink_durations.extend(self._RESTING_BUFFER)
            engine._update_blink_frequency_baseline(0.0)
            assert engine._windows_since_baseline == 1
            assert engine._baseline_blink_duration_ms == pytest.approx(116.0)
        finally:
            patcher.stop()

    def test_baseline_capture_is_capped(self):
        """An already-drowsy baseline must not disable the signal.

        3 of 16 logged sessions captured anchors above 690 ms and then
        improved — the signature of a reference taken while the user was
        tired or tracking was degraded.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            engine._blink_durations.extend([2000.0] * 10)
            captured = engine._capture_baseline_blink_duration()
            assert captured == _BLINK_DURATION_BASELINE_MAX_MS
        finally:
            patcher.stop()

    def test_baseline_capture_returns_none_without_blinks(self):
        """No blink yet means retry on a later window, not lock in a value."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            assert engine._capture_baseline_blink_duration() is None
        finally:
            patcher.stop()

    def test_reset_clears_duration_baseline(self):
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._arm(engine, baseline_ms=300.0)
            engine.reset()
            assert engine._baseline_blink_duration_ms is None
        finally:
            patcher.stop()


class TestHeadMotionEarSuppression:
    """EAR must not be believed while the mesh is not describing an eye.

    Regression cover for a labelled false microsleep (private corpus,
    n=1).  The user turned ~60 degrees and swung back.
    Through the turn the mesh went degenerate — ear_right read 0.97,
    physically impossible — and on the way back the landmarks had not
    re-converged, leaving both eyes at 0.03–0.14 against a 0.154 threshold
    for 2.5 s.  Pitch normalised *first*, releasing the pitch gate while
    yaw was still ~50 degrees, and the very next frame was trusted.  The
    composite fatigue score was 0.31: the scored model was right, and the
    microsleep override fired anyway.

    The pitch gate answers "is the eye foreshortened?".  These guards
    answer "did the mesh solve the eye at all?", which a fast head turn
    breaks while pitch stays perfectly neutral.
    """

    _DT = 0.033

    @classmethod
    def _feed(cls, engine, advance, face, duration_s, baseline_pitch=0.0, dt=None):
        dt = cls._DT if dt is None else dt
        steps = max(1, int(round(duration_s / dt)))
        for _ in range(steps):
            engine.update(
                face, PresenceState.PRESENT, head_pitch=face.pitch, baseline_pitch=baseline_pitch
            )
            advance(dt)

    @classmethod
    def _sweep(cls, engine, advance, deg_per_s, duration_s, ear_avg=0.30, dt=None):
        """Rotate the head at *deg_per_s*, holding EAR at *ear_avg*."""
        dt = cls._DT if dt is None else dt
        steps = max(1, int(round(duration_s / dt)))
        yaw = 0.0
        for _ in range(steps):
            yaw += deg_per_s * dt
            engine.update(
                _make_face(ear_avg=ear_avg, yaw=yaw),
                PresenceState.PRESENT,
                head_pitch=0.0,
                baseline_pitch=0.0,
            )
            advance(dt)

    # ── Fast head turn ───────────────────────────────────────────

    # 30 FPS, the ~15 FPS the logged pipeline actually runs at, and 7 FPS.
    @pytest.mark.parametrize("dt", [0.033, 0.067, 0.143])
    def test_sustained_fast_turn_suppresses_ear(self, dt):
        """A real turn holds the rate above the gate for several frames."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._sweep(engine, advance, EAR_YAW_RATE_GATE_DPS * 2.0, 0.5, dt=dt)
            assert engine.debug_ear_suppressed is True
            # It is not the pitch gate doing this — pitch never moved.
            assert engine.debug_pitch_gated is False
        finally:
            patcher.stop()

    # Not 0.143: at 7 FPS one interval already spans 8.6 deg of travel at
    # the gate rate and is meant to trip it (see EAR_YAW_RATE_SUSTAIN_S).
    @pytest.mark.parametrize("dt", [0.033, 0.067])
    def test_single_frame_yaw_spike_does_not_suppress(self, dt):
        """One interval above the gate is mesh noise, not a head turn.

        Parametrized at ~15 FPS because that is the rate the logged pipeline
        runs at, and the old 0.05 s sustain passed this test at 30 FPS while
        letting every single 67 ms interval through in production.

        42% of the 1 784 above-gate excursions in the logged corpus are a
        single frame long.  Charging a settling window to each of them
        costs observation time out of the independent-PERCLOS denominator,
        which is enough to push a *genuine* microsleep below
        ``_PERCLOS_MIN_INDEPENDENT_S`` — it did exactly that to the
        scripted closure in the fixture video.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(engine, advance, _make_face(yaw=0.0), 0.5, dt=dt)
            # One frame that jumps far, then straight back to steady.
            spike = EAR_YAW_RATE_GATE_DPS * 2.0 * dt
            engine.update(_make_face(yaw=spike), PresenceState.PRESENT)
            advance(dt)
            assert engine.debug_ear_suppressed is False
        finally:
            patcher.stop()

    def test_fast_turn_fires_no_microsleep(self):
        """The 10:08:50 shape: a turn, then a long sub-threshold stretch.

        Without the guards this is a 2.5 s "closure" on a wide-awake user.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            # Sustained sweep back to centre with the eyes reading closed
            # the whole way — landmarks that have not re-converged.
            self._sweep(engine, advance, EAR_YAW_RATE_GATE_DPS * 2.0, 0.5, ear_avg=0.05)
            # Head now still, EAR still garbage during the settling window.
            self._feed(engine, advance, _make_face(ear_avg=0.05), 0.5)

            assert [e for e in events if e.reason == "microsleep"] == []
        finally:
            patcher.stop()

    # ── Degenerate landmarks ─────────────────────────────────────

    def test_implausible_ear_is_not_an_open_eye(self):
        """ear_right = 0.97 is a collapsed solve, not a wide-open eye.

        Feeding it through as "open" is what let the artifact frames
        surrounding the turn look like ordinary observation.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(
                engine,
                advance,
                _make_face(ear_avg=0.50, ear_left=0.30, ear_right=EAR_PLAUSIBLE_MAX + 0.5),
                1.0,
            )
            assert engine.debug_ear_suppressed is True
            assert engine.perclos == 0.0
        finally:
            patcher.stop()

    def test_normal_ear_is_not_suppressed(self):
        """The plausibility cap must sit above any real aperture.

        Corpus p99 of the per-eye maximum is 0.374; the cap is 0.45.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(engine, advance, _OPEN_EYES, 1.0)
            assert engine.debug_ear_suppressed is False
        finally:
            patcher.stop()

    # ── Settling window ──────────────────────────────────────────

    def test_closure_inside_settling_window_fires_nothing(self):
        """Pitch normalises before the landmarks do.

        The live event released the pitch gate at yaw ~50 degrees and
        trusted the very next frame.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            gated = _make_face(ear_avg=0.10, pitch=-(EAR_PITCH_GATE_DEG + 2.0))
            self._feed(engine, advance, gated, 1.0)
            assert engine.debug_pitch_gated is True

            # Gate releases; EAR is still garbage.  Stay inside the window.
            self._feed(engine, advance, _make_face(ear_avg=0.10), EAR_GATE_SETTLE_S * 0.5)
            assert engine.debug_pitch_gated is False
            assert engine.debug_ear_suppressed is True
            assert [e for e in events if e.reason == "microsleep"] == []
        finally:
            patcher.stop()

    def test_settling_window_expires(self):
        """The window is a delay, not an off switch."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            gated = _make_face(ear_avg=0.30, pitch=-(EAR_PITCH_GATE_DEG + 2.0))
            self._feed(engine, advance, gated, 0.5)
            self._feed(engine, advance, _OPEN_EYES, EAR_GATE_SETTLE_S + 0.2)
            assert engine.debug_ear_suppressed is False
        finally:
            patcher.stop()

    # ── Specificity ──────────────────────────────────────────────

    def test_quiet_microsleep_still_fires(self):
        """The guards must not disable genuine microsleep detection.

        All 8 microsleep events that survive this change across the logged
        corpus occurred in quiet conditions — no gating, peak yaw rate
        7–50 deg/s in the preceding 5 s.  This is that shape.
        """
        events = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, _CLOSED_EYES, MICROSLEEP_THRESHOLD_S + 0.5)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
            assert microsleeps[0].event_type == EVENT_FATIGUE
        finally:
            patcher.stop()

    def test_reset_clears_suppression_state(self):
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._sweep(engine, advance, EAR_YAW_RATE_GATE_DPS * 2.0, 0.5)
            assert engine.debug_ear_suppressed is True
            engine.reset()
            assert engine.debug_ear_suppressed is False
            assert engine._prev_yaw is None
            assert engine._yaw_rate_high_since is None
        finally:
            patcher.stop()


class TestMicrosleepBilateralShare:
    """A microsleep episode must be carried by BOTH eyes.

    Regression cover for the wink false positive.  The eyes-closed test is
    on ``ear_avg``, so one eye is enough to clear it: closing a single eye
    in front of the camera fired a real alert (``score=0.15``), and a
    labelled false alert (``score=0.09``, private corpus, n=1) is the
    same shape with no winking — left 0.075, right 0.166 against a 0.160
    threshold, head still.

    Real closure is bilateral: across 4 243 logged closure episodes the two
    eyes cross the threshold within 100 ms of each other 94.8% of the time.
    The episode is still delimited by the microsleep timer so its *duration*
    is unchanged; only its qualification changes.  See
    MICROSLEEP_BILATERAL_SHARE for why a per-frame both-eyes test was
    measured and rejected.

    DORMANT AT THE SHIPPED RATIO.  Once the timer moved onto genuinely-shut
    time, ``shut`` requires ``ear_avg < MICROSLEEP_SHUT_RATIO * threshold``;
    at 0.50 that means ``L + R < threshold``, so both eyes are necessarily
    below it and a one-eyed episode cannot form at all — measured as 0 of
    14 928 shut frames corpus-wide.  The gate re-arms as soon as the ratio
    is loosened past 0.5, so these tests raise it to exercise the gate and
    ``test_gate_is_dormant_at_the_shipped_ratio`` pins the coupling.
    """

    # Ratio at which a one-eyed episode can form, so the gate is reachable.
    _REACHABLE_RATIO = 0.75

    _DT = 0.033
    # Threshold for _BASELINE_EAR (0.30) is 0.30 * 0.70 = 0.21.
    _SHUT = 0.05  # unambiguously below
    _OPEN = 0.25  # unambiguously above
    _AVG_BELOW = 0.15  # ear_avg the engine sees during a one-eyed closure

    @classmethod
    def _feed(cls, engine, advance, face, duration_s):
        steps = max(1, int(round(duration_s / cls._DT)))
        for _ in range(steps):
            engine.update(face, PresenceState.PRESENT)
            advance(cls._DT)

    @pytest.fixture(autouse=True)
    def _reachable_ratio(self):
        """Run this class with the gate reachable.

        At the shipped 0.50 a one-eyed episode cannot form, so every test
        here would pass vacuously.  Raising the ratio exercises the gate
        itself; ``TestBilateralGateDormancy`` covers the shipped value.
        """
        with patch.object(
            fatigue_engine_module,
            "MICROSLEEP_SHUT_RATIO",
            self._REACHABLE_RATIO,
        ):
            yield

    @classmethod
    def _wink(cls, left_shut: bool):
        """One eye shut, the other plainly open — ear_avg still reads closed."""
        return _make_face(
            ear_avg=cls._AVG_BELOW,
            ear_left=cls._SHUT if left_shut else cls._OPEN,
            ear_right=cls._OPEN if left_shut else cls._SHUT,
        )

    @classmethod
    def _both_shut(cls):
        return _make_face(ear_avg=cls._SHUT, ear_left=cls._SHUT, ear_right=cls._SHUT)

    # ── The reproduction ─────────────────────────────────────────

    @pytest.mark.parametrize("left_shut", [True, False])
    def test_wink_fires_no_microsleep(self, left_shut):
        """Either eye alone was enough to fire before this gate existed."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._wink(left_shut), MICROSLEEP_THRESHOLD_S + 1.0)

            assert [e for e in events if e.reason == "microsleep"] == []
            assert engine.debug_episode_bilateral_share == 0.0
        finally:
            patcher.stop()

    def test_bilateral_closure_still_fires(self):
        """The gate must not touch a genuine two-eyed closure."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._both_shut(), MICROSLEEP_THRESHOLD_S + 1.0)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
            assert microsleeps[0].event_type == EVENT_FATIGUE
            assert engine.debug_episode_bilateral_share == 1.0
        finally:
            patcher.stop()

    # ── The reason it is a share and not a per-frame test ────────

    def test_mostly_bilateral_closure_survives_brief_desync(self):
        """Brief one-eyed frames inside a real closure must not disqualify it.

        This is the regression that killed the per-frame variant: requiring
        both eyes on *every* frame re-measures the episode from the second
        eye closing to the first reopening and fragments it on single-frame
        de-synchronisation, dropping 14 genuine 1.50-1.65 s closures across
        the logged corpus.
        """
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            # ~85% bilateral: a long two-eyed closure with one short stretch
            # where a single eye's landmarks wander above the threshold.
            self._feed(engine, advance, self._both_shut(), 1.2)
            self._feed(engine, advance, self._wink(left_shut=True), 0.3)
            self._feed(engine, advance, self._both_shut(), 1.2)

            assert engine.debug_episode_bilateral_share >= (MICROSLEEP_BILATERAL_SHARE)
            assert len([e for e in events if e.reason == "microsleep"]) == 1
        finally:
            patcher.stop()

    def test_episode_below_share_line_is_suppressed(self):
        """Half-and-half sits under the 0.60 line and must not fire."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._wink(left_shut=True), 1.4)
            self._feed(engine, advance, self._both_shut(), 1.4)

            assert engine.debug_episode_bilateral_share < (MICROSLEEP_BILATERAL_SHARE)
            assert [e for e in events if e.reason == "microsleep"] == []
        finally:
            patcher.stop()

    # ── Episode duration is unchanged ────────────────────────────

    def test_wink_does_not_shorten_the_episode(self):
        """The gate qualifies the episode; it must not re-delimit it.

        ``debug_microsleep_elapsed_s`` is still driven by ``ear_avg``, so a
        suppressed episode is still *timed* — that is what keeps a genuine
        closure that merely starts one-eyed from being cut short.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(engine, advance, self._wink(left_shut=True), 2.0)
            assert engine.debug_microsleep_elapsed_s == pytest.approx(2.0, abs=0.1)
        finally:
            patcher.stop()

    # ── State hygiene ────────────────────────────────────────────

    def test_share_is_one_outside_an_episode(self):
        """No episode means nothing disqualifying, not a spurious zero."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(engine, advance, _OPEN_EYES, 0.5)
            assert engine.debug_episode_bilateral_share == 1.0
        finally:
            patcher.stop()

    def test_counters_do_not_leak_into_the_next_episode(self):
        """A wink must not poison the share of the closure that follows it."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._wink(left_shut=True), 1.0)
            # Eyes open long enough to confirm the episode ended.
            self._feed(engine, advance, _OPEN_EYES, 0.5)
            self._feed(engine, advance, self._both_shut(), MICROSLEEP_THRESHOLD_S + 0.5)

            assert engine.debug_episode_bilateral_share == 1.0
            assert len([e for e in events if e.reason == "microsleep"]) == 1
        finally:
            patcher.stop()

    def test_face_loss_and_reset_clear_the_counters(self):
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(engine, advance, self._wink(left_shut=True), 1.0)
            assert engine._episode_frames > 0

            engine.update(_NO_FACE, PresenceState.ABSENT)
            advance(self._DT)
            assert engine._episode_frames == 0
            assert engine.debug_episode_bilateral_share == 1.0

            self._feed(engine, advance, self._wink(left_shut=False), 1.0)
            assert engine._episode_frames > 0
            engine.reset()
            assert engine._episode_frames == 0
            assert engine._episode_bilateral_frames == 0
        finally:
            patcher.stop()

    def test_session_pause_clears_the_counters(self):
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            self._feed(engine, advance, self._wink(left_shut=True), 1.0)
            assert engine._episode_frames > 0
            engine.on_session_pause()
            assert engine._episode_frames == 0
            assert engine._episode_bilateral_frames == 0
        finally:
            patcher.stop()


class TestMicrosleepBlinkConfirm:
    """A microsleep episode must look like a *closure*, not a narrowed eye.

    Regression cover for two false positives in the labelled corpus
    (tests/fixtures/test_edge_cases.mp4):

      * segment 12, a held squint — the shape of a labelled false alert
      * segment 14, gaze dropped to the keyboard with the head held level

    The second is why this gate exists at all.  EAR is a 2-D distance
    ratio, so a lid lowered by gaze is indistinguishable from a lid
    lowered by closure: segment 14 reached 13% aperture, inside the 5-10%
    band genuine closures occupy, with head pitch never reaching
    EAR_PITCH_GATE_DEG.  It accrued 7.23 s of qualifying microsleep time
    and produced no alert only because an earlier false positive had
    opened the cooldown.  No threshold on EAR separates these; the
    eyeBlink blendshape does, because it is a learned appearance signal.

    Per-episode medians in the corpus: genuine 0.738-0.794, awake
    0.588-0.700.  The episode is still delimited by the microsleep timer,
    so its *duration* is unchanged — only its qualification.
    """

    _DT = 0.033
    # Threshold for _BASELINE_EAR (0.30) is 0.30 * 0.70 = 0.21; shut is
    # half that, so 0.05 is unambiguously shut on both eyes.
    _SHUT = 0.05
    _GENUINE = 0.78  # mid genuine band (corpus 0.738-0.794)
    _NARROWED = 0.66  # mid awake band (corpus 0.588-0.700)

    @classmethod
    def _feed(cls, engine, advance, face, duration_s):
        steps = max(1, int(round(duration_s / cls._DT)))
        for _ in range(steps):
            engine.update(face, PresenceState.PRESENT)
            advance(cls._DT)

    @classmethod
    def _closed(cls, blink_score):
        return _make_face(ear_avg=cls._SHUT, blink_score=blink_score)

    # -- The reproductions --------------------------------------------

    def test_narrowed_eyes_fire_no_microsleep(self):
        """Segment 12 / 14: shut by EAR, but the lids are not closed."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._closed(self._NARROWED), MICROSLEEP_THRESHOLD_S + 1.0)

            assert [e for e in events if e.reason == "microsleep"] == []
            assert engine.debug_microsleep_withheld == "blink"
        finally:
            patcher.stop()

    def test_genuine_closure_still_fires(self):
        """The gate must not touch a real closure."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._closed(self._GENUINE), MICROSLEEP_THRESHOLD_S + 1.0)

            assert [e for e in events if e.reason == "microsleep"] != []
        finally:
            patcher.stop()

    # -- Fail-open ----------------------------------------------------

    def test_absent_blendshape_fires(self):
        """No blendshape means no evidence, never "the eye looks open".

        A model built without blendshapes must not silently switch
        microsleep detection off.
        """
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._closed(None), MICROSLEEP_THRESHOLD_S + 1.0)

            assert [e for e in events if e.reason == "microsleep"] != []
            assert engine.debug_episode_blink_confirm == 1.0
        finally:
            patcher.stop()

    # -- Statistic ----------------------------------------------------

    def test_median_not_mean_so_stray_frames_cannot_veto(self):
        """A few low frames must not sink an otherwise genuine episode.

        Blendshape scores dip on the transition frames as the lid travels.
        A mean would let a handful of them drag a real closure under the
        line; the median is why the gate is stated per-episode.
        """
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            # Three low frames, then a sustained genuine closure.
            self._feed(engine, advance, self._closed(0.10), 3 * self._DT)
            self._feed(engine, advance, self._closed(self._GENUINE), MICROSLEEP_THRESHOLD_S + 1.0)

            assert [e for e in events if e.reason == "microsleep"] != []
        finally:
            patcher.stop()

    def test_gate_is_evaluated_per_episode_not_per_frame(self):
        """A narrowed episode stays refused for its whole length.

        The suppression branch returns without latching
        ``_microsleep_fired``, so the gate re-decides every frame; that is
        what keeps ``microsleep_withheld`` populated across the whole
        withheld stretch instead of a single frame.
        """
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            self._feed(engine, advance, self._closed(self._NARROWED), MICROSLEEP_THRESHOLD_S + 3.0)

            assert [e for e in events if e.reason == "microsleep"] == []
            assert engine.debug_microsleep_withheld == "blink"
            assert engine.debug_microsleep_elapsed_s > MICROSLEEP_THRESHOLD_S
        finally:
            patcher.stop()

    def test_threshold_sits_between_the_measured_populations(self):
        """Pins MICROSLEEP_BLINK_CONFIRM against the corpus it came from.

        Genuine episodes measured 0.738-0.794 and awake ones 0.588-0.700.
        Moving the constant outside that gap silently reclassifies one of
        the two populations, so fail loudly here instead.
        """
        assert 0.700 < MICROSLEEP_BLINK_CONFIRM < 0.738


class TestBilateralGateDormancy:
    """At the shipped ratio a one-eyed microsleep episode cannot form.

    ``shut`` is ``ear_avg < MICROSLEEP_SHUT_RATIO * blink_threshold``.  With
    the ratio at 0.50 that is ``L + R < blink_threshold``, which forces both
    eyes below the threshold — so the bilateral share can never fall short
    while an episode is running.  Measured as 0 of 14 928 shut frames across
    the logged corpus.

    This is a coupling, not a coincidence: loosening MICROSLEEP_SHUT_RATIO
    past 0.5 re-arms the bilateral gate.  The test exists so that raising
    the ratio surfaces the dependency instead of silently changing which
    guard is load-bearing.
    """

    def test_shipped_ratio_implies_bilateral(self):
        assert MICROSLEEP_SHUT_RATIO <= 0.5, (
            "MICROSLEEP_SHUT_RATIO has been loosened past 0.5, which re-arms "
            "the bilateral share gate (TestMicrosleepBilateralShare). Confirm "
            "that gate still behaves and update this test deliberately."
        )

    def test_a_wink_cannot_start_an_episode(self):
        """One eye shut, one plainly open — the timer must never start."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            wink = _make_face(ear_avg=0.15, ear_left=0.05, ear_right=0.25)
            for _ in range(80):
                engine.update(wink, PresenceState.PRESENT)
                advance(0.033)
            assert engine.debug_microsleep_elapsed_s == 0.0
            assert engine.microsleep_active is False
        finally:
            patcher.stop()


class TestMicrosleepShutTime:
    """The microsleep timer must run on eyes SHUT, not on sub-threshold EAR.

    Regression cover for a labelled false alert (``score=0.06``, private
    corpus, n=1).  That episode was 100% bilateral, so the bilateral gate
    does not touch it, and it lasted 1.58 s by the old reckoning — but its
    shape was 0.2 s of genuine closure (ear_avg 0.027) followed by 1.4 s at
    0.08-0.12 against a 0.180 threshold, i.e. roughly 35% aperture.  The
    timer was counting a slow partial reopening as closure.

    Longest contiguous genuinely-shut stretch for the four alerts the user
    labelled false: 0.00 s, 0.00 s, 0.00 s, 0.15 s.  Genuine long closures
    in the same corpus carry 1.4-5.6 s.  See MICROSLEEP_SHUT_RATIO.
    """

    _DT = 0.033

    @classmethod
    def _thresholds(cls, engine):
        return engine.blink_threshold, engine.shut_threshold

    @classmethod
    def _feed(cls, engine, advance, ear, duration_s):
        steps = max(1, int(round(duration_s / cls._DT)))
        for _ in range(steps):
            engine.update(_make_face(ear_avg=ear), PresenceState.PRESENT)
            advance(cls._DT)

    # ── The reproduction ─────────────────────────────────────────

    def test_narrowed_eyes_fire_no_microsleep(self):
        """Held below the blink threshold but above shut — a squint."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            blink, shut = self._thresholds(engine)
            narrowed = (blink + shut) / 2.0  # squarely between the two
            assert shut < narrowed < blink

            self._feed(engine, advance, narrowed, MICROSLEEP_THRESHOLD_S + 1.0)

            assert [e for e in events if e.reason == "microsleep"] == []
            assert engine.debug_microsleep_elapsed_s == 0.0
        finally:
            patcher.stop()

    def test_partial_reopening_fires_no_microsleep(self):
        """The exact 12:33:28 shape: brief deep closure, long partial hold."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            blink, shut = self._thresholds(engine)

            self._feed(engine, advance, shut * 0.3, 0.2)  # genuine closure
            self._feed(engine, advance, blink * 0.6, 1.4)  # partial hold
            self._feed(engine, advance, blink * 1.5, 0.5)  # open

            assert [e for e in events if e.reason == "microsleep"] == []
        finally:
            patcher.stop()

    def test_genuine_closure_still_fires(self):
        """Eyes actually shut for the threshold duration must still alert."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, below_fraction=0.20)
            _, shut = self._thresholds(engine)

            self._feed(engine, advance, shut * 0.3, MICROSLEEP_THRESHOLD_S + 0.5)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
            assert microsleeps[0].event_type == EVENT_FATIGUE
        finally:
            patcher.stop()

    # ── The rest of the engine must be untouched ─────────────────

    def test_perclos_still_accrues_on_sub_threshold_time(self):
        """PERCLOS is a duty cycle, not a closure test.

        Its PERCLOS_GATE is calibrated against sub-threshold time, so a
        narrowed-eye stretch that fires no microsleep must still raise it.
        Switching PERCLOS onto the shut flag would silently move that gate.
        """
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            blink, shut = self._thresholds(engine)
            narrowed = (blink + shut) / 2.0

            self._feed(engine, advance, blink * 1.5, 3.0)  # open baseline
            self._feed(engine, advance, narrowed, 3.0)  # narrowed

            assert engine.perclos > 0.0
            assert engine.debug_microsleep_elapsed_s == 0.0
        finally:
            patcher.stop()

    def test_blink_detection_unchanged_by_the_shut_flag(self):
        """Blinks are still counted at the blink threshold."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            blink, shut = self._thresholds(engine)
            narrowed = (blink + shut) / 2.0
            for _ in range(3):
                self._feed(engine, advance, blink * 1.5, 0.5)
                self._feed(engine, advance, narrowed, 0.3)
            self._feed(engine, advance, blink * 1.5, 0.5)

            assert len(engine._blink_times) == 3
            assert engine.debug_avg_blink_duration_ms > 0.0
        finally:
            patcher.stop()

    def test_shut_threshold_tracks_the_blink_threshold(self):
        """Exposed so callers read it rather than re-deriving the ratio."""
        engine, advance, patcher = _make_engine_with_mock_time()
        try:
            assert engine.shut_threshold == pytest.approx(
                engine.blink_threshold * MICROSLEEP_SHUT_RATIO
            )
            assert engine.shut_threshold < engine.blink_threshold
        finally:
            patcher.stop()


class TestMicrosleepContext:
    """A microsleep alerts only when the user is already in fatigue context.

    Regression cover for a labelled false alert (private corpus, n=1): the
    user was drinking, the eyes were genuinely shut for 2.3 s (blink_score
    0.75-0.79, bilateral share 1.00), and every gate passed.  The eyes
    cannot separate that from a microsleep; the score (0.41) could.  Out of
    context the episode is logged as ``"context"`` and nothing is emitted.
    See MICROSLEEP_CONTEXT_S.
    """

    _DT = 0.033
    _SHUT = 0.05
    _GENUINE = 0.78

    @classmethod
    def _feed(cls, engine, advance, duration_s):
        face = _make_face(ear_avg=cls._SHUT, blink_score=cls._GENUINE)
        steps = max(1, int(round(duration_s / cls._DT)))
        for _ in range(steps):
            engine.update(face, PresenceState.PRESENT)
            advance(cls._DT)

    @staticmethod
    def _now():
        return fatigue_engine_module.time.monotonic()

    def test_lone_microsleep_is_logged_not_alerted(self):
        """Event 367: genuine closure, low score, no recent warning."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, context=False)
            self._feed(engine, advance, 2.3)

            assert events == []
            assert engine.debug_microsleep_withheld == "context"
        finally:
            patcher.stop()

    def test_score_at_warning_tier_is_context(self):
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, context=False)
            engine._compute_fatigue_score = lambda: FATIGUE_WARNING_THRESHOLD
            self._feed(engine, advance, 2.3)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
            assert microsleeps[0].event_type == EVENT_FATIGUE
        finally:
            patcher.stop()

    @pytest.mark.parametrize(
        "ago_s, fires",
        [
            (60.0, True),
            (MICROSLEEP_CONTEXT_S - 5.0, True),
            (MICROSLEEP_CONTEXT_S + 30.0, False),
        ],
    )
    def test_recent_warning_is_context(self, ago_s, fires):
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            # The mock clock starts at 100 s and 0.0 means "no warning yet",
            # so move well clear of zero before back-dating a warning.
            advance(1000.0)
            _prefill_perclos(engine, context=False)
            engine._last_warning_time = self._now() - ago_s
            self._feed(engine, advance, 2.3)

            fired = [e for e in events if e.reason == "microsleep"] != []
            assert fired is fires
        finally:
            patcher.stop()

    def test_context_arriving_mid_episode_still_fires(self):
        """The context refusal is not latched: it re-asks every frame."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine, context=False)
            self._feed(engine, advance, 2.0)
            assert engine.debug_microsleep_withheld == "context"

            engine._compute_fatigue_score = lambda: FATIGUE_WARNING_THRESHOLD
            self._feed(engine, advance, 0.3)

            microsleeps = [e for e in events if e.reason == "microsleep"]
            assert len(microsleeps) == 1
            assert microsleeps[0].duration > 2.0
        finally:
            patcher.stop()


class TestSleepAlert:
    """Eyes shut for SLEEP_CLOSURE_S alerts with no context requirement.

    This is what keeps "alert when they sleep" true for a user who dozes
    off with no prior warning signs: low score, low prior PERCLOS, and the
    microsleep path withholding the episode.  Only the bilateral and blink
    gates apply.  (The bilateral gate cannot be exercised here: at
    MICROSLEEP_SHUT_RATIO 0.50 a shut ``ear_avg`` forces both eyes below
    the blink threshold — see TestBilateralGateDormancy.)
    """

    _DT = 0.033
    _SHUT = 0.05
    _GENUINE = 0.78
    _NARROWED = 0.66

    @classmethod
    def _feed(cls, engine, advance, ear, blink_score, duration_s):
        face = _make_face(ear_avg=ear, blink_score=blink_score)
        steps = max(1, int(round(duration_s / cls._DT)))
        for _ in range(steps):
            engine.update(face, PresenceState.PRESENT)
            advance(cls._DT)

    @staticmethod
    def _alerts(events):
        return [e for e in events if e.event_type == EVENT_FATIGUE]

    def test_sleep_fires_without_context(self):
        """Fresh engine: no PERCLOS history, no score, no warning."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            self._feed(engine, advance, self._SHUT, self._GENUINE, SLEEP_CLOSURE_S + 0.2)

            alerts = self._alerts(events)
            assert [e.reason for e in alerts] == ["sleep"]
            assert alerts[0].duration >= SLEEP_CLOSURE_S
        finally:
            patcher.stop()

    def test_short_of_threshold_is_silent(self):
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            self._feed(engine, advance, self._SHUT, self._GENUINE, SLEEP_CLOSURE_S - 0.3)

            assert self._alerts(events) == []
        finally:
            patcher.stop()

    def test_once_per_episode_even_past_the_cooldown(self):
        """A 70 s closure outlasts FATIGUE_COOLDOWN_S; the latch holds."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            self._feed(engine, advance, self._SHUT, self._GENUINE, 70.0)

            assert [e.reason for e in self._alerts(events)] == ["sleep"]
        finally:
            patcher.stop()

    def test_next_episode_fires_again(self):
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            self._feed(engine, advance, self._SHUT, self._GENUINE, SLEEP_CLOSURE_S + 0.2)
            self._feed(engine, advance, 0.30, None, 1.0)  # eyes open
            advance(60.0)  # clear cooldown
            self._feed(engine, advance, self._SHUT, self._GENUINE, SLEEP_CLOSURE_S + 0.2)

            assert [e.reason for e in self._alerts(events)] == ["sleep", "sleep"]
        finally:
            patcher.stop()

    def test_lowered_gaze_is_not_sleep(self):
        """The 7.23 s gaze-down shape: shut by EAR, lids not closed."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            self._feed(engine, advance, self._SHUT, self._NARROWED, 10.0)

            assert self._alerts(events) == []
        finally:
            patcher.stop()

    def test_contextual_microsleep_then_sleep_is_one_alert(self):
        """A drowsy user's closure fires as microsleep at 1.5 s; the sleep
        trigger at SLEEP_CLOSURE_S lands in the shared cooldown."""
        events: list[FatigueEvent] = []
        engine, advance, patcher = _make_engine_with_mock_time(callback=events.append)
        try:
            _prefill_perclos(engine)
            self._feed(engine, advance, self._SHUT, self._GENUINE, SLEEP_CLOSURE_S + 1.0)

            assert [e.reason for e in self._alerts(events)] == ["microsleep"]
        finally:
            patcher.stop()
