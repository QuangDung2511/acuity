# Tests for session manager.
# ─────────────────────────────────────────────────────────────────
# Two layers:
#   1. Unit tests   — mock CV pipeline + DB, test state machine
#   2. Integration  — real video file (test_full_session.mp4) fed
#                     through the full pipeline instead of live camera.
#                     Self-skips without the private fixture; the
#                     CI-safe counterpart is tests/test_session_e2e.py.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from acuity.core.calibration import CalibrationResult
from acuity.core.cv_pipeline import EventData
from acuity.core.session_manager import SessionManager, SessionState
from acuity.utils.constants import (
    EVENT_DISTRACTION_POSE,
    EVENT_FATIGUE,
    REST_TYPE_FLUID,
    REST_TYPE_LONG,
    REST_TYPE_SHORT,
    SESSION_MODE_FLUID,
    SESSION_MODE_STRUCTURED,
)

# ── Paths ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "private" / "test_full_session.mp4"
_FACE_MODEL = _PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"
_POSE_MODEL = _PROJECT_ROOT / "assets" / "models" / "pose_landmarker.task"

_HAS_VIDEO = _VIDEO_PATH.exists()
_HAS_MODELS = _FACE_MODEL.exists() and _POSE_MODEL.exists()


# ── Helpers ─────────────────────────────────────────────────────

_FAKE_CAL = CalibrationResult(
    baseline_ear=0.30,
    baseline_ear_std=0.02,
    baseline_mar=0.05,
    baseline_mar_std=0.01,
    baseline_pitch=0.0,
    baseline_yaw=0.0,
)


def _make_event(
    event_type: str = EVENT_DISTRACTION_POSE,
    reason: str = "",
    duration: float = 5.0,
) -> EventData:
    return EventData(
        event_type=event_type,
        reason=reason,
        duration=duration,
        timestamp=time.time(),
    )


@pytest.fixture()
def mock_pipeline():
    """Patch CVPipelineThread so no real camera or models are used."""
    with patch("acuity.core.session_manager.CVPipelineThread") as MockCls:
        instance = MagicMock()
        instance.wait = MagicMock()
        instance.stop = MagicMock()
        instance.start = MagicMock()
        instance.deleteLater = MagicMock()
        # Signal mocks — connect is a no-op, we'll invoke handlers directly
        instance.calibration_complete = MagicMock()
        instance.event_detected = MagicMock()
        instance.face_lost = MagicMock()
        MockCls.return_value = instance
        yield MockCls, instance


@pytest.fixture()
def mock_db():
    """Patch all DB query functions used by SessionManager."""
    with patch("acuity.core.session_manager.queries") as q:
        q.create_session.return_value = 1
        q.start_focus_block.return_value = 10
        q.end_focus_block = MagicMock()
        q.start_rest.return_value = 100
        q.end_rest = MagicMock()
        q.skip_rest = MagicMock()
        q.insert_event.return_value = 1000
        q.end_session = MagicMock()
        yield q


@pytest.fixture()
def mock_config():
    """Patch ConfigManager with deterministic defaults."""
    with patch("acuity.core.session_manager.ConfigManager") as MockCfg:
        cfg = MagicMock()
        _defaults = {
            "session_mode": SESSION_MODE_STRUCTURED,
            "camera_index": 0,
            "head_pose_threshold": 15.0,
            "face_lost_threshold": 15.0,
            "desk_writing_mode": False,
            "structured_work_min": 25.0,
            "structured_rest_min": 5.0,
            "structured_long_rest_min": 15.0,
            "blocks_before_long_rest": 4,
            "fluid_work_min": 40.0,
        }
        cfg.get_str.side_effect = lambda k: str(_defaults[k])
        cfg.get_int.side_effect = lambda k: int(_defaults[k])
        cfg.get_float.side_effect = lambda k: float(_defaults[k])
        cfg.get_bool.side_effect = lambda k: bool(_defaults[k])

        # Allow tests to override defaults
        cfg._defaults = _defaults
        MockCfg.return_value = cfg
        yield cfg


@pytest.fixture()
def mgr(qtbot, mock_pipeline, mock_db, mock_config):
    """Create a SessionManager with all dependencies mocked."""
    sm = SessionManager()
    yield sm
    # QObject cleanup — stop any running timers.
    sm._stop_all_timers()


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS — state machine logic, no camera or models
# ═══════════════════════════════════════════════════════════════


class TestSessionLifecycle:
    """Tests for the basic session lifecycle."""

    def test_initial_state_is_idle(self, mgr):
        assert mgr.state == SessionState.IDLE

    def test_start_session_transitions_to_calibrating(self, mgr, mock_db):
        """start_session must create a DB session and enter CALIBRATING
        (intent: calibration happens before any focus tracking)."""
        mgr.start_session()
        assert mgr.state == SessionState.CALIBRATING
        mock_db.create_session.assert_called_once()

    def test_start_session_in_non_idle_is_noop(self, mgr):
        """Calling start_session when not IDLE must be ignored
        (intent: prevent duplicate sessions)."""
        mgr.start_session()
        mgr.start_session()  # should be ignored
        assert mgr.state == SessionState.CALIBRATING

    def test_calibration_complete_transitions_to_focus(self, mgr, mock_db):
        """After calibration completes, state must be FOCUS with a
        new focus block in the DB (intent: tracking begins
        immediately after baselines are established)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        assert mgr.state == SessionState.FOCUS
        assert mgr.block_number == 1
        mock_db.start_focus_block.assert_called_once_with(
            session_id=1,
            block_number=1,
        )

    def test_end_session_from_focus(self, mgr, mock_db):
        """end_session during FOCUS must finalise the block and
        session in the DB (intent: clean shutdown preserves data)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.state == SessionState.FOCUS

        mgr.end_session()
        # end_session transitions ENDED → IDLE so a fresh session can start.
        assert mgr.state == SessionState.IDLE
        mock_db.end_focus_block.assert_called_once()
        mock_db.end_session.assert_called_once()

    def test_end_session_emits_signal(self, qtbot, mgr):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        with qtbot.waitSignal(mgr.session_ended, timeout=1000):
            mgr.end_session()

    def test_end_session_from_idle_is_noop(self, mgr, mock_db):
        mgr.end_session()
        assert mgr.state == SessionState.IDLE
        mock_db.end_session.assert_not_called()


class TestPauseResume:
    """Tests for the pause/resume flow."""

    def test_pause_transitions_to_paused(self, mgr):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.pause()
        assert mgr.state == SessionState.PAUSED

    def test_resume_transitions_to_focus(self, mgr):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.pause()
        mgr.resume()
        assert mgr.state == SessionState.FOCUS

    def test_pause_in_non_focus_is_noop(self, mgr):
        """pause() outside FOCUS must be ignored."""
        mgr.start_session()
        mgr.pause()  # still CALIBRATING
        assert mgr.state == SessionState.CALIBRATING

    def test_resume_in_non_paused_is_noop(self, mgr):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.resume()  # not paused
        assert mgr.state == SessionState.FOCUS

    def test_end_session_from_paused(self, mgr, mock_db):
        """end_session from PAUSED must finalise the block
        (intent: paused time is excluded from duration)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.pause()
        mgr.end_session()
        # end_session transitions ENDED → IDLE so a fresh session can start.
        assert mgr.state == SessionState.IDLE
        mock_db.end_focus_block.assert_called_once()

    def test_short_pause_keeps_the_pipeline(self, mgr, mock_pipeline):
        """A pause under the threshold only suspends the pipeline
        (intent: quick interruptions resume instantly, no re-calibration)."""
        MockCls, instance = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.pause()
        instance.pause.assert_called_once()
        instance.stop.assert_not_called()

        mgr.resume()
        assert mgr.state == SessionState.FOCUS
        instance.resume.assert_called_once()
        # No second pipeline — the calibrated one is still alive.
        MockCls.assert_called_once()

    def test_long_pause_releases_the_camera(self, mgr, mock_pipeline, mock_db):
        """Past PAUSE_RECALIBRATE_S the pipeline is torn down
        (intent: don't hold the camera through a long break)."""
        _, instance = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.pause()

        mgr._on_pause_release()  # fire the one-shot timer directly

        instance.stop.assert_called_once()
        assert mgr.state == SessionState.PAUSED  # still paused, just camera-less
        assert mgr._pause_needs_recal is True
        mock_db.end_focus_block.assert_not_called()  # block stays open

    def test_resume_after_long_pause_recalibrates_same_block(
        self,
        mgr,
        mock_pipeline,
        mock_db,
    ):
        """Resuming a long pause re-calibrates but keeps the block
        (intent: the user may have re-seated, but this is not a new block)."""
        MockCls, _ = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.block_number == 1

        mgr.pause()
        mgr._on_pause_release()
        mgr.resume()

        assert mgr.state == SessionState.CALIBRATING
        assert mgr._resume_existing_block is True
        assert MockCls.call_count == 2  # fresh pipeline for the re-calibration

        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.state == SessionState.FOCUS
        assert mgr.block_number == 1  # same block!
        mock_db.start_focus_block.assert_called_once()

    def test_ending_a_long_pause_clears_the_recal_flag(self, mgr, mock_pipeline):
        """The flag must not survive into the next session, where it would
        make a short pause spawn a second pipeline."""
        MockCls, _ = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.pause()
        mgr._on_pause_release()
        mgr.end_session()
        assert mgr._pause_needs_recal is False
        assert mgr._resume_existing_block is False

        # A short pause in the next session resumes in place.
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        built_before = MockCls.call_count
        mgr.pause()
        mgr.resume()
        assert mgr.state == SessionState.FOCUS
        assert MockCls.call_count == built_before

    def test_pause_release_outside_paused_is_noop(self, mgr, mock_pipeline):
        """A late timer firing must not tear down a running pipeline."""
        _, instance = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        mgr._on_pause_release()  # never paused

        instance.stop.assert_not_called()
        assert mgr._pause_needs_recal is False


class TestEventLogging:
    """Tests for event detection → DB logging."""

    def test_event_detected_inserts_to_db(self, mgr, mock_db):
        """Events during FOCUS must be persisted to the DB
        (intent: all events are logged for analytics)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        event = _make_event(EVENT_DISTRACTION_POSE, duration=5.0)
        mgr._on_event_detected(event)

        mock_db.insert_event.assert_called_once()
        call_kwargs = mock_db.insert_event.call_args
        assert call_kwargs[1]["event_type"] == EVENT_DISTRACTION_POSE

    def test_event_reason_persisted(self, mgr, mock_db):
        """The engine's reason must reach the DB
        (intent: Insights splits fatigue events by cause)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        event = _make_event(EVENT_FATIGUE, reason="microsleep", duration=0.5)
        mgr._on_event_detected(event)

        assert mock_db.insert_event.call_args[1]["reason"] == "microsleep"

    def test_empty_reason_stored_as_null(self, mgr, mock_db):
        """An engine with no reason (distraction) must store NULL, not ''
        (intent: NULL is the single 'no recorded cause' value)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        event = _make_event(EVENT_DISTRACTION_POSE, reason="", duration=5.0)
        mgr._on_event_detected(event)

        assert mock_db.insert_event.call_args[1]["reason"] is None

    def test_event_emits_event_logged_signal(self, qtbot, mgr):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        event = _make_event(EVENT_FATIGUE, reason="microsleep", duration=0.5)
        with qtbot.waitSignal(mgr.event_logged, timeout=1000):
            mgr._on_event_detected(event)

    def test_event_ignored_outside_focus(self, mgr, mock_db):
        """Events outside FOCUS/CALIBRATING must be dropped
        (intent: events only occur during focus blocks)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.end_session()

        event = _make_event()
        mgr._on_event_detected(event)
        mock_db.insert_event.assert_not_called()


class TestRestTypeLogic:
    """Tests for _compute_rest_params — rest type determination."""

    def test_structured_short_rest_default(self, mgr):
        """Blocks 1-3 in a 4-block cycle get Short rest."""
        mgr._mode = SESSION_MODE_STRUCTURED
        mgr._block_number = 1
        mgr._focus_active_s = 100.0  # not used in structured
        rest_type, planned = mgr._compute_rest_params()
        assert rest_type == REST_TYPE_SHORT
        assert planned == 5.0 * 60.0  # 5 minutes

    def test_structured_long_rest_at_cycle_boundary(self, mgr):
        """Block 4 in a 4-block cycle gets Long rest
        (intent: Pomodoro cycle triggers extended break)."""
        mgr._mode = SESSION_MODE_STRUCTURED
        mgr._block_number = 4
        rest_type, planned = mgr._compute_rest_params()
        assert rest_type == REST_TYPE_LONG
        assert planned == 15.0 * 60.0  # 15 minutes

    def test_structured_long_rest_at_cycle_multiple(self, mgr):
        """Block 8 (second cycle) also gets Long rest."""
        mgr._mode = SESSION_MODE_STRUCTURED
        mgr._block_number = 8
        rest_type, planned = mgr._compute_rest_params()
        assert rest_type == REST_TYPE_LONG

    def test_fluid_rest_is_quarter_of_focus(self, mgr):
        """Fluid rest = focus_time / 4 (intent: dynamic rest
        scales with how long the user has been focusing)."""
        mgr._mode = SESSION_MODE_FLUID
        mgr._focus_active_s = 2400.0  # 40 minutes
        rest_type, planned = mgr._compute_rest_params()
        assert rest_type == REST_TYPE_FLUID
        assert planned == 600.0  # 10 minutes


class TestSkipRest:
    """Tests for skip-rest."""

    def test_skip_rest_records_actual_time(self, mgr, mock_db):
        """skip_rest must record actual elapsed time before skip
        (intent: partial rest data preserved for analytics)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        # Manually enter rest state
        mgr._enter_rest()
        assert mgr.state == SessionState.REST

        mgr.skip_rest()
        mock_db.skip_rest.assert_called_once()

    def test_skip_rest_in_non_rest_is_noop(self, mgr, mock_db):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.skip_rest()  # in FOCUS, not REST
        mock_db.skip_rest.assert_not_called()

    def test_skip_rest_does_not_announce(self, mgr, mock_db):
        """skip_rest must not fire rest_timer_expired (intent: the user
        just clicked Skip — announcing it would be noise)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._enter_rest()

        fired = []
        mgr.rest_timer_expired.connect(lambda: fired.append(1))
        mgr.skip_rest()

        assert fired == []


class TestRestTimerExpired:
    """Tests for the rest-end announcement signal."""

    def test_expiry_fires_signal_before_rest_complete(self, mgr, mock_db):
        """rest_timer_expired must fire, and must precede rest_complete
        (intent: _on_rest_complete opens a modal that blocks the event
        loop, so the toast+sound have to be dispatched first)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._enter_rest()

        order = []
        mgr.rest_timer_expired.connect(lambda: order.append("expired"))
        mgr.rest_complete.connect(lambda: order.append("complete"))

        # Force the countdown past zero.
        mgr._rest_start_mono = time.monotonic() - (mgr._rest_planned_s + 1.0)
        mgr._on_rest_tick()

        assert order == ["expired", "complete"]

    def test_no_signal_while_rest_still_running(self, mgr, mock_db):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._enter_rest()

        fired = []
        mgr.rest_timer_expired.connect(lambda: fired.append(1))
        mgr._on_rest_tick()  # only just started

        assert fired == []


class TestAwayState:
    """Tests for the AWAY state."""

    def test_face_lost_transitions_to_away(self, mgr, mock_db):
        """face_lost during FOCUS must transition to AWAY and pause the
        current block without finalising it in the DB
        (intent: block is resumed if user clicks Continue)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._on_face_lost()
        assert mgr.state == SessionState.AWAY
        # Block is paused, not ended — no DB row should be closed yet.
        mock_db.end_focus_block.assert_not_called()

    def test_face_lost_outside_focus_is_noop(self, mgr):
        mgr.start_session()
        mgr._on_face_lost()  # CALIBRATING, not FOCUS
        assert mgr.state == SessionState.CALIBRATING

    def test_end_from_away(self, mgr, mock_db):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._on_face_lost()
        mgr.end_from_away()
        # end_session (called internally) resolves to IDLE.
        assert mgr.state == SessionState.IDLE
        mock_db.end_session.assert_called_once()

    def test_continue_from_away_resumes_same_block(self, mgr, mock_pipeline, mock_db):
        """continue_from_away must re-calibrate but keep the same block
        number and not create a new DB row
        (intent: away is a pause, not a block boundary)."""
        MockCls, _ = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.block_number == 1

        mgr._on_face_lost()
        assert mgr.state == SessionState.AWAY
        mock_db.end_focus_block.assert_not_called()  # block still open

        mgr.continue_from_away()
        assert mgr.state == SessionState.CALIBRATING
        assert mgr._resume_existing_block is True

        # Simulate calibration finishing.
        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.state == SessionState.FOCUS
        assert mgr.block_number == 1  # same block!
        # start_focus_block called exactly once (the original block).
        mock_db.start_focus_block.assert_called_once()

    def test_continue_from_away_preserves_focus_time(self, mgr, mock_pipeline):
        """Accumulated focus time from before away must be preserved
        after the user continues
        (intent: Structured countdown resumes from where it left off)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        # Inject 5 minutes of already-counted focus time.
        mgr._focus_active_s = 300.0
        mgr._focus_start_mono = time.monotonic()

        mgr._on_face_lost()
        pre_away_accum = mgr._focus_active_s
        assert pre_away_accum >= 300.0  # accumulated before the away

        mgr.continue_from_away()
        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.state == SessionState.FOCUS
        # After resume, accumulated time must be at least what we had.
        assert mgr._focus_active_s >= 300.0

    def test_end_from_away_finalises_block(self, mgr, mock_db):
        """Choosing End from the away dialog must finalise the open block
        and then end the session
        (intent: data integrity when user quits from away)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._on_face_lost()
        mgr.end_from_away()
        mock_db.end_focus_block.assert_called_once()
        mock_db.end_session.assert_called_once()

    def test_end_session_from_away_finalises_block(self, mgr, mock_db):
        """end_session() called directly while in AWAY must also close
        the open block
        (intent: covers quit-from-tray while away)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._on_face_lost()
        mgr.end_session()
        mock_db.end_focus_block.assert_called_once()
        mock_db.end_session.assert_called_once()

    def test_continue_from_non_away_is_noop(self, mgr):
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.continue_from_away()  # in FOCUS, not AWAY
        assert mgr.state == SessionState.FOCUS


class TestFluidMode:
    """Tests for Fluid mode specifics."""

    def test_request_rest_in_fluid(self, mgr, mock_db, mock_config):
        mock_config._defaults["session_mode"] = SESSION_MODE_FLUID
        mock_config.get_str.side_effect = lambda k: str(mock_config._defaults[k])

        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.state == SessionState.FOCUS

        mgr.request_rest()
        assert mgr.state == SessionState.REST

    def test_request_rest_in_structured_enters_rest(self, mgr, mock_db):
        """The manual rest button is shown in both modes, so request_rest
        must work in Structured too (intent: one button, one meaning).

        Structured keeps its fixed-length rest — cutting a block short
        must not shrink the break that follows.
        """
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        assert mgr.state == SessionState.FOCUS

        mgr._focus_active_s = 300.0  # 5 min in — a quarter would be 75 s
        mgr.request_rest()
        assert mgr.state == SessionState.REST

        # Short rest of the configured length, not a Fluid proportion.
        _, kwargs = mock_db.start_rest.call_args
        assert kwargs["rest_type"] == REST_TYPE_SHORT
        assert kwargs["planned_duration"] == 5.0 * 60.0

    def test_request_rest_outside_focus_is_noop(self, mgr, mock_db):
        """The FOCUS guard is the one guard request_rest still has."""
        mgr.start_session()  # CALIBRATING, not FOCUS
        mgr.request_rest()
        assert mgr.state != SessionState.REST


class TestFocusTick:
    """Tests for _on_focus_tick — countdown vs. count-up behaviour."""

    def test_structured_focus_tick_emits_countdown(self, mgr, mock_config):
        """In Structured mode, focus_tick must emit remaining seconds
        (countdown), not elapsed
        (intent: Pomodoro timer counts down to 00:00)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._focus_active_s = 0.0
        mgr._focus_start_mono = time.monotonic()

        ticks = []
        mgr.focus_tick.connect(ticks.append)
        mgr._on_focus_tick()

        assert len(ticks) == 1
        # With 25-min target and ~0s elapsed, remaining should be near 1500.
        assert ticks[0] > 1400

    def test_fluid_focus_tick_emits_elapsed(self, mgr, mock_config):
        """In Fluid mode, focus_tick must emit elapsed seconds (count-up)
        (intent: Fluid mode is an open-ended stopwatch)."""
        mock_config._defaults["session_mode"] = SESSION_MODE_FLUID
        mock_config.get_str.side_effect = lambda k: str(mock_config._defaults[k])

        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._focus_active_s = 0.0
        mgr._focus_start_mono = time.monotonic()

        ticks = []
        mgr.focus_tick.connect(ticks.append)
        mgr._on_focus_tick()

        assert len(ticks) == 1
        # Near 0s elapsed — must be a small count-up value.
        assert ticks[0] < 5

    def test_structured_tick_respects_pre_away_accumulation(self, mgr, mock_config):
        """After resuming from away, countdown must account for time
        already elapsed before the away period
        (intent: 5 min done before away → only 20 min left, not 25)."""
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        # Simulate 5 min already focused before the away.
        mgr._focus_active_s = 300.0
        mgr._focus_start_mono = time.monotonic()

        ticks = []
        mgr.focus_tick.connect(ticks.append)
        mgr._on_focus_tick()

        assert len(ticks) == 1
        # 25 min target − 5 min elapsed ≈ 1200s remaining.
        assert ticks[0] < 1210  # at most ~20 min remaining
        assert ticks[0] > 1180  # but not less than that


class TestStateChangedSignal:
    """Tests for the state_changed signal."""

    def test_state_changed_emits_on_transition(self, qtbot, mgr):
        """state_changed must emit the new state name on every transition."""
        states = []
        mgr.state_changed.connect(states.append)

        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)

        assert "CALIBRATING" in states
        assert "FOCUS" in states

    def test_repr_contains_state(self, mgr):
        assert "IDLE" in repr(mgr)
        mgr.start_session()
        assert "CALIBRATING" in repr(mgr)


class TestPipelineLifecycle:
    """Tests for pipeline creation and teardown."""

    def test_pipeline_created_on_start(self, mgr, mock_pipeline):
        MockCls, instance = mock_pipeline
        mgr.start_session()
        MockCls.assert_called_once()
        instance.start.assert_called_once()

    def test_pipeline_stopped_on_end_session(self, mgr, mock_pipeline):
        _, instance = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr.end_session()
        instance.stop.assert_called()
        instance.wait.assert_called()

    def test_pipeline_stopped_on_enter_rest(self, mgr, mock_pipeline):
        """Pipeline must be stopped when entering REST
        (intent: camera released during rest)."""
        _, instance = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._enter_rest()
        instance.stop.assert_called()

    def test_pipeline_stopped_on_enter_away(self, mgr, mock_pipeline):
        """Pipeline must be stopped when entering AWAY
        (intent: camera released during away)."""
        _, instance = mock_pipeline
        mgr.start_session()
        mgr._on_calibration_complete(_FAKE_CAL)
        mgr._on_face_lost()
        instance.stop.assert_called()


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION TEST — real video instead of live camera
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not (_HAS_VIDEO and _HAS_MODELS),
    reason="Requires test_full_session.mp4 and MediaPipe model files.",
)
class TestSessionManagerIntegration:
    """Run a short session lifecycle against recorded video footage.

    Instead of a live camera, we feed test_full_session.mp4 through
    the real CV pipeline by monkeypatching CameraManager to read
    from the video file.
    """

    def test_full_lifecycle_with_video(self, qtbot, tmp_path):
        """Feed recorded footage through the real pipeline and verify
        the session manager transitions through a complete lifecycle:
        IDLE → CALIBRATING → FOCUS → (events detected) → ENDED.

        Intent: end-to-end proof that the orchestration works with
        real CV output, not just mocks.
        """
        import cv2

        from acuity.core.camera import CameraManager
        from acuity.db.database import DatabaseManager

        # conftest.py points ACUITY_DATA_DIR at tmp_path, so this is an
        # isolated DB and tests don't pollute prod data.
        test_db = DatabaseManager()
        test_db.connect()

        video_cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert video_cap.isOpened(), f"Cannot open {_VIDEO_PATH}"

        # Skip 3s pre-roll (matches test_calibration.py convention).
        fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        for _ in range(int(3 * fps)):
            video_cap.read()

        # ── Monkeypatch CameraManager to read from video ────────
        original_open = CameraManager.open
        original_read = CameraManager.read_frame
        original_release = CameraManager.release

        def fake_open(self, device_index=0):
            self._cap = video_cap
            self._device_index = device_index
            self._consecutive_failures = 0

        def fake_read(self):
            if self._cap is None:
                return False, None
            ret, frame = self._cap.read()
            if ret:
                return True, frame
            return False, None

        def fake_release(self):
            # Don't actually release the video cap — we reuse it.
            self._cap = None
            self._device_index = -1
            self._consecutive_failures = 0

        try:
            CameraManager.open = fake_open
            CameraManager.read_frame = fake_read
            CameraManager.release = fake_release

            # Patch the DB singleton so queries go to our test DB.
            with patch("acuity.db.queries.get_db", return_value=test_db):
                sm = SessionManager()

                # Collect state transitions.
                states = []
                sm.state_changed.connect(states.append)

                events_logged = []
                sm.event_logged.connect(events_logged.append)

                # Start session — this creates the pipeline and begins
                # calibration against the video frames.
                sm.start_session()
                assert sm.state == SessionState.CALIBRATING

                # Wait for calibration to complete (the pipeline thread
                # reads frames from the video in the background).
                # Calibration takes ~10s of video time but runs at CPU
                # speed, so typically completes in 1-3s wall time.
                calibration_done = False

                def on_cal_done(_result):
                    nonlocal calibration_done
                    calibration_done = True

                # Connect to the pipeline's calibration_complete signal.
                if sm.pipeline is not None:
                    sm.pipeline.calibration_complete.connect(on_cal_done)

                # Wait for calibration — up to 30s wall time.
                qtbot.waitUntil(lambda: sm.state == SessionState.FOCUS, timeout=30_000)

                assert "CALIBRATING" in states
                assert "FOCUS" in states
                assert sm.block_number == 1

                # Let tracking run for a few seconds to collect events.
                import time as _time

                _time.sleep(3)

                # ── End the session ─────────────────────────────────
                sm.end_session()
                # end_session transitions ENDED → IDLE (by design);
                # the ENDED state is still emitted as a signal.
                assert sm.state == SessionState.IDLE
                assert "ENDED" in states

                # ── Verify DB state ─────────────────────────────────
                conn = test_db.connection

                # Session row exists and is finalised.
                row = conn.execute(
                    "SELECT * FROM Sessions WHERE session_id = ?",
                    (sm.session_id,),
                ).fetchone()
                assert row is not None
                assert row["end_time"] is not None
                assert row["mode"] == SESSION_MODE_STRUCTURED

                # At least one focus block was created.
                blocks = conn.execute(
                    "SELECT * FROM FocusBlocks WHERE session_id = ?",
                    (sm.session_id,),
                ).fetchall()
                assert len(blocks) >= 1
                assert blocks[0]["end_time"] is not None
                assert blocks[0]["duration"] > 0

        finally:
            CameraManager.open = original_open
            CameraManager.read_frame = original_read
            CameraManager.release = original_release
            video_cap.release()
            test_db.close()
