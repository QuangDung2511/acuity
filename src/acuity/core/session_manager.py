# Session state machine: Idle, Calibrating, Focus, Rest, Away.
# ─────────────────────────────────────────────────────────────────
# Drives the entire session lifecycle:
#
#   IDLE → CALIBRATING → FOCUS ⇄ REST → … → ENDED
#                          ↕
#                        AWAY
#
# Responsibilities:
#   • Creates / ends Session, FocusBlock, Rest, Event DB rows
#   • Starts / stops the CVPipelineThread per focus block
#   • Releases camera on REST / AWAY
#   • Determines rest type (Short / Long / Fluid) based on
#     block count, cycle length, and session mode
#   • Handles skip-rest (records actual elapsed time)
#   • Handles pause / resume (paused time excluded from duration)
#   • On return from AWAY, triggers silent re-calibration
#
# Thread: GUI.
#   SessionManager is a QObject living on the GUI thread.  It
#   communicates with CVPipelineThread via signal/slot connections.
#   QTimers (focus countdown, rest countdown, away countdown) also
#   run on the GUI thread.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from datetime import datetime
from enum import Enum, auto

from PySide6.QtCore import QObject, QTimer, Signal

from acuity.core.calibration import CalibrationResult
from acuity.core.cv_pipeline import CVPipelineThread, EventData
from acuity.db import queries
from acuity.utils.config import ConfigManager
from acuity.utils.constants import (
    EVENT_AWAY,
    EVENT_FATIGUE,
    FLUID_REST_DIVISOR,
    PAUSE_RECALIBRATE_S,
    REST_TYPE_FLUID,
    REST_TYPE_LONG,
    REST_TYPE_SHORT,
    SESSION_MODE_FLUID,
    SESSION_MODE_STRUCTURED,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)


# ── State enum ──────────────────────────────────────────────────


class SessionState(Enum):
    """Session lifecycle states."""

    IDLE = auto()
    CALIBRATING = auto()
    FOCUS = auto()
    REST = auto()
    AWAY = auto()
    PAUSED = auto()
    ENDED = auto()


# ── SessionManager ──────────────────────────────────────────────


class SessionManager(QObject):
    """Orchestrates the full session lifecycle via a state machine.

    Signals
    -------
    state_changed(str)
        Emitted on every state transition with the new state name.
    focus_tick(int)
        Elapsed active focus seconds in the current block (1 Hz).
    rest_tick(int)
        Remaining rest seconds in the current rest period (1 Hz).
    away_tick(int)
        Seconds spent in AWAY state (1 Hz).  When this reaches the
        away threshold, the ``user_away`` signal fires.
    user_away()
        The ABSENT presence state persisted beyond the threshold.
    session_ended()
        The session has been finalised in the database.
    event_logged(str)
        Human-readable description of a logged event for the live
        event log widget.
    """

    state_changed = Signal(str)
    focus_tick = Signal(int)
    rest_tick = Signal(int)
    away_tick = Signal(int)
    user_away = Signal()
    session_ended = Signal()
    event_logged = Signal(str)
    fatigue_rest_prompt = Signal()
    # Emitted when a rest period ends (naturally or via skip-rest).
    # MainWindow handles this by showing a prompt before starting the
    # next focus cycle.  No auto-transition occurs without user input.
    rest_complete = Signal()
    # Emitted when a structured-mode focus block hits its target and rest
    # begins automatically.  Carries (block_number, planned_rest_seconds)
    # so MainWindow can announce the transition via toast + sound.
    #
    # Rest START is announced rather than prompted: the user is by
    # definition present (they just finished a focus block) and nothing
    # downstream requires their input.  Rest END uses a modal prompt
    # instead, because the next step is a 10 s calibration that needs the
    # user physically present and in position — see rest_complete.
    rest_auto_started = Signal(int, int)
    # Emitted when a rest countdown reaches zero, immediately BEFORE
    # rest_complete.  MainWindow routes it to AlertManager.announce_rest_end.
    #
    # Deliberately separate from rest_complete rather than a flag on it:
    # skip_rest() also emits rest_complete, and a skip is a click the user
    # just made — announcing it would be noise.  The split mirrors
    # rest_auto_started (announcement) vs rest_complete (control flow).
    # Order matters: _on_rest_complete opens a modal, which blocks on
    # exec(), so the announcement must be dispatched first.
    rest_timer_expired = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = SessionState.IDLE
        self._cfg = ConfigManager()

        # ── Session bookkeeping ─────────────────────────────────
        self._session_id: int = 0
        self._mode: str = ""
        self._block_number: int = 0
        self._block_id: int = 0
        self._rest_id: int = 0

        # ── Time tracking ───────────────────────────────────────
        self._focus_start_mono: float = 0.0  # monotonic clock
        self._focus_active_s: float = 0.0  # accumulated active focus
        self._pause_start_mono: float = 0.0
        self._total_focus_s: float = 0.0  # session-wide accumulator
        self._total_rest_s: float = 0.0

        self._rest_start_mono: float = 0.0
        self._rest_planned_s: float = 0.0

        self._away_start_mono: float = 0.0

        # ── Block-resume flag ───────────────────────────────────
        # Set True by continue_from_away() and by resume() after a long
        # pause, so that _on_calibration_complete knows to resume the
        # current block rather than create a new one.
        self._resume_existing_block: bool = False

        # ── Long-pause flag ─────────────────────────────────────
        # Set True by _on_pause_release() once a manual pause has run
        # past PAUSE_RECALIBRATE_S and the pipeline has been torn down;
        # tells resume() to re-calibrate instead of unpausing in place.
        self._pause_needs_recal: bool = False

        # ── Calibration baselines (for re-calibration) ──────────
        self._calibration: CalibrationResult | None = None

        # ── CV pipeline ─────────────────────────────────────────
        self._pipeline: CVPipelineThread | None = None

        # ── GUI timers (1 Hz ticks) ─────────────────────────────
        self._focus_timer = QTimer(self)
        self._focus_timer.setInterval(1000)
        self._focus_timer.timeout.connect(self._on_focus_tick)

        self._rest_timer = QTimer(self)
        self._rest_timer.setInterval(1000)
        self._rest_timer.timeout.connect(self._on_rest_tick)

        self._away_timer = QTimer(self)
        self._away_timer.setInterval(1000)
        self._away_timer.timeout.connect(self._on_away_tick)

        # One-shot: fires once a manual pause has run long enough to be
        # treated as an absence (camera released, re-calibrate on resume).
        self._pause_release_timer = QTimer(self)
        self._pause_release_timer.setSingleShot(True)
        self._pause_release_timer.setInterval(int(PAUSE_RECALIBRATE_S * 1000))
        self._pause_release_timer.timeout.connect(self._on_pause_release)

    # ── Properties ──────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def block_number(self) -> int:
        return self._block_number

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def pipeline(self) -> CVPipelineThread | None:
        """Access the current CV pipeline (for signal wiring by the UI)."""
        return self._pipeline

    @property
    def current_focus_elapsed_s(self) -> float:
        """Elapsed active seconds in the current focus block."""
        if self._state == SessionState.FOCUS:
            return self._focus_active_s + (time.monotonic() - self._focus_start_mono)
        return self._focus_active_s

    # ── Public API ──────────────────────────────────────────────

    def start_session(self) -> None:
        """Begin a new session: create DB row → start calibration."""
        if self._state != SessionState.IDLE:
            log.warning("start_session called in state %s — ignored.", self._state.name)
            return

        self._mode = self._cfg.get_str("session_mode")
        self._session_id = queries.create_session(mode=self._mode)
        self._block_number = 0
        self._total_focus_s = 0.0
        self._total_rest_s = 0.0
        self._calibration = None

        log.info(
            "Session %d started (mode=%s).",
            self._session_id,
            self._mode,
        )
        self._start_focus_cycle(calibration_result=None)

    def end_session(self) -> None:
        """End the current session from any active state."""
        if self._state == SessionState.IDLE or self._state == SessionState.ENDED:
            return

        # Finalise any in-progress block.
        if self._state in (SessionState.FOCUS, SessionState.PAUSED):
            self._end_current_block()
        elif self._state == SessionState.AWAY:
            # _enter_away paused the block without finalising it;
            # close the DB row now using the accumulated time.
            queries.end_focus_block(self._block_id, self._focus_active_s)
            self._total_focus_s += self._focus_active_s
            log.info(
                "Focus block #%d finalised from end_session (AWAY) (%.0fs active).",
                self._block_number,
                self._focus_active_s,
            )
        if self._state == SessionState.REST:
            self._finalise_rest(skipped=True)

        self._stop_pipeline()
        self._stop_all_timers()
        # Ending mid-recalibration (from AWAY or from a long pause) would
        # otherwise leave the flag set for the next session's first
        # calibration, which would then resume a block that never started.
        self._resume_existing_block = False
        self._pause_needs_recal = False

        queries.end_session(
            self._session_id,
            self._total_focus_s,
            self._total_rest_s,
        )
        self._set_state(SessionState.ENDED)
        self.session_ended.emit()
        log.info(
            "Session %d ended (focus=%.0fs, rest=%.0fs).",
            self._session_id,
            self._total_focus_s,
            self._total_rest_s,
        )
        # Reset to IDLE so the user can start a fresh session without
        # restarting the application.
        self._set_state(SessionState.IDLE)

    def pause(self) -> None:
        """Manually pause the current focus block.

        The pipeline is only suspended, not destroyed — a short pause
        resumes instantly.  ``_pause_release_timer`` escalates to a full
        teardown if the pause outlives ``PAUSE_RECALIBRATE_S``.
        """
        if self._state != SessionState.FOCUS:
            return
        self._pause_start_mono = time.monotonic()
        self._focus_active_s += self._pause_start_mono - self._focus_start_mono
        self._focus_timer.stop()
        self._pause_needs_recal = False
        if self._pipeline is not None:
            self._pipeline.pause()
            self._pipeline.on_session_pause()
        self._pause_release_timer.start()
        self._set_state(SessionState.PAUSED)
        log.info("Focus block #%d paused.", self._block_number)

    def resume(self) -> None:
        """Resume after a manual pause.

        Short pause: the pipeline is still alive with valid baselines, so
        it simply unpauses.  Long pause (``_pause_needs_recal``): the
        camera was released, and the user has most likely moved, so this
        re-runs calibration first — the same treatment as returning from
        AWAY, and it resumes the *same* block either way.
        """
        if self._state != SessionState.PAUSED:
            return
        self._pause_release_timer.stop()

        if self._pause_needs_recal:
            self._pause_needs_recal = False
            self._resume_existing_block = True
            log.info(
                "Resuming block #%d after a long pause — re-calibrating (accumulated=%.0fs).",
                self._block_number,
                self._focus_active_s,
            )
            self._start_focus_cycle(calibration_result=None)
            return

        if self._pipeline is not None:
            self._pipeline.on_session_resume()
            self._pipeline.resume()
        self._focus_start_mono = time.monotonic()
        self._focus_timer.start()
        self._set_state(SessionState.FOCUS)
        log.info("Focus block #%d resumed.", self._block_number)

    def _on_pause_release(self) -> None:
        """A manual pause has run past ``PAUSE_RECALIBRATE_S``.

        Tears the pipeline down to release the camera and flags
        the resume path to re-calibrate.  The focus block is left open in
        the database with ``_focus_active_s`` already accumulated by
        ``pause()`` — the same arrangement ``_enter_away`` relies on, so
        ``end_session`` from here still closes the block correctly.
        """
        if self._state != SessionState.PAUSED:
            return
        self._stop_pipeline()
        self._pause_needs_recal = True
        log.info(
            "Pause exceeded %.0fs — camera released, block #%d will re-calibrate on resume.",
            PAUSE_RECALIBRATE_S,
            self._block_number,
        )

    def skip_rest(self) -> None:
        """Skip the current rest period."""
        if self._state != SessionState.REST:
            return
        self._finalise_rest(skipped=True)
        # Emit rest_complete so MainWindow can prompt before the next block.
        self.rest_complete.emit()

    def begin_next_block(self) -> None:
        """Start the next focus cycle with a fresh calibration.

        Called by the UI after the user confirms they are ready to
        continue following the rest-complete prompt.  Passing
        ``calibration_result=None`` forces a new 10-second calibration
        so that baselines reflect the user's current posture and EAR
        after the rest break.
        """
        if self._state != SessionState.REST:
            log.warning(
                "begin_next_block called outside REST state (%s) — ignored.",
                self._state.name,
            )
            return
        self._start_focus_cycle(calibration_result=None)

    # ── Away state (called by UI when user clicks Continue/End) ─

    def continue_from_away(self) -> None:
        """User clicked 'Continue Session' after away detection.

        Triggers a silent re-calibration then resumes
        the *same* focus block — block number and accumulated focus time
        are preserved.  The flag ``_resume_existing_block`` tells
        ``_on_calibration_complete`` to call ``_resume_focus_block``
        instead of ``_begin_focus_block``.
        """
        if self._state != SessionState.AWAY:
            return
        self._away_timer.stop()
        self._resume_existing_block = True
        log.info(
            "Continuing session from AWAY — re-calibrating (block #%d, accumulated=%.0fs).",
            self._block_number,
            self._focus_active_s,
        )
        # Re-calibration only — no new block created here.
        self._start_focus_cycle(calibration_result=None)

    def end_from_away(self) -> None:
        """User clicked 'End Session' after away detection.

        The focus block was paused (not finalised) in ``_enter_away``.
        ``end_session`` handles closing the open block via its AWAY branch,
        so we just stop the away timer and delegate.
        """
        if self._state != SessionState.AWAY:
            return
        self._away_timer.stop()
        self.end_session()

    # ── CV pipeline signal handlers ─────────────────────────────

    def _on_calibration_complete(self, result: CalibrationResult) -> None:
        """Calibration finished — transition to FOCUS.

        If ``_resume_existing_block`` is set (return from AWAY, or from a
        long manual pause), resumes the existing focus block — preserving
        accumulated time and block number — instead of starting a new one.
        """
        self._calibration = result
        if self._resume_existing_block:
            self._resume_existing_block = False
            self._resume_focus_block()
        else:
            self._begin_focus_block()

    def _on_event_detected(self, event_data: EventData) -> None:
        """Engine fired an event — log it to the database."""
        if self._state not in (SessionState.FOCUS, SessionState.CALIBRATING):
            return

        ts = datetime.fromtimestamp(event_data.timestamp).isoformat(timespec="seconds")
        queries.insert_event(
            session_id=self._session_id,
            block_id=self._block_id,
            event_type=event_data.event_type,
            duration=event_data.duration,
            timestamp=ts,
            reason=event_data.reason or None,
        )

        # Build a human-readable log entry.
        reason = f" ({event_data.reason})" if event_data.reason else ""
        msg = f"[{ts.split('T')[1]}] {event_data.event_type}{reason} — {event_data.duration:.1f}s"
        self.event_logged.emit(msg)
        log.debug("Event logged: %s", msg)

        # ── Fatigue Rest Trigger (Fluid Mode) ───────────────────
        if (
            event_data.event_type == EVENT_FATIGUE
            and self._mode == SESSION_MODE_FLUID
            and self._state == SessionState.FOCUS
        ):
            self.fatigue_rest_prompt.emit()

    def _on_face_lost(self) -> None:
        """Presence entered ABSENT — start the away countdown."""
        if self._state != SessionState.FOCUS:
            return
        self._enter_away()

    # ── Internal state transitions ──────────────────────────────

    def _set_state(self, new_state: SessionState) -> None:
        old = self._state
        self._state = new_state
        self.state_changed.emit(new_state.name)
        log.debug("State: %s → %s", old.name, new_state.name)

    def _start_focus_cycle(
        self,
        calibration_result: CalibrationResult | None,
    ) -> None:
        """Create a new pipeline and enter CALIBRATING.

        If *calibration_result* is provided the pipeline skips
        calibration and jumps straight to tracking.
        """
        device_index = self._cfg.get_int("camera_index")

        self._pipeline = CVPipelineThread(
            device_index=device_index,
            calibration_result=calibration_result,
        )
        # Wire signals.
        self._pipeline.calibration_complete.connect(self._on_calibration_complete)
        self._pipeline.event_detected.connect(self._on_event_detected)
        self._pipeline.face_lost.connect(self._on_face_lost)

        if calibration_result is not None:
            # Skip calibration — go straight to focus.
            self._pipeline.start()
            self._begin_focus_block()
        else:
            self._set_state(SessionState.CALIBRATING)
            self._pipeline.start()

    def _begin_focus_block(self) -> None:
        """Create a FocusBlock DB row and start the focus timer."""
        self._block_number += 1
        self._block_id = queries.start_focus_block(
            session_id=self._session_id,
            block_number=self._block_number,
        )
        self._focus_start_mono = time.monotonic()
        self._focus_active_s = 0.0
        self._focus_timer.start()
        self._set_state(SessionState.FOCUS)
        log.info("Focus block #%d started (block_id=%d).", self._block_number, self._block_id)

    def _resume_focus_block(self) -> None:
        """Resume the current focus block after re-calibration from AWAY.

        Unlike ``_begin_focus_block``, this does **not** increment the block
        number or create a new DB row.  The accumulated ``_focus_active_s``
        from before the away period is preserved, so Structured mode will
        correctly count down only the *remaining* time.
        """
        self._focus_start_mono = time.monotonic()
        # _focus_active_s intentionally NOT reset — retains pre-away value.
        self._focus_timer.start()
        self._set_state(SessionState.FOCUS)
        log.info(
            "Focus block #%d resumed after away (accumulated=%.0fs, block_id=%d).",
            self._block_number,
            self._focus_active_s,
            self._block_id,
        )

    def _end_current_block(self) -> None:
        """Finalise the current focus block in the DB."""
        if self._state == SessionState.FOCUS:
            self._focus_active_s += time.monotonic() - self._focus_start_mono
        # (If PAUSED, _focus_active_s already has the accumulated value.)

        self._focus_timer.stop()
        queries.end_focus_block(self._block_id, self._focus_active_s)
        self._total_focus_s += self._focus_active_s
        log.info(
            "Focus block #%d ended (%.0fs active).",
            self._block_number,
            self._focus_active_s,
        )

    def _enter_rest(self) -> None:
        """Transition from FOCUS → REST: end block, release camera, start rest."""
        self._end_current_block()
        self._stop_pipeline()

        # Determine rest type and planned duration.
        rest_type, planned_s = self._compute_rest_params()

        self._rest_id = queries.start_rest(
            session_id=self._session_id,
            block_id=self._block_id,
            rest_type=rest_type,
            planned_duration=planned_s,
        )
        self._rest_start_mono = time.monotonic()
        self._rest_planned_s = planned_s
        self._rest_timer.start()
        self._set_state(SessionState.REST)
        log.info(
            "Rest started (type=%s, planned=%.0fs).",
            rest_type,
            planned_s,
        )

    def _finalise_rest(self, *, skipped: bool) -> None:
        """Close the current rest row in the DB."""
        actual_s = time.monotonic() - self._rest_start_mono
        self._rest_timer.stop()

        if skipped:
            queries.skip_rest(self._rest_id, actual_s)
        else:
            queries.end_rest(self._rest_id, actual_s)

        self._total_rest_s += actual_s
        log.info(
            "Rest ended (actual=%.0fs, skipped=%s).",
            actual_s,
            skipped,
        )

    def _enter_away(self) -> None:
        """Transition from FOCUS → AWAY: pause tracking, release camera.

        The current focus block is **paused** but not finalised in the DB.
        ``_focus_active_s`` is updated with the elapsed time so far, and
        the DB row remains open.  If the user chooses Continue, the block
        is resumed via ``_resume_focus_block``; if they choose End (or
        ``end_session`` is called directly), the block is closed then.
        """
        # Accumulate elapsed focus time without finalising the DB row.
        self._focus_active_s += time.monotonic() - self._focus_start_mono
        self._focus_timer.stop()
        self._stop_pipeline()

        self._away_start_mono = time.monotonic()
        self._away_timer.start()
        self._set_state(SessionState.AWAY)
        log.info(
            "Entered AWAY state (block #%d, accumulated=%.0fs).",
            self._block_number,
            self._focus_active_s,
        )

    def _fire_away_event(self) -> None:
        """Log an Away event when the threshold is exceeded."""
        duration = time.monotonic() - self._away_start_mono
        queries.insert_event(
            session_id=self._session_id,
            block_id=self._block_id,
            event_type=EVENT_AWAY,
            duration=duration,
        )
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] User Away — {duration:.0f}s"
        self.event_logged.emit(msg)
        self.user_away.emit()
        log.info("Away event fired (%.0fs).", duration)

    # ── Timer callbacks ─────────────────────────────────────────

    def _on_focus_tick(self) -> None:
        """1 Hz tick during FOCUS — emit time to display and check block end.

        Structured mode: emits remaining seconds (countdown from target).
        Fluid mode:      emits elapsed seconds (count-up stopwatch).
        """
        elapsed = self._focus_active_s + (time.monotonic() - self._focus_start_mono)

        if self._mode == SESSION_MODE_STRUCTURED:
            target_s = self._cfg.get_float("structured_work_min") * 60.0
            remaining = max(0, int(target_s - elapsed))
            self.focus_tick.emit(remaining)
            if elapsed >= target_s:
                # Capture the block number before _enter_rest() closes it.
                block = self._block_number
                self._enter_rest()
                self.rest_auto_started.emit(block, int(self._rest_planned_s))
        else:
            # Fluid mode: count up (elapsed seconds).
            self.focus_tick.emit(int(elapsed))

    def _on_rest_tick(self) -> None:
        """1 Hz tick during REST — emit remaining seconds."""
        elapsed = time.monotonic() - self._rest_start_mono
        remaining = max(0, int(self._rest_planned_s - elapsed))
        self.rest_tick.emit(remaining)

        if remaining <= 0:
            # Natural rest completion — stop timer, finalise the DB row,
            # then emit rest_complete so MainWindow can prompt the user
            # before the next calibration+focus cycle begins.
            self._finalise_rest(skipped=False)
            # Announce before rest_complete: its handler opens a modal
            # that blocks the event loop until the user answers.
            self.rest_timer_expired.emit()
            self.rest_complete.emit()

    def _on_away_tick(self) -> None:
        """1 Hz tick during AWAY — emit elapsed, fire event at threshold."""
        elapsed = time.monotonic() - self._away_start_mono
        self.away_tick.emit(int(elapsed))

        threshold = self._cfg.get_float("face_lost_threshold")
        if elapsed >= threshold:
            self._away_timer.stop()
            self._fire_away_event()

    # ── Rest type logic ─────────────────────────────────────────

    def _compute_rest_params(self) -> tuple[str, float]:
        """Determine rest type and planned duration for the current block.

        Returns
        -------
        (rest_type, planned_seconds)
        """
        if self._mode == SESSION_MODE_FLUID:
            planned = self._focus_active_s / FLUID_REST_DIVISOR
            return REST_TYPE_FLUID, planned

        # Structured mode — Short unless end of Pomodoro cycle → Long.
        cycle_len = self._cfg.get_int("blocks_before_long_rest")
        if self._block_number % cycle_len == 0:
            planned = self._cfg.get_float("structured_long_rest_min") * 60.0
            return REST_TYPE_LONG, planned

        planned = self._cfg.get_float("structured_rest_min") * 60.0
        return REST_TYPE_SHORT, planned

    # ── Pipeline lifecycle ──────────────────────────────────────

    def _stop_pipeline(self) -> None:
        """Stop and discard the CV pipeline, releasing the camera."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline.wait(5000)  # wait up to 5 s for thread exit
            self._pipeline.deleteLater()
            self._pipeline = None

    def _stop_all_timers(self) -> None:
        self._focus_timer.stop()
        self._rest_timer.stop()
        self._away_timer.stop()
        self._pause_release_timer.stop()

    # ── Manual rest trigger (both modes) ─────────────────────────

    def request_rest(self) -> None:
        """Start a rest now, ending the current focus block early.

        The manual rest entry point for **both** modes, called by the
        Focus tab's rest button and by the Fluid fatigue handler.
        _compute_rest_params() already sizes the rest correctly for each:
        elapsed / FLUID_REST_DIVISOR in Fluid, the configured Short (or
        cycle-end Long) rest in Structured — Structured's contract is
        fixed-length rests, so cutting a block short does not shrink the
        break that follows.  The block number still increments per block
        however the block ended, so the long-rest cycle is unaffected.

        Deliberately does *not* emit rest_auto_started: that signal
        announces an unrequested transition, and the user asked for this
        one.  The UI gates the button behind rest_button_unlock_min so a
        rest cannot be claimed seconds into a block.
        """
        if self._state != SessionState.FOCUS:
            return
        self._enter_rest()
        # Note: rest completion (natural or skipped) always routes through
        # rest_complete signal → MainWindow prompt → begin_next_block().

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<SessionManager state={self._state.name} "
            f"session={self._session_id} block=#{self._block_number}>"
        )
