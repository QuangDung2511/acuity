# Pomodoro countdown and Fluid stopwatch + dynamic rest calc.
# ─────────────────────────────────────────────────────────────────
# Two timer classes for the two session modes:
#
#   StructuredTimer — Pomodoro countdown (Mode A)
#     Focus duration counts down.  After completion the timer
#     reports which rest type is due (Short or Long) based on the
#     completed block count within the Pomodoro cycle.
#
#   FluidTimer — open-ended stopwatch (Mode B)
#     Counts up from zero.  No automatic finish.  Provides
#     compute_rest_duration() → elapsed / 4.
#
# Both timers:
#   • Emit tick(int) at 1 Hz   (remaining seconds / elapsed seconds)
#   • Emit finished()          (countdown hit 0 / manually triggered)
#   • Support pause / resume   (paused time excluded)
#   • Are QObject-based — must live on the GUI thread
#
# Thread: GUI.
#
# Usage (inside SessionManager):
#
#   timer = StructuredTimer(focus_m=25, short_rest_m=5,
#                           long_rest_m=15, cycle_length=4)
#   timer.tick.connect(ui.update_countdown)
#   timer.finished.connect(session._enter_rest)
#   timer.start_focus()
#
#   timer = FluidTimer(target_m=40)
#   timer.tick.connect(ui.update_stopwatch)
#   timer.start()
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time

from PySide6.QtCore import QObject, QTimer, Signal

from acuity.utils.constants import (
    BLOCKS_BEFORE_LONG_REST_DEFAULT,
    FLUID_REST_DIVISOR,
    FLUID_WORK_DEFAULT_M,
    REST_TYPE_LONG,
    REST_TYPE_SHORT,
    STRUCTURED_LONG_REST_DEFAULT_M,
    STRUCTURED_REST_DEFAULT_M,
    STRUCTURED_WORK_DEFAULT_M,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
#  STRUCTURED TIMER (Pomodoro)
# ════════════════════════════════════════════════════════════════


class StructuredTimer(QObject):
    """Pomodoro countdown with short/long rest cycle tracking.

    Manages a single countdown (focus or rest) at a time.  The
    session manager calls :meth:`start_focus` or :meth:`start_rest`
    to begin each phase; the timer counts down and emits
    ``finished`` when it reaches zero.

    Block counting is handled internally so the timer always knows
    whether the next rest should be Short or Long.

    Parameters
    ----------
    focus_m : float
        Focus block duration in minutes.
    short_rest_m : float
        Short rest duration in minutes.
    long_rest_m : float
        Long rest duration in minutes.
    cycle_length : int
        Number of focus blocks before a long rest replaces the
        short rest (Pomodoro cycle length).
    parent : QObject or None
        Qt parent.

    Signals
    -------
    tick(int)
        Remaining seconds in the current countdown (1 Hz).
    finished()
        The current countdown reached zero.
    """

    tick = Signal(int)
    finished = Signal()

    def __init__(
        self,
        focus_m: float = STRUCTURED_WORK_DEFAULT_M,
        short_rest_m: float = STRUCTURED_REST_DEFAULT_M,
        long_rest_m: float = STRUCTURED_LONG_REST_DEFAULT_M,
        cycle_length: int = BLOCKS_BEFORE_LONG_REST_DEFAULT,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._focus_s = focus_m * 60.0
        self._short_rest_s = short_rest_m * 60.0
        self._long_rest_s = long_rest_m * 60.0
        self._cycle_length = max(1, cycle_length)

        self._completed_blocks = 0

        # ── Countdown state ─────────────────────────────────────
        self._target_s: float = 0.0
        self._active_s: float = 0.0  # accumulated active time
        self._start_mono: float = 0.0  # monotonic anchor
        self._paused = False
        self._running = False

        self._qtimer = QTimer(self)
        self._qtimer.setInterval(1000)
        self._qtimer.timeout.connect(self._on_tick)

    # ── Properties ──────────────────────────────────────────────

    @property
    def focus_duration_s(self) -> float:
        """Configured focus block duration in seconds."""
        return self._focus_s

    @focus_duration_s.setter
    def focus_duration_s(self, value: float) -> None:
        self._focus_s = max(0.0, value)

    @property
    def short_rest_duration_s(self) -> float:
        return self._short_rest_s

    @short_rest_duration_s.setter
    def short_rest_duration_s(self, value: float) -> None:
        self._short_rest_s = max(0.0, value)

    @property
    def long_rest_duration_s(self) -> float:
        return self._long_rest_s

    @long_rest_duration_s.setter
    def long_rest_duration_s(self, value: float) -> None:
        self._long_rest_s = max(0.0, value)

    @property
    def cycle_length(self) -> int:
        return self._cycle_length

    @cycle_length.setter
    def cycle_length(self, value: int) -> None:
        self._cycle_length = max(1, value)

    @property
    def completed_blocks(self) -> int:
        """Number of focus blocks completed in this session."""
        return self._completed_blocks

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def remaining_s(self) -> float:
        """Seconds remaining in the current countdown."""
        if not self._running:
            return 0.0
        return max(0.0, self._target_s - self._elapsed_s())

    @property
    def elapsed_s(self) -> float:
        """Active seconds elapsed in the current countdown."""
        if not self._running:
            return 0.0
        return self._elapsed_s()

    @property
    def next_rest_type(self) -> str:
        """Rest type that follows the *next* completed focus block.

        Based on `(completed_blocks + 1) % cycle_length`.
        """
        if (self._completed_blocks + 1) % self._cycle_length == 0:
            return REST_TYPE_LONG
        return REST_TYPE_SHORT

    @property
    def next_rest_duration_s(self) -> float:
        """Planned rest duration (seconds) for the next rest."""
        if self.next_rest_type == REST_TYPE_LONG:
            return self._long_rest_s
        return self._short_rest_s

    # ── Public API ──────────────────────────────────────────────

    def start_focus(self) -> None:
        """Start a focus-block countdown."""
        self._start_countdown(self._focus_s)
        log.info(
            "Structured focus started (%.0fs, block #%d).",
            self._focus_s,
            self._completed_blocks + 1,
        )

    def start_rest(self) -> None:
        """Start a rest countdown (type determined by block count)."""
        rest_type = self.next_rest_type
        duration = self.next_rest_duration_s
        self._start_countdown(duration)
        log.info(
            "Structured rest started (type=%s, %.0fs).",
            rest_type,
            duration,
        )

    def mark_block_complete(self) -> None:
        """Record that the current focus block finished.

        Call this when the focus countdown emits ``finished`` or
        when the session manager ends the block for other reasons.
        """
        self._completed_blocks += 1
        log.debug("Blocks completed: %d", self._completed_blocks)

    def pause(self) -> None:
        """Pause the countdown.  Paused time is excluded."""
        if not self._running or self._paused:
            return
        self._active_s += time.monotonic() - self._start_mono
        self._paused = True
        self._qtimer.stop()
        log.debug("Structured timer paused (active=%.1fs).", self._active_s)

    def resume(self) -> None:
        """Resume after a pause."""
        if not self._running or not self._paused:
            return
        self._start_mono = time.monotonic()
        self._paused = False
        self._qtimer.start()
        log.debug("Structured timer resumed.")

    def stop(self) -> None:
        """Stop the countdown without emitting ``finished``."""
        self._qtimer.stop()
        self._running = False
        self._paused = False

    def reset(self) -> None:
        """Reset block count and stop any running countdown."""
        self.stop()
        self._completed_blocks = 0
        self._active_s = 0.0
        log.debug("Structured timer fully reset.")

    # ── Private helpers ─────────────────────────────────────────

    def _start_countdown(self, target_s: float) -> None:
        self.stop()
        self._target_s = target_s
        self._active_s = 0.0
        self._start_mono = time.monotonic()
        self._running = True
        self._paused = False
        self._qtimer.start()

    def _elapsed_s(self) -> float:
        if self._paused:
            return self._active_s
        return self._active_s + (time.monotonic() - self._start_mono)

    def _on_tick(self) -> None:
        remaining = max(0, int(self._target_s - self._elapsed_s()))
        self.tick.emit(remaining)
        if remaining <= 0:
            self._qtimer.stop()
            self._running = False
            self.finished.emit()

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        state = "paused" if self._paused else ("running" if self._running else "idle")
        return (
            f"<StructuredTimer {state} "
            f"blocks={self._completed_blocks} "
            f"remaining={self.remaining_s:.0f}s>"
        )


# ════════════════════════════════════════════════════════════════
#  FLUID TIMER (Stopwatch + Dynamic Rest)
# ════════════════════════════════════════════════════════════════


class FluidTimer(QObject):
    """Open-ended stopwatch with dynamic rest calculation.

    Counts up from zero with no automatic finish.  The session
    manager calls :meth:`finish` explicitly when the user or the
    fatigue engine requests a rest.

    Parameters
    ----------
    target_m : float
        Focus target in minutes (informational — shown on UI,
        does not auto-stop the timer).
    parent : QObject or None
        Qt parent.

    Signals
    -------
    tick(int)
        Elapsed active seconds (1 Hz).
    finished()
        Emitted when :meth:`finish` is called.
    """

    tick = Signal(int)
    finished = Signal()

    def __init__(
        self,
        target_m: float = FLUID_WORK_DEFAULT_M,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._target_s = target_m * 60.0

        # ── Stopwatch state ─────────────────────────────────────
        self._active_s: float = 0.0
        self._start_mono: float = 0.0
        self._paused = False
        self._running = False

        self._qtimer = QTimer(self)
        self._qtimer.setInterval(1000)
        self._qtimer.timeout.connect(self._on_tick)

    # ── Properties ──────────────────────────────────────────────

    @property
    def target_s(self) -> float:
        """Focus target in seconds (informational)."""
        return self._target_s

    @target_s.setter
    def target_s(self, value: float) -> None:
        self._target_s = max(0.0, value)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def elapsed_s(self) -> float:
        """Active seconds elapsed since start."""
        if not self._running:
            return self._active_s  # return final value after stop
        return self._elapsed_s()

    @property
    def rest_duration_s(self) -> float:
        """Dynamic rest duration: elapsed_focus / FLUID_REST_DIVISOR."""
        return self.elapsed_s / FLUID_REST_DIVISOR

    # ── Public API ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the stopwatch from zero."""
        self.stop()
        self._active_s = 0.0
        self._start_mono = time.monotonic()
        self._running = True
        self._paused = False
        self._qtimer.start()
        log.info("Fluid timer started (target=%.0fs).", self._target_s)

    def pause(self) -> None:
        """Pause the stopwatch.  Paused time is excluded."""
        if not self._running or self._paused:
            return
        self._active_s += time.monotonic() - self._start_mono
        self._paused = True
        self._qtimer.stop()
        log.debug("Fluid timer paused (active=%.1fs).", self._active_s)

    def resume(self) -> None:
        """Resume after a pause."""
        if not self._running or not self._paused:
            return
        self._start_mono = time.monotonic()
        self._paused = False
        self._qtimer.start()
        log.debug("Fluid timer resumed.")

    def finish(self) -> None:
        """Manually end the focus period and emit ``finished``.

        Called by the session manager when the user requests rest
        or when a fatigue event triggers an automatic rest prompt.
        """
        if not self._running:
            return
        self._active_s = self._elapsed_s()
        self._qtimer.stop()
        self._running = False
        self._paused = False
        self.finished.emit()
        log.info(
            "Fluid focus ended (elapsed=%.0fs, rest=%.0fs).",
            self._active_s,
            self.rest_duration_s,
        )

    def stop(self) -> None:
        """Stop without emitting ``finished``."""
        if self._running:
            self._active_s = self._elapsed_s()
        self._qtimer.stop()
        self._running = False
        self._paused = False

    def reset(self) -> None:
        """Stop and clear elapsed time."""
        self.stop()
        self._active_s = 0.0
        log.debug("Fluid timer reset.")

    # ── Private helpers ─────────────────────────────────────────

    def _elapsed_s(self) -> float:
        if self._paused:
            return self._active_s
        return self._active_s + (time.monotonic() - self._start_mono)

    def _on_tick(self) -> None:
        self.tick.emit(int(self._elapsed_s()))

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        state = "paused" if self._paused else ("running" if self._running else "idle")
        return f"<FluidTimer {state} elapsed={self.elapsed_s:.0f}s target={self._target_s:.0f}s>"
