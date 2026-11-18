# Head-pose distraction: one continuous-duration threshold timer.
# ─────────────────────────────────────────────────────────────────
# Responsibilities:
#   • Time how long the head stays turned away from its calibrated
#     baseline.  The timer starts when the deviation begins and resets
#     when it clears.  Past the user-configurable threshold, one
#     Distraction_Pose event fires per episode via the callback.
#
# Thread: pure.  No Qt and no I/O; the CV worker calls update() once
# per frame.  It reads time.monotonic() itself (ported behaviour).
#
# Design:
#   • The original's second timer (phone in frame) is gone with the
#     phone detector.
#   • The torso parameters (baseline_torso_pitch, torso_pitch_deg) stay
#     in the signature but the pipeline no longer passes them, so both
#     default to 0.0 and head pitch is compared with its baseline
#     directly.  Ticket 11 deletes them with the head-pose rework.
#
# Inputs:
#   - face_mesh.FaceResult   (pitch, yaw, face_detected)
#   - presence.PresenceState (PRESENT / PRESENT_NO_FACE / ABSENT)
#   - calibration.CalibrationResult (baseline_pitch, baseline_yaw)
#
# Thresholds sourced from acuity/utils/constants.py.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from acuity.core.face_mesh import FaceResult
from acuity.core.presence import PresenceState
from acuity.utils.angles import wrap_deg
from acuity.utils.constants import (
    EVENT_DISTRACTION_POSE,
    HEAD_POSE_THRESHOLD_DEFAULT_S,
    HEAD_POSE_THRESHOLD_MAX_S,
    HEAD_POSE_THRESHOLD_MIN_S,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# Separate tolerances — pitch is more forgiving because residual
# expression noise and natural reading posture shifts primarily
# affect the vertical axis.  Yaw (left/right) is a cleaner
# signal for "looking away from screen".
_HEAD_POSE_PITCH_TOLERANCE_DEG = 50.0
_HEAD_POSE_YAW_TOLERANCE_DEG = 40.0


# ── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class DistractionEvent:
    """Emitted when a distraction threshold is exceeded.

    Attributes
    ----------
    event_type : str
        ``EVENT_DISTRACTION_POSE``.
    duration : float
        Continuous duration (seconds) of the distraction at the
        moment the event fires.
    timestamp : float
        ``time.time()`` when the event was created.
    """

    event_type: str
    duration: float
    timestamp: float


# Type alias for the event callback.
DistractionCallback = Callable[[DistractionEvent], None]


# ── DistractionEngine ──────────────────────────────────────────


class DistractionEngine:
    """Head-pose distraction detector with one threshold timer.

    Parameters
    ----------
    baseline_pitch : float
        Calibrated neutral head pitch (degrees).
    baseline_yaw : float
        Calibrated neutral head yaw (degrees).
    head_pose_threshold_s : float
        Seconds of continuous head-pose deviation before firing.
        Clamped to ``[HEAD_POSE_THRESHOLD_MIN_S, HEAD_POSE_THRESHOLD_MAX_S]``.
    head_pose_pitch_tolerance_deg : float
        Pitch tolerance (degrees) for vertical head deviation.
        Higher than yaw to accommodate natural reading posture
        shifts and residual expression noise.
    head_pose_yaw_tolerance_deg : float
        Yaw tolerance (degrees) for horizontal head deviation.
        Tighter than pitch — left/right turns are a cleaner
        signal for "looking away from screen".
    callback : DistractionCallback or None
        Called once per distraction episode when a threshold is
        exceeded.  If ``None``, events are only logged.
    desk_writing_mode : bool
        When ``True``, the head-pose timer is completely suppressed —
        including during ``PRESENT_NO_FACE``.  Defaults to ``False``.

    Example
    -------
    ::

        from acuity.core.calibration import CalibrationResult

        engine = DistractionEngine(
            baseline_pitch=cal.baseline_pitch,
            baseline_yaw=cal.baseline_yaw,
            callback=my_handler,
        )

        # Per-frame loop (with presence fusion):
        from acuity.core.presence import determine_presence, PresenceState
        presence = determine_presence(face_result, pose_result)
        engine.update(face_result, presence)
    """

    def __init__(
        self,
        baseline_pitch: float,
        baseline_yaw: float,
        baseline_torso_pitch: float = 0.0,
        head_pose_threshold_s: float = HEAD_POSE_THRESHOLD_DEFAULT_S,
        head_pose_pitch_tolerance_deg: float = _HEAD_POSE_PITCH_TOLERANCE_DEG,
        head_pose_yaw_tolerance_deg: float = _HEAD_POSE_YAW_TOLERANCE_DEG,
        callback: DistractionCallback | None = None,
        desk_writing_mode: bool = False,
    ) -> None:
        self._baseline_pitch = baseline_pitch
        self._baseline_yaw = baseline_yaw
        self._baseline_torso_pitch = baseline_torso_pitch
        self._head_pose_pitch_tolerance_deg = head_pose_pitch_tolerance_deg
        self._head_pose_yaw_tolerance_deg = head_pose_yaw_tolerance_deg
        self._callback = callback
        self._desk_writing_mode = desk_writing_mode

        # Clamp thresholds to spec bounds.
        self._head_pose_threshold_s = _clamp(
            head_pose_threshold_s,
            HEAD_POSE_THRESHOLD_MIN_S,
            HEAD_POSE_THRESHOLD_MAX_S,
        )

        # ── Timer state (head pose) ─────────────────────────────
        self._pose_start: float | None = None
        self._pose_fired: bool = False

        # Body-relative pitch values for debug overlay.
        self._last_current_relative_pitch: float = 0.0
        self._last_calibrated_relative_pitch: float = 0.0

        log.info(
            "DistractionEngine initialised "
            "(pose_thresh=%.1fs, "
            "pitch_tol=%.1f°, yaw_tol=%.1f°, baseline pitch=%.1f° yaw=%.1f°, "
            "desk_writing_mode=%s).",
            self._head_pose_threshold_s,
            self._head_pose_pitch_tolerance_deg,
            self._head_pose_yaw_tolerance_deg,
            self._baseline_pitch,
            self._baseline_yaw,
            self._desk_writing_mode,
        )

    # ── Properties ──────────────────────────────────────────────

    @property
    def desk_writing_mode(self) -> bool:
        """Whether desk-writing mode is active.

        When ``True``, the head-pose timer is suppressed entirely.
        """
        return self._desk_writing_mode

    @desk_writing_mode.setter
    def desk_writing_mode(self, value: bool) -> None:
        old = self._desk_writing_mode
        self._desk_writing_mode = bool(value)
        if old != self._desk_writing_mode:
            # Reset the pose timer when toggling to avoid stale state.
            self._pose_start = None
            self._pose_fired = False
            log.info(
                "Desk writing mode %s.",
                "enabled" if self._desk_writing_mode else "disabled",
            )

    @property
    def head_pose_threshold_s(self) -> float:
        """Current head-pose threshold (seconds)."""
        return self._head_pose_threshold_s

    @head_pose_threshold_s.setter
    def head_pose_threshold_s(self, value: float) -> None:
        self._head_pose_threshold_s = _clamp(
            value,
            HEAD_POSE_THRESHOLD_MIN_S,
            HEAD_POSE_THRESHOLD_MAX_S,
        )
        log.info(
            "Head-pose threshold updated to %.1f s.",
            self._head_pose_threshold_s,
        )

    @property
    def pose_elapsed_s(self) -> float:
        """Seconds the head-pose timer has been running (0.0 if idle)."""
        if self._pose_start is None:
            return 0.0
        return time.monotonic() - self._pose_start

    # ── Debug diagnostic properties ─────────────────────
    # Read-only getters that expose internal state for the debug
    # overlay snapshot.  These add zero overhead when debug mode is
    # off because _build_debug_snapshot() is never called.

    @property
    def debug_baseline_pitch(self) -> float:
        """Calibrated neutral head pitch (degrees)."""
        return self._baseline_pitch

    @property
    def debug_baseline_yaw(self) -> float:
        """Calibrated neutral head yaw (degrees)."""
        return self._baseline_yaw

    @property
    def debug_head_pose_pitch_tolerance_deg(self) -> float:
        """Pitch tolerance (degrees) before the distraction timer starts."""
        return self._head_pose_pitch_tolerance_deg

    @property
    def debug_head_pose_yaw_tolerance_deg(self) -> float:
        """Yaw tolerance (degrees) before the distraction timer starts."""
        return self._head_pose_yaw_tolerance_deg

    @property
    def debug_pose_fired(self) -> bool:
        """True if the head-pose timer already fired in the current episode."""
        return self._pose_fired

    @property
    def debug_current_relative_pitch(self) -> float:
        """Body-relative head pitch (degrees)."""
        return self._last_current_relative_pitch

    @property
    def debug_calibrated_relative_pitch(self) -> float:
        """Calibrated body-relative head pitch (degrees)."""
        return self._last_calibrated_relative_pitch

    # ── Public API ──────────────────────────────────────────────

    def update(
        self,
        face: FaceResult,
        presence: PresenceState = PresenceState.PRESENT,
        torso_pitch_deg: float = 0.0,
    ) -> None:
        """Process one frame's results and advance the head-pose timer.

        Parameters
        ----------
        face : FaceResult
            Output from :class:`FaceMeshAnalyzer.process`.
        presence : PresenceState
            Fused presence signal from face + pose detection.
            When ``PRESENT_NO_FACE``, the head-pose timer keeps
            running (user is present but looking away).
            Defaults to ``PRESENT`` for backward compatibility.
        torso_pitch_deg : float
            Current torso pitch (degrees). Used to normalize
            head pitch against body movement. Defaults to 0.0.

        Call this once per frame during a focus block.  When the timer
        exceeds its threshold, the registered callback fires once.
        The timer continues running (so ``pose_elapsed_s`` reflects
        total episode duration) but will not fire again until the
        condition clears and recurs.
        """
        effective_torso_pitch = (
            torso_pitch_deg if torso_pitch_deg != 0.0 else self._baseline_torso_pitch
        )

        now = time.monotonic()
        if not self._desk_writing_mode:
            self._update_pose_timer(face, now, presence, effective_torso_pitch)

    def reset(self) -> None:
        """Reset the head-pose timer to idle.

        Use when the session pauses, enters a rest period, or
        re-calibrates after an Away state.
        """
        self._pose_start = None
        self._pose_fired = False
        log.debug("DistractionEngine timers reset.")

    def set_baselines(
        self, baseline_pitch: float, baseline_yaw: float, baseline_torso_pitch: float = 0.0
    ) -> None:
        """Update calibration baselines (e.g. after re-calibration).

        Also resets the timer to avoid stale state.
        """
        self._baseline_pitch = baseline_pitch
        self._baseline_yaw = baseline_yaw
        self._baseline_torso_pitch = baseline_torso_pitch
        self.reset()
        log.info(
            "Baselines updated (pitch=%.1f°, yaw=%.1f°).",
            baseline_pitch,
            baseline_yaw,
        )

    # ── Private helpers ─────────────────────────────────────────

    def _is_head_deviated(
        self,
        face: FaceResult,
        presence: PresenceState,
        torso_pitch_deg: float = 0.0,
    ) -> bool:
        """True if the head pose has deviated beyond tolerance."""
        if presence == PresenceState.PRESENT_NO_FACE:
            return True
        if not face.face_detected:
            return False

        # Body-relative pitch: head pitch minus torso pitch
        # This stays stable when the whole body moves
        current_relative_pitch = face.pitch - torso_pitch_deg
        calibrated_relative_pitch = self._baseline_pitch - self._baseline_torso_pitch
        self._last_current_relative_pitch = current_relative_pitch
        self._last_calibrated_relative_pitch = calibrated_relative_pitch

        # Shortest-arc differences — a plain subtraction across the
        # ±180° boundary reports a few degrees of movement as ~360°
        # and trips the tolerance on every crossing.
        pitch_delta = abs(wrap_deg(current_relative_pitch - calibrated_relative_pitch))
        yaw_delta = abs(wrap_deg(face.yaw - self._baseline_yaw))

        return (
            pitch_delta > self._head_pose_pitch_tolerance_deg
            or yaw_delta > self._head_pose_yaw_tolerance_deg
        )

    def _update_pose_timer(
        self,
        face: FaceResult,
        now: float,
        presence: PresenceState,
        torso_pitch_deg: float = 0.0,
    ) -> None:
        """Advance or reset the head-pose deviation timer."""
        if self._is_head_deviated(face, presence, torso_pitch_deg):
            # Condition active — start or continue timing.
            if self._pose_start is None:
                self._pose_start = now
                self._pose_fired = False
                log.debug("Head-pose deviation started.")

            elapsed = now - self._pose_start
            if elapsed >= self._head_pose_threshold_s and not self._pose_fired:
                self._pose_fired = True
                self._fire(EVENT_DISTRACTION_POSE, elapsed)
        else:
            # Condition cleared — log episode duration if one was active.
            if self._pose_start is not None:
                elapsed = now - self._pose_start
                log.debug("Head-pose deviation cleared after %.1f s.", elapsed)
                self._pose_start = None
                self._pose_fired = False

    def _fire(self, event_type: str, duration: float) -> None:
        """Create a DistractionEvent and dispatch it."""
        event = DistractionEvent(
            event_type=event_type,
            duration=duration,
            timestamp=time.time(),
        )
        log.info("Distraction event: %s (%.1f s).", event_type, duration)
        if self._callback is not None:
            self._callback(event)

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<DistractionEngine "
            f"pose={self.pose_elapsed_s:.1f}s/{self._head_pose_threshold_s:.0f}s>"
        )


# ── Module-level helpers ────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to ``[lo, hi]``."""
    return max(lo, min(hi, value))
