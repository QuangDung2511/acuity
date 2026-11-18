# QThread worker - runs MediaPipe per frame, emits signals.
# ─────────────────────────────────────────────────────────────────
# Orchestrates the unified CV pipeline:
#   camera → face_mesh → pose_tracker → presence → engines
#
# Thread: the run() loop and everything it calls run on the worker
# thread.  The public setters are called from the GUI thread and only
# write plain attributes the worker reads once per frame.  Results
# leave through Qt signals only.
#
# Two phases per focus block:
#   1. Calibration (10 s) — collects baselines, emits frames
#      for the video preview, reports calibration_progress(0..100).
#   2. Tracking — runs engines per frame, emits event_detected for
#      any threshold breach.  No frames emitted (video feed stops
#      after calibration).
#
# Signals:
#   frame_ready(QImage)         — BGR frame converted to QImage
#   event_detected(EventData)   — distraction / fatigue
#   face_lost()                 — presence entered ABSENT state
#   calibration_progress(int)   — 0..100 during calibration phase
#
# Resource management:
#   Camera is released via stop() or release_camera() when the
#   session enters rest, away, or ends.  The pipeline can be
#   restarted for the next focus block via start().
#
# Frame source: the camera is opened through the system camera adapter
# (acuity.system.camera.open_source), so a test can substitute a
# synthetic or recorded source without touching this file.
#
# Port notes (ticket 03): the phone detector, posture engine and their
# per-frame calls are gone.  The pose landmarker stays for presence and
# the calibration skeleton.  The distraction engine no longer receives a
# torso pitch (PoseResult lost it in ticket 01), so head pitch is
# compared with its baseline directly; ticket 11 reworks head pose.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from acuity.core.calibration import CalibrationResult, CalibrationRunner
from acuity.core.camera import CameraError, CameraManager
from acuity.core.data_logger import DataLogger
from acuity.core.debug_overlay import DebugOverlayRenderer
from acuity.core.debug_snapshot import DebugSnapshot
from acuity.core.distraction_engine import DistractionEngine, DistractionEvent
from acuity.core.face_mesh import FaceMeshAnalyzer, FaceResult
from acuity.core.fatigue_engine import FatigueEngine, FatigueEvent
from acuity.core.pose_tracker import PoseAnalyzer, PoseResult
from acuity.core.presence import PresenceState, determine_presence
from acuity.utils.angles import wrap_deg
from acuity.utils.config import ConfigManager
from acuity.utils.constants import PERCLOS_GATE
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# Target loop interval (~30 FPS).
_FRAME_INTERVAL_S = 1.0 / 30.0


# ── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class EventData:
    """Unified event payload emitted via ``event_detected``.

    Wraps the engine-specific event types into a single signal-safe
    object so the UI / session manager has one connection point.

    Attributes
    ----------
    event_type : str
        One of the ``EVENT_*`` constants from ``constants.py``.
    reason : str
        Sub-type detail (e.g. ``"microsleep"``).
        Empty string for distraction events.
    duration : float
        Duration in seconds at the moment the event fired.
    timestamp : float
        ``time.time()`` when the event was created.
    debug_detail : str
        Raw metric values included when debug mode is ON.
        Empty string in normal operation — zero cost to consumers
        that don't use debug mode.
    """

    event_type: str
    reason: str
    duration: float
    timestamp: float
    debug_detail: str = ""  # populated only when debug_mode is True


# ── CVPipelineThread ────────────────────────────────────────────


class CVPipelineThread(QThread):
    """Background CV pipeline: camera → analyzers → engines → signals.

    Typical lifecycle (driven by session manager)::

        pipeline = CVPipelineThread()
        pipeline.frame_ready.connect(ui.show_frame)
        pipeline.event_detected.connect(session.on_event)
        pipeline.face_lost.connect(session.on_face_lost)
        pipeline.calibration_progress.connect(ui.update_progress)
        pipeline.start()     # begins calibration → tracking
        ...
        pipeline.stop()      # end of focus block / session

    Parameters
    ----------
    device_index : int
        Camera device index (from ConfigManager / Settings dropdown).
    calibration_result : CalibrationResult or None
        If provided, skip calibration and jump straight to tracking
        using these baselines (useful for testing / re-calibration
        that already completed externally).
    """

    # ── Signals ─────────────────────────────────────────────────
    frame_ready = Signal(QImage)
    event_detected = Signal(object)  # EventData
    face_lost = Signal()
    calibration_progress = Signal(int)
    calibration_complete = Signal(object)  # CalibrationResult

    def __init__(
        self,
        device_index: int = 0,
        calibration_result: CalibrationResult | None = None,
        parent: QThread | None = None,
    ) -> None:
        super().__init__(parent)
        self._device_index = device_index
        self._pre_calibration = calibration_result
        self._running = False
        self._paused = False
        # Edge detector for _paused, read only by the worker thread so the
        # engine resets on entering a pause happen off the GUI thread.
        self._was_paused = False

        # Populated on the worker thread during run().
        self._camera: CameraManager | None = None
        self._face_mesh: FaceMeshAnalyzer | None = None
        self._pose_tracker: PoseAnalyzer | None = None
        self._distraction_engine: DistractionEngine | None = None
        self._fatigue_engine: FatigueEngine | None = None

        # Stored after calibration completes so _build_debug_snapshot()
        # can reference baselines during the tracking phase.
        self._cal_result: CalibrationResult | None = None

        # Presence tracking for face_lost signal.
        self._prev_presence: PresenceState = PresenceState.PRESENT

        # ── Debug overlay ────────────────────────────────
        # Read from config at construction so the pipeline respects the
        # persisted setting on startup without waiting for a UI toggle.
        cfg = ConfigManager()

        # ── Camera mirroring ─────────────────────────────────────
        # Applied to the raw frame immediately after capture, so the
        # analyzers, calibration baselines and the debug overlay all
        # operate on the same orientation.  Toggling this mid-session
        # would invert the sign of the calibrated yaw baseline, so the
        # UI only offers it while idle.
        self._flip_horizontal: bool = cfg.get_bool("camera_flip")

        self._debug_mode: bool = cfg.get_bool("debug_overlay")
        self._overlay: DebugOverlayRenderer | None = (
            DebugOverlayRenderer(
                show_landmarks=True,
                frame_skip=cfg.get_int("debug_overlay_frame_skip"),
            )
            if self._debug_mode
            else None
        )

        # Last-known per-frame results — cached so engine callbacks can
        # build debug_detail strings without needing per-frame locals.
        self._last_face: FaceResult | None = None
        self._last_pose: PoseResult | None = None

        # Last event fired this frame — set by _on_* callbacks, consumed
        # and cleared by the tracking loop when logging to CSV.
        self._last_event: EventData | None = None

        # ── Per-frame CSV logger ───────────────────────────────
        # Created alongside the overlay; closed in _cleanup().
        self._data_logger: DataLogger | None = DataLogger() if self._debug_mode else None
        if self._data_logger is not None:
            log.info("Data logger started: %s", self._data_logger.path)

    # ── Public control API (called from GUI thread) ─────────────

    def stop(self) -> None:
        """Request the pipeline loop to stop and release resources."""
        self._running = False

    def pause(self) -> None:
        """Suspend engine processing without stopping the camera thread.

        Frames are still read to prevent camera buffer stalls, but no
        analyzer or engine work is done and no signals are emitted.
        """
        self._paused = True
        log.debug("CV pipeline paused.")

    def resume(self) -> None:
        """Resume engine processing after a pause."""
        self._paused = False
        log.debug("CV pipeline resumed.")

    def release_camera(self) -> None:
        """Release the camera without stopping the thread.

        Used when entering REST or AWAY states, so the camera is not
        held through a break, while keeping the thread alive for a quick restart.
        """
        if self._camera is not None:
            self._camera.release()

    def set_flip_horizontal(self, enabled: bool) -> None:
        """Mirror the camera image horizontally (selfie view).

        Safe to call from the GUI thread — ``_flip_horizontal`` is a
        plain bool read once per frame by the worker.

        Intended to be set while idle: the flip is applied before
        analysis, so changing it after calibration would invalidate the
        sign of the yaw baseline.  The Focus tab
        disables its switch during a session for that reason.
        """
        self._flip_horizontal = enabled
        log.debug("Camera flip set to %s.", enabled)

    def set_desk_writing_mode(self, enabled: bool) -> None:
        """Update desk-writing mode on the live distraction engine."""
        if self._distraction_engine is not None:
            self._distraction_engine.desk_writing_mode = enabled

    def on_session_pause(self) -> None:
        """Forward session-pause notification to the fatigue engine."""
        if self._fatigue_engine:
            self._fatigue_engine.on_session_pause()

    def on_session_resume(self) -> None:
        """Forward session-resume notification to the fatigue engine."""
        if self._fatigue_engine:
            self._fatigue_engine.on_session_resume()

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable the debug overlay at runtime.

        Safe to call from the GUI thread while the worker thread is
        running — ``_debug_mode`` is a plain bool; the GIL guarantees
        atomic reads on the worker side.

        When enabling: creates a fresh :class:`DebugOverlayRenderer`
        (respecting the current ``debug_overlay_frame_skip`` config
        value) and starts emitting annotated frames from the tracking
        loop.

        When disabling: destroys the renderer and stops emitting frames.
        The ``VideoFeedWidget`` returns to the static 'Tracking Active'
        indicator (handled by :meth:`MainWindow._on_debug_overlay_changed`).
        """
        self._debug_mode = enabled
        if enabled:
            cfg = ConfigManager()
            self._overlay = DebugOverlayRenderer(
                show_landmarks=True,
                frame_skip=cfg.get_int("debug_overlay_frame_skip"),
            )
            log.debug(
                "Debug overlay enabled (frame_skip=%d).", cfg.get_int("debug_overlay_frame_skip")
            )
        else:
            self._overlay = None
            log.debug("Debug overlay disabled.")

    def update_thresholds(
        self,
        *,
        head_pose_s: float | None = None,
    ) -> None:
        """Live-update engine thresholds from the Settings UI."""
        if head_pose_s is not None and self._distraction_engine is not None:
            self._distraction_engine.head_pose_threshold_s = head_pose_s

    # ── QThread entry point ─────────────────────────────────────

    def run(self) -> None:  # noqa: C901 — orchestration is inherently sequential
        """Main loop: calibration → tracking.  Runs on the worker thread."""
        self._running = True

        try:
            self._init_analyzers()
            self._open_camera()
        except (CameraError, FileNotFoundError) as exc:
            log.error("Pipeline start-up failed: %s", exc)
            self._cleanup()
            return

        try:
            # ── Phase 1: Calibration ────────────────────────────
            cal = self._pre_calibration
            if cal is None:
                cal = self._run_calibration()
                if cal is None:
                    # Calibration aborted (stop() called or camera lost).
                    return

            # ── Phase 2: Tracking ───────────────────────────────
            self._init_engines(cal)
            self._run_tracking()
        finally:
            self._cleanup()

    # ── Frame acquisition ───────────────────────────────────────

    def _read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame from the camera, mirrored if the user asked.

        Every consumer downstream (analyzers, engines, overlay, preview)
        sees the flipped frame, so no coordinate translation is needed
        anywhere else.
        """
        ok, frame = self._camera.read_frame()
        if ok and frame is not None and self._flip_horizontal:
            frame = cv2.flip(frame, 1)
        return ok, frame

    # ── Phase 1: Calibration ────────────────────────────────────

    def _run_calibration(self) -> CalibrationResult | None:
        """Collect baselines for 10 s, emit frames + progress."""
        runner = CalibrationRunner()
        log.info("Calibration phase starting.")

        while self._running:
            t_start = time.monotonic()

            ok, frame = self._read_frame()
            if not ok:
                if not self._camera.is_opened:
                    log.error("Camera lost during calibration.")
                    return None
                self._sleep_until(t_start)
                continue

            face = self._face_mesh.process(frame)
            pose = self._pose_tracker.process(frame)

            runner.feed(face)

            # ── Debug calibration overlay ─────
            # When debug mode is ON, annotate the frame with live EAR,
            # pitch/yaw, sample counts, and pose skeleton before emitting.
            # The overlay renderer already exists (created in __init__ or
            # set_debug_mode) so this is a no-alloc hot path when debug
            # is OFF.
            if self._debug_mode and self._overlay is not None:
                annotated = self._overlay.draw_calibration(
                    frame.copy(),
                    face,
                    pose,
                    face_sample_count=len(runner._ears),
                    elapsed_s=runner.elapsed_s,
                    duration_s=runner._duration_s,
                )
                emit_frame = annotated
            else:
                emit_frame = frame

            # Emit frame for the video preview.
            self.frame_ready.emit(self._frame_to_qimage(emit_frame))

            # Report progress (0..100).
            progress = min(100, int(runner.elapsed_s / runner._duration_s * 100))
            self.calibration_progress.emit(progress)

            if runner.is_complete:
                self.calibration_progress.emit(100)
                self.calibration_complete.emit(runner.result)
                log.info("Calibration phase complete.")
                return runner.result

            self._sleep_until(t_start)

        return None  # stop() called during calibration

    # ── Phase 2: Tracking ───────────────────────────────────────

    def _run_tracking(self) -> None:
        """Per-frame analysis loop: analyzers → presence → engines."""
        log.info("Tracking phase starting.")

        while self._running:
            t_start = time.monotonic()

            ok, frame = self._read_frame()
            if not ok:
                if not self._camera.is_opened:
                    log.error("Camera lost during tracking.")
                    self.face_lost.emit()
                    return
                self._sleep_until(t_start)
                continue

            # While paused, drain frames to prevent camera buffer stalls
            # but skip all analysis and signal emission.
            #
            # The distraction timer anchors its episodes to
            # an absolute time.monotonic() value, so leaving them running
            # across a pause makes the whole pause count as deviation: the
            # first tracked frame after resume sees an elapsed/accumulated
            # time far past the threshold and fires immediately.  Clearing
            # them on the pause edge — on this thread, so it cannot
            # interleave with an update() in flight — starts the episodes
            # fresh on resume.  FatigueEngine is handled separately via
            # on_session_pause/on_session_resume, which need the pause
            # duration to decay the score rather than discard it.
            if self._paused:
                if not self._was_paused:
                    self._was_paused = True
                    self._distraction_engine.reset()
                self._sleep_until(t_start)
                continue
            self._was_paused = False

            # ── Analyzers ───────────────────────────────────────
            face = self._face_mesh.process(frame)
            pose = self._pose_tracker.process(frame)

            # Cache for debug callbacks.
            self._last_face = face
            self._last_pose = pose

            # ── Presence fusion ─────────────────────────
            presence = determine_presence(face, pose)

            # Emit face_lost when presence transitions to ABSENT.
            if presence == PresenceState.ABSENT and self._prev_presence != PresenceState.ABSENT:
                self.face_lost.emit()
            self._prev_presence = presence

            # ── Engines ─────────────────────────────────────────
            self._distraction_engine.update(face, presence)
            self._fatigue_engine.update(
                face,
                presence,
                head_pitch=face.pitch,
                baseline_pitch=self._cal_result.baseline_pitch,
                mar=face.mar,
            )

            # ── Debug overlay ────────────────────────────
            # Check self._debug_mode dynamically every frame so that
            # live toggling via Ctrl+D or the Settings checkbox takes
            # effect immediately without restarting the pipeline.
            if self._debug_mode and self._overlay is not None:
                snapshot = self._build_debug_snapshot(face, pose, presence)
                annotated = self._overlay.draw(
                    frame.copy(),
                    snapshot,
                    face_landmarks_px=face.landmarks_px,
                    pose_landmarks_px=pose.landmarks,
                )
                self.frame_ready.emit(self._frame_to_qimage(annotated))

                # ── CSV logging ───────────────────────────────
                if self._data_logger is not None:
                    ev = self._last_event
                    self._last_event = None
                    self._data_logger.log(
                        snapshot,
                        event_fired=ev is not None,
                        event_type=ev.event_type if ev is not None else "",
                        event_reason=ev.reason if ev is not None else "",
                    )

            self._sleep_until(t_start)

        log.info("Tracking loop stopped.")

    # ── Initialisation helpers ──────────────────────────────────

    def _init_analyzers(self) -> None:
        """Create the two CV analyzers (worker thread)."""
        self._face_mesh = FaceMeshAnalyzer()
        self._pose_tracker = PoseAnalyzer()
        log.info("CV analyzers initialised.")

    def _open_camera(self) -> None:
        """Open the camera device."""
        self._camera = CameraManager()
        self._camera.open(self._device_index)

    def _init_engines(self, cal: CalibrationResult) -> None:
        """Create the two detection engines using calibration baselines."""
        cfg = ConfigManager()

        # Persist for _build_debug_snapshot() during tracking.
        self._cal_result = cal

        self._distraction_engine = DistractionEngine(
            baseline_pitch=cal.baseline_pitch,
            baseline_yaw=cal.baseline_yaw,
            head_pose_threshold_s=cfg.get_float("head_pose_threshold"),
            desk_writing_mode=cfg.get_bool("desk_writing_mode"),
            callback=self._on_distraction,
        )
        self._fatigue_engine = FatigueEngine(
            baseline_ear=cal.baseline_ear,
            baseline_ear_std=cal.baseline_ear_std,
            baseline_mar=cal.baseline_mar,
            baseline_mar_std=cal.baseline_mar_std,
            callback=self._on_fatigue,
        )
        log.info("Detection engines initialised.")

    # ── Debug snapshot builder ──────────────────────────

    def _build_debug_snapshot(
        self,
        face,
        pose: PoseResult,
        presence: PresenceState,
    ) -> DebugSnapshot:
        """Assemble a :class:`DebugSnapshot` from current engine state.

        Called on every frame **only** when debug mode is active, so the
        per-call cost (~20 attribute reads) has no impact in normal use.

        Parameters
        ----------
        face :
            :class:`~acuity.core.face_mesh.FaceResult` for the current frame.
        pose :
            :class:`~acuity.core.pose_tracker.PoseResult` for the current frame.
        presence :
            Fused :class:`~acuity.core.presence.PresenceState`.
        """
        cal = self._cal_result
        de = self._distraction_engine
        fe = self._fatigue_engine

        # Guard: if engines aren't ready yet, return empty snapshot.
        if cal is None or de is None or fe is None:
            return DebugSnapshot(presence=presence.name)

        # Smoothed EAR from the fatigue engine's internal buffer.
        # The buffer is cleared while the engine is paused (face loss,
        # pitch gate), where debug_smoothed_ear reads 0.0 — which would
        # otherwise be logged as a closed eye.  Gate on observability so
        # the CSV never labels a paused frame as eyes_closed.
        smoothed_ear = fe.debug_smoothed_ear
        eyes_closed = fe.debug_ear_observed and smoothed_ear < fe.blink_threshold
        # Stricter than eyes_closed — this is what the microsleep timer runs
        # on, so the CSV can tell a squint from a real closure.
        eyes_shut = fe.debug_ear_observed and smoothed_ear < fe.shut_threshold

        return DebugSnapshot(
            # ── Presence ────────────────────────────────────────
            presence=presence.name,
            # ── Face mesh ────────────────────────────────────────
            face_detected=face.face_detected,
            ear_left=face.ear_left,
            ear_right=face.ear_right,
            ear_avg=face.ear_avg,
            pitch=face.pitch,
            yaw=face.yaw,
            roll=face.roll,
            blink_score=face.blink_score,
            # ── Calibration baselines ─────────────────────────────
            baseline_pitch=cal.baseline_pitch,
            baseline_yaw=cal.baseline_yaw,
            baseline_ear=cal.baseline_ear,
            baseline_ear_std=cal.baseline_ear_std,
            # ── Distraction — head pose ───────────────────────────
            head_pose_deviated=(
                abs(wrap_deg(face.pitch - cal.baseline_pitch))
                > de.debug_head_pose_pitch_tolerance_deg
                or abs(wrap_deg(face.yaw - cal.baseline_yaw)) > de.debug_head_pose_yaw_tolerance_deg
            )
            if face.face_detected
            else False,
            pitch_delta=abs(wrap_deg(face.pitch - cal.baseline_pitch)),
            yaw_delta=abs(wrap_deg(face.yaw - cal.baseline_yaw)),
            head_pose_pitch_tolerance_deg=de.debug_head_pose_pitch_tolerance_deg,
            head_pose_yaw_tolerance_deg=de.debug_head_pose_yaw_tolerance_deg,
            pose_timer_elapsed_s=de.pose_elapsed_s,
            pose_timer_threshold_s=de.head_pose_threshold_s,
            pose_timer_fired=de.debug_pose_fired,
            desk_writing_mode=de.desk_writing_mode,
            # ── Fatigue ───────────────────────────────────────────
            pitch_gated=fe.debug_pitch_gated,
            ear_suppressed=fe.debug_ear_suppressed,
            blink_threshold=fe.blink_threshold,
            smoothed_ear=smoothed_ear,
            eyes_closed=eyes_closed,
            eyes_shut=eyes_shut,
            microsleep_active=fe.microsleep_active,
            microsleep_elapsed_s=fe.debug_microsleep_elapsed_s,
            episode_bilateral_share=fe.debug_episode_bilateral_share,
            episode_blink_confirm=fe.debug_episode_blink_confirm,
            microsleep_withheld=fe.debug_microsleep_withheld,
            perclos=fe.perclos,
            perclos_gate=PERCLOS_GATE,
            blink_rate=fe.blink_rate,
            baseline_blink_rate=fe.baseline_blink_rate,
            fatigue_cooldown_remaining=fe.cooldown_remaining,
            # ── Fatigue scoring model (v2) ────────────────────────
            fatigue_score=fe.fatigue_score,
            signal_perclos=fe.debug_signal_perclos,
            signal_blink_dur=fe.debug_signal_blink_dur,
            signal_yawn=fe.debug_signal_yawn,
            signal_blink_freq=fe.debug_signal_blink_freq,
            signal_head_nod=fe.debug_signal_head_nod,
            mar=face.mar,
            mar_smoothed=fe.debug_mar_smoothed,
            yawn_threshold=fe.yawn_threshold,
            yawn_active=fe.debug_yawn_active,
            yawn_count_5min=fe.debug_yawn_count,
            avg_blink_duration_ms=fe.debug_avg_blink_duration_ms,
            head_nod_count_1min=fe.debug_nod_count,
            fatigue_warning_active=fe.debug_warning_active,
            fatigue_alert_active=fe.debug_alert_active,
            # ── Pose landmarker (presence input) ──────────────────
            pose_detected=pose.pose_detected,
            # ── Distraction — body relative pitch ────────────────
            body_relative_pitch=de.debug_current_relative_pitch,
            calibrated_relative_pitch=de.debug_calibrated_relative_pitch,
        )

    # ── Engine callbacks → signal emission ──────────────────────

    def _on_distraction(self, event: DistractionEvent) -> None:
        detail = ""
        if self._debug_mode and self._distraction_engine is not None:
            face = self._last_face
            de = self._distraction_engine
            if face is not None and face.face_detected:
                pitch_d = abs(wrap_deg(face.pitch - de.debug_baseline_pitch))
                yaw_d = abs(wrap_deg(face.yaw - de.debug_baseline_yaw))
                detail = (
                    f"pitch={face.pitch:.1f}° (Δ{pitch_d:.1f}°), "
                    f"yaw={face.yaw:.1f}° (Δ{yaw_d:.1f}°), "
                    f"elapsed={event.duration:.1f}s"
                )
            else:
                detail = f"elapsed={event.duration:.1f}s (face not detected)"
        self.event_detected.emit(
            EventData(
                event_type=event.event_type,
                reason="",
                duration=event.duration,
                timestamp=event.timestamp,
                debug_detail=detail,
            )
        )
        self._last_event = EventData(
            event_type=event.event_type,
            reason="",
            duration=event.duration,
            timestamp=event.timestamp,
            debug_detail=detail,
        )

    def _on_fatigue(self, event: FatigueEvent) -> None:
        detail = ""
        if self._debug_mode and self._fatigue_engine is not None:
            fe = self._fatigue_engine
            detail = (
                f"reason={event.reason}, "
                f"score={event.fatigue_score:.2f}, "
                f"perclos={event.perclos:.2f}, "
                f"blink_dur={event.avg_blink_duration_ms:.0f}ms, "
                f"yawns={event.yawn_count}, "
                f"blink_rate={fe.blink_rate:.1f}/min"
            )
        self.event_detected.emit(
            EventData(
                event_type=event.event_type,
                reason=event.reason,
                duration=event.duration,
                timestamp=event.timestamp,
                debug_detail=detail,
            )
        )
        self._last_event = EventData(
            event_type=event.event_type,
            reason=event.reason,
            duration=event.duration,
            timestamp=event.timestamp,
            debug_detail=detail,
        )

    # ── Cleanup ─────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Release all resources (worker thread)."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._pose_tracker is not None:
            self._pose_tracker.close()
            self._pose_tracker = None
        self._distraction_engine = None
        self._fatigue_engine = None
        if self._data_logger is not None:
            self._data_logger.close()
            self._data_logger = None
            log.info("Data logger closed.")
        self._running = False
        log.info("Pipeline resources released.")

    # ── Frame conversion ────────────────────────────────────────

    @staticmethod
    def _frame_to_qimage(frame: np.ndarray) -> QImage:
        """Convert a BGR OpenCV frame to a ``QImage`` (RGB888)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()  # .copy() detaches from the numpy buffer

    # ── Timing ──────────────────────────────────────────────────

    def _sleep_until(self, frame_start: float) -> None:
        """Sleep for the remainder of the frame interval."""
        elapsed = time.monotonic() - frame_start
        remaining = _FRAME_INTERVAL_S - elapsed
        if remaining > 0:
            self.msleep(int(remaining * 1000))
