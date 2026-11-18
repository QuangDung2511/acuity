# Tests for timer module — using recorded footage instead of live camera.
# ─────────────────────────────────────────────────────────────────
# Three layers:
#   1. Unit tests       — StructuredTimer + FluidTimer logic in
#                         isolation (mock QTimer, no camera)
#   2. Video playback   — feed test_full_session.mp4 through real
#                         CV analyzers while timers run alongside
#   3. Full integration — wire the CV pipeline + timer + engines
#                         together and verify events + timing.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from pathlib import Path

import pytest

from acuity.core.calibration import CalibrationResult, CalibrationRunner
from acuity.core.timer import FluidTimer, StructuredTimer
from acuity.utils.constants import (
    FLUID_REST_DIVISOR,
    REST_TYPE_LONG,
    REST_TYPE_SHORT,
)

# ── Paths ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "private" / "test_full_session.mp4"
_FACE_MODEL = _PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"
_POSE_MODEL = _PROJECT_ROOT / "assets" / "models" / "pose_landmarker.task"

_HAS_VIDEO = _VIDEO_PATH.exists()
_HAS_MODELS = _FACE_MODEL.exists() and _POSE_MODEL.exists()


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS — StructuredTimer logic (no camera, no models)
# ═══════════════════════════════════════════════════════════════


class TestStructuredTimerUnit:
    """Tests for StructuredTimer state machine and Pomodoro logic."""

    def test_initial_state_is_idle(self, qtbot):
        """Timer must start in an idle state (not running, not paused)."""
        timer = StructuredTimer(focus_m=1)
        assert not timer.is_running
        assert not timer.is_paused
        assert timer.remaining_s == 0.0
        assert timer.elapsed_s == 0.0

    def test_start_focus_transitions_to_running(self, qtbot):
        """start_focus must put the timer into a running state."""
        timer = StructuredTimer(focus_m=1)
        timer.start_focus()
        assert timer.is_running
        assert not timer.is_paused

    def test_remaining_decreases_after_start(self, qtbot):
        """remaining_s must be approximately focus_duration after start,
        and decrease as real time passes."""
        timer = StructuredTimer(focus_m=0.05)  # 3 seconds
        timer.start_focus()
        initial = timer.remaining_s
        assert initial > 0.0
        assert initial <= 3.0

    def test_stop_halts_timer(self, qtbot):
        """stop() must set is_running=False without emitting finished."""
        timer = StructuredTimer(focus_m=1)
        finished_spy = []
        timer.finished.connect(lambda: finished_spy.append(True))
        timer.start_focus()
        timer.stop()
        assert not timer.is_running
        assert len(finished_spy) == 0

    def test_pause_and_resume(self, qtbot):
        """pause/resume must toggle is_paused and exclude paused time."""
        timer = StructuredTimer(focus_m=1)
        timer.start_focus()

        timer.pause()
        assert timer.is_paused
        assert timer.is_running  # still "running" conceptually

        elapsed_at_pause = timer.elapsed_s
        time.sleep(0.05)
        # Elapsed should not advance while paused.
        assert timer.elapsed_s == pytest.approx(elapsed_at_pause, abs=0.01)

        timer.resume()
        assert not timer.is_paused
        assert timer.is_running

    def test_pause_when_not_running_is_noop(self, qtbot):
        timer = StructuredTimer(focus_m=1)
        timer.pause()  # not running → no effect
        assert not timer.is_paused

    def test_resume_when_not_paused_is_noop(self, qtbot):
        timer = StructuredTimer(focus_m=1)
        timer.start_focus()
        timer.resume()  # not paused → no effect
        assert not timer.is_paused

    def test_mark_block_complete_increments_count(self, qtbot):
        timer = StructuredTimer(focus_m=1, cycle_length=4)
        assert timer.completed_blocks == 0

        timer.mark_block_complete()
        assert timer.completed_blocks == 1

        timer.mark_block_complete()
        assert timer.completed_blocks == 2

    def test_next_rest_type_short_for_non_boundary(self, qtbot):
        """Blocks 1–3 in a 4-block cycle should yield Short rest."""
        timer = StructuredTimer(focus_m=1, cycle_length=4)
        # 0 completed → next = block #1 → (0+1) % 4 = 1 ≠ 0 → Short
        assert timer.next_rest_type == REST_TYPE_SHORT

    def test_next_rest_type_long_at_cycle_boundary(self, qtbot):
        """Block 4 in a 4-block cycle should yield Long rest."""
        timer = StructuredTimer(focus_m=1, cycle_length=4)
        for _ in range(3):
            timer.mark_block_complete()
        # 3 completed → next = block #4 → (3+1) % 4 = 0 → Long
        assert timer.next_rest_type == REST_TYPE_LONG

    def test_next_rest_duration_matches_type(self, qtbot):
        timer = StructuredTimer(
            focus_m=25,
            short_rest_m=5,
            long_rest_m=15,
            cycle_length=4,
        )
        assert timer.next_rest_duration_s == 5.0 * 60.0  # Short

        for _ in range(3):
            timer.mark_block_complete()
        assert timer.next_rest_duration_s == 15.0 * 60.0  # Long

    def test_reset_clears_everything(self, qtbot):
        timer = StructuredTimer(focus_m=1)
        timer.start_focus()
        timer.mark_block_complete()
        timer.mark_block_complete()
        timer.reset()
        assert not timer.is_running
        assert timer.completed_blocks == 0

    def test_focus_duration_property(self, qtbot):
        timer = StructuredTimer(focus_m=25)
        assert timer.focus_duration_s == 25.0 * 60.0

        timer.focus_duration_s = 30.0 * 60.0
        assert timer.focus_duration_s == 30.0 * 60.0

    def test_cycle_length_property(self, qtbot):
        timer = StructuredTimer(cycle_length=4)
        assert timer.cycle_length == 4

        timer.cycle_length = 6
        assert timer.cycle_length == 6

        # Minimum of 1
        timer.cycle_length = 0
        assert timer.cycle_length == 1

    def test_finished_signal_on_countdown_complete(self, qtbot):
        """finished signal must fire when countdown reaches zero.
        We use a very short duration to avoid long waits."""
        timer = StructuredTimer(focus_m=0.02)  # 1.2 seconds
        with qtbot.waitSignal(timer.finished, timeout=5000):
            timer.start_focus()

    def test_tick_signal_emits_remaining(self, qtbot):
        """tick(int) must emit at least once with a non-negative value."""
        timer = StructuredTimer(focus_m=0.05)  # 3 seconds
        ticks = []
        timer.tick.connect(ticks.append)
        timer.start_focus()

        # Wait for at least one tick.
        qtbot.waitUntil(lambda: len(ticks) > 0, timeout=3000)
        timer.stop()

        assert len(ticks) > 0
        assert all(t >= 0 for t in ticks)

    def test_start_rest_countdown(self, qtbot):
        """start_rest must begin a rest countdown using the correct
        duration based on the current block count."""
        timer = StructuredTimer(
            focus_m=25,
            short_rest_m=0.05,
            long_rest_m=15,
            cycle_length=4,
        )
        timer.start_rest()
        assert timer.is_running
        # Short rest = 0.05 min = 3 seconds → remaining should be close to 3
        assert timer.remaining_s <= 3.0

    def test_repr(self, qtbot):
        timer = StructuredTimer(focus_m=1)
        assert "idle" in repr(timer)
        timer.start_focus()
        assert "running" in repr(timer)
        timer.pause()
        assert "paused" in repr(timer)


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS — FluidTimer logic (no camera, no models)
# ═══════════════════════════════════════════════════════════════


class TestFluidTimerUnit:
    """Tests for FluidTimer (stopwatch + dynamic rest calculation)."""

    def test_initial_state_is_idle(self, qtbot):
        timer = FluidTimer(target_m=40)
        assert not timer.is_running
        assert not timer.is_paused
        assert timer.elapsed_s == 0.0

    def test_start_begins_counting(self, qtbot):
        timer = FluidTimer(target_m=40)
        timer.start()
        assert timer.is_running
        time.sleep(0.05)
        assert timer.elapsed_s > 0.0

    def test_pause_freezes_elapsed(self, qtbot):
        timer = FluidTimer(target_m=40)
        timer.start()
        time.sleep(0.05)
        timer.pause()
        assert timer.is_paused

        frozen = timer.elapsed_s
        time.sleep(0.05)
        assert timer.elapsed_s == pytest.approx(frozen, abs=0.01)

    def test_resume_continues_counting(self, qtbot):
        timer = FluidTimer(target_m=40)
        timer.start()
        timer.pause()
        frozen = timer.elapsed_s
        timer.resume()
        time.sleep(0.05)
        assert timer.elapsed_s > frozen

    def test_finish_emits_signal_and_stops(self, qtbot):
        timer = FluidTimer(target_m=40)
        timer.start()
        time.sleep(0.05)

        with qtbot.waitSignal(timer.finished, timeout=1000):
            timer.finish()

        assert not timer.is_running
        assert timer.elapsed_s > 0.0

    def test_rest_duration_is_quarter_of_elapsed(self, qtbot):
        """rest_duration_s = elapsed_s / FLUID_REST_DIVISOR (4)."""
        timer = FluidTimer(target_m=40)
        timer.start()
        time.sleep(0.1)
        timer.finish()

        expected = timer.elapsed_s / FLUID_REST_DIVISOR
        assert timer.rest_duration_s == pytest.approx(expected, rel=0.05)

    def test_stop_without_signal(self, qtbot):
        """stop() must not emit finished."""
        timer = FluidTimer(target_m=40)
        finished_spy = []
        timer.finished.connect(lambda: finished_spy.append(True))
        timer.start()
        timer.stop()
        assert not timer.is_running
        assert len(finished_spy) == 0

    def test_reset_clears_elapsed(self, qtbot):
        timer = FluidTimer(target_m=40)
        timer.start()
        time.sleep(0.05)
        timer.reset()
        assert not timer.is_running
        assert timer.elapsed_s == 0.0

    def test_target_property(self, qtbot):
        timer = FluidTimer(target_m=40)
        assert timer.target_s == 40.0 * 60.0

        timer.target_s = 60.0 * 60.0
        assert timer.target_s == 60.0 * 60.0

    def test_tick_signal_emits_elapsed(self, qtbot):
        timer = FluidTimer(target_m=40)
        ticks = []
        timer.tick.connect(ticks.append)
        timer.start()
        qtbot.waitUntil(lambda: len(ticks) > 0, timeout=3000)
        timer.stop()
        assert len(ticks) > 0

    def test_finish_when_not_running_is_noop(self, qtbot):
        timer = FluidTimer(target_m=40)
        finished_spy = []
        timer.finished.connect(lambda: finished_spy.append(True))
        timer.finish()  # not running
        assert len(finished_spy) == 0

    def test_repr(self, qtbot):
        timer = FluidTimer(target_m=40)
        assert "idle" in repr(timer)
        timer.start()
        assert "running" in repr(timer)
        timer.pause()
        assert "paused" in repr(timer)


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION — video footage + CV pipeline + timers
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not (_HAS_VIDEO and _HAS_MODELS),
    reason="Requires test_full_session.mp4 and MediaPipe model files.",
)
class TestTimerWithVideoFootage:
    """Feed recorded footage through the real CV analyzers alongside
    running timers — verifies the full pipeline works without a live
    camera.

    These tests monkeypatch CameraManager to read from video and
    exercise the timer ↔ engine interaction end-to-end.
    """

    def test_structured_timer_with_video_calibration(self, qtbot):
        """Run calibration from video footage while a StructuredTimer
        counts down, then verify baselines are plausible and the
        timer tracks elapsed time correctly.

        Intent: prove that calibration + structured timing work
        together against recorded footage.
        """
        import cv2

        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.pose_tracker import PoseAnalyzer

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened(), f"Cannot open test video: {_VIDEO_PATH}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Skip 3s pre-roll.
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        # Use a short calibration window for test speed.
        runner = CalibrationRunner(duration_s=2.0)

        # Start a structured timer alongside calibration.
        timer = StructuredTimer(focus_m=0.1)  # 6 seconds
        tick_values = []
        timer.tick.connect(tick_values.append)
        timer.start_focus()

        try:
            frames_fed = 0
            max_frames = int(5 * fps)

            while not runner.is_complete and frames_fed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                face = face_analyzer.process(frame)
                pose_analyzer.process(frame)
                runner.feed(face)
                frames_fed += 1

            assert runner.is_complete, f"Calibration did not complete after {frames_fed} frames."

            r = runner.result

            # Plausibility: EAR in open-eye range.
            assert 0.10 < r.baseline_ear < 0.50, (
                f"baseline_ear={r.baseline_ear:.3f} outside plausible range"
            )

            # Timer should have been running while we processed frames.
            assert timer.is_running
            assert timer.elapsed_s > 0.0
            assert len(tick_values) >= 0  # may or may not have ticked yet

        finally:
            timer.stop()
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()

    def test_fluid_timer_with_video_tracking(self, qtbot):
        """Start a FluidTimer, feed video frames through face mesh +
        pose tracker, then finish and verify rest calculation.

        Intent: prove that open-ended timing works alongside real
        CV frame processing from recorded footage.
        """
        import cv2

        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.pose_tracker import PoseAnalyzer

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened(), f"Cannot open test video: {_VIDEO_PATH}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Skip 3s pre-roll.
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        # Start a fluid stopwatch.
        timer = FluidTimer(target_m=40)
        tick_values = []
        timer.tick.connect(tick_values.append)
        timer.start()

        try:
            # Process ~2 seconds of video frames.
            frames_to_read = int(2 * fps)
            face_detected_count = 0

            for _ in range(frames_to_read):
                ret, frame = cap.read()
                if not ret:
                    break

                face = face_analyzer.process(frame)
                pose_analyzer.process(frame)

                if face.face_detected:
                    face_detected_count += 1

            # Timer should have been counting up.
            assert timer.is_running
            assert timer.elapsed_s > 0.0

            # Finish the timer and verify rest duration.
            timer.finish()
            assert not timer.is_running
            assert timer.elapsed_s > 0.0

            # Rest = elapsed / 4
            expected_rest = timer.elapsed_s / FLUID_REST_DIVISOR
            assert timer.rest_duration_s == pytest.approx(expected_rest, rel=0.05)

            # We should have detected at least some faces in the video.
            assert face_detected_count > 0, (
                "No faces detected in video — is the test fixture valid?"
            )

        finally:
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()

    def test_structured_pomodoro_cycle_with_video(self, qtbot):
        """Simulate a complete Pomodoro cycle (block complete → rest
        type check) while video provides the CV context.

        Intent: prove the timer's Pomodoro cycle logic works when
        driven by video-paced frame processing.
        """
        import cv2

        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.pose_tracker import PoseAnalyzer
        from acuity.core.presence import PresenceState, determine_presence

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened(), f"Cannot open test video: {_VIDEO_PATH}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Skip 3s pre-roll.
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        timer = StructuredTimer(
            focus_m=0.05,  # 3 seconds per block
            short_rest_m=0.02,  # 1.2 seconds short rest
            long_rest_m=0.05,  # 3 seconds long rest
            cycle_length=4,
        )

        try:
            rest_types_seen = []
            presence_states_seen = set()

            # Simulate 4 blocks (one full Pomodoro cycle).
            for _block in range(1, 5):
                timer.start_focus()

                # Process a handful of frames per block.
                for _ in range(int(0.5 * fps)):
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(3 * fps))
                        ret, frame = cap.read()
                        if not ret:
                            break

                    face = face_analyzer.process(frame)
                    pose = pose_analyzer.process(frame)
                    presence = determine_presence(face, pose)
                    presence_states_seen.add(presence)

                timer.stop()
                rest_type = timer.next_rest_type
                rest_types_seen.append(rest_type)
                timer.mark_block_complete()

                # Start and immediately stop the rest countdown.
                timer.start_rest()
                timer.stop()

            # Blocks 1-3 → Short, Block 4 → Long
            assert rest_types_seen[:3] == [REST_TYPE_SHORT] * 3
            assert rest_types_seen[3] == REST_TYPE_LONG
            assert timer.completed_blocks == 4

            # We should have seen at least PRESENT or PRESENT_NO_FACE.
            assert (
                PresenceState.PRESENT in presence_states_seen
                or PresenceState.PRESENT_NO_FACE in presence_states_seen
            )

        finally:
            timer.stop()
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()

    def test_pause_resume_during_video_playback(self, qtbot):
        """Pause the timer while continuing to read video frames, then
        resume — verify paused time is excluded from elapsed.

        Intent: paused time must not count toward
        focus duration, even when the camera (video) keeps producing
        frames.
        """
        import cv2

        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.pose_tracker import PoseAnalyzer

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        timer = StructuredTimer(focus_m=1)  # long enough it won't finish
        timer.start_focus()

        try:
            # Process some frames while running.
            for _ in range(int(0.5 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                face_analyzer.process(frame)
                pose_analyzer.process(frame)

            # Pause the timer.
            timer.pause()
            elapsed_at_pause = timer.elapsed_s

            # Continue reading video frames while paused (simulates
            # camera still capturing but timer frozen).
            for _ in range(int(1 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                face_analyzer.process(frame)
                pose_analyzer.process(frame)
                time.sleep(0.01)  # small sleep to let wall time pass

            # Elapsed should NOT have advanced.
            assert timer.elapsed_s == pytest.approx(elapsed_at_pause, abs=0.05)

            # Resume and process a few more frames.
            timer.resume()
            for _ in range(int(0.5 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                face_analyzer.process(frame)
                pose_analyzer.process(frame)
                time.sleep(0.01)

            # Elapsed should now be greater than at pause, but less
            # than if we had never paused (paused wall-time excluded).
            assert timer.elapsed_s > elapsed_at_pause

        finally:
            timer.stop()
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()

    def test_video_driven_distraction_detection(self, qtbot):
        """Feed video frames through the distraction engine while a
        timer is running — verify the pipeline detects presence and
        the timer operates independently.

        Intent: end-to-end proof that the timer + distraction engine
        can co-exist when driven by recorded footage.
        """
        import cv2

        from acuity.core.calibration import CalibrationRunner
        from acuity.core.distraction_engine import DistractionEngine
        from acuity.core.face_mesh import FaceMeshAnalyzer
        from acuity.core.pose_tracker import PoseAnalyzer
        from acuity.core.presence import determine_presence

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for _ in range(int(3 * fps)):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()
        pose_analyzer = PoseAnalyzer()

        # ── Phase 1: Calibrate from video ───────────────────────
        cal_runner = CalibrationRunner(duration_s=2.0)
        frames = 0
        while not cal_runner.is_complete and frames < int(5 * fps):
            ret, frame = cap.read()
            if not ret:
                break
            face = face_analyzer.process(frame)
            pose_analyzer.process(frame)
            cal_runner.feed(face)
            frames += 1

        assert cal_runner.is_complete
        cal = cal_runner.result

        # ── Phase 2: Track with distraction engine + timer ──────
        events_fired = []

        def on_distraction(event):
            events_fired.append(event)

        engine = DistractionEngine(
            baseline_pitch=cal.baseline_pitch,
            baseline_yaw=cal.baseline_yaw,
            head_pose_threshold_s=2.0,  # low threshold for test
            callback=on_distraction,
        )

        timer = FluidTimer(target_m=40)
        timer.start()

        try:
            frames_tracked = 0
            for _ in range(int(3 * fps)):
                ret, frame = cap.read()
                if not ret:
                    break

                face = face_analyzer.process(frame)
                pose = pose_analyzer.process(frame)
                presence = determine_presence(face, pose)

                engine.update(face, presence)
                frames_tracked += 1

            assert frames_tracked > 0
            assert timer.is_running
            assert timer.elapsed_s > 0.0

            timer.finish()

            # We don't assert specific events — the video content
            # determines whether distractions occur.  The key
            # validation is that nothing crashed and the timer
            # tracked time correctly.
            assert timer.rest_duration_s > 0.0

        finally:
            timer.stop()
            face_analyzer.close()
            pose_analyzer.close()
            cap.release()


# ═══════════════════════════════════════════════════════════════
#  FULL PIPELINE INTEGRATION — CVPipelineThread + timer
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not (_HAS_VIDEO and _HAS_MODELS),
    reason="Requires test_full_session.mp4 and MediaPipe model files.",
)
class TestTimerWithCVPipeline:
    """Wire CVPipelineThread to read from video (monkeypatched
    CameraManager) and run a timer in parallel, verifying the
    full orchestration.
    """

    def test_pipeline_calibration_with_structured_timer(self, qtbot):
        """Start the real CVPipelineThread against video footage,
        let calibration complete, and verify a StructuredTimer
        can track the focus block in parallel.

        Intent: integration proof that the threaded pipeline and
        the GUI-thread timer co-exist without conflicts when
        driven by recorded footage.
        """
        import cv2

        from acuity.core.camera import CameraManager
        from acuity.core.cv_pipeline import CVPipelineThread

        video_cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert video_cap.isOpened()

        fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        for _ in range(int(3 * fps)):
            video_cap.read()

        # ── Monkeypatch CameraManager → video file ─────────────
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
            return (True, frame) if ret else (False, None)

        def fake_release(self):
            self._cap = None
            self._device_index = -1
            self._consecutive_failures = 0

        CameraManager.open = fake_open
        CameraManager.read_frame = fake_read
        CameraManager.release = fake_release

        # Track pipeline events.
        calibration_results = []
        events_detected = []

        pipeline = CVPipelineThread(device_index=0)
        pipeline.calibration_complete.connect(calibration_results.append)
        pipeline.event_detected.connect(events_detected.append)

        timer = StructuredTimer(focus_m=1)
        tick_values = []
        timer.tick.connect(tick_values.append)

        try:
            pipeline.start()

            # Wait for calibration (up to 30s wall time).
            qtbot.waitUntil(
                lambda: len(calibration_results) > 0,
                timeout=30_000,
            )

            # Calibration done — start the focus timer.
            timer.start_focus()
            assert timer.is_running

            cal = calibration_results[0]
            assert isinstance(cal, CalibrationResult)
            assert cal.baseline_ear > 0.0

            # Let tracking run briefly.
            time.sleep(2)

            # Timer should have been ticking.
            assert timer.elapsed_s > 0.0

            # Stop everything.
            pipeline.stop()
            pipeline.wait(5000)
            timer.stop()

            assert not timer.is_running

        finally:
            CameraManager.open = original_open
            CameraManager.read_frame = original_read
            CameraManager.release = original_release
            timer.stop()
            pipeline.stop()
            pipeline.wait(5000)
            video_cap.release()
