# Tests for calibration module.
# ─────────────────────────────────────────────────────────────────
# Two layers:
#   1. Unit tests   — synthetic FaceResult, no models
#   2. Integration  — real video + MediaPipe models (skipped if
#                     fixtures/models are missing)
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from acuity.core.calibration import _MIN_SAMPLES, CalibrationResult, CalibrationRunner
from acuity.core.face_mesh import FaceResult

# ── Paths ───────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VIDEO_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "private" / "test_full_session.mp4"
_FACE_MODEL = _PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"

_HAS_VIDEO = _VIDEO_PATH.exists()
_HAS_MODELS = _FACE_MODEL.exists()


# ── Helpers ─────────────────────────────────────────────────────


def _make_face(
    ear: float = 0.30,
    pitch: float = 0.0,
    yaw: float = 0.0,
    detected: bool = True,
) -> FaceResult:
    """Create a FaceResult with sensible defaults."""
    return FaceResult(
        ear_left=ear,
        ear_right=ear,
        ear_avg=ear,
        mar=0.0,
        pitch=pitch,
        yaw=yaw,
        roll=0.0,
        face_detected=detected,
    )


_NO_FACE = _make_face(detected=False)


def _make_runner_with_mock_time(duration_s: float = 2.0):
    """Create a CalibrationRunner with a mock clock.

    Returns (runner, advance_time_fn).  Call advance_time_fn(seconds)
    to move the mock clock forward.  This lets tests accumulate
    samples before the window elapses.
    """
    fake_time = [100.0]

    patcher = patch("acuity.core.calibration.time")
    mock_time = patcher.start()
    mock_time.monotonic = lambda: fake_time[0]

    runner = CalibrationRunner(duration_s=duration_s)

    def advance(seconds: float) -> None:
        fake_time[0] += seconds

    return runner, advance, patcher


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS — synthetic data, no external dependencies
# ═══════════════════════════════════════════════════════════════


class TestCalibrationRunnerUnit:
    """Tests for CalibrationRunner logic in isolation."""

    def test_not_complete_before_duration(self):
        """Runner must not finalise before the time window elapses."""
        runner, advance, patcher = _make_runner_with_mock_time(5.0)
        try:
            runner.feed(_make_face())
            assert not runner.is_complete
            assert runner.result is None
        finally:
            patcher.stop()

    def test_completes_after_duration(self):
        """Runner finalises once elapsed time ≥ duration_s."""
        runner, advance, patcher = _make_runner_with_mock_time(1.0)
        try:
            # Feed enough samples within the window
            for _ in range(_MIN_SAMPLES):
                runner.feed(_make_face())
            assert not runner.is_complete

            # Advance past the window
            advance(1.5)
            runner.feed(_make_face())

            assert runner.is_complete
            assert runner.result is not None
            assert isinstance(runner.result, CalibrationResult)
        finally:
            patcher.stop()

    def test_median_baselines_are_correct(self):
        """Baselines must be the median of collected samples (intent:
        median is robust against outlier frames like blinks)."""
        runner, advance, patcher = _make_runner_with_mock_time(1.0)
        try:
            # Feed values where the median is knowable:
            # EAR:   [0.20, 0.25, 0.30, 0.35, 0.40] → median = 0.30
            # pitch: [-5, -2, 0, 2, 5]               → median = 0.0
            # yaw:   [-10, -5, 0, 5, 10]              → median = 0.0
            ears = [0.20, 0.25, 0.30, 0.35, 0.40]
            pitches = [-5.0, -2.0, 0.0, 2.0, 5.0]
            yaws = [-10.0, -5.0, 0.0, 5.0, 10.0]

            for i in range(5):
                runner.feed(_make_face(ear=ears[i], pitch=pitches[i], yaw=yaws[i]))

            # Advance past the window and trigger finalisation
            advance(1.5)
            runner.feed(_make_face(ear=0.30))

            assert runner.is_complete
            r = runner.result
            assert r.baseline_ear == pytest.approx(0.30)
            assert r.baseline_pitch == pytest.approx(0.0)
            assert r.baseline_yaw == pytest.approx(0.0)
        finally:
            patcher.stop()

    def test_skips_undetected_face_samples(self):
        """Frames where face_detected=False must not pollute baselines."""
        runner, advance, patcher = _make_runner_with_mock_time(1.0)
        try:
            # Feed _MIN_SAMPLES good faces + some no-face frames
            for _ in range(_MIN_SAMPLES):
                runner.feed(_make_face(ear=0.30))
            for _ in range(3):
                runner.feed(_NO_FACE)

            # Advance past window
            advance(1.5)
            runner.feed(_make_face(ear=0.30))

            assert runner.is_complete
            # EAR baseline should be 0.30, not dragged toward 0.0
            assert runner.result.baseline_ear == pytest.approx(0.30)
        finally:
            patcher.stop()

    def test_insufficient_face_samples_fallback(self):
        """When fewer than _MIN_SAMPLES faces are detected, face
        baselines should fall back to 0.0 (intent: prevent unstable
        baselines from influencing downstream engines)."""
        runner, advance, patcher = _make_runner_with_mock_time(1.0)
        try:
            # Feed 1 face sample in a window of _MIN_SAMPLES frames
            runner.feed(_make_face(ear=0.35))
            for _ in range(_MIN_SAMPLES - 1):
                runner.feed(_NO_FACE)

            advance(1.5)
            runner.feed(_NO_FACE)

            assert runner.is_complete
            assert runner.result.baseline_ear == 0.0
            assert runner.result.baseline_pitch == 0.0
            assert runner.result.baseline_yaw == 0.0
        finally:
            patcher.stop()

    def test_feed_after_complete_is_noop(self):
        """Calling feed() after finalisation must not change the result."""
        runner, advance, patcher = _make_runner_with_mock_time(1.0)
        try:
            for _ in range(_MIN_SAMPLES):
                runner.feed(_make_face(ear=0.30))

            advance(1.5)
            runner.feed(_make_face(ear=0.30))
            assert runner.is_complete
            first_result = runner.result

            # Feed wildly different data — should be ignored
            for _ in range(10):
                runner.feed(_make_face(ear=0.99))

            assert runner.result is first_result  # same object, unchanged
        finally:
            patcher.stop()

    def test_elapsed_before_first_feed(self):
        """elapsed_s must be 0.0 before any feed() call."""
        runner, _, patcher = _make_runner_with_mock_time(10.0)
        try:
            assert runner.elapsed_s == 0.0
        finally:
            patcher.stop()

    def test_elapsed_advances_after_feed(self):
        """elapsed_s must reflect time since the first feed() call."""
        runner, advance, patcher = _make_runner_with_mock_time(10.0)
        try:
            runner.feed(_make_face())
            advance(3.0)
            assert runner.elapsed_s == pytest.approx(3.0)
        finally:
            patcher.stop()

    def test_repr_before_and_after_complete(self):
        """__repr__ should reflect the runner's state."""
        runner, advance, patcher = _make_runner_with_mock_time(1.0)
        try:
            assert "CalibrationRunner" in repr(runner)

            for _ in range(_MIN_SAMPLES):
                runner.feed(_make_face())

            advance(1.5)
            runner.feed(_make_face())
            assert "complete" in repr(runner)
        finally:
            patcher.stop()

    def test_custom_duration(self):
        """Runner should respect a custom duration_s value."""
        runner, advance, patcher = _make_runner_with_mock_time(2.0)
        try:
            # Feed at t=100.0
            runner.feed(_make_face())
            assert not runner.is_complete

            # Feed at t=101.0 — still not 2s
            advance(1.0)
            for _ in range(_MIN_SAMPLES):
                runner.feed(_make_face())
            assert not runner.is_complete

            # Feed at t=102.0 — exactly 2s
            advance(1.0)
            runner.feed(_make_face())
            assert runner.is_complete
        finally:
            patcher.stop()


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION TEST — real video + MediaPipe models
# ═══════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not (_HAS_VIDEO and _HAS_MODELS),
    reason="Requires tests/fixtures/private/test_full_session.mp4 (private) and the face model.",
)
class TestCalibrationIntegration:
    """Run CalibrationRunner against the calibration segment of the
    recorded test video (segment 1: 3s–18s, user sitting straight)."""

    def test_calibration_from_video(self):
        """Feed real frames from the calibration segment through
        FaceMeshAnalyzer → CalibrationRunner and verify
        that baselines are physiologically plausible."""
        import cv2

        from acuity.core.face_mesh import FaceMeshAnalyzer

        cap = cv2.VideoCapture(str(_VIDEO_PATH))
        assert cap.isOpened(), f"Cannot open test video: {_VIDEO_PATH}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Skip the 3-second pre-roll countdown to reach segment 1.
        preroll_frames = int(3 * fps)
        for _ in range(preroll_frames):
            cap.read()

        face_analyzer = FaceMeshAnalyzer()

        # Use a short calibration window (2s) — we only need enough
        # frames to validate the pipeline, not the full 10s.  It is timed
        # on video time (1/fps per frame), not the wall clock: analysis
        # runs faster than real time, so a wall-clock window would depend
        # on how fast this machine runs FaceMesh.
        runner, advance, clock_patcher = _make_runner_with_mock_time(2.0)

        try:
            frames_fed = 0
            max_frames = int(5 * fps)  # safety cap: don't read forever

            while not runner.is_complete and frames_fed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                face_result = face_analyzer.process(frame)
                runner.feed(face_result)
                advance(1.0 / fps)
                frames_fed += 1

            assert runner.is_complete, (
                f"CalibrationRunner did not complete after {frames_fed} frames."
            )

            r = runner.result

            # ── Plausibility checks ─────────────────────────────
            # These verify intent: the calibration completed with
            # real data and produced finite, non-degenerate baselines.

            # EAR: typical open-eye range is 0.15–0.45
            assert 0.10 < r.baseline_ear < 0.50, (
                f"baseline_ear={r.baseline_ear:.3f} outside plausible range"
            )

            # Head pose: solvePnP + decomposeProjectionMatrix can
            # produce angles in any range depending on camera/face
            # geometry.  We only need them to be finite — downstream
            # engines compare *deviations* from baseline, not absolute
            # values.
            assert math.isfinite(r.baseline_pitch), (
                f"baseline_pitch={r.baseline_pitch} is not finite"
            )
            assert math.isfinite(r.baseline_yaw), f"baseline_yaw={r.baseline_yaw} is not finite"

        finally:
            clock_patcher.stop()
            face_analyzer.close()
            cap.release()
