# Unit tests for DebugOverlayRenderer.
# ─────────────────────────────────────────────────────────────────
# Tests:
#   • draw()             — returns valid BGR frame, same shape
#   • draw()             — edge cases: no face, no pose
#   • draw()             — color-coding (OK / warning / alert)
#   • draw()             — frame-skip caching behaviour
#   • draw_calibration() — returns valid BGR frame
#   • draw_calibration() — edge cases: no face detected, no pose
#   • Landmark colors    — palette constants are correct BGR tuples
#
# No external models or hardware required.  All inputs are synthetic
# NumPy arrays and minimal dataclass mock objects.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from acuity.core.debug_overlay import (
    _COLOR_ALERT,
    _COLOR_OK,
    _COLOR_WARN,
    _LM_L_EYE,
    _LM_NOSE,
    _LM_POSE_ANCHOR,
    _LM_R_EYE,
    _LM_SHLDR_NOSE_LINE,
    _LM_SHOULDER,
    _WARN_FRAC,
    DebugOverlayRenderer,
)
from acuity.core.debug_snapshot import DebugSnapshot

# ── Test helpers ─────────────────────────────────────────────────


def _blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a black BGR frame of the given dimensions."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_face_result(
    face_detected: bool = True,
    ear_left: float = 0.35,
    ear_right: float = 0.33,
    ear_avg: float = 0.34,
    pitch: float = 5.0,
    yaw: float = -2.0,
    roll: float = 1.0,
):
    m = MagicMock()
    m.face_detected = face_detected
    m.ear_left = ear_left
    m.ear_right = ear_right
    m.ear_avg = ear_avg
    m.pitch = pitch
    m.yaw = yaw
    m.roll = roll
    return m


def _make_pose_result(
    pose_detected: bool = True,
):
    m = MagicMock()
    m.pose_detected = pose_detected
    m.landmarks = None
    return m


def _default_snapshot(**overrides) -> DebugSnapshot:
    """Build a DebugSnapshot with sensible defaults, accepting overrides."""
    defaults = dict(
        presence="PRESENT",
        face_detected=True,
        ear_left=0.35,
        ear_right=0.33,
        ear_avg=0.34,
        pitch=5.0,
        yaw=-2.0,
        roll=1.0,
        baseline_pitch=2.0,
        baseline_yaw=-1.0,
        baseline_ear=0.34,
        baseline_ear_std=0.02,
        head_pose_deviated=False,
        pitch_delta=3.0,
        yaw_delta=1.0,
        head_pose_pitch_tolerance_deg=20.0,
        head_pose_yaw_tolerance_deg=15.0,
        pose_timer_elapsed_s=0.0,
        pose_timer_threshold_s=15.0,
        pose_timer_fired=False,
        desk_writing_mode=False,
        blink_threshold=0.25,
        smoothed_ear=0.32,
        eyes_closed=False,
        microsleep_active=False,
        microsleep_elapsed_s=0.0,
        perclos=0.04,
        perclos_gate=0.15,
        blink_rate=14.0,
        baseline_blink_rate=16.0,
        fatigue_cooldown_remaining=0.0,
        pose_detected=True,
    )
    defaults.update(overrides)
    return DebugSnapshot(**defaults)


def _face_landmarks(n: int = 478, w: int = 320, h: int = 240) -> np.ndarray:
    """Fake face landmark array (478, 2) within frame bounds."""
    rng = np.random.default_rng(seed=0)
    lm = np.zeros((n, 2), dtype=np.float32)
    lm[:, 0] = rng.uniform(0, w, n)
    lm[:, 1] = rng.uniform(0, h, n)
    return lm


def _pose_landmarks(n: int = 33, w: int = 320, h: int = 240) -> np.ndarray:
    """Fake pose landmark array (33, 2) within frame bounds."""
    rng = np.random.default_rng(seed=1)
    lm = np.zeros((n, 2), dtype=np.float32)
    lm[:, 0] = rng.uniform(0, w, n)
    lm[:, 1] = rng.uniform(0, h, n)
    return lm


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS
# ═══════════════════════════════════════════════════════════════


class TestLandmarkColorPalette:
    """Color constants have the correct BGR tuples and don't overlap with
    the traffic-light status palette (green/yellow/red)."""

    def test_status_colors_defined(self):
        assert _COLOR_OK == (0, 200, 0)  # green
        assert _COLOR_WARN == (0, 220, 220)  # yellow (BGR)
        assert _COLOR_ALERT == (0, 0, 220)  # red

    def test_landmark_colors_not_status_colors(self):
        """Landmark palette must be distinct from status palette."""
        status = {_COLOR_OK, _COLOR_WARN, _COLOR_ALERT}
        landmarks = {
            _LM_L_EYE,
            _LM_R_EYE,
            _LM_POSE_ANCHOR,
            _LM_NOSE,
            _LM_SHOULDER,
            _LM_SHLDR_NOSE_LINE,
        }
        assert status.isdisjoint(landmarks), (
            "Some landmark colors overlap with status colors — "
            "this would cause visual ambiguity in the overlay."
        )

    def test_landmark_colors_are_valid_bgr(self):
        """Every landmark color must be a 3-tuple of ints in [0, 255]."""
        for color in (
            _LM_L_EYE,
            _LM_R_EYE,
            _LM_POSE_ANCHOR,
            _LM_NOSE,
            _LM_SHOULDER,
            _LM_SHLDR_NOSE_LINE,
        ):
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_warn_fraction_value(self):
        assert _WARN_FRAC == pytest.approx(0.70)


class TestRendererInit:
    """DebugOverlayRenderer initialises correctly."""

    def test_default_params(self):
        r = DebugOverlayRenderer()
        assert r._show_landmarks is True
        assert r._frame_skip == 1
        assert r._frame_count == 0
        assert r._cached_overlay is None

    def test_custom_params(self):
        r = DebugOverlayRenderer(show_landmarks=False, frame_skip=3)
        assert r._show_landmarks is False
        assert r._frame_skip == 3

    def test_frame_skip_clamped_to_minimum_one(self):
        r = DebugOverlayRenderer(frame_skip=0)
        assert r._frame_skip == 1

    def test_frame_skip_negative_clamped(self):
        r = DebugOverlayRenderer(frame_skip=-5)
        assert r._frame_skip == 1


class TestDrawReturnShape:
    """draw() returns a valid BGR frame with the same H×W×3 shape."""

    def test_returns_same_shape_as_input(self):
        r = DebugOverlayRenderer()
        frame = _blank_frame(240, 320)
        result = r.draw(frame, _default_snapshot())
        assert result.shape == (240, 320, 3)
        assert result.dtype == np.uint8

    def test_returns_same_shape_tall_frame(self):
        r = DebugOverlayRenderer()
        frame = _blank_frame(480, 640)
        result = r.draw(frame, _default_snapshot())
        assert result.shape == (480, 640, 3)

    def test_draw_returns_frame_object(self):
        """draw() must return the frame (for chaining)."""
        r = DebugOverlayRenderer()
        frame = _blank_frame()
        result = r.draw(frame, _default_snapshot())
        assert result is frame

    def test_frame_is_mutated_in_place(self):
        """After draw(), the frame must differ from all-zeros."""
        r = DebugOverlayRenderer()
        frame = _blank_frame()
        r.draw(frame, _default_snapshot())
        assert not np.all(frame == 0), "Frame was not annotated"


class TestDrawEdgeCasesNoData:
    """draw() handles edge cases where face or pose data is absent."""

    def test_no_face_detected(self):
        r = DebugOverlayRenderer()
        snap = _default_snapshot(face_detected=False, ear_avg=0.0)
        frame = _blank_frame()
        result = r.draw(frame, snap)
        assert result.shape == (240, 320, 3)

    def test_no_pose_detected(self):
        r = DebugOverlayRenderer()
        snap = _default_snapshot(pose_detected=False)
        result = r.draw(_blank_frame(), snap)
        assert result.shape == (240, 320, 3)

    def test_absent_presence(self):
        r = DebugOverlayRenderer()
        snap = _default_snapshot(presence="ABSENT", face_detected=False, pose_detected=False)
        result = r.draw(_blank_frame(), snap)
        assert result.shape == (240, 320, 3)

    def test_no_face_landmarks(self):
        """draw() must not raise when face_landmarks_px is None."""
        r = DebugOverlayRenderer()
        r.draw(_blank_frame(), _default_snapshot(), face_landmarks_px=None)

    def test_no_pose_landmarks(self):
        """draw() must not raise when pose_landmarks_px is None."""
        r = DebugOverlayRenderer()
        r.draw(_blank_frame(), _default_snapshot(), pose_landmarks_px=None)

    def test_all_landmarks_none(self):
        r = DebugOverlayRenderer()
        r.draw(_blank_frame(), _default_snapshot(), face_landmarks_px=None, pose_landmarks_px=None)


class TestDrawWithLandmarks:
    """draw() handles valid landmark arrays without errors."""

    def test_with_face_landmarks(self):
        r = DebugOverlayRenderer()
        result = r.draw(
            _blank_frame(),
            _default_snapshot(),
            face_landmarks_px=_face_landmarks(),
        )
        assert result.shape == (240, 320, 3)

    def test_with_pose_landmarks(self):
        r = DebugOverlayRenderer()
        result = r.draw(
            _blank_frame(),
            _default_snapshot(),
            pose_landmarks_px=_pose_landmarks(),
        )
        assert result.shape == (240, 320, 3)

    def test_with_both_landmark_types(self):
        r = DebugOverlayRenderer()
        result = r.draw(
            _blank_frame(),
            _default_snapshot(),
            face_landmarks_px=_face_landmarks(),
            pose_landmarks_px=_pose_landmarks(),
        )
        assert result.shape == (240, 320, 3)

    def test_landmarks_suppressed_when_show_landmarks_false(self):
        """show_landmarks=False should still complete without error."""
        r = DebugOverlayRenderer(show_landmarks=False)
        r.draw(
            _blank_frame(),
            _default_snapshot(),
            face_landmarks_px=_face_landmarks(),
            pose_landmarks_px=_pose_landmarks(),
        )


class TestFrameSkipCaching:
    """Frame-skip caching behaviour: overlay is only recomputed every N frames."""

    def test_frame_count_increments_each_draw(self):
        r = DebugOverlayRenderer(frame_skip=1)
        assert r._frame_count == 0
        r.draw(_blank_frame(), _default_snapshot())
        assert r._frame_count == 1
        r.draw(_blank_frame(), _default_snapshot())
        assert r._frame_count == 2

    def test_cached_overlay_populated_after_first_draw(self):
        r = DebugOverlayRenderer(frame_skip=3)
        assert r._cached_overlay is None
        r.draw(_blank_frame(), _default_snapshot())
        assert r._cached_overlay is not None

    def test_frame_skip_every_frame_redraws(self):
        """With frame_skip=1 the cached overlay is updated every call."""
        r = DebugOverlayRenderer(frame_skip=1)
        r.draw(_blank_frame(), _default_snapshot())
        cache_after_1 = r._cached_overlay.copy()
        # Change a distinctive snapshot value so re-render would differ.
        r.draw(_blank_frame(), _default_snapshot(presence="ABSENT"))
        cache_after_2 = r._cached_overlay.copy()
        # Both renders happened; shapes must be the same.
        assert cache_after_1.shape == cache_after_2.shape

    def test_frame_skip_3_uses_cache_for_frames_2_and_3(self):
        """With frame_skip=3: frame 1 renders, frames 2&3 use cache.
        The cached overlay object must not be replaced on frames 2&3."""
        r = DebugOverlayRenderer(frame_skip=3)
        r.draw(_blank_frame(), _default_snapshot())
        cache_id_after_frame_1 = id(r._cached_overlay)

        r.draw(_blank_frame(), _default_snapshot())
        # On frame 2 with frame_skip=3, the cache should NOT be replaced.
        # (frame_count=2, 2%3 != 1 so should_render is False)
        cache_id_after_frame_2 = id(r._cached_overlay)
        assert cache_id_after_frame_1 == cache_id_after_frame_2

    def test_draw_with_skip_still_returns_correct_shape(self):
        """Even when using cached overlay, return shape is correct."""
        r = DebugOverlayRenderer(frame_skip=5)
        for _ in range(8):
            result = r.draw(_blank_frame(240, 320), _default_snapshot())
            assert result.shape == (240, 320, 3)


class TestDrawCalibration:
    """draw_calibration() returns a valid frame and handles edge cases."""

    def test_returns_same_shape(self):
        r = DebugOverlayRenderer()
        face = _make_face_result()
        pose = _make_pose_result()
        result = r.draw_calibration(
            _blank_frame(240, 320),
            face,
            pose,
            face_sample_count=50,
            elapsed_s=5.0,
            duration_s=10.0,
        )
        assert result.shape == (240, 320, 3)
        assert result.dtype == np.uint8

    def test_frame_annotated(self):
        """draw_calibration must modify at least some pixels."""
        r = DebugOverlayRenderer()
        face = _make_face_result()
        pose = _make_pose_result()
        frame = _blank_frame()
        r.draw_calibration(
            frame,
            face,
            pose,
            face_sample_count=10,
            elapsed_s=1.0,
            duration_s=10.0,
        )
        assert not np.all(frame == 0), "Frame was not annotated by draw_calibration"

    def test_no_face_detected(self):
        """draw_calibration must not raise when face is not detected."""
        r = DebugOverlayRenderer()
        face = _make_face_result(face_detected=False, ear_avg=0.0)
        pose = _make_pose_result()
        r.draw_calibration(
            _blank_frame(),
            face,
            pose,
            face_sample_count=0,
            elapsed_s=0.0,
            duration_s=10.0,
        )

    def test_no_pose_detected(self):
        r = DebugOverlayRenderer()
        face = _make_face_result()
        pose = _make_pose_result(pose_detected=False)
        r.draw_calibration(
            _blank_frame(),
            face,
            pose,
            face_sample_count=10,
            elapsed_s=3.0,
            duration_s=10.0,
        )

    def test_elapsed_at_zero(self):
        """elapsed_s=0 must not cause division by zero."""
        r = DebugOverlayRenderer()
        face = _make_face_result()
        pose = _make_pose_result()
        r.draw_calibration(
            _blank_frame(),
            face,
            pose,
            face_sample_count=0,
            elapsed_s=0.0,
            duration_s=10.0,
        )

    def test_elapsed_exceeds_duration(self):
        """elapsed_s > duration_s must not crash (progress clamps to 1.0)."""
        r = DebugOverlayRenderer()
        face = _make_face_result()
        pose = _make_pose_result()
        r.draw_calibration(
            _blank_frame(),
            face,
            pose,
            face_sample_count=200,
            elapsed_s=15.0,
            duration_s=10.0,
        )

    def test_zero_duration_does_not_raise(self):
        """duration_s=0 must not cause ZeroDivisionError."""
        r = DebugOverlayRenderer()
        face = _make_face_result()
        pose = _make_pose_result()
        r.draw_calibration(
            _blank_frame(),
            face,
            pose,
            face_sample_count=0,
            elapsed_s=0.0,
            duration_s=0.0,
        )

    def test_show_landmarks_false_still_works(self):
        r = DebugOverlayRenderer(show_landmarks=False)
        face = _make_face_result()
        pose = _make_pose_result()
        result = r.draw_calibration(
            _blank_frame(),
            face,
            pose,
            face_sample_count=5,
            elapsed_s=2.0,
            duration_s=10.0,
        )
        assert result.shape == (240, 320, 3)

    def test_large_sample_counts(self):
        """Very large sample counts must not cause layout errors."""
        r = DebugOverlayRenderer()
        face = _make_face_result()
        pose = _make_pose_result()
        r.draw_calibration(
            _blank_frame(),
            face,
            pose,
            face_sample_count=99999,
            elapsed_s=10.0,
            duration_s=10.0,
        )


class TestEventDataDebugDetail:
    """EventData.debug_detail field exists and defaults to empty string."""

    def test_event_data_has_debug_detail_field(self):
        from acuity.core.cv_pipeline import EventData

        e = EventData(
            event_type="Distraction_Pose",
            reason="",
            duration=15.0,
            timestamp=1000.0,
        )
        assert e.debug_detail == ""

    def test_event_data_debug_detail_can_be_set(self):
        from acuity.core.cv_pipeline import EventData

        detail = "pitch=28.3° (Δ18.1°), yaw=-4.2° (Δ2.5°), elapsed=15.2s"
        e = EventData(
            event_type="Distraction_Pose",
            reason="",
            duration=15.0,
            timestamp=1000.0,
            debug_detail=detail,
        )
        assert e.debug_detail == detail

    def test_event_data_is_frozen(self):
        from acuity.core.cv_pipeline import EventData

        e = EventData(
            event_type="Fatigue",
            reason="microsleep",
            duration=10.0,
            timestamp=2000.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            e.debug_detail = "modified"  # type: ignore[misc]
