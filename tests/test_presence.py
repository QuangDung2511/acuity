# Tests for presence fusion (face + pose detection -> PresenceState).
# ─────────────────────────────────────────────────────────────────
# Pure-logic unit tests on synthetic FaceResult / PoseResult; no models.
# ─────────────────────────────────────────────────────────────────
"""Tests for the presence fusion module."""

from __future__ import annotations

from acuity.core.face_mesh import FaceResult
from acuity.core.pose_tracker import PoseResult
from acuity.core.presence import PresenceState, determine_presence


def _face(detected: bool = True) -> FaceResult:
    return FaceResult(
        ear_left=0.3,
        ear_right=0.3,
        ear_avg=0.3,
        mar=0.0,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        face_detected=detected,
    )


def _pose(detected: bool = True) -> PoseResult:
    return PoseResult(landmarks=None, pose_detected=detected)


class TestDeterminePresence:
    """Tests for the determine_presence() fusion function."""

    def test_both_detected_returns_present(self):
        """Face + pose → PRESENT."""
        assert determine_presence(_face(True), _pose(True)) == PresenceState.PRESENT

    def test_face_only_returns_present(self):
        """Face + no pose → PRESENT (face is sufficient)."""
        assert determine_presence(_face(True), _pose(False)) == PresenceState.PRESENT

    def test_pose_only_returns_present_no_face(self):
        """No face + pose → PRESENT_NO_FACE (user is there, looking away)."""
        assert determine_presence(_face(False), _pose(True)) == PresenceState.PRESENT_NO_FACE

    def test_neither_detected_returns_absent(self):
        """No face + no pose → ABSENT (user left)."""
        assert determine_presence(_face(False), _pose(False)) == PresenceState.ABSENT
