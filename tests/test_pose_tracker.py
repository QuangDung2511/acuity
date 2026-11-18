# Tests for PoseAnalyzer: the body-presence signal that presence fusion needs.
# ─────────────────────────────────────────────────────────────────
# Seam: PoseAnalyzer.process() with MediaPipe's landmarker mocked at
# create_from_options, so no model inference runs. Landmarks go in
# normalised, the way MediaPipe reports them, and are checked in pixels.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from acuity.core.pose_tracker import PoseAnalyzer

_W, _H = 640, 480


def _make_mock_landmarks(nose_pt, l_shoulder_pt, r_shoulder_pt):
    """Create a mock MediaPipe pose_landmarks array with 33 points.

    Points are given in pixels and stored normalised, the way MediaPipe
    reports them, so the analyzer has to scale them back.
    """

    class DummyLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    lms = [DummyLandmark(0.0, 0.0) for _ in range(33)]
    lms[0] = DummyLandmark(nose_pt[0] / _W, nose_pt[1] / _H)
    lms[11] = DummyLandmark(l_shoulder_pt[0] / _W, l_shoulder_pt[1] / _H)
    lms[12] = DummyLandmark(r_shoulder_pt[0] / _W, r_shoulder_pt[1] / _H)
    return [lms]


def _process(mock_landmarker, pose_landmarks):
    mock_results = MagicMock()
    mock_results.pose_landmarks = pose_landmarks
    mock_landmarker.return_value.detect_for_video.return_value = mock_results
    analyzer = PoseAnalyzer()
    return analyzer.process(np.zeros((_H, _W, 3), dtype=np.uint8))


class TestPoseAnalyzerPresence:
    """The pose result is what tells PRESENT_NO_FACE apart from ABSENT
    when FaceMesh loses the face, and its landmarks feed the debug
    overlay's skeleton."""

    @patch("acuity.core.pose_tracker.mp.tasks.vision.PoseLandmarker.create_from_options")
    def test_detected_pose_reports_presence_with_pixel_landmarks(self, mock_landmarker):
        """A detected body must report presence, with landmarks scaled
        back to the frame's pixel coordinates."""
        result = _process(
            mock_landmarker,
            _make_mock_landmarks(
                nose_pt=(100, 100), l_shoulder_pt=(80, 200), r_shoulder_pt=(120, 200)
            ),
        )

        assert result.pose_detected is True
        assert result.landmarks.shape == (33, 2)
        np.testing.assert_allclose(result.landmarks[0], (100, 100))
        np.testing.assert_allclose(result.landmarks[11], (80, 200))
        np.testing.assert_allclose(result.landmarks[12], (120, 200))

    @patch("acuity.core.pose_tracker.mp.tasks.vision.PoseLandmarker.create_from_options")
    def test_no_pose_detected(self, mock_landmarker):
        """No body in frame must read as not present, with no landmarks,
        so presence fusion can fall through to ABSENT."""
        result = _process(mock_landmarker, [])

        assert result.pose_detected is False
        assert result.landmarks is None
