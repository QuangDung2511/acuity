# Fuses face + pose detection into a unified presence signal.
# ─────────────────────────────────────────────────────────────────
# Resolves the ambiguity of ``face_detected=False`` by cross-
# referencing the PoseTracker's detection result.  MediaPipe
# FaceMesh loses tracking during large head movements (~41% of
# frames in testing), but PoseTracker (shoulders/torso) maintained
# 100% detection rate across all test scenarios.
#
# Tri-state presence signal:
#   PRESENT         — Face visible → normal tracking
#   PRESENT_NO_FACE — Face lost, body visible → user is looking away
#   ABSENT          — Neither visible → user may have left
#
# Thread: pure; called from the CV worker thread.
#
# Used by:
#   - cv_pipeline.py         (per-frame presence determination)
#   - distraction_engine.py  (decides whether to count as deviation)
#   - fatigue_engine.py      (pauses blink tracker during no-face)
#
# Design rationale:
#   This module is intentionally a standalone pure-logic module with
#   zero ML dependencies.  Both cv_pipeline.py and distraction_engine
#   need the fusion logic; placing it here avoids circular imports
#   and keeps the rule in exactly one place.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from enum import Enum, auto

from acuity.core.face_mesh import FaceResult
from acuity.core.pose_tracker import PoseResult


class PresenceState(Enum):
    """Tri-state presence signal derived from face + pose detection.

    This enum represents the fused presence state of the user,
    combining face detection (MediaPipe FaceMesh) and body detection
    (MediaPipe Pose) into a single actionable signal.

    Members
    -------
    PRESENT
        Face is visible — all engines operate normally.
    PRESENT_NO_FACE
        Body (shoulders/torso) is visible but face is not — the
        user is physically present but looking away from the screen.
        The distraction timer should run; the Away timer should NOT
        start.
    ABSENT
        Neither face nor body is visible — the user may have left.
        The Away countdown should begin.
    """

    PRESENT = auto()
    PRESENT_NO_FACE = auto()
    ABSENT = auto()


def determine_presence(
    face: FaceResult,
    pose: PoseResult,
) -> PresenceState:
    """Determine user presence from face and pose detection results.

    Fuses the binary ``face_detected`` and ``pose_detected`` flags
    into a single tri-state signal.  The logic is intentionally
    simple and deterministic — no timers, no hysteresis, no state:

    - Face visible (regardless of pose) → ``PRESENT``
    - Face not visible + body visible → ``PRESENT_NO_FACE``
    - Neither visible → ``ABSENT``

    Parameters
    ----------
    face : FaceResult
        Output from :class:`~acuity.core.face_mesh.FaceMeshAnalyzer`.
    pose : PoseResult
        Output from :class:`~acuity.core.pose_tracker.PoseAnalyzer`.

    Returns
    -------
    PresenceState
        The fused presence state for this frame.

    Examples
    --------
    >>> from acuity.core.face_mesh import FaceResult
    >>> from acuity.core.pose_tracker import PoseResult
    >>> face = FaceResult(0.3, 0.3, 0.3, 0.0, 0.0, 0.0, True)
    >>> pose = PoseResult(None, True)
    >>> determine_presence(face, pose)
    <PresenceState.PRESENT: 1>
    """
    if face.face_detected:
        return PresenceState.PRESENT
    if pose.pose_detected:
        return PresenceState.PRESENT_NO_FACE
    return PresenceState.ABSENT
