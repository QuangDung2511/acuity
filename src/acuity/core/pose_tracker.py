# MediaPipe Pose wrapper - body presence detection.
# ─────────────────────────────────────────────────────────────────
# Wraps MediaPipe's PoseLandmarker (Tasks API, 33 landmarks) to
# provide:
#   • Whether a body is in frame — the signal presence fusion uses to
#     tell "looking away" (body, no face) from "left the desk"
#   • Raw landmark coordinates for downstream use (debug skeleton)
#
# Thread: CV worker (one analyzer per pipeline).
#
# Design rationale:
#   FaceMesh loses the face on large head movements while Pose keeps
#   the shoulders, so body detection alone is what this module is kept
#   for.  The posture metrics and the lateral "torso pitch" angle the
#   original build derived here were cut with posture detection: a
#   frontal camera cannot see real posture problems.
#
# Used by:
#   - cv_pipeline.py     (per-frame presence)
#   - presence.py        (face + pose fusion)
#
# Model: assets/models/pose_landmarker.task (lite)
#   Download: https://storage.googleapis.com/mediapipe-models/
#             pose_landmarker/pose_landmarker_lite/float16/1/
#             pose_landmarker_lite.task
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from acuity.utils.logger import get_logger

log = get_logger(__name__)

# ── Resolve model path ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (src/acuity/core/..)
_MODEL_PATH = _PROJECT_ROOT / "assets" / "models" / "pose_landmarker.task"


# ── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class PoseResult:
    """Per-frame output from :class:`PoseAnalyzer`.

    Attributes
    ----------
    landmarks : Optional[np.ndarray]
        Pixel-space coordinates of all 33 pose landmarks as an
        ``(33, 2)`` array, or ``None`` if no pose was detected.
    pose_detected : bool
        True if at least one pose was found in the frame.
    """

    landmarks: np.ndarray | None
    pose_detected: bool


# Sentinel returned when no pose is detected in the frame.
_NO_POSE = PoseResult(landmarks=None, pose_detected=False)


# ── PoseAnalyzer ────────────────────────────────────────────────


class PoseAnalyzer:
    """MediaPipe Pose Landmarker wrapper reporting body presence per frame.

    Uses the MediaPipe Tasks API (``PoseLandmarker``) in VIDEO running
    mode with the lite model for efficient real-time tracking.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the ``pose_landmarker.task`` model file.  Defaults to
        ``assets/models/pose_landmarker.task`` in the project root.
    min_detection_confidence : float
        Minimum confidence for the pose-detection model.
    min_tracking_confidence : float
        Minimum confidence for the landmark-tracking model.

    Example
    -------
    ::

        analyzer = PoseAnalyzer()
        result = analyzer.process(frame)
        if result.pose_detected:
            print(f"nose at {result.landmarks[0]}")
        analyzer.close()
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_path = str(model_path or _MODEL_PATH)

        if not Path(resolved_path).exists():
            raise FileNotFoundError(
                f"PoseLandmarker model not found at '{resolved_path}'.  "
                "Download it from: https://storage.googleapis.com/"
                "mediapipe-models/pose_landmarker/pose_landmarker_lite/"
                "float16/1/pose_landmarker_lite.task  and place it in "
                "assets/models/ (rename to pose_landmarker.task)."
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=resolved_path,
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

        # Monotonically increasing timestamp for VIDEO mode (ms).
        self._timestamp_ms: int = 0

        log.info(
            "PoseAnalyzer initialised (model=%s, det=%.2f, trk=%.2f).",
            Path(resolved_path).name,
            min_detection_confidence,
            min_tracking_confidence,
        )

    # ── Public API ──────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> PoseResult:
        """Analyse a single BGR frame and report body presence.

        Parameters
        ----------
        frame : np.ndarray
            An OpenCV BGR image (``H × W × 3``, ``uint8``).

        Returns
        -------
        PoseResult
            The detection flag and the pixel-space landmark array.
            If no pose is detected, ``landmarks`` is ``None``.
        """
        if self._landmarker is None:
            return _NO_POSE

        h, w = frame.shape[:2]

        # Convert BGR → RGB and wrap in MediaPipe Image.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # Advance the monotonic timestamp (VIDEO mode requires this).
        self._timestamp_ms += 33  # ~30 FPS step

        results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not results.pose_landmarks:
            return _NO_POSE

        # Use the first detected pose.
        pose_lms = results.pose_landmarks[0]

        # Convert normalised landmarks → pixel coordinates (x, y).
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h) for lm in pose_lms],
            dtype=np.float64,
        )

        return PoseResult(landmarks=landmarks_px, pose_detected=True)

    def close(self) -> None:
        """Release MediaPipe resources.

        Safe to call multiple times.
        """
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            log.info("PoseAnalyzer closed.")

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        state = "active" if self._landmarker is not None else "closed"
        return f"<PoseAnalyzer {state}>"

    def __del__(self) -> None:
        """Safety net — release on garbage collection."""
        self.close()
