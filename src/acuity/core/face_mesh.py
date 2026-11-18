# MediaPipe Face Mesh wrapper - EAR calc, MAR calc, head-pose.
# ─────────────────────────────────────────────────────────────────
# Wraps MediaPipe's FaceLandmarker (Tasks API, 478 landmarks) to
# provide:
#   • Per-eye and average Eye Aspect Ratio (EAR)
#   • Mouth Aspect Ratio (MAR) for yawn detection
#   • Head pose estimation (pitch, yaw, roll) via cv2.solvePnP
#   • Face detection status
#
# Thread: CV worker (one analyzer per pipeline).
#
# Used by:
#   - cv_pipeline.py   (per-frame face analysis)
#   - calibration.py   (baseline EAR, MAR & head-pose collection)
#   - fatigue_engine.py (EAR → blink / micro-sleep, MAR → yawning)
#   - distraction_engine.py (head-pose → gaze deviation)
#
# Reference landmarks (MediaPipe 478-point model):
#   Left  eye:  [362, 385, 387, 263, 373, 380]
#   Right eye:  [33,  160, 158, 133, 153, 144]
#   Pose anchors (solvePnP): nose tip (1), chin (199),
#       left eye corner (33), right eye corner (263),
#       left mouth corner (61), right mouth corner (291)
#
# Model: assets/models/face_landmarker.task (float16)
#   Download: https://storage.googleapis.com/mediapipe-models/
#             face_landmarker/face_landmarker/float16/1/
#             face_landmarker.task
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from acuity.utils.angles import ema_angle_deg, wrap_deg
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# ── Resolve model path ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (src/acuity/core/..)
_MODEL_PATH = _PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"

# Exponential Moving Average alpha for head-pose smoothing.
# Lower = more smoothing.  0.4 balances responsiveness vs jitter.
_EMA_ALPHA = 0.4


# ── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class FaceResult:
    """Per-frame output from :class:`FaceMeshAnalyzer`.

    Attributes
    ----------
    ear_left : float
        Eye Aspect Ratio for the left eye (viewer's left = subject's right).
    ear_right : float
        Eye Aspect Ratio for the right eye.
    ear_avg : float
        Mean of ``ear_left`` and ``ear_right``.
    mar : float
        Mouth Aspect Ratio — vertical lip opening divided by horizontal
        mouth width.  Low (≈ 0.0 – 0.3) when the mouth is closed;
        high (≈ 0.6+) during a wide yawn.  Used by
        :class:`FatigueEngine` to detect yawning events.
    pitch : float
        Head rotation around the X-axis (degrees). Positive = looking up.
    yaw : float
        Head rotation around the Y-axis (degrees). Positive = looking right.
    roll : float
        Head rotation around the Z-axis (degrees). Positive = tilting right.
    face_detected : bool
        True if at least one face was found in the frame.
    landmarks_px : np.ndarray or None
        Pixel-space coordinates of all 478 face-mesh landmarks as an
        ``(478, 2)`` float64 array, or ``None`` when no face is detected.
        Used by the debug overlay to draw eye contours and pose
        anchors.  ``None`` for the ``_NO_FACE`` sentinel so callers that
        don't use debug mode incur zero allocation on missed frames.
    blink_score : float or None
        Mean of MediaPipe's ``eyeBlinkLeft`` / ``eyeBlinkRight``
        blendshapes for this frame, in [0.0, 1.0].  ``None`` when the
        model reported no blendshapes.

        This is a *learned appearance* signal, deliberately independent of
        EAR: EAR is a 2-D distance ratio, so a lowered lid looks identical
        whether the eye is shut or the gaze has simply dropped.  The
        fatigue engine uses it to qualify microsleep episodes — see
        MICROSLEEP_BLINK_CONFIRM.

        ``None`` means *no evidence*, never "the eye looks open"; every
        consumer must fail open on it, or a model without blendshapes
        would silently disable microsleep detection.
    """

    ear_left: float
    ear_right: float
    ear_avg: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    face_detected: bool
    landmarks_px: np.ndarray | None = None  # (478, 2) float64, debug only
    blink_score: float | None = None  # mean eyeBlink blendshape


# Sentinel returned when no face is detected in the frame.
_NO_FACE = FaceResult(
    ear_left=0.0,
    ear_right=0.0,
    ear_avg=0.0,
    mar=0.0,
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
    face_detected=False,
    landmarks_px=None,
)


# ── Landmark index constants ────────────────────────────────────

# MediaPipe 478-point Face Mesh landmark indices for EAR.
# Each tuple: (P1-outer, P2-top, P3-top, P4-inner, P5-bottom, P6-bottom)
# arranged for the standard EAR formula:
#
#   EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

# Left eye (subject's left eye = viewer's right)
_LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Right eye (subject's right eye = viewer's left)
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Inner-lip landmarks for Mouth Aspect Ratio (MAR).
# Vertical pairs (top landmark, bottom landmark) — three pairs for
# robustness against asymmetric jaw movement:
#   (13, 14)   upper/lower lip centre
#   (82, 87)   left inner lip
#   (312, 317) right inner lip
# Horizontal pair: mouth corners.
_MOUTH_VERTICAL = [(13, 14), (82, 87), (312, 317)]
_MOUTH_HORIZONTAL = (78, 308)

# Six canonical 3-D reference points for solvePnP head-pose estimation.
# Indices: nose tip, forehead center, left eye outer, right eye outer,
#          nasal bridge, forehead bridge.
# All points are on rigid bone/cartilage structures — no expression-
# sensitive landmarks (mouth corners, chin) to avoid smile/talk noise.
_POSE_LANDMARKS = [1, 10, 33, 263, 6, 168]

# Approximate 3-D model points of a generic face (in mm, arbitrary
# coordinate system centred roughly at the nose tip).  These match
# the order of ``_POSE_LANDMARKS`` above.
_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # Nose tip
        [0.0, 65.0, -10.0],  # Forehead center
        [-43.3, 32.7, -26.0],  # Left eye outer corner
        [43.3, 32.7, -26.0],  # Right eye outer corner
        [0.0, 15.0, -15.0],  # Nasal bridge (between eyes)
        [0.0, 45.0, -12.0],  # Forehead bridge (above nose bridge)
    ],
    dtype=np.float64,
)


# ── Helper functions ────────────────────────────────────────────


def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(p1 - p2))


def _compute_ear(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """Compute the Eye Aspect Ratio for a single eye.

    Uses the standard formula from Soukupová & Čech (2016):

        EAR = (||P2 − P6|| + ||P3 − P5||) / (2 · ||P1 − P4||)

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape ``(N, 2)`` with pixel-space (x, y) for each
        of the 478 face-mesh landmarks.
    eye_indices : list[int]
        Six landmark indices in the order
        ``[P1, P2, P3, P4, P5, P6]``.

    Returns
    -------
    float
        The EAR value (typically 0.15 – 0.45 when eyes are open).
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]

    vertical_a = _euclidean(p2, p6)
    vertical_b = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (vertical_a + vertical_b) / (2.0 * horizontal)


def _extract_blink_score(results) -> float | None:
    """Mean of the eyeBlinkLeft / eyeBlinkRight blendshapes, or None.

    Returns ``None`` — meaning *no evidence* — whenever the model did not
    report both categories, rather than a number that could be mistaken
    for an open eye.  The microsleep gate fails open on ``None``, so a
    model built without blendshapes keeps today's behaviour instead of
    silently suppressing every microsleep.

    Averaged across the two eyes for the same reason EAR is: a one-sided
    landmark collapse must not decide the episode on its own.  The
    bilateral share test (MICROSLEEP_BILATERAL_SHARE) remains the guard
    against genuinely one-eyed episodes.
    """
    shapes = getattr(results, "face_blendshapes", None)
    if not shapes:
        return None

    scores = {
        category.category_name: category.score
        for category in shapes[0]
        if category.category_name in ("eyeBlinkLeft", "eyeBlinkRight")
    }
    if len(scores) != 2:
        return None
    return (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2.0


def _compute_mar(landmarks: np.ndarray) -> float:
    """Compute the Mouth Aspect Ratio (MAR) for yawn detection.

    Uses the same geometric ratio approach as EAR, applied to inner-lip
    landmarks.  Three vertical pairs are averaged for robustness against
    asymmetric jaw movement and landmark jitter.

        MAR = Σ||vertical_pair|| / (3 · ||horizontal||)

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape ``(478, 2)`` with pixel-space (x, y) coordinates.

    Returns
    -------
    float
        MAR value — typically 0.0 – 0.3 when the mouth is closed,
        rising to 0.6+ during a wide yawn.  Returns 0.0 if the
        horizontal mouth distance is degenerate (< 1e-6 px).
    """
    vertical_sum = sum(_euclidean(landmarks[top], landmarks[bot]) for top, bot in _MOUTH_VERTICAL)
    horizontal = _euclidean(
        landmarks[_MOUTH_HORIZONTAL[0]],
        landmarks[_MOUTH_HORIZONTAL[1]],
    )
    if horizontal < 1e-6:
        return 0.0
    return vertical_sum / (3.0 * horizontal)


# ── FaceMeshAnalyzer ────────────────────────────────────────────


class FaceMeshAnalyzer:
    """MediaPipe Face Landmarker wrapper providing EAR and head-pose per frame.

    Uses the MediaPipe Tasks API (``FaceLandmarker``) in VIDEO running
    mode with the 478-landmark model for maximum EAR accuracy.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the ``face_landmarker.task`` model file.  Defaults to
        ``assets/models/face_landmarker.task`` in the project root.
    num_faces : int
        Maximum number of faces to detect (default 1 — single user).
    min_detection_confidence : float
        Minimum confidence for the face-detection model.
    min_tracking_confidence : float
        Minimum confidence for the landmark-tracking model.

    Example
    -------
    ::

        analyzer = FaceMeshAnalyzer()
        result = analyzer.process(frame)
        if result.face_detected:
            print(f"EAR={result.ear_avg:.3f}  yaw={result.yaw:.1f}°")
        analyzer.close()
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_path = str(model_path or _MODEL_PATH)

        if not Path(resolved_path).exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at '{resolved_path}'.  "
                "Download it from: https://storage.googleapis.com/"
                "mediapipe-models/face_landmarker/face_landmarker/"
                "float16/1/face_landmarker.task  and place it in "
                "assets/models/."
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=resolved_path,
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            # Blendshapes carry eyeBlinkLeft/Right, which is what the
            # microsleep gate qualifies episodes on (MICROSLEEP_BLINK_CONFIRM).
            # Measured cost over 380 frames of test_edge_cases.mp4: median
            # detect_for_video 9.08 ms -> 9.27 ms, p90 unchanged at 10.6 ms.
            # 0.19 ms against a ~33 ms frame budget.
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        # Monotonically increasing timestamp for VIDEO mode (ms).
        self._timestamp_ms: int = 0

        # Camera matrix is lazily initialised on the first frame
        # because we need the frame dimensions.
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._frame_size: tuple[int, int] = (0, 0)  # (w, h)

        # EMA state for head-pose smoothing (reduces landmark jitter).
        self._ema_pitch: float | None = None
        self._ema_yaw: float | None = None
        self._ema_roll: float | None = None

        log.info(
            "FaceMeshAnalyzer initialised (model=%s, faces=%d, det=%.2f, trk=%.2f).",
            Path(resolved_path).name,
            num_faces,
            min_detection_confidence,
            min_tracking_confidence,
        )

    # ── Public API ──────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> FaceResult:
        """Analyse a single BGR frame and return face metrics.

        Parameters
        ----------
        frame : np.ndarray
            An OpenCV BGR image (``H × W × 3``, ``uint8``).

        Returns
        -------
        FaceResult
            Contains EAR values, head-pose angles, and detection flag.
            If no face is detected, all numeric fields are ``0.0`` and
            ``face_detected`` is ``False``.
        """
        if self._landmarker is None:
            return _NO_FACE

        h, w = frame.shape[:2]

        # Lazily build / rebuild the camera matrix when resolution changes.
        if (w, h) != self._frame_size:
            self._build_camera_matrix(w, h)

        # Convert BGR → RGB and wrap in MediaPipe Image.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # Advance the monotonic timestamp (VIDEO mode requires this).
        self._timestamp_ms += 33  # ~30 FPS step

        results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not results.face_landmarks:
            return _NO_FACE

        # Use the first detected face.
        face_lms = results.face_landmarks[0]

        # Convert normalised landmarks → pixel coordinates (x, y).
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h) for lm in face_lms],
            dtype=np.float64,
        )

        # ── EAR ─────────────────────────────────────────────────
        ear_left = _compute_ear(landmarks_px, _LEFT_EYE)
        ear_right = _compute_ear(landmarks_px, _RIGHT_EYE)
        ear_avg = (ear_left + ear_right) / 2.0

        # ── MAR ─────────────────────────────────────────────────
        mar = _compute_mar(landmarks_px)

        # ── Blink blendshape ────────────────────────────────────
        # Stays None if the model reported nothing, which the fatigue
        # engine treats as "no evidence" rather than as an open eye.
        blink_score = _extract_blink_score(results)

        # ── Head Pose (solvePnP) ────────────────────────────────
        pitch, yaw, roll = self._estimate_head_pose(landmarks_px)

        # ── EMA smoothing ───────────────────────────────────────
        # Dampens frame-to-frame jitter from landmark noise.
        # Angles are blended along the shortest arc — a linear blend
        # would sweep the long way round whenever the raw signal
        # crosses ±180°, injecting excursions far larger than any
        # real head movement.
        if self._ema_pitch is None:
            self._ema_pitch = pitch
            self._ema_yaw = yaw
            self._ema_roll = roll
        else:
            self._ema_pitch = ema_angle_deg(self._ema_pitch, pitch, _EMA_ALPHA)
            self._ema_yaw = ema_angle_deg(self._ema_yaw, yaw, _EMA_ALPHA)
            self._ema_roll = ema_angle_deg(self._ema_roll, roll, _EMA_ALPHA)

        pitch = self._ema_pitch
        yaw = self._ema_yaw
        roll = self._ema_roll

        return FaceResult(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            face_detected=True,
            landmarks_px=landmarks_px,  # (478, 2) — used by debug overlay
            blink_score=blink_score,
        )

    def close(self) -> None:
        """Release MediaPipe resources.

        Safe to call multiple times.
        """
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            log.info("FaceMeshAnalyzer closed.")

    # ── Private helpers ─────────────────────────────────────────

    def _build_camera_matrix(self, w: int, h: int) -> None:
        """Construct an approximate camera intrinsic matrix.

        Uses a pinhole model with focal length ≈ frame width and
        the principal point at the frame centre.  This is good enough
        for relative head-pose tracking — we only need angular
        *changes* from the calibrated baseline, not absolute world
        coordinates.
        """
        focal_length = float(w)
        cx, cy = w / 2.0, h / 2.0

        self._camera_matrix = np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self._frame_size = (w, h)
        log.debug("Camera matrix built for %dx%d.", w, h)

    def _estimate_head_pose(self, landmarks_px: np.ndarray) -> tuple[float, float, float]:
        """Estimate head pitch, yaw, and roll via ``cv2.solvePnP``.

        Parameters
        ----------
        landmarks_px : np.ndarray
            Pixel-space landmarks of shape ``(478, 2)``.

        Returns
        -------
        (pitch, yaw, roll)
            Euler angles in degrees.
        """
        # Extract the 2-D image points for the six pose-anchor landmarks.
        image_points = np.array(
            [landmarks_px[i] for i in _POSE_LANDMARKS],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            log.debug("solvePnP failed — returning zero angles.")
            return 0.0, 0.0, 0.0

        # Convert the rotation vector to a rotation matrix, then
        # decompose into Euler angles.
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

        return pitch, yaw, roll

    @staticmethod
    def _rotation_matrix_to_euler(
        rmat: np.ndarray,
    ) -> tuple[float, float, float]:
        """Convert a 3 × 3 rotation matrix to Euler angles (degrees).

        Uses ``cv2.decomposeProjectionMatrix`` for consistency with
        OpenCV conventions.

        Returns
        -------
        (pitch, yaw, roll)
            Euler angles in degrees.
        """
        # Build a 3×4 projection matrix for decomposeProjectionMatrix.
        proj = np.hstack((rmat, np.zeros((3, 1), dtype=np.float64)))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)

        pitch = float(euler[0, 0])
        yaw = float(euler[1, 0])
        roll = float(euler[2, 0])

        # ── Re-centre pitch on 0° ────────────────────────────────
        # The camera frame has +Y down while the face model has +Y
        # up, so the recovered rotation carries a 180° flip about X:
        # a user looking straight at the camera decomposes to
        # pitch ≈ ±180°, sitting right on the wrap boundary.  Every
        # small upward head movement then flips the sign and any
        # downstream subtraction reads it as a ~360° swing.
        #
        # Subtracting the flip is a pure change of representation —
        # it is monotonic near neutral, so the sign convention
        # (positive = looking up) and all calibrated thresholds keep
        # their existing meaning.
        pitch = wrap_deg(pitch - 180.0)

        # Roll picks up the same flip whenever the decomposition
        # crosses gimbal configurations; wrap it for consistency.
        roll = wrap_deg(roll)
        yaw = wrap_deg(yaw)

        return pitch, yaw, roll

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        state = "active" if self._landmarker is not None else "closed"
        return f"<FaceMeshAnalyzer {state}>"

    def __del__(self) -> None:
        """Safety net — release on garbage collection."""
        self.close()
