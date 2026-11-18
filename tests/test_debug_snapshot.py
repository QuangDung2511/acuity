# Unit tests for DebugSnapshot construction.
# ─────────────────────────────────────────────────────────────────
# Tests that _build_debug_snapshot() correctly reads from each engine
# and assembles a DebugSnapshot with the expected field values.
#
# Strategy: patch CVPipelineThread's engine references directly (no
# camera, no Qt event loop) using unittest.mock.MagicMock.  The method
# under test performs ~20 attribute reads — we assert every group.
#
# No external models or hardware required.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from acuity.core.debug_snapshot import DebugSnapshot
from acuity.core.presence import PresenceState
from acuity.utils.constants import MICROSLEEP_SHUT_RATIO

# ── Helpers ──────────────────────────────────────────────────────


def _make_face(
    face_detected: bool = True,
    ear_left: float = 0.35,
    ear_right: float = 0.33,
    ear_avg: float = 0.34,
    pitch: float = 5.0,
    yaw: float = -2.0,
    roll: float = 1.0,
    landmarks_px=None,
):
    """Minimal FaceResult stand-in (namespace object)."""
    m = MagicMock()
    m.face_detected = face_detected
    m.ear_left = ear_left
    m.ear_right = ear_right
    m.ear_avg = ear_avg
    m.pitch = pitch
    m.yaw = yaw
    m.roll = roll
    m.landmarks_px = landmarks_px
    m.blink_score = None
    return m


def _make_pose(
    pose_detected: bool = True,
    landmarks=None,
):
    """Minimal PoseResult stand-in."""
    m = MagicMock()
    m.pose_detected = pose_detected
    m.landmarks = landmarks
    return m


def _make_cal(
    baseline_pitch: float = 2.0,
    baseline_yaw: float = -1.0,
    baseline_ear: float = 0.34,
    baseline_ear_std: float = 0.02,
):
    """Minimal CalibrationResult stand-in."""
    m = MagicMock()
    m.baseline_pitch = baseline_pitch
    m.baseline_yaw = baseline_yaw
    m.baseline_ear = baseline_ear
    m.baseline_ear_std = baseline_ear_std
    return m


def _make_distraction_engine(
    head_pose_pitch_tolerance_deg: float = 20.0,
    head_pose_yaw_tolerance_deg: float = 15.0,
    pose_elapsed_s: float = 3.5,
    head_pose_threshold_s: float = 15.0,
    pose_fired: bool = False,
    desk_writing_mode: bool = False,
):
    m = MagicMock()
    m.debug_head_pose_pitch_tolerance_deg = head_pose_pitch_tolerance_deg
    m.debug_head_pose_yaw_tolerance_deg = head_pose_yaw_tolerance_deg
    m.pose_elapsed_s = pose_elapsed_s
    m.head_pose_threshold_s = head_pose_threshold_s
    m.debug_pose_fired = pose_fired
    m.desk_writing_mode = desk_writing_mode
    return m


def _make_fatigue_engine(
    blink_threshold: float = 0.25,
    smoothed_ear: float = 0.32,
    microsleep_active: bool = False,
    microsleep_elapsed_s: float = 0.0,
    perclos: float = 0.05,
    blink_rate: float = 14.0,
    baseline_blink_rate=None,
    cooldown_remaining: float = 0.0,
):
    m = MagicMock()
    m.blink_threshold = blink_threshold
    # Mirrors FatigueEngine.shut_threshold — the microsleep timer's stricter
    # boundary.  A MagicMock here would compare as an object, not a float.
    m.shut_threshold = blink_threshold * MICROSLEEP_SHUT_RATIO
    m.debug_smoothed_ear = smoothed_ear
    m.microsleep_active = microsleep_active
    m.debug_microsleep_elapsed_s = microsleep_elapsed_s
    m.perclos = perclos
    m.blink_rate = blink_rate
    m.baseline_blink_rate = baseline_blink_rate
    m.cooldown_remaining = cooldown_remaining
    return m


def _make_pipeline_with_engines(
    cal=None,
    distraction=None,
    fatigue=None,
):
    """Create a CVPipelineThread instance with engines injected directly.

    Patches ConfigManager so no Qt or QSettings is needed.
    """

    cfg_mock = MagicMock()
    cfg_mock.get_bool.return_value = False
    cfg_mock.get_int.return_value = 1

    with patch("acuity.core.cv_pipeline.ConfigManager", return_value=cfg_mock):
        from acuity.core.cv_pipeline import CVPipelineThread

        pipeline = CVPipelineThread.__new__(CVPipelineThread)

    # Inject state without calling __init__ (avoids Qt/camera setup).
    pipeline._cal_result = cal
    pipeline._distraction_engine = distraction
    pipeline._fatigue_engine = fatigue
    return pipeline


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS
# ═══════════════════════════════════════════════════════════════


class TestDebugSnapshotDefaults:
    """DebugSnapshot can be instantiated with all-default values."""

    def test_default_instantiation(self):
        s = DebugSnapshot()
        assert s.presence == "PRESENT"
        assert s.face_detected is False
        assert s.ear_avg == 0.0
        assert s.pose_detected is False

    def test_frozen_cannot_mutate(self):
        s = DebugSnapshot()
        with pytest.raises((AttributeError, TypeError)):
            s.presence = "ABSENT"  # type: ignore[misc]

    def test_custom_fields_round_trip(self):
        s = DebugSnapshot(
            presence="ABSENT",
            face_detected=True,
            ear_avg=0.31,
            pitch=12.5,
            pose_detected=True,
        )
        assert s.presence == "ABSENT"
        assert s.face_detected is True
        assert s.ear_avg == pytest.approx(0.31)
        assert s.pose_detected is True


class TestSnapshotGuardEnginesNotReady:
    """_build_debug_snapshot must return a minimal snapshot when engines
    are not yet initialised (i.e., during the calibration phase or early
    start-up)."""

    def test_returns_snapshot_when_cal_is_none(self):
        """If _cal_result is None, return an empty snapshot."""
        pipeline = _make_pipeline_with_engines(
            cal=None,
            distraction=_make_distraction_engine(),
            fatigue=_make_fatigue_engine(),
        )
        snapshot = pipeline._build_debug_snapshot(
            face=_make_face(),
            pose=_make_pose(),
            presence=PresenceState.PRESENT,
        )
        assert isinstance(snapshot, DebugSnapshot)
        assert snapshot.presence == "PRESENT"
        # Fields that require engine data should be at defaults.
        assert snapshot.blink_threshold == 0.0

    def test_returns_snapshot_when_all_engines_none(self):
        """All three guards fail → minimal snapshot."""
        pipeline = _make_pipeline_with_engines(
            cal=None,
            distraction=None,
            fatigue=None,
        )
        snapshot = pipeline._build_debug_snapshot(
            face=_make_face(),
            pose=_make_pose(),
            presence=PresenceState.ABSENT,
        )
        assert snapshot.presence == "ABSENT"

    def test_returns_snapshot_when_distraction_engine_none(self):
        """Only distraction engine missing → guard fires."""
        pipeline = _make_pipeline_with_engines(
            cal=_make_cal(),
            distraction=None,
            fatigue=_make_fatigue_engine(),
        )
        snapshot = pipeline._build_debug_snapshot(
            face=_make_face(),
            pose=_make_pose(),
            presence=PresenceState.PRESENT,
        )
        assert isinstance(snapshot, DebugSnapshot)


class TestSnapshotPresenceField:
    """Presence field reflects the current PresenceState enum name."""

    @pytest.mark.parametrize(
        "state",
        [
            PresenceState.PRESENT,
            PresenceState.PRESENT_NO_FACE,
            PresenceState.ABSENT,
        ],
    )
    def test_presence_name(self, state):
        pipeline = _make_pipeline_with_engines(
            cal=_make_cal(),
            distraction=_make_distraction_engine(),
            fatigue=_make_fatigue_engine(),
        )
        s = pipeline._build_debug_snapshot(
            _make_face(),
            _make_pose(),
            state,
        )
        assert s.presence == state.name


class TestSnapshotFaceMeshFields:
    """Face mesh values are read from FaceResult verbatim."""

    def test_ear_and_pose_values_copied(self):
        face = _make_face(
            ear_left=0.40, ear_right=0.38, ear_avg=0.39, pitch=10.0, yaw=-5.0, roll=2.5
        )
        pipeline = _make_pipeline_with_engines(
            _make_cal(),
            _make_distraction_engine(),
            _make_fatigue_engine(),
        )
        s = pipeline._build_debug_snapshot(face, _make_pose(), PresenceState.PRESENT)
        assert s.face_detected is True
        assert s.ear_left == pytest.approx(0.40)
        assert s.ear_right == pytest.approx(0.38)
        assert s.ear_avg == pytest.approx(0.39)
        assert s.pitch == pytest.approx(10.0)
        assert s.yaw == pytest.approx(-5.0)
        assert s.roll == pytest.approx(2.5)

    def test_no_face_detected_fields(self):
        face = _make_face(face_detected=False, ear_avg=0.0, pitch=0.0, yaw=0.0)
        pipeline = _make_pipeline_with_engines(
            _make_cal(),
            _make_distraction_engine(),
            _make_fatigue_engine(),
        )
        s = pipeline._build_debug_snapshot(face, _make_pose(), PresenceState.PRESENT_NO_FACE)
        assert s.face_detected is False
        # head_pose_deviated must be False when face not detected.
        assert s.head_pose_deviated is False


class TestSnapshotCalibrationBaselines:
    """Calibration baselines are copied from CalibrationResult."""

    def test_baseline_fields_match_cal(self):
        cal = _make_cal(
            baseline_pitch=3.5,
            baseline_yaw=-1.2,
            baseline_ear=0.31,
            baseline_ear_std=0.025,
        )
        pipeline = _make_pipeline_with_engines(
            cal,
            _make_distraction_engine(),
            _make_fatigue_engine(),
        )
        s = pipeline._build_debug_snapshot(
            _make_face(),
            _make_pose(),
            PresenceState.PRESENT,
        )
        assert s.baseline_pitch == pytest.approx(3.5)
        assert s.baseline_yaw == pytest.approx(-1.2)
        assert s.baseline_ear == pytest.approx(0.31)
        assert s.baseline_ear_std == pytest.approx(0.025)


class TestSnapshotHeadPoseDeviation:
    """head_pose_deviated flag logic is computed correctly."""

    def test_deviated_when_pitch_exceeds_tolerance(self):
        """pitch delta = |10 - 2| = 8 > tolerance 5 → deviated."""
        cal = _make_cal(baseline_pitch=2.0, baseline_yaw=0.0)
        de = _make_distraction_engine(head_pose_pitch_tolerance_deg=5.0)
        face = _make_face(pitch=10.0, yaw=0.0)
        pipeline = _make_pipeline_with_engines(cal, de, _make_fatigue_engine())
        s = pipeline._build_debug_snapshot(
            face,
            _make_pose(),
            PresenceState.PRESENT,
        )
        assert s.head_pose_deviated is True
        assert s.pitch_delta == pytest.approx(8.0)

    def test_not_deviated_when_within_tolerance(self):
        """pitch delta = |4 - 2| = 2 < tolerance 5 → not deviated."""
        cal = _make_cal(baseline_pitch=2.0, baseline_yaw=0.0)
        de = _make_distraction_engine(
            head_pose_pitch_tolerance_deg=5.0, head_pose_yaw_tolerance_deg=5.0
        )
        face = _make_face(pitch=4.0, yaw=0.0)
        pipeline = _make_pipeline_with_engines(cal, de, _make_fatigue_engine())
        s = pipeline._build_debug_snapshot(
            face,
            _make_pose(),
            PresenceState.PRESENT,
        )
        assert s.head_pose_deviated is False

    def test_deviated_on_yaw_alone(self):
        """yaw delta alone can trigger deviation."""
        cal = _make_cal(baseline_pitch=0.0, baseline_yaw=0.0)
        de = _make_distraction_engine(head_pose_yaw_tolerance_deg=10.0)
        face = _make_face(pitch=0.0, yaw=15.0)
        pipeline = _make_pipeline_with_engines(cal, de, _make_fatigue_engine())
        s = pipeline._build_debug_snapshot(
            face,
            _make_pose(),
            PresenceState.PRESENT,
        )
        assert s.head_pose_deviated is True
        assert s.yaw_delta == pytest.approx(15.0)


class TestSnapshotFatigueFields:
    """Fatigue engine metrics appear in the snapshot."""

    def test_fatigue_fields_copied(self):
        fe = _make_fatigue_engine(
            blink_threshold=0.25,
            smoothed_ear=0.31,
            microsleep_active=True,
            microsleep_elapsed_s=0.6,
            perclos=0.12,
            blink_rate=9.5,
            baseline_blink_rate=16.0,
            cooldown_remaining=3.2,
        )
        pipeline = _make_pipeline_with_engines(
            _make_cal(),
            _make_distraction_engine(),
            fe,
        )
        s = pipeline._build_debug_snapshot(
            _make_face(),
            _make_pose(),
            PresenceState.PRESENT,
        )
        assert s.blink_threshold == pytest.approx(0.25)
        assert s.smoothed_ear == pytest.approx(0.31)
        assert s.microsleep_active is True
        assert s.microsleep_elapsed_s == pytest.approx(0.6)
        assert s.perclos == pytest.approx(0.12)
        assert s.blink_rate == pytest.approx(9.5)
        assert s.baseline_blink_rate == pytest.approx(16.0)
        assert s.fatigue_cooldown_remaining == pytest.approx(3.2)

    def test_eyes_closed_when_ear_below_threshold(self):
        """eyes_closed is derived: smoothed_ear < blink_threshold."""
        fe = _make_fatigue_engine(blink_threshold=0.25, smoothed_ear=0.18)
        pipeline = _make_pipeline_with_engines(
            _make_cal(),
            _make_distraction_engine(),
            fe,
        )
        s = pipeline._build_debug_snapshot(
            _make_face(),
            _make_pose(),
            PresenceState.PRESENT,
        )
        assert s.eyes_closed is True

    def test_eyes_open_when_ear_above_threshold(self):
        fe = _make_fatigue_engine(blink_threshold=0.25, smoothed_ear=0.34)
        pipeline = _make_pipeline_with_engines(
            _make_cal(),
            _make_distraction_engine(),
            fe,
        )
        s = pipeline._build_debug_snapshot(
            _make_face(),
            _make_pose(),
            PresenceState.PRESENT,
        )
        assert s.eyes_closed is False


class TestDataLoggerRowWidth:
    """The CSV row must stay the same width as the header.

    data_logger asserts the header length at import, which catches a
    forgotten _HEADERS entry but NOT the opposite and more likely slip:
    adding a header without adding the matching value to _build_row.  That
    produces a silently misaligned CSV — every column after the gap reads
    as its neighbour, which is exactly the kind of corruption the tuning
    corpus cannot detect by eye.
    """

    def test_row_width_matches_header_width(self):
        from acuity.core.data_logger import _HEADERS, DataLogger

        # _frame_number is the only instance state _build_row reads, so a
        # stub avoids opening a real file just to count columns.
        stub = SimpleNamespace(_frame_number=0)
        row = DataLogger._build_row(
            stub,
            DebugSnapshot(),
            False,
            "",
            "",
        )
        assert len(row) == len(_HEADERS), (
            f"_build_row emitted {len(row)} values for {len(_HEADERS)} "
            "headers — the CSV would be misaligned."
        )

    def test_microsleep_gate_columns_are_logged(self):
        """The gate's evidence has to reach the CSV to be scoreable."""
        from acuity.core.data_logger import _HEADERS

        for column in (
            "episode_bilateral_share",
            "episode_blink_confirm",
            "microsleep_withheld",
            "blink_score",
        ):
            assert column in _HEADERS
