# 10-second calibration phase - captures baseline EAR, MAR, head pose.
# ─────────────────────────────────────────────────────────────────
# Collects FaceResult samples over the calibration window (default
# 10 s) and computes robust median baselines for downstream
# engines:
#   - fatigue_engine.py   → baseline_ear, baseline_mar
#   - distraction_engine.py → baseline_pitch, baseline_yaw
#
# Thread: pure; fed from the CV worker thread.  The posture baselines
# (shoulder, neck ratio, nose-forward, shoulder roll) and the lateral
# torso angle were cut with posture detection, so only face samples
# are collected.
#
# Usage (inside the CV pipeline loop):
#   runner = CalibrationRunner()
#   while not runner.is_complete:
#       face = face_analyzer.process(frame)
#       runner.feed(face)
#   baselines = runner.result  # CalibrationResult
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from acuity.core.face_mesh import FaceResult
from acuity.utils.angles import circular_median_deg
from acuity.utils.constants import CALIBRATION_DURATION_S
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# Minimum number of valid samples required before we accept a
# calibration.  Guards against degenerate cases where the user is
# absent for most of the window.
_MIN_SAMPLES = 5


# ── Data structures ─────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationResult:
    """Median baselines computed during the calibration window.

    Attributes
    ----------
    baseline_ear : float
        Median average EAR — resting eye-openness.
    baseline_ear_std : float
        Standard deviation of EAR samples — spread of eye-openness.
        Used by FatigueEngine to derive a per-user adaptive blink
        threshold via ``baseline_ear - 2 * baseline_ear_std``.
        Set to ``0.0`` when fewer than ``_MIN_SAMPLES`` face frames
        were collected (fallback to ratio-based threshold).
    baseline_mar : float
        Median Mouth Aspect Ratio — resting (closed-mouth) MAR.
        Used by FatigueEngine to derive a per-user yawn threshold
        via ``baseline_mar + YAWN_MAR_DELTA``, floored at
        ``YAWN_MAR_MIN``.  Set to ``0.0`` when fewer than
        ``_MIN_SAMPLES`` face frames were collected — the floor keeps
        yawn detection working in that case.
    baseline_mar_std : float
        Standard deviation of MAR samples during calibration.
        Useful for diagnosing unusual mouth-movement during baseline
        collection (e.g. the user was talking).
    baseline_pitch : float
        Median head pitch (degrees) — neutral vertical gaze.
    baseline_yaw : float
        Median head yaw (degrees) — neutral horizontal gaze.
    """

    baseline_ear: float
    baseline_ear_std: float
    baseline_mar: float
    baseline_mar_std: float
    baseline_pitch: float
    baseline_yaw: float


# ── CalibrationRunner ───────────────────────────────────────────


class CalibrationRunner:
    """Collects per-frame results and computes median baselines.

    Parameters
    ----------
    duration_s : float
        Length of the calibration window in seconds.  Defaults to
        ``CALIBRATION_DURATION_S`` (10 s).

    The runner is single-use: once :pyattr:`is_complete` becomes True,
    call :pyattr:`result` to retrieve the baselines, then discard the
    runner.  For a re-calibration (e.g. return from Away state), create
    a new instance.
    """

    def __init__(self, duration_s: float = CALIBRATION_DURATION_S) -> None:
        self._duration_s = duration_s

        # Sample accumulators — only frames with a detected face are
        # included.
        self._ears: list[float] = []
        self._mars: list[float] = []
        self._pitches: list[float] = []
        self._yaws: list[float] = []

        self._start_time: float | None = None
        self._result: CalibrationResult | None = None

        log.info("CalibrationRunner created (window=%.1f s).", duration_s)

    # ── Properties ──────────────────────────────────────────────

    @property
    def is_complete(self) -> bool:
        """True once the calibration window has elapsed and baselines
        have been computed."""
        return self._result is not None

    @property
    def elapsed_s(self) -> float:
        """Seconds elapsed since the first :meth:`feed` call."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def result(self) -> CalibrationResult | None:
        """The computed baselines, or ``None`` if not yet complete."""
        return self._result

    # ── Public API ──────────────────────────────────────────────

    def feed(self, face: FaceResult) -> None:
        """Ingest one frame's worth of results.

        Parameters
        ----------
        face : FaceResult
            Output from :class:`FaceMeshAnalyzer.process`.

        The method silently skips samples where no face was detected
        (those frames carry no useful baseline information).

        Once the calibration window has elapsed, the method computes
        median baselines and makes :pyattr:`result` available.  Further
        calls after completion are no-ops.
        """
        if self._result is not None:
            return  # already finalised

        # Start the clock on the first call.
        if self._start_time is None:
            self._start_time = time.monotonic()
            log.info("Calibration started.")

        # ── Accumulate valid samples ────────────────────────────
        if face.face_detected:
            self._ears.append(face.ear_avg)
            self._mars.append(face.mar)
            self._pitches.append(face.pitch)
            self._yaws.append(face.yaw)

        # ── Check if the window has elapsed ─────────────────────
        if self.elapsed_s < self._duration_s:
            return

        self._finalise()

    # ── Private helpers ─────────────────────────────────────────

    def _finalise(self) -> None:
        """Compute median baselines from accumulated samples."""
        face_ok = len(self._ears) >= _MIN_SAMPLES

        if not face_ok:
            log.warning(
                "Calibration: only %d face samples (need %d). "
                "Using fallback zeros for face baselines.",
                len(self._ears),
                _MIN_SAMPLES,
            )

        self._result = CalibrationResult(
            baseline_ear=float(np.median(self._ears)) if face_ok else 0.0,
            baseline_ear_std=float(np.std(self._ears)) if face_ok else 0.0,
            baseline_mar=float(np.median(self._mars)) if face_ok else 0.0,
            baseline_mar_std=float(np.std(self._mars)) if face_ok else 0.0,
            # Head-pose baselines use a circular median: a plain median
            # over samples straddling the ±180° wrap collapses toward 0°,
            # i.e. the opposite of the true resting direction.
            baseline_pitch=circular_median_deg(self._pitches) if face_ok else 0.0,
            baseline_yaw=circular_median_deg(self._yaws) if face_ok else 0.0,
        )

        log.info(
            "Calibration complete — %d face samples.  "
            "Baselines: EAR=%.3f±%.3f  MAR=%.3f±%.3f  pitch=%.1f°  yaw=%.1f°",
            len(self._ears),
            self._result.baseline_ear,
            self._result.baseline_ear_std,
            self._result.baseline_mar,
            self._result.baseline_mar_std,
            self._result.baseline_pitch,
            self._result.baseline_yaw,
        )

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        if self._result is not None:
            return "<CalibrationRunner complete>"
        return f"<CalibrationRunner elapsed={self.elapsed_s:.1f}s / {self._duration_s:.1f}s>"
