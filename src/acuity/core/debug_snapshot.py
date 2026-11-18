# Per-frame diagnostic snapshot for the debug overlay.
# ─────────────────────────────────────────────────────────────────
# A single frozen dataclass that aggregates all internal state from
# every CV engine into one object per frame.  Used exclusively by
# the debug overlay pipeline:
#
#   Producer : CVPipelineThread._build_debug_snapshot()
#   Consumer : DebugOverlayRenderer.draw()
#
# Fields are grouped by engine / subsystem with inline comments
# explaining units and semantics.  All fields have defaults so
# the dataclass can be instantiated partially in unit tests.
#
# This file has NO runtime impact when debug mode is off — the
# snapshot is only constructed when self._debug_mode is True in
# CVPipelineThread.
#
# Thread: pure data.  Built on the CV worker, read by the overlay and
# the data logger on the same thread.
#
# The original's phone and posture fields, the shoulder baselines and
# torso_pitch_deg are gone with the detectors that produced them.
# pose_detected stays: the pose landmarker still feeds presence.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DebugSnapshot:
    """All per-frame diagnostic state needed by the debug overlay.

    Attributes
    ----------
    presence : str
        Current presence fusion state name: ``"PRESENT"``,
        ``"PRESENT_NO_FACE"``, or ``"ABSENT"``.

    face_detected : bool
        Whether MediaPipe FaceLandmarker detected a face this frame.
    ear_left : float
        Raw left-eye EAR (subject's left = viewer's right).
    ear_right : float
        Raw right-eye EAR.
    ear_avg : float
        Mean of ear_left and ear_right.
    pitch : float
        Head pitch in degrees (positive = looking up).
    yaw : float
        Head yaw in degrees (positive = looking right).
    roll : float
        Head roll in degrees (positive = tilting right).
    blink_score : float or None
        This frame's mean eyeBlink blendshape (``FaceResult.blink_score``),
        ``None`` when the model reported none.  Logged per frame so the
        blink-confirm gate can be re-measured against the corpus — until
        now only its episode median (``episode_blink_confirm``) was kept,
        which cannot be re-thresholded or normalised offline.

    baseline_pitch : float
        Calibrated neutral head pitch (degrees).
    baseline_yaw : float
        Calibrated neutral head yaw (degrees).
    baseline_ear : float
        Calibrated resting EAR (median).
    baseline_ear_std : float
        Standard deviation of EAR samples during calibration.

    head_pose_deviated : bool
        Whether the current pitch/yaw exceeds the tolerance threshold.
    pitch_delta : float
        ``abs(pitch − baseline_pitch)`` in degrees.
    yaw_delta : float
        ``abs(yaw − baseline_yaw)`` in degrees.
    head_pose_pitch_tolerance_deg : float
        Pitch tolerance before the distraction timer starts (°).
    head_pose_yaw_tolerance_deg : float
        Yaw tolerance before the distraction timer starts (°).
    pose_timer_elapsed_s : float
        Seconds the head-pose distraction timer has been running.
    pose_timer_threshold_s : float
        Threshold that triggers a Distraction_Pose event.
    pose_timer_fired : bool
        True if the pose timer already fired this episode.
    desk_writing_mode : bool
        Whether desk-writing mode is active (suppresses Trigger A).

    blink_threshold : float
        EAR value below which the eyes are considered closed.
    smoothed_ear : float
        Median-filtered EAR (5-sample window) used for threshold comparison.
    eyes_closed : bool
        True if smoothed EAR is below the blink threshold this frame.
    microsleep_active : bool
        True if a confirmed microsleep episode is currently in progress.
    microsleep_elapsed_s : float
        How long the current eyes-closed episode has been running (0 if open).
    episode_bilateral_share : float
        Fraction of the current eyes-shut episode's frames on which BOTH eyes
        read below the blink threshold.  Reads **1.0 outside an episode**, not
        0.0 — the engine's convention is "nothing disqualifying" (see
        FatigueEngine.debug_episode_bilateral_share), so a CSV row with no
        episode in progress carries 1.0.  Logged so a CSV can show *why* an
        episode qualified, or did not, at the MICROSLEEP_BILATERAL_SHARE gate.
    episode_blink_confirm : float
        Median eyeBlink blendshape across the current eyes-shut episode.
        Follows the same 1.0-means-nothing-disqualifying convention, which
        here also covers "the model reported no blendshapes" — so 1.0 is
        never evidence that a closure was confirmed.  See
        MICROSLEEP_BLINK_CONFIRM.
    microsleep_withheld : str
        Why a qualifying microsleep did not fire this frame: ``"bilateral"``,
        ``"blink"``, ``"perclos"``, ``"observation"``, ``"context"``,
        ``"cooldown"``, or ``""``.  Observation only — a suppressed episode otherwise leaves no
        trace in the CSV, and scoring runs undercount false positives.
    perclos : float
        Fraction of recent frames with eyes below threshold (0.0 – 1.0).
    perclos_gate : float
        Minimum PERCLOS required to allow a microsleep event to fire.
    blink_rate : float
        Current rolling blink rate (blinks/min).
    baseline_blink_rate : float or None
        Baseline blink rate captured from the first full window. ``None``
        until the window fills.
    fatigue_cooldown_remaining : float
        Seconds until the next fatigue event may fire (0.0 if ready).

    pose_detected : bool
        Whether MediaPipe Pose detected a body this frame (presence input).

    body_relative_pitch : float
        Body-relative head pitch (degrees).
    calibrated_relative_pitch : float
        Calibrated body-relative head pitch (degrees).

    fatigue_score : float
        Composite fatigue score in [0.0, 1.0] from the weighted 5-signal model.
    signal_perclos : float
        Normalized PERCLOS sub-signal in [0.0, 1.0].
    signal_blink_dur : float
        Normalized blink-duration sub-signal in [0.0, 1.0].
    signal_yawn : float
        Normalized yawn-count sub-signal in [0.0, 1.0].
    signal_blink_freq : float
        Normalized blink-frequency-drop sub-signal in [0.0, 1.0].
    signal_head_nod : float
        Normalized head-nod sub-signal in [0.0, 1.0].
    mar : float
        Raw Mouth Aspect Ratio from the current frame.
    mar_smoothed : float
        Median-filtered MAR (5-sample window) used for yawn detection.
    yawn_threshold : float
        MAR value above which the mouth is considered open.
    yawn_active : bool
        True if the mouth is currently open past the yawn threshold.
    yawn_count_5min : int
        Number of yawns detected in the last 5 minutes.
    avg_blink_duration_ms : float
        Rolling average blink duration (ms) from the last 50 blinks.
    head_nod_count_1min : int
        Number of head nods detected in the last 60 seconds.
    fatigue_warning_active : bool
        True if the score has been above the warning threshold long enough
        to be in a sustained warning state.
    fatigue_alert_active : bool
        True if the score has been above the alert threshold long enough
        to be in a sustained alert state.
    """

    # ── Presence ─────────────────────────────────────────────────
    presence: str = "PRESENT"

    # ── Face Mesh raw values ──────────────────────────────────────
    face_detected: bool = False
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    blink_score: float | None = None  # eyeBlink blendshape; None = no evidence

    # ── Calibration baselines ─────────────────────────────────────
    baseline_pitch: float = 0.0
    baseline_yaw: float = 0.0
    baseline_ear: float = 0.0
    baseline_ear_std: float = 0.0

    # ── Distraction engine — head pose ────────────────────────────
    head_pose_deviated: bool = False
    pitch_delta: float = 0.0  # abs(pitch - baseline_pitch)
    yaw_delta: float = 0.0  # abs(yaw - baseline_yaw)
    head_pose_pitch_tolerance_deg: float = 20.0
    head_pose_yaw_tolerance_deg: float = 15.0
    pose_timer_elapsed_s: float = 0.0
    pose_timer_threshold_s: float = 15.0
    pose_timer_fired: bool = False
    desk_writing_mode: bool = False

    # ── Fatigue engine ────────────────────────────────────────────
    pitch_gated: bool = False  # head pitched down past the gate
    ear_suppressed: bool = False  # any EAR guard active (superset of above)
    blink_threshold: float = 0.0
    smoothed_ear: float = 0.0  # after 5-sample median filter
    eyes_closed: bool = False  # smoothed_ear < blink_threshold
    eyes_shut: bool = False  # smoothed_ear < shut_threshold (stricter)
    microsleep_active: bool = False
    microsleep_elapsed_s: float = 0.0  # 0.0 when eyes are open
    episode_bilateral_share: float = 1.0  # both-eyes share; 1.0 = no episode
    episode_blink_confirm: float = 1.0  # median eyeBlink; 1.0 = no samples
    microsleep_withheld: str = ""  # gate that refused this frame
    perclos: float = 0.0  # fraction in [0.0, 1.0]
    perclos_gate: float = 0.15
    blink_rate: float = 0.0  # blinks/min rolling window
    baseline_blink_rate: float | None = None
    fatigue_cooldown_remaining: float = 0.0

    # ── Fatigue scoring model ─────────────────────────────────────
    fatigue_score: float = 0.0
    signal_perclos: float = 0.0
    signal_blink_dur: float = 0.0
    signal_yawn: float = 0.0
    signal_blink_freq: float = 0.0
    signal_head_nod: float = 0.0
    mar: float = 0.0
    mar_smoothed: float = 0.0
    yawn_threshold: float = 0.0
    yawn_active: bool = False
    yawn_count_5min: int = 0
    avg_blink_duration_ms: float = 0.0
    head_nod_count_1min: int = 0
    fatigue_warning_active: bool = False
    fatigue_alert_active: bool = False

    # ── Pose landmarker (presence input) ──────────────────────────
    pose_detected: bool = False

    # ── Distraction engine — body relative pitch ────────────────────
    body_relative_pitch: float = 0.0
    calibrated_relative_pitch: float = 0.0
