# Per-frame CSV data logger for benchmarking and analysis.
# ─────────────────────────────────────────────────────────────────
# Activated when debug mode is ON.  Writes one row per frame to
# <log_dir>/session_YYYYMMDD_HHMMSS.csv (log_dir() from the
# paths adapter, so ACUITY_DATA_DIR redirects it), capturing all raw
# engine metrics and fatigue scoring signals in a single flat table.
#
# Thread: CV worker (not thread-safe).  No Qt.
#
# Column contract: tools/replay_csv_fatigue.py re-drives the fatigue
# engine from these files.  Every column in its _REQUIRED list, plus
# blink_score, yawn_threshold, fatigue_score and timestamp_wall, must
# keep its name (tests/test_data_logger.py checks this).
#
# 50 columns grouped as:
#   Timing (3)       : timestamp_mono, timestamp_wall, frame_number
#   Presence (1)     : presence
#   Face mesh (12)   : face_detected, ear_left, ear_right, ear_avg,
#                      mar, mar_smoothed, pitch, yaw, roll, blink_score,
#                      pitch_gated, ear_suppressed
#   Fatigue raw (12) : smoothed_ear, blink_threshold, eyes_closed,
#                      eyes_shut, perclos, blink_rate,
#                      baseline_blink_rate, avg_blink_duration_ms,
#                      yawn_threshold, yawn_active, yawn_count_5min,
#                      head_nod_count_1min
#   Fatigue scoring  : fatigue_score, signal_perclos, signal_blink_dur,
#     (13)             signal_yawn, signal_blink_freq, signal_head_nod,
#                      fatigue_warning_active, fatigue_alert_active,
#                      microsleep_active, microsleep_elapsed_s,
#                      episode_bilateral_share, episode_blink_confirm,
#                      microsleep_withheld
#   Distraction (5)  : head_pose_deviated, pitch_delta, yaw_delta,
#                      pose_timer_elapsed_s, pose_timer_fired
#   Pose (1)         : pose_detected (presence input)
#   Events (3)       : event_fired, event_type, event_reason
#
# The original's phone and posture columns are gone with their
# detectors.
#
# Usage:
#   logger = DataLogger()            # opens file, writes header
#   logger.log(snapshot,             # call each frame
#              event_fired=True,
#              event_type="Fatigue",
#              event_reason="score_threshold")
#   logger.close()                   # flush + close (call in cleanup)
#
# The logger is a plain Python class with zero Qt dependency.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

from acuity.core.debug_snapshot import DebugSnapshot
from acuity.system import paths

# ── Column header order (must match _row() exactly) ──────────────
_HEADERS: list[str] = [
    # Timing
    "timestamp_mono",
    "timestamp_wall",
    "frame_number",
    # Presence
    "presence",
    # Face mesh
    "face_detected",
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "mar_smoothed",
    "pitch",
    "yaw",
    "roll",
    "blink_score",
    "pitch_gated",
    "ear_suppressed",
    # Fatigue raw
    "smoothed_ear",
    "blink_threshold",
    "eyes_closed",
    "eyes_shut",
    "perclos",
    "blink_rate",
    "baseline_blink_rate",
    "avg_blink_duration_ms",
    "yawn_threshold",
    "yawn_active",
    "yawn_count_5min",
    "head_nod_count_1min",
    # Fatigue scoring
    "fatigue_score",
    "signal_perclos",
    "signal_blink_dur",
    "signal_yawn",
    "signal_blink_freq",
    "signal_head_nod",
    "fatigue_warning_active",
    "fatigue_alert_active",
    "microsleep_active",
    "microsleep_elapsed_s",
    "episode_bilateral_share",
    "episode_blink_confirm",
    "microsleep_withheld",
    # Distraction
    "head_pose_deviated",
    "pitch_delta",
    "yaw_delta",
    "pose_timer_elapsed_s",
    "pose_timer_fired",
    # Pose
    "pose_detected",
    # Events
    "event_fired",
    "event_type",
    "event_reason",
]

assert len(_HEADERS) == 50, f"Expected 50 columns, got {len(_HEADERS)}"


class DataLogger:
    """Writes one CSV row per frame to a timestamped session file.

    The file goes in ``paths.log_dir()``, created if absent.

    Notes
    -----
    * ``close()`` **must** be called (or use as a context manager) to
      guarantee the file is flushed and closed.
    * The logger is not thread-safe; it is expected to be called from
      the CV pipeline thread only.
    """

    def __init__(self) -> None:
        base = paths.log_dir()
        base.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = base / f"session_{stamp}.csv"

        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_HEADERS)

        self._frame_number: int = 0

    # ── Public API ───────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Absolute path of the CSV file being written."""
        return self._path.resolve()

    def log(
        self,
        snapshot: DebugSnapshot,
        *,
        event_fired: bool = False,
        event_type: str = "",
        event_reason: str = "",
    ) -> None:
        """Write one row for the current frame.

        Parameters
        ----------
        snapshot : DebugSnapshot
            Frozen per-frame diagnostic state from the CV pipeline.
        event_fired : bool
            True if any event was emitted this frame.
        event_type : str
            The ``EventData.event_type`` string, or ``""`` if none.
        event_reason : str
            The ``EventData.reason`` string, or ``""`` if none.
        """
        self._frame_number += 1
        row = self._build_row(snapshot, event_fired, event_type, event_reason)
        self._writer.writerow(row)

    def close(self) -> None:
        """Flush and close the CSV file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    # ── Context manager support ──────────────────────────────────

    def __enter__(self) -> DataLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Internal ─────────────────────────────────────────────────

    def _build_row(
        self,
        s: DebugSnapshot,
        event_fired: bool,
        event_type: str,
        event_reason: str,
    ) -> list:
        """Build the ordered list of values matching _HEADERS."""
        return [
            # Timing
            f"{time.monotonic():.6f}",
            datetime.now().isoformat(timespec="milliseconds"),
            self._frame_number,
            # Presence
            s.presence,
            # Face mesh
            int(s.face_detected),
            f"{s.ear_left:.4f}",
            f"{s.ear_right:.4f}",
            f"{s.ear_avg:.4f}",
            f"{s.mar:.4f}",
            f"{s.mar_smoothed:.4f}",
            f"{s.pitch:.2f}",
            f"{s.yaw:.2f}",
            f"{s.roll:.2f}",
            "" if s.blink_score is None else f"{s.blink_score:.3f}",
            int(s.pitch_gated),
            int(s.ear_suppressed),
            # Fatigue raw
            f"{s.smoothed_ear:.4f}",
            f"{s.blink_threshold:.4f}",
            int(s.eyes_closed),
            int(s.eyes_shut),
            f"{s.perclos:.4f}",
            f"{s.blink_rate:.2f}",
            f"{s.baseline_blink_rate:.2f}" if s.baseline_blink_rate is not None else "",
            f"{s.avg_blink_duration_ms:.1f}",
            f"{s.yawn_threshold:.4f}",
            int(s.yawn_active),
            s.yawn_count_5min,
            s.head_nod_count_1min,
            # Fatigue scoring
            f"{s.fatigue_score:.4f}",
            f"{s.signal_perclos:.4f}",
            f"{s.signal_blink_dur:.4f}",
            f"{s.signal_yawn:.4f}",
            f"{s.signal_blink_freq:.4f}",
            f"{s.signal_head_nod:.4f}",
            int(s.fatigue_warning_active),
            int(s.fatigue_alert_active),
            int(s.microsleep_active),
            f"{s.microsleep_elapsed_s:.3f}",
            f"{s.episode_bilateral_share:.3f}",
            f"{s.episode_blink_confirm:.3f}",
            s.microsleep_withheld,
            # Distraction
            int(s.head_pose_deviated),
            f"{s.pitch_delta:.2f}",
            f"{s.yaw_delta:.2f}",
            f"{s.pose_timer_elapsed_s:.3f}",
            int(s.pose_timer_fired),
            # Pose
            int(s.pose_detected),
            # Events
            int(event_fired),
            event_type,
            event_reason,
        ]
