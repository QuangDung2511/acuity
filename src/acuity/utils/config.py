# QSettings wrapper - persists user preferences to disk.
# ─────────────────────────────────────────────────────────────────
# Uses PySide6.QtCore.QSettings (INI-backed on Windows) so every
# user-tunable value survives restarts without an external file.
#
# Thread: GUI.  Construct and use it on the GUI thread; the CV worker
# receives settings through its constructor and setters.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from acuity.utils.constants import (
    # Alert defaults
    ALERT_DEFAULT_DISTRACTION,
    ALERT_DEFAULT_FATIGUE,
    ALERT_DEFAULT_FATIGUE_WARNING,
    APP_NAME,
    APP_ORG,
    BLOCKS_BEFORE_LONG_REST_DEFAULT,
    # Camera
    CAMERA_FLIP_DEFAULT,
    # Debug overlay
    DEBUG_OVERLAY_DEFAULT,
    DEBUG_OVERLAY_FRAME_SKIP_DEFAULT,
    # Feature toggles
    DESK_WRITING_MODE_DEFAULT,
    FACE_LOST_THRESHOLD_DEFAULT_S,
    FATIGUE_REST_DEFAULT,
    FLUID_WORK_DEFAULT_M,
    # Threshold defaults
    HEAD_POSE_THRESHOLD_DEFAULT_S,
    REST_BUTTON_UNLOCK_DEFAULT_M,
    # Session mode
    SESSION_MODE_DEFAULT,
    STRUCTURED_LONG_REST_DEFAULT_M,
    STRUCTURED_REST_DEFAULT_M,
    # Timer defaults
    STRUCTURED_WORK_DEFAULT_M,
)


class ConfigManager:
    """Thin wrapper around QSettings for typed, default-aware access.

    All keys are organised under logical groups:
        thresholds/   – detection time thresholds (seconds)
        alerts/       – alert behaviour per category
        timer/        – Pomodoro / Fluid timer parameters
        camera/       – selected camera device index
        insights/     – focus-goal hours, etc.
        audio/        – custom audio file paths per alert type
    """

    # ── Key registry ────────────────────────────────────────────
    # Maps a friendly constant name → (QSettings key, default value)
    _DEFAULTS: dict[str, tuple[str, Any]] = {
        # Thresholds (seconds)
        "head_pose_threshold": ("thresholds/head_pose_s", HEAD_POSE_THRESHOLD_DEFAULT_S),
        "face_lost_threshold": ("thresholds/face_lost_s", FACE_LOST_THRESHOLD_DEFAULT_S),
        # Feature toggles
        "desk_writing_mode": ("features/desk_writing_mode", DESK_WRITING_MODE_DEFAULT),
        # Alert modes
        "alert_distraction": ("alerts/distraction", ALERT_DEFAULT_DISTRACTION),
        "alert_fatigue": ("alerts/fatigue", ALERT_DEFAULT_FATIGUE),
        "alert_fatigue_warning": ("alerts/fatigue_warning_mode", ALERT_DEFAULT_FATIGUE_WARNING),
        # Event review: ask the user to rate each alert at rest and session
        # end, so Insights can show each detector's precision on them.
        "review_events": ("alerts/review_events", False),
        # Timer
        "session_mode": ("timer/session_mode", SESSION_MODE_DEFAULT),
        "structured_work_min": ("timer/structured_work_min", STRUCTURED_WORK_DEFAULT_M),
        "structured_rest_min": ("timer/structured_rest_min", STRUCTURED_REST_DEFAULT_M),
        "structured_long_rest_min": (
            "timer/structured_long_rest_min",
            STRUCTURED_LONG_REST_DEFAULT_M,
        ),
        "blocks_before_long_rest": (
            "timer/blocks_before_long_rest",
            BLOCKS_BEFORE_LONG_REST_DEFAULT,
        ),
        "fluid_work_min": ("timer/fluid_work_min", FLUID_WORK_DEFAULT_M),
        "fatigue_rest_behaviour": ("timer/fatigue_rest_behaviour", FATIGUE_REST_DEFAULT),
        # Manual rest button unlock gate — applies to BOTH modes.
        "rest_button_unlock_min": ("timer/rest_button_unlock_min", REST_BUTTON_UNLOCK_DEFAULT_M),
        # Camera
        "camera_index": ("camera/device_index", 0),
        "camera_flip": ("camera/flip_horizontal", CAMERA_FLIP_DEFAULT),
        # Insights
        "daily_focus_goal_h": ("insights/daily_focus_goal_h", 4.0),
        # Custom audio paths (empty string = use system default)
        "audio_distraction": ("audio/distraction_path", ""),
        "audio_fatigue": ("audio/fatigue_path", ""),
        "audio_fatigue_warning": ("audio/fatigue_warning_path", ""),
        "audio_rest_start": ("audio/rest_start_path", ""),
        "audio_rest_end": ("audio/rest_end_path", ""),
        # Appearance
        "theme": ("appearance/theme", "dark"),
        # Debug overlay
        "debug_overlay": ("debug/overlay_enabled", DEBUG_OVERLAY_DEFAULT),
        "debug_overlay_frame_skip": ("debug/overlay_frame_skip", DEBUG_OVERLAY_FRAME_SKIP_DEFAULT),
    }

    def __init__(self) -> None:
        self._settings = QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            APP_ORG,
            APP_NAME,
        )

    # ── Generic helpers ─────────────────────────────────────────

    def _resolve(self, name: str) -> tuple[str, Any]:
        """Return (qsettings_key, default) for a registered name."""
        try:
            return self._DEFAULTS[name]
        except KeyError:
            raise KeyError(
                f"Unknown config key '{name}'. Registered keys: {sorted(self._DEFAULTS)}"
            ) from None

    # ── Public getters ──────────────────────────────────────────

    def get(self, name: str) -> Any:
        """Return the stored value (or default) for *name*."""
        key, default = self._resolve(name)
        return self._settings.value(key, default)

    def get_int(self, name: str) -> int:
        """Return value cast to int."""
        key, default = self._resolve(name)
        return int(self._settings.value(key, default))

    def get_float(self, name: str) -> float:
        """Return value cast to float."""
        key, default = self._resolve(name)
        return float(self._settings.value(key, default))

    def get_str(self, name: str) -> str:
        """Return value cast to str."""
        key, default = self._resolve(name)
        return str(self._settings.value(key, default))

    def get_bool(self, name: str) -> bool:
        """Return value cast to bool.

        QSettings may deserialise booleans as the strings ``'true'`` /
        ``'false'``, so a simple ``bool()`` cast is unreliable.  This
        method normalises all representations.
        """
        key, default = self._resolve(name)
        val = self._settings.value(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    def get_path(self, name: str) -> Path | None:
        """Return value as a Path, or None if empty/unset."""
        raw = self.get_str(name)
        return Path(raw) if raw else None

    # ── Public setters ──────────────────────────────────────────

    def set(self, name: str, value: Any) -> None:
        """Persist *value* under the registered *name*."""
        key, _ = self._resolve(name)
        self._settings.setValue(key, value)

    # ── Bulk operations ─────────────────────────────────────────

    def reset_all(self) -> None:
        """Restore every key to its factory default."""
        for key, default in self._DEFAULTS.values():
            self._settings.setValue(key, default)
        self._settings.sync()

    def sync(self) -> None:
        """Flush pending writes to persistent storage."""
        self._settings.sync()

    # ── Introspection ───────────────────────────────────────────

    @classmethod
    def registered_keys(cls) -> list[str]:
        """Return all registered config key names."""
        return sorted(cls._DEFAULTS)

    def as_dict(self) -> dict[str, Any]:
        """Snapshot of every key → current value (useful for logging)."""
        return {name: self.get(name) for name in self._DEFAULTS}

    def settings_path(self) -> str:
        """Return the filesystem path QSettings is writing to."""
        return self._settings.fileName()
