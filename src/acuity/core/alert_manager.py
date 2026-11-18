# Routes events to correct alert type (log / toast / sound / both).
# ─────────────────────────────────────────────────────────────────
# Receives engine events through ``handle_event`` (MainWindow connects
# each focus block's pipeline ``event_detected`` to it) and dispatches
# alerts based on the user's per-category preference from
# ConfigManager.
#
# Thread: GUI.  Toasts go through QSystemTrayIcon and sounds through
# QMediaPlayer, both GUI-thread objects.
#
# Alert modes (constants from acuity/utils/constants.py):
#   log_only             — event is logged only (no user-facing alert)
#   notification_only    — OS-level toast notification via QSystemTrayIcon
#   sound_only           — audio alert via SoundPlayer
#   notification_and_sound — both toast and sound
#
# Per-category config keys (from ConfigManager):
#   alert_distraction     → applies to Distraction_Pose + Distraction_Window
#   alert_fatigue         → applies to Fatigue (alert) events
#   alert_fatigue_warning → applies to Fatigue_Warning events
#
# The two distraction channels share one setting and one sound: to the
# user both mean "you stopped working", and a separate window setting
# would ask them to tune two things for one habit.  Each keeps its own
# toast text.
#
# Custom audio config keys (each falls back to the bundled default
# sound in assets/sounds/ when unset — see SoundPlayer):
#   audio_distraction, audio_fatigue_warning, audio_fatigue
#
# Usage (inside MainWindow):
#
#   alert_mgr = AlertManager(tray_icon=tray)
#   pipeline.event_detected.connect(alert_mgr.handle_event)
#   # — or directly —
#   alert_mgr.handle_event(event_data)
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QSystemTrayIcon

from acuity.core.cv_pipeline import EventData
from acuity.core.sound_player import SoundPlayer
from acuity.utils.config import ConfigManager
from acuity.utils.constants import (
    ALERT_LOG_ONLY,
    ALERT_NOTIFICATION_AND_SOUND,
    ALERT_NOTIFICATION_ONLY,
    ALERT_SOUND_ONLY,
    APP_NAME,
    EVENT_DISTRACTION_POSE,
    EVENT_DISTRACTION_WINDOW,
    EVENT_FATIGUE,
    EVENT_FATIGUE_WARNING,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# ── Event type → alert category mapping ─────────────────────────
# Maps EVENT_* constants to the config key prefix used for both
# the alert mode and custom audio path lookups.

_EVENT_CATEGORY: dict[str, str] = {
    EVENT_DISTRACTION_POSE: "distraction",
    EVENT_DISTRACTION_WINDOW: "distraction",
    EVENT_FATIGUE: "fatigue",
    EVENT_FATIGUE_WARNING: "fatigue_warning",  # separate category from alert-tier fatigue
}

# Human-readable titles for toast notifications.
_NOTIFY_TITLES: dict[str, str] = {
    EVENT_DISTRACTION_POSE: "Distraction — Head Pose",
    EVENT_DISTRACTION_WINDOW: "Distraction — Window",
    EVENT_FATIGUE: "Fatigue Alert",
    EVENT_FATIGUE_WARNING: "Fatigue Warning",
}

# Notification body templates.
_NOTIFY_BODIES: dict[str, str] = {
    EVENT_DISTRACTION_POSE: "Head deviated for {duration:.0f}s.",
    # reason is the app's executable name; window titles never reach it.
    EVENT_DISTRACTION_WINDOW: "Away from your focus apps for {duration:.0f}s ({reason}).",
    EVENT_FATIGUE: "{reason} ({duration:.1f}s).",  # score is debug-only — see plan Decision #5
    EVENT_FATIGUE_WARNING: "Signs of fatigue detected. Consider adjusting.",
}


class AlertManager(QObject):
    """Dispatches alerts based on event type and user preferences.

    Reads the user's alert mode for each category from
    :class:`ConfigManager` on every event, so changes made in the
    Settings UI take effect immediately without restart.

    Parameters
    ----------
    tray_icon : QSystemTrayIcon or None
        If provided, OS-level toast notifications are sent via
        ``tray_icon.showMessage()``.  If ``None``, notification
        dispatch is silently skipped (useful for headless testing
        or before the tray is initialised).
    parent : QObject or None
        Qt parent.
    """

    def __init__(
        self,
        tray_icon: QSystemTrayIcon | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._tray_icon = tray_icon
        self._cfg = ConfigManager()
        self._sound_player = SoundPlayer(self)

        log.info("AlertManager initialised (tray=%s).", tray_icon is not None)

    # ── Properties ──────────────────────────────────────────────

    @property
    def tray_icon(self) -> QSystemTrayIcon | None:
        return self._tray_icon

    @tray_icon.setter
    def tray_icon(self, icon: QSystemTrayIcon | None) -> None:
        """Set or replace the tray icon (e.g. after delayed init)."""
        self._tray_icon = icon

    @property
    def sound_player(self) -> SoundPlayer:
        """Access the underlying SoundPlayer (for volume control, etc.)."""
        return self._sound_player

    # ── Public API ──────────────────────────────────────────────

    def handle_event(self, event: EventData) -> None:
        """Dispatch an alert for *event* based on user preferences.

        Parameters
        ----------
        event : EventData
            The engine event to dispatch.  Events whose
            ``event_type`` is not in the known category map
            (e.g. ``EVENT_AWAY``) are logged only — they have no
            configurable alert.
        """
        category = _EVENT_CATEGORY.get(event.event_type)
        if category is None:
            # Away events and other non-alertable types are log-only.
            log.debug(
                "Event '%s' has no alert category — log only.",
                event.event_type,
            )
            return

        mode = self._get_alert_mode(category)

        log.debug(
            "Alert dispatch: type=%s category=%s mode=%s",
            event.event_type,
            category,
            mode,
        )

        if mode == ALERT_LOG_ONLY:
            # Already logged by SessionManager — nothing else to do.
            return

        if mode in (ALERT_NOTIFICATION_ONLY, ALERT_NOTIFICATION_AND_SOUND):
            self._send_notification(event)

        if mode in (ALERT_SOUND_ONLY, ALERT_NOTIFICATION_AND_SOUND):
            self._play_sound(category)

    def announce_rest_start(self, block_number: int, rest_seconds: int) -> None:
        """Announce an automatic focus→rest transition (structured mode).

        Always sends both a toast and a sound, deliberately bypassing the
        per-category ``alert_*`` preferences.  Those settings govern
        *warnings* — advice the user may reasonably want silenced.  This
        is a state change: the session has already switched to rest, and a
        user who misses it keeps working through a break they scheduled.
        Silencing it would leave the app in a state the user cannot infer
        from the screen they are looking at.

        The sound honours a custom ``audio_rest_start`` path like every
        other category, so it can still be replaced — just not muted here.
        """
        minutes = rest_seconds // 60
        seconds = rest_seconds % 60
        duration = f"{minutes} min" if seconds == 0 else f"{minutes}:{seconds:02d}"

        if self._tray_icon is not None and self._tray_icon.isSystemTrayAvailable():
            self._tray_icon.showMessage(
                f"Focus block {block_number} complete",
                f"Rest for {duration}. Press Skip Rest to keep working.",
                QSystemTrayIcon.MessageIcon.Information,
                5000,
            )
        else:
            log.warning("System tray unavailable — rest-start toast skipped.")

        self._play_sound("rest_start")
        log.info(
            "Rest-start announced (block=%d, rest=%ds).",
            block_number,
            rest_seconds,
        )

    def announce_rest_end(self) -> None:
        """Announce that a rest period's timer has run out.

        The counterpart to ``announce_rest_start`` and unconditional for
        the same reason: this is a state change, not advice.  The user is
        by definition *away* — that is what a rest is — so the screen is
        the one surface that cannot be relied on to deliver it, and a
        missed cue means the break silently overruns.  The per-category
        ``alert_*`` preferences govern warnings and do not apply.

        Fires only when the countdown reaches zero, never on skip-rest:
        a skip is a click the user just made, so it needs no announcing.
        See ``SessionManager.rest_timer_expired``.

        The sound honours a custom ``audio_rest_end`` path like every
        other category, so it can still be replaced — just not muted here.
        """
        if self._tray_icon is not None and self._tray_icon.isSystemTrayAvailable():
            self._tray_icon.showMessage(
                "Rest complete",
                "Time's up — switch back to Acuity to start the next block.",
                QSystemTrayIcon.MessageIcon.Information,
                5000,
            )
        else:
            log.warning("System tray unavailable — rest-end toast skipped.")

        self._play_sound("rest_end")
        log.info("Rest-end announced.")

    # ── Private helpers ─────────────────────────────────────────

    def _get_alert_mode(self, category: str) -> str:
        """Look up the user's alert preference for *category*.

        Re-reads from ConfigManager on every call so Settings
        changes are picked up immediately.
        """
        key = f"alert_{category}"
        return self._cfg.get_str(key)

    def _send_notification(self, event: EventData) -> None:
        """Send an OS-level toast notification via QSystemTrayIcon."""
        if self._tray_icon is None:
            log.debug(
                "No tray icon — skipping notification for '%s'.",
                event.event_type,
            )
            return

        if not self._tray_icon.isSystemTrayAvailable():
            log.warning("System tray not available — skipping notification.")
            return

        title = _NOTIFY_TITLES.get(event.event_type, APP_NAME)
        body_template = _NOTIFY_BODIES.get(event.event_type, "")
        body = body_template.format(
            duration=event.duration,
            reason=event.reason or event.event_type,
        )

        self._tray_icon.showMessage(
            title,
            body,
            QSystemTrayIcon.MessageIcon.Information,
            5000,  # display for 5 seconds
        )
        log.debug("Toast sent: %s — %s", title, body)

    def _play_sound(self, category: str) -> None:
        """Play the alert sound for *category* (custom or default)."""
        custom_path = self._get_custom_audio_path(category)
        self._sound_player.play(category, custom_path)

    def _get_custom_audio_path(self, category: str) -> Path | None:
        """Resolve the user's custom audio file for *category*.

        Returns None if no custom path is configured or the file
        does not exist.
        """
        path = self._cfg.get_path(f"audio_{category}")
        if path is not None and path.is_file():
            return path
        return None

    # ── Dunder helpers ──────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<AlertManager tray={'yes' if self._tray_icon else 'no'} sound={self._sound_player!r}>"
        )
