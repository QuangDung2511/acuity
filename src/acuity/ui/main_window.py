# QMainWindow - hosts tab widget, toolbar, system-tray wiring.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   MainWindow() -> QMainWindow
#   MainWindow.tray -> AcuityTrayIcon   (read-only)
#
# Window behaviour:
#   Close (X)  → hide from taskbar; app stays alive in tray.
#   Minimize(-) → minimize to OS taskbar (default Qt behaviour).
#   Tray Quit  → QApplication.quit() (handled by AcuityTrayIcon).
#
# Wiring responsibility:
#   MainWindow owns SessionManager and AlertManager.
#   _setup_wiring() connects all signals between FocusTab ↔ SessionManager
#   ↔ CVPipelineThread (via session_manager.pipeline property after start).
#   Pipeline signals (frame_ready, calibration_progress, calibration_complete)
#   are connected lazily in _on_state_changed when CALIBRATING is entered,
#   because the pipeline object doesn't exist until SessionManager creates it.
#
# Event review (review_events setting, default off):
#   When a rest starts, MainWindow opens EventReviewDialog for the block
#   just finished so the user can mark each event Correct / False alarm
#   while it is still fresh — the rest is the one moment the user has
#   already stopped working, so asking costs no focus.  At session end it
#   reviews whatever is left (the final block, or a skipped review).
#   The labels feed the per-detector precision on the Insights tab.
#   The setting is read when each review would open, so turning it on
#   mid-session applies from the next rest.
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QMainWindow, QTabWidget

from acuity.core.alert_manager import AlertManager
from acuity.core.cv_pipeline import EventData
from acuity.core.session_manager import SessionManager, SessionState
from acuity.db import queries
from acuity.ui.system_tray import AcuityTrayIcon
from acuity.ui.tabs.focus_tab import FocusTab
from acuity.ui.tabs.insights_tab import InsightsTab
from acuity.ui.tabs.settings_tab import SettingsTab
from acuity.ui.widgets.event_review_dialog import EventReviewDialog
from acuity.utils.config import ConfigManager
from acuity.utils.constants import (
    APP_NAME,
    EVENT_AWAY,
    EVENT_FATIGUE,
    FATIGUE_REST_DYNAMIC,
    FLUID_REST_DIVISOR,
    SESSION_MODE_FLUID,
    SESSION_MODE_STRUCTURED,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)


class MainWindow(QMainWindow):
    """Main application window for Acuity.

    Hosts a QTabWidget with three tabs:
      0 – Focus    (timer, calibration feed, live event log)
      1 – Insights (charts and analytics)
      2 – Settings (alert config, thresholds, camera, audio)

    Owns:
      • AcuityTrayIcon — tray icon (accessible via self.tray)
      • SessionManager — session lifecycle state machine
      • AlertManager   — routes events to toast / sound / log

    Window management:
      • Closing (X) hides the window to the system tray rather than
        terminating the application.
      • Minimising (-) falls through to standard OS taskbar behaviour
        — no override needed; Qt does this by default.
      • The tray icon's Show action calls show_from_tray() to restore
        the window.
    """

    def __init__(self) -> None:
        super().__init__()

        self._setup_window()
        self._setup_tabs()

        # Tray is created after the window so it can hold a valid
        # reference to self (parent window for Qt ownership tree).
        self._tray = AcuityTrayIcon(self)
        self._tray.contextMenu().actions()[0].triggered.disconnect()
        self._tray.contextMenu().actions()[0].triggered.connect(self.show_from_tray)
        self._tray.show()

        # Session manager + alert manager
        self._session_mgr = SessionManager(parent=self)
        self._alert_mgr = AlertManager(tray_icon=self._tray, parent=self)

        # Previous state name, so _on_state_changed can tell *which*
        # transition into a state it is handling (see the FOCUS branch).
        self._prev_state_name: str = SessionState.IDLE.name

        # Events already shown in a review the user saved, so the next
        # review (next rest, or session end) only lists what is new.
        # A skipped review adds nothing, so its events come back.
        self._reviewed_event_ids: set[int] = set()

        self._setup_wiring()

        log.debug("MainWindow initialised.")

    # ── Public property ────────────────────────────────────────────

    @property
    def tray(self) -> AcuityTrayIcon:
        """The system-tray icon owned by this window."""
        return self._tray

    # ── Setup ──────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        """Configure window title and minimum size."""
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(860, 580)
        # No custom flags: Qt's default gives minimise → taskbar for free.

    def _setup_tabs(self) -> None:
        """Build the central QTabWidget with the main tabs."""
        tabs = QTabWidget()
        tabs.setObjectName("mainTabs")

        self._focus_tab = FocusTab(tabs)
        self._insights_tab = InsightsTab(tabs)
        self._settings_tab = SettingsTab(tabs)
        tabs.addTab(self._focus_tab, "Focus")
        tabs.addTab(self._insights_tab, "Insights")
        tabs.addTab(self._settings_tab, "Settings")

        tabs.setCurrentIndex(0)  # Focus tab shown on launch
        self.setCentralWidget(tabs)
        self._tabs = tabs

        # Insights reads the DB on demand: redraw whenever it is shown so
        # it reflects blocks and events logged since it was last visible.
        tabs.currentChanged.connect(self._on_tab_changed)
        self._settings_tab.theme_changed.connect(self._insights_tab.set_theme)

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._insights_tab:
            self._insights_tab.refresh()

    def _setup_wiring(self) -> None:
        """Connect FocusTab signals → SessionManager and vice-versa.

        Pipeline signals (frame_ready, calibration_progress,
        calibration_complete) cannot be wired here because the pipeline
        object is created by SessionManager only when a session starts.
        Those connections are made in _on_state_changed when the state
        transitions to CALIBRATING.
        """
        ft = self._focus_tab
        sm = self._session_mgr

        # ── FocusTab → SessionManager ─────────────────────────────
        ft.start_requested.connect(self._on_start_requested)
        ft.stop_requested.connect(sm.end_session)
        ft.pause_requested.connect(sm.pause)
        ft.resume_requested.connect(sm.resume)
        ft.skip_rest_requested.connect(sm.skip_rest)
        ft.rest_requested.connect(sm.request_rest)
        ft.flip_camera_changed.connect(self._on_flip_camera_changed)

        # Seed the mirror switch from the persisted setting.
        ft.sync_flip_checkbox(ConfigManager().get_bool("camera_flip"))

        # ── SessionManager → FocusTab ─────────────────────────────
        sm.focus_tick.connect(ft.on_tick)
        # Second slot on the same 1 Hz tick: drives the manual rest
        # button's unlock gate.  The tick argument is unusable here
        # (remaining in Structured, elapsed in Fluid), so the slot
        # reads sm.current_focus_elapsed_s instead.
        sm.focus_tick.connect(self._update_rest_button_gate)
        sm.rest_tick.connect(ft.on_rest_tick)
        sm.event_logged.connect(ft.on_event)
        sm.session_ended.connect(ft.on_session_ended)
        sm.session_ended.connect(self._insights_tab.refresh)
        sm.session_ended.connect(self._on_session_ended_review)

        # Away signal: show dialog, then route user choice back to SM.
        sm.user_away.connect(self._on_user_away)

        # Rest complete: prompt the user before starting the next block.
        sm.rest_complete.connect(self._on_rest_complete)
        sm.rest_auto_started.connect(self._on_rest_auto_started)
        # Rest countdown hit zero: announce it before the modal opens.
        sm.rest_timer_expired.connect(self._alert_mgr.announce_rest_end)

        # State changes: update mode label + Skip Rest button + lazy
        # pipeline wiring.
        sm.state_changed.connect(self._on_state_changed)
        sm.fatigue_rest_prompt.connect(self._on_fatigue_rest_prompt)

        # ── SettingsTab → CVPipelineThread (desk writing mode) ──
        # The pipeline is created lazily per focus cycle; forward the
        # signal through the session manager's current pipeline reference.
        self._settings_tab.desk_writing_mode_changed.connect(self._on_desk_writing_mode_changed)

        # ── SettingsTab → CVPipelineThread (debug overlay) ────
        self._settings_tab.debug_overlay_changed.connect(self._on_debug_overlay_changed)

        # ── Ctrl+D shortcut → toggle debug overlay ─────────────────
        self._debug_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        self._debug_shortcut.activated.connect(self._toggle_debug_overlay)

        log.debug("Signal wiring complete.")
        # Seed idle phase so the button row is correct on first paint.
        ft.set_context_phase("idle")

    # ── Slot: Start requested ──────────────────────────────────────

    def _on_desk_writing_mode_changed(self, enabled: bool) -> None:
        """Forward desk-writing-mode toggle to the live pipeline.

        The pipeline is created per focus cycle.  If no session is
        running the config is already persisted by SettingsTab, so the
        next pipeline will pick it up from ConfigManager.
        """
        pipeline = self._session_mgr.pipeline
        if pipeline is not None:
            pipeline.set_desk_writing_mode(enabled)
            log.debug("Desk writing mode forwarded to pipeline: %s", enabled)

    def _on_flip_camera_changed(self, enabled: bool) -> None:
        """Persist the mirror-camera switch and forward it to the pipeline.

        The Focus tab disables the switch during a session, so in
        practice this only fires while idle and the next pipeline picks
        the value up from config at construction.  The forward covers
        the case where a pipeline is alive but not yet calibrating.
        """
        ConfigManager().set("camera_flip", enabled)

        pipeline = self._session_mgr.pipeline
        if pipeline is not None:
            pipeline.set_flip_horizontal(enabled)
        log.debug("Camera flip set to %s.", enabled)

    def _on_debug_overlay_changed(self, enabled: bool) -> None:
        """Forward debug overlay toggle to the live pipeline.

        When enabled mid-tracking: tells the pipeline to start emitting
        annotated frames and switches the VideoFeedWidget back to live
        mode.  When disabled: stops annotated frame emission and restores
        the static 'Tracking Active' indicator.
        """
        pipeline = self._session_mgr.pipeline
        if pipeline is not None:
            pipeline.set_debug_mode(enabled)
            log.debug("Debug overlay forwarded to pipeline: %s", enabled)

        # Manage the video feed widget state.
        is_tracking = self._session_mgr.state == SessionState.FOCUS
        if is_tracking:
            if enabled:
                # Keep / restore live feed so annotated frames display.
                self._focus_tab.on_calibration_start()
            else:
                # Return to the static indicator.
                self._focus_tab.on_calibration_done()

    def _toggle_debug_overlay(self) -> None:
        """Ctrl+D handler — toggle debug overlay and sync the Settings checkbox.

        Reads the current persisted value, flips it, updates config,
        syncs the Settings checkbox (without re-emitting its signal), then
        calls _on_debug_overlay_changed() to do the actual pipeline work.
        """
        cfg = ConfigManager()
        current = cfg.get_bool("debug_overlay")
        new_value = not current
        cfg.set("debug_overlay", new_value)
        # Sync the Settings checkbox without triggering its signal again.
        self._settings_tab.sync_debug_checkbox(new_value)
        self._on_debug_overlay_changed(new_value)
        log.debug("Ctrl+D: debug overlay toggled to %s", new_value)

    def _on_start_requested(self, mode: str) -> None:
        """Write the chosen mode to config then start the session."""
        cfg = ConfigManager()
        cfg.set("session_mode", mode)
        self._session_mgr.start_session()

    # ── Slot: State changed ────────────────────────────────────────

    def _on_state_changed(self, state_name: str) -> None:
        """React to every SessionManager state transition.

        Responsibilities:
        • Update the timer mode label in FocusTab.
        • Enable/disable the Skip Rest button.
        • Wire pipeline signals on CALIBRATING (first time pipeline exists).
        • Disconnect pipeline signals when the pipeline is torn down.
        """
        ft = self._focus_tab
        sm = self._session_mgr
        prev = self._prev_state_name
        self._prev_state_name = state_name

        if state_name == SessionState.CALIBRATING.name:
            # Pipeline was just created — wire its signals to the UI.
            ft.on_calibration_start()
            self._wire_pipeline()
            ft.set_context_phase("calibrating")

        elif state_name == SessionState.FOCUS.name:
            # A short pause returns straight to FOCUS without passing
            # through CALIBRATING, so nothing else undoes the blanked
            # feed from the PAUSED branch below.  Done before the timer
            # label is set, because the restore path can overwrite it.
            if prev == SessionState.PAUSED.name:
                self._restore_feed_after_pause()

            block = sm.block_number
            mode = sm.mode
            if mode == SESSION_MODE_STRUCTURED:
                ft.on_timer_label(f"Focus Block {block}")
            else:
                # Fluid mode is a continuous session — no block numbering
                # shown to the user; blocks still increment in the DB.
                ft.on_timer_label("Focus — Fluid")
            ft.set_context_phase("focus")
            # set_context_phase hid the rest button; bring it back with
            # the correct enabled state now rather than a second later.
            self._update_rest_button_gate(0)

        elif state_name == SessionState.REST.name:
            ft.on_timer_label("Rest")
            ft.set_context_phase("rest")
            # The pipeline is stopped on REST, so no further frames arrive.
            # Blank the feed rather than leaving the last frame frozen.
            ft.on_camera_released("Camera off — resting")
            # Deferred: REST is announced from inside _enter_rest, before
            # the rest countdown starts.
            QTimer.singleShot(0, self._review_block)

        elif state_name == SessionState.PAUSED.name:
            ft.on_timer_label("Paused")
            ft.set_context_phase("paused")
            # No frames are emitted while paused, so the widget would keep
            # painting the last annotated frame — a frozen overlay that
            # reads as live tracking.  Blank it as REST and AWAY do.
            ft.on_camera_released("Paused — not tracking")

        elif state_name == SessionState.AWAY.name:
            ft.on_timer_label("Away…")
            # During away the dialog takes over; hide the ctx button.
            ft.set_context_phase("calibrating")  # reuse hidden state
            ft.on_camera_released("Camera off — away")

        elif state_name == SessionState.ENDED.name:
            # on_session_ended() resets to idle phase directly.
            pass

    # ── Pipeline lazy wiring ───────────────────────────────────────

    def _wire_pipeline(self) -> None:
        """Connect CVPipelineThread signals to FocusTab slots.

        Called once per focus cycle (each time CALIBRATING is entered),
        because SessionManager creates a fresh CVPipelineThread each time.
        """
        pipeline = self._session_mgr.pipeline
        if pipeline is None:
            log.warning("_wire_pipeline called but pipeline is None.")
            return

        ft = self._focus_tab
        pipeline.frame_ready.connect(ft.on_frame)
        pipeline.calibration_progress.connect(ft.on_calibration_progress)
        pipeline.calibration_complete.connect(self._on_calibration_complete)

        # Route raw events to AlertManager.
        pipeline.event_detected.connect(self._alert_mgr.handle_event)

        log.debug("Pipeline signals wired.")

    def _on_calibration_complete(self, _result) -> None:
        """Calibration finished — switch the video feed to the appropriate state.

        When debug overlay is ON: keep the VideoFeedWidget in live mode
        so annotated tracking frames continue displaying.
        When debug overlay is OFF: switch to the static 'Tracking Active'
        indicator (normal behaviour).
        """
        cfg = ConfigManager()
        if cfg.get_bool("debug_overlay"):
            # Debug mode: don't call on_calibration_done(); feed stays live.
            log.debug("Calibration complete — debug overlay ON, keeping live feed.")
        else:
            self._focus_tab.on_calibration_done()

    def _restore_feed_after_pause(self) -> None:
        """Undo the blanked feed when a short pause resumes.

        Same two cases as _on_debug_overlay_changed: with the overlay ON
        the widget must go back to live mode or push_qimage() drops every
        annotated frame; with it OFF the static 'Tracking Active'
        indicator returns.
        """
        if ConfigManager().get_bool("debug_overlay"):
            self._focus_tab.on_calibration_start()
        else:
            self._focus_tab.on_calibration_done()

    # ── Event review (rest start / session end) ────────────────────

    def _review_block(self) -> None:
        """Rest started — review the events of the block just finished."""
        if not ConfigManager().get_bool("review_events"):
            return
        if self._session_mgr.state != SessionState.REST:
            return  # the rest was already left (e.g. skipped instantly)
        self._ask_labels(
            self._session_mgr.session_id,
            f"Block {self._session_mgr.block_number} review.",
        )

    def _ask_labels(self, session_id: int, status: str) -> bool:
        """Show the review for this session's not-yet-reviewed events.

        Returns False when there was nothing to review.
        """
        # Away is presence bookkeeping, not an engine judgement.
        events = [
            e
            for e in queries.get_events_for_session(session_id)
            if e.event_type != EVENT_AWAY and e.event_id not in self._reviewed_event_ids
        ]
        if not events:
            return False
        dialog = EventReviewDialog(events, status, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            labels = dialog.labels()
            for event_id, label in labels.items():
                queries.set_event_label(event_id, label)
            # Blank rows were seen and deliberately left: don't re-ask.
            self._reviewed_event_ids.update(e.event_id for e in events)
            log.info(
                "Session %d review: %d of %d events labelled.",
                session_id,
                len(labels),
                len(events),
            )
        return True

    def _on_session_ended_review(self) -> None:
        """Queue the review of whatever this session has left.

        Deferred one event-loop turn: session_ended is emitted from inside
        end_session, which only returns the state machine to IDLE after
        the emit.  A modal opened here would hold it in ENDED.
        """
        session_id = self._session_mgr.session_id
        QTimer.singleShot(0, lambda: self._review_session(session_id))

    def _review_session(self, session_id: int) -> None:
        """Ask for labels on the session's not-yet-reviewed events."""
        if ConfigManager().get_bool("review_events"):
            self._ask_labels(session_id, "Session review.")
        self._reviewed_event_ids.clear()

    # ── Slot: User Away ────────────────────────────────────────────

    def _on_user_away(self) -> None:
        """Show the AwayDialog and route the user's choice to SessionManager.

        A modal dialog with Continue / End Session.
        Continue → silent re-calibration (handled by session_manager).
        End      → end_session().
        """
        # Log the event in the Focus tab.
        self._focus_tab.on_event("User Away Detected — session paused.")

        wants_continue = self._focus_tab.show_away_dialog()
        if wants_continue:
            self._session_mgr.continue_from_away()
        else:
            self._session_mgr.end_from_away()

    def _update_rest_button_gate(self, _tick: int) -> None:
        """Show and gate the manual rest button during FOCUS (1 Hz).

        The button is available in both modes — it shares a slot with
        nothing mode-specific, so it cannot mean one thing in Fluid and
        be dead in Structured.  What differs is only the tooltip, since
        Fluid can name the duration it is about to grant and Structured
        cannot without exposing the long-rest cycle counter.

        rest_button_unlock_min is re-read every tick so a Settings
        change applies mid-session, matching AlertManager._get_alert_mode.
        """
        sm = self._session_mgr
        if sm.state != SessionState.FOCUS:
            return

        gate_s = ConfigManager().get_int("rest_button_unlock_min") * 60
        elapsed = sm.current_focus_elapsed_s

        if elapsed < gate_s:
            remaining = int(gate_s - elapsed)
            self._focus_tab.set_rest_button(
                visible=True,
                enabled=False,
                tooltip=(
                    f"Available after {gate_s // 60} min of focus "
                    f"({remaining // 60}:{remaining % 60:02d} to go)"
                ),
            )
            return

        if sm.mode == SESSION_MODE_FLUID:
            rest_s = int(elapsed / FLUID_REST_DIVISOR)
            tooltip = (
                f"Start a rest now — about {rest_s // 60}:{rest_s % 60:02d} "
                f"(focus time ÷ {FLUID_REST_DIVISOR})"
            )
        else:
            tooltip = "End this block early and start your rest"

        self._focus_tab.set_rest_button(visible=True, enabled=True, tooltip=tooltip)

    def _on_rest_complete(self) -> None:
        """Rest period ended — prompt the user before starting the next block.

        Shows a modal 'Rest Complete' dialog.  The user must explicitly
        choose to start the next focus block (which triggers a fresh
        10-second calibration) or end the session entirely.
        """
        ft = self._focus_tab
        ft.on_timer_label("Rest Complete")
        ft.on_event("Rest complete — awaiting confirmation to start next block.")

        wants_continue = ft.show_rest_complete_dialog()
        if wants_continue:
            self._session_mgr.begin_next_block()
        else:
            self._session_mgr.end_session()

    def _on_rest_auto_started(self, block_number: int, rest_seconds: int) -> None:
        """Structured mode hit its block target and rest began automatically.

        Announced, not prompted.  In structured mode the user already
        committed to the work/rest split when they configured it — a modal
        at every boundary re-opens a settled decision, steals keyboard
        focus mid-sentence, and (if the user has stepped away) delays the
        break from starting at all.  Skip Rest remains available for the
        whole rest period as the escape hatch.
        """
        self._alert_mgr.announce_rest_start(block_number, rest_seconds)
        self._focus_tab.on_event(
            f"Focus block {block_number} complete — rest started ({rest_seconds // 60} min)."
        )

    def _on_fatigue_rest_prompt(self) -> None:
        """Handle fatigue event in Fluid mode (prompt or dynamic rest)."""
        cfg = ConfigManager()
        behaviour = cfg.get_str("fatigue_rest_behaviour")

        if behaviour == FATIGUE_REST_DYNAMIC:
            # Auto-rest with notification
            self._alert_mgr.handle_event(
                EventData(
                    event_type=EVENT_FATIGUE,
                    reason="auto_rest",
                    duration=0.0,
                    timestamp=time.time(),
                )
            )
            self._session_mgr.request_rest()
        else:
            # Show prompt dialog
            rest_s = self._session_mgr.current_focus_elapsed_s / FLUID_REST_DIVISOR
            wants_rest = self._focus_tab.show_fatigue_rest_dialog(rest_s)
            if wants_rest:
                self._session_mgr.request_rest()

    # ── Window-management overrides ────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Intercept the close button (X).

        Closing must hide the window to the
        system tray, NOT quit the application.  The app continues
        running; the user quits via Tray → Quit.
        """
        event.ignore()  # prevent actual window destruction
        self.hide()  # remove from taskbar and screen

        # Send a one-shot OS toast the first time the window is hidden.
        if not getattr(self, "_hide_toast_sent", False):
            self._tray.notify(
                APP_NAME,
                "Acuity is still running in the system tray.",
            )
            self._hide_toast_sent = True

        log.debug("Close event intercepted — window hidden to tray.")

    def show_from_tray(self) -> None:
        """Restore the window when the tray icon's Show action fires."""
        self.showNormal()
        self.raise_()
        self.activateWindow()
        log.debug("Window restored from tray.")
