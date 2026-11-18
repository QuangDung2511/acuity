# Timer display, calibration video feed, live event log.
# ─────────────────────────────────────────────────────────────────
# Public API (consumed by MainWindow wiring):
#
#   FocusTab()
#   ── Slots (driven by SessionManager / CVPipelineThread signals) ──
#   .on_frame(img: QImage)              – forward frame to video feed
#   .on_calibration_start()             – switch feed to live mode
#   .on_calibration_progress(pct: int)  – update calibration countdown
#   .on_calibration_done()              – switch feed to Tracking Active
#   .on_tick(seconds: int)              – update timer (focus elapsed)
#   .on_rest_tick(seconds: int)         – update timer (rest remaining)
#   .on_timer_label(text: str)          – update mode label
#   .on_event(text: str)                – append to event log
#   .on_session_ended()                 – reset all widgets
#   .show_away_dialog() -> bool         – show modal; True=Continue
#   .set_context_phase(phase: str)      – drive contextual right button
#       phase values: "focus" | "paused" | "rest" | "calibrating" | "idle"
#   .sync_flip_checkbox(on: bool)       – set mirror switch, no signal
#   ── Signals (wired by MainWindow to session_manager) ─────────────
#   .start_requested(mode: str)         – "Structured" | "Fluid"
#   .stop_requested
#   .pause_requested
#   .resume_requested
#   .skip_rest_requested
#   .flip_camera_changed(on: bool)      – mirror-camera switch toggled
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import warnings

from PySide6.QtCore import Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from acuity.ui.widgets.away_dialog import AwayDialog
from acuity.ui.widgets.event_log import EventLog
from acuity.ui.widgets.fatigue_rest_dialog import FatigueRestDialog
from acuity.ui.widgets.rest_complete_dialog import RestCompleteDialog
from acuity.ui.widgets.timer_display import TimerDisplay
from acuity.ui.widgets.video_feed import VideoFeedWidget
from acuity.utils.constants import SESSION_MODE_FLUID, SESSION_MODE_STRUCTURED
from acuity.utils.logger import get_logger

log = get_logger(__name__)


class FocusTab(QWidget):
    """Focus tab — Tab 0 of the main window.

    Layout (matches ui_sketches/focus_tab.png):
      ┌──────────────────────────────┬─────────────────────┐
      │                              │  Mode: [combo]      │
      │   VideoFeedWidget            │  ─────────────────  │
      │   (calibration live feed     │  logs               │
      │    OR 'Tracking Active')     │  (event log)        │
      │                              │  ─────────────────  │
      │                              │       00:00         │
      │                              │      [Take a Rest]  │
      │                              │  [End]  [ctx_btn]   │
      └──────────────────────────────┴─────────────────────┘

    Button layout by phase (right slot only):
      idle        → [End disabled]  [Start Session]
      calibrating → [End Session ]  [hidden]
      focus       → [End Session ]  [Pause]
      paused      → [End Session ]  [Resume]
      rest        → [End Session ]  [Skip Rest]

    The "Take a Rest" button above the action row is separate from
    the contextual slot: it is shown only during FOCUS, in both
    session modes, and its enabled state is driven by MainWindow
    (which owns the unlock gate).  set_context_phase() hides it for
    every phase, so MainWindow re-shows it when entering FOCUS.

    The tab owns no session logic. It only:
      • emits Qt signals when the user clicks buttons, and
      • exposes slots for external controllers to drive UI state.
    """

    # ── Signals ────────────────────────────────────────────────────
    start_requested = Signal(str)  # arg: "Structured" | "Fluid"
    stop_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    skip_rest_requested = Signal()
    rest_requested = Signal()  # manual rest, both modes
    flip_camera_changed = Signal(bool)  # mirror camera preview

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._paused = False
        self._session_active = False
        self._build_ui()
        self._set_session_active(False)  # initialise button states

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Two-column layout matching focus_tab.png sketch."""
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # Left column — video feed (dominant, stretches to fill height)
        self._video_feed = VideoFeedWidget()
        root.addWidget(self._video_feed, stretch=3)

        # Right column — stacked controls
        root.addLayout(self._make_right_column(), stretch=1)

    def _make_right_column(self) -> QVBoxLayout:
        """Right panel: mode selector / event log / timer / action buttons."""
        col = QVBoxLayout()
        col.setSpacing(10)

        # ── Mode selector (top of right column) ──────────────────
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)

        mode_label = QLabel("Mode:")
        mode_label.setProperty("secondary", True)
        mode_row.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("focusTabModeCombo")
        self._mode_combo.addItem("Structured (Pomodoro)", SESSION_MODE_STRUCTURED)
        self._mode_combo.addItem("Fluid (Stopwatch)", SESSION_MODE_FLUID)
        mode_row.addWidget(self._mode_combo, stretch=1)
        col.addLayout(mode_row)

        # ── Mirror-camera switch ─────────────────────────────────
        # The flip is applied to the raw frame before analysis, so it
        # must stay fixed for the lifetime of a session — otherwise the
        # calibrated yaw / lean baselines would have the wrong sign.
        # Locked during a session alongside the mode selector.
        self._flip_cb = QCheckBox("Mirror camera")
        self._flip_cb.setObjectName("focusTabFlipCb")
        self._flip_cb.setToolTip(
            "Show the camera as a mirror (selfie view).\n"
            "Can only be changed between sessions — the calibration "
            "baselines depend on the image orientation."
        )
        self._flip_cb.toggled.connect(self.flip_camera_changed)
        col.addWidget(self._flip_cb)

        # ── Event log (fills remaining vertical space) ────────────
        self._event_log = EventLog()
        self._event_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        col.addWidget(self._event_log, stretch=1)

        # ── Timer display ─────────────────────────────────────────
        self._timer = TimerDisplay()
        col.addWidget(self._timer)

        # Manual rest button — sits in the slot the old "Hide Timer"
        # toggle occupied, right-aligned under the clock.  Kept out of
        # the action row below because that row's right slot is already
        # multiplexed by set_context_phase(); this button's visibility
        # and enabled state are driven independently by MainWindow.
        rest_row = QHBoxLayout()
        rest_row.setContentsMargins(0, 0, 0, 0)
        rest_row.addStretch()

        self._rest_btn = QPushButton("Take a Rest")
        self._rest_btn.setObjectName("focusTabRestBtn")
        self._rest_btn.setProperty("accent", "true")
        self._rest_btn.setVisible(False)
        self._rest_btn.clicked.connect(self.rest_requested)
        rest_row.addWidget(self._rest_btn)

        col.addLayout(rest_row)

        # ── Action row: [End Session] [contextual right slot] ─────
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self._stop_btn = QPushButton("End Session")
        self._stop_btn.setObjectName("focusTabStopBtn")
        self._stop_btn.setProperty("role", "danger")
        self._stop_btn.clicked.connect(self.stop_requested)
        action_row.addWidget(self._stop_btn)

        # Contextual right-slot button — wired by set_context_phase().
        self._ctx_btn = QPushButton()
        self._ctx_btn.setObjectName("focusTabCtxBtn")
        action_row.addWidget(self._ctx_btn)

        col.addLayout(action_row)
        return col

    # ── Button state management ────────────────────────────────────

    # Phase → (button label, visible)
    _CTX_PHASES: dict[str, tuple[str, bool]] = {
        "idle": ("Start Session", True),
        "calibrating": ("", False),
        "focus": ("Pause", True),
        "paused": ("Resume", True),
        "rest": ("Skip Rest", True),
    }

    def _set_session_active(self, active: bool) -> None:
        """Enable/disable widgets that must be locked during a session."""
        self._session_active = active
        self._mode_combo.setEnabled(not active)
        self._flip_cb.setEnabled(not active)
        self._stop_btn.setEnabled(active)

    def sync_flip_checkbox(self, enabled: bool) -> None:
        """Set the mirror switch without emitting flip_camera_changed.

        Used by MainWindow to seed the switch from persisted config.
        """
        self._flip_cb.blockSignals(True)
        self._flip_cb.setChecked(enabled)
        self._flip_cb.blockSignals(False)

    def set_rest_button(self, visible: bool, enabled: bool = True, tooltip: str = "") -> None:
        """Drive the manual rest button.

        Called by MainWindow, which owns the unlock gate (the tab has no
        access to session state or config).  set_context_phase() hides
        the button for every phase, so a caller entering FOCUS must call
        this afterwards to bring it back.
        """
        self._rest_btn.setVisible(visible)
        self._rest_btn.setEnabled(enabled)
        self._rest_btn.setToolTip(tooltip)

    def set_context_phase(self, phase: str) -> None:
        """Drive the contextual right-slot button for the given phase.

        phase must be one of: "idle", "calibrating", "focus", "paused", "rest".
        """
        label, visible = self._CTX_PHASES.get(phase, ("", False))

        # Hide the manual rest button on every phase change; MainWindow
        # re-shows it when the phase is FOCUS.  This also resets it for
        # free via on_session_ended -> set_context_phase("idle").
        self._rest_btn.setVisible(False)

        # With no connection, PyQt6 raised here and PySide6 instead warns
        # (RuntimeWarning "Failed to disconnect"); both mean "nothing to do".
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Failed to disconnect", category=RuntimeWarning
            )
            try:
                self._ctx_btn.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass  # no connection yet

        self._ctx_btn.setText(label)
        self._ctx_btn.setVisible(visible)

        if phase == "idle":
            self._ctx_btn.setProperty("accent", "true")
            self._ctx_btn.clicked.connect(self._on_start_clicked)
        elif phase in ("focus", "paused"):
            self._ctx_btn.setProperty("accent", None)
            self._ctx_btn.clicked.connect(self._on_pause_clicked)
        elif phase == "rest":
            self._ctx_btn.setProperty("accent", None)
            self._ctx_btn.clicked.connect(self.skip_rest_requested)

        # Force Qt to re-evaluate style sheet properties.
        self._ctx_btn.style().unpolish(self._ctx_btn)
        self._ctx_btn.style().polish(self._ctx_btn)

    # ── Button click handlers ──────────────────────────────────────

    def _on_start_clicked(self) -> None:
        mode = self._mode_combo.currentData()
        self._set_session_active(True)
        self.start_requested.emit(mode)

    def _on_pause_clicked(self) -> None:
        if self._paused:
            self._paused = False
            self.resume_requested.emit()
        else:
            self._paused = True
            self.pause_requested.emit()

    # ── Public slots (driven by session_manager / cv_pipeline) ────

    def on_frame(self, image: QImage) -> None:
        """Receive a QImage frame during calibration.

        Connected to CVPipelineThread.frame_ready (which emits QImage).
        """
        self._video_feed.push_qimage(image)

    def on_calibration_start(self) -> None:
        """Called when calibration phase begins."""
        self._video_feed.show_calibration_view()
        self._timer.set_label("Calibrating…")
        self._timer.set_time(10)

    def on_calibration_progress(self, percent: int) -> None:
        """Called each second during calibration (0–100).

        Counts down from 10 s.
        """
        remaining = max(0, round(10 * (1 - percent / 100)))
        self._timer.set_time(remaining)

    def on_calibration_done(self) -> None:
        """Called when calibration finishes; feed switches to indicator."""
        self._video_feed.show_tracking_active()

    def on_tick(self, seconds: int) -> None:
        """Called on each focus_tick; updates the clock with elapsed seconds."""
        self._timer.set_time(seconds)

    def on_rest_tick(self, seconds: int) -> None:
        """Called on each rest_tick; updates the clock with remaining seconds."""
        self._timer.set_time(seconds)

    def on_camera_released(self, message: str = "Camera off") -> None:
        """Blank the video feed when the pipeline is torn down."""
        self._video_feed.show_camera_off(message)

    def on_timer_label(self, text: str) -> None:
        """Update the mode label above the clock (e.g. 'Focus Block 2')."""
        self._timer.set_label(text)

    def on_event(self, text: str) -> None:
        """Append an event string to the live event log."""
        self._event_log.append_event(text)

    def on_session_ended(self) -> None:
        """Reset all widgets to idle state after session ends."""
        self._set_session_active(False)
        self._paused = False
        self._timer.reset()
        self._video_feed.show_idle()
        self._event_log.clear_log()
        self.set_context_phase("idle")

    def show_away_dialog(self) -> bool:
        """Show the modal AwayDialog.

        Returns:
            True  if the user chose Continue Session.
            False if the user chose End Session.
        """
        dialog = AwayDialog(parent=self)
        accepted = dialog.exec() == AwayDialog.DialogCode.Accepted
        log.info("AwayDialog result: %s", "Continue" if accepted else "End")
        return accepted

    def show_fatigue_rest_dialog(self, rest_duration_s: float) -> bool:
        """Show the fatigue rest prompt dialog. Returns True if user wants rest."""
        dialog = FatigueRestDialog(rest_duration_s, parent=self)
        return dialog.exec_and_get_result()

    def show_rest_complete_dialog(self) -> bool:
        """Show the rest-complete prompt dialog.

        Returns:
            True  if the user chose Start Next Block.
            False if the user chose End Session.
        """
        dialog = RestCompleteDialog(parent=self)
        accepted = dialog.exec() == RestCompleteDialog.DialogCode.Accepted
        log.info("RestCompleteDialog result: %s", "Start" if accepted else "End")
        return accepted
