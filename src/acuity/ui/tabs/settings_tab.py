# Settings Tab — alert config, thresholds, camera, audio, theme.
# ─────────────────────────────────────────────────────────────────
# Implements every user-configurable setting: camera device, per-category
# alert behaviour and audio, detection thresholds, desk writing mode, theme
# and the debug overlay.
#
# Public API:
#   SettingsTab(parent=None) → QWidget
#
#   Signals:
#     desk_writing_mode_changed(bool) — emitted when checkbox toggled.
#       MainWindow connects this to pipeline.set_desk_writing_mode().
#     theme_changed(str)              — "dark" | "light"
#
# All values read/written via ConfigManager (QSettings-backed).
#
# Thread: GUI.
#
# The original's phone and posture alert rows, thresholds and audio
# pickers are gone with their detectors.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from acuity.core.camera import CameraManager
from acuity.ui.styles.theme import apply_theme
from acuity.utils.config import ConfigManager
from acuity.utils.constants import (
    ALERT_LOG_ONLY,
    ALERT_NOTIFICATION_AND_SOUND,
    ALERT_NOTIFICATION_ONLY,
    ALERT_SOUND_ONLY,
    BLOCKS_BEFORE_LONG_REST_MAX,
    BLOCKS_BEFORE_LONG_REST_MIN,
    FACE_LOST_THRESHOLD_MAX_S,
    FACE_LOST_THRESHOLD_MIN_S,
    FATIGUE_REST_DYNAMIC,
    FATIGUE_REST_PROMPT,
    HEAD_POSE_THRESHOLD_MAX_S,
    HEAD_POSE_THRESHOLD_MIN_S,
    REST_BUTTON_UNLOCK_MAX_M,
    REST_BUTTON_UNLOCK_MIN_M,
    STRUCTURED_LONG_REST_MAX_M,
    STRUCTURED_LONG_REST_MIN_M,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# Alert option tuples: (display label, config key value)
_ALERT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Log Only", ALERT_LOG_ONLY),
    ("Notification Only", ALERT_NOTIFICATION_ONLY),
    ("Sound Only", ALERT_SOUND_ONLY),
    ("Notification + Sound", ALERT_NOTIFICATION_AND_SOUND),
)


class _NoScrollSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


class _NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class _NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class _NoScrollSpinBox(QSpinBox):
    """QSpinBox that does not respond to scroll-wheel events.

    Prevents accidental value changes when the user scrolls the
    Settings page past a spin box.
    """

    def wheelEvent(self, event):
        event.ignore()


class SettingsTab(QWidget):
    """Settings tab — Tab 2 of the main window.

    Sections (matching ui_sketches/settings_tab.png):
      1. Alert Config — radio buttons (Log Only / Notification Only /
         Sound Only / Notification + Sound) for Distractions, Fatigue
         Warning, Fatigue Alert.  Desk Writing Mode checkbox inside
         Distractions row.
      2. Thresholds — sliders for Trigger A (head pose) and Away.
      3. Audio — custom audio file pickers per alert type.
      4. Camera — device dropdown (populated from CameraManager.list_devices).
      5. Insights — daily focus goal (hours) spin box.
      6. Appearance — theme toggle (Dark / Light).

    All widgets are wired to ConfigManager so values persist across
    restarts.
    """

    # ── Signals ────────────────────────────────────────────────────
    desk_writing_mode_changed = Signal(bool)
    theme_changed = Signal(str)  # "dark" | "light"
    debug_overlay_changed = Signal(bool)  # emitted when debug overlay toggled

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = ConfigManager()
        self._build_ui()
        self._load_from_config()

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the scrollable settings layout."""
        # Outer layout holds only the scroll area.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        # Inner container — all sections go here.
        container = QWidget()
        scroll.setWidget(container)

        root = QVBoxLayout(container)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)

        root.addWidget(self._build_alert_config_group())
        root.addWidget(self._build_timer_group())
        root.addWidget(self._build_threshold_group())
        root.addWidget(self._build_audio_group())
        root.addWidget(self._build_camera_group())
        root.addWidget(self._build_insights_group())
        root.addWidget(self._build_appearance_group())
        root.addWidget(self._build_developer_group())
        root.addStretch(1)

    # ── Section builders ───────────────────────────────────────────

    def _build_alert_config_group(self) -> QGroupBox:
        """Alert Config group: radio buttons for each event category."""
        group = QGroupBox("Alert Config")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # Row helper stores (QButtonGroup, list[QRadioButton]) per category.
        self._alert_groups: dict[str, tuple[QButtonGroup, list[QRadioButton]]] = {}

        # ── Distractions row ────────────────────────────────────
        dist_row = self._make_alert_row("Distractions", "alert_distraction")
        layout.addLayout(dist_row)

        # Desk Writing Mode checkbox — lives inside the Distraction section,
        # since it only changes distraction detection.
        desk_row = QHBoxLayout()
        desk_row.setContentsMargins(120, 0, 0, 0)  # indent under label
        self._desk_writing_cb = QCheckBox("Desk Writing Mode")
        self._desk_writing_cb.setToolTip(
            "Suppress head-pose distraction alerts while writing at your desk."
        )
        self._desk_writing_cb.toggled.connect(self._on_desk_writing_toggled)
        desk_row.addWidget(self._desk_writing_cb)
        desk_row.addStretch(1)
        layout.addLayout(desk_row)

        # ── Separator ────────────────────────────────────────────
        layout.addWidget(self._make_hline())

        # ── Fatigue Warning row ──────────────────────────────────
        layout.addLayout(self._make_alert_row("Fatigue Warning", "alert_fatigue_warning"))

        layout.addWidget(self._make_hline())

        # ── Fatigue Alert row ────────────────────────────────────
        layout.addLayout(self._make_alert_row("Fatigue Alert", "alert_fatigue"))

        layout.addWidget(self._make_hline())

        # ── Event review (labels feed per-detector precision) ────
        self._review_events_cb = QCheckBox("Ask me to rate alerts at each rest")
        self._review_events_cb.setToolTip(
            "At each rest and at session end, mark each alert Correct or "
            "False alarm. The Insights tab then shows how often each "
            "detector is right for you."
        )
        self._review_events_cb.toggled.connect(self._on_review_events_toggled)
        layout.addWidget(self._review_events_cb)

        return group

    def _make_alert_row(self, label: str, config_key: str) -> QHBoxLayout:
        """Build one alert-category row with 4 radio buttons.

        Stores the button group in self._alert_groups[config_key].
        """
        row = QHBoxLayout()
        row.setSpacing(12)

        cat_label = QLabel(label)
        cat_label.setMinimumWidth(110)
        cat_label.setProperty("secondary", True)
        row.addWidget(cat_label)

        btn_group = QButtonGroup(self)
        radios: list[QRadioButton] = []

        for display, value in _ALERT_OPTIONS:
            rb = QRadioButton(display)
            rb.setProperty("_alert_value", value)
            btn_group.addButton(rb)
            radios.append(rb)
            row.addWidget(rb)

        row.addStretch(1)
        self._alert_groups[config_key] = (btn_group, radios)

        # Connect after storing so _load_from_config can set state first.
        btn_group.buttonToggled.connect(
            lambda btn, checked, k=config_key: self._on_alert_toggled(btn, checked, k)
        )

        return row

    def _build_timer_group(self) -> QGroupBox:
        """Timer group: Structured (Pomodoro) settings + Fluid mode fatigue behaviour."""
        group = QGroupBox("Timer Settings")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # Registry for integer spinboxes (config key → widget).
        # Populated by _make_spinbox_row; read by _load_from_config.
        self._spinboxes: dict[str, _NoScrollSpinBox] = {}

        # ── Structured Mode (Pomodoro) section ─────────────────────
        struct_lbl = QLabel("Structured Mode (Pomodoro)")
        struct_lbl.setProperty("secondary", True)
        layout.addWidget(struct_lbl)

        layout.addLayout(
            self._make_spinbox_row(
                "Focus Duration",
                "structured_work_min",
                5,
                480,
                "min",
            )
        )
        layout.addLayout(
            self._make_spinbox_row(
                "Short Rest",
                "structured_rest_min",
                1,
                60,
                "min",
            )
        )
        layout.addLayout(
            self._make_spinbox_row(
                "Long Rest",
                "structured_long_rest_min",
                STRUCTURED_LONG_REST_MIN_M,
                STRUCTURED_LONG_REST_MAX_M,
                "min",
            )
        )
        layout.addLayout(
            self._make_spinbox_row(
                "Blocks before Long Rest",
                "blocks_before_long_rest",
                BLOCKS_BEFORE_LONG_REST_MIN,
                BLOCKS_BEFORE_LONG_REST_MAX,
                "blocks",
            )
        )

        layout.addWidget(self._make_hline())

        # ── Fluid Mode section ────────────────────────────────
        fluid_lbl = QLabel("Fluid Mode")
        fluid_lbl.setProperty("secondary", True)
        layout.addWidget(fluid_lbl)

        fatigue_row = QHBoxLayout()
        fatigue_row.setSpacing(12)

        lbl = QLabel("When fatigue is detected:")
        lbl.setProperty("secondary", True)
        fatigue_row.addWidget(lbl)

        self._fatigue_rest_combo = _NoScrollComboBox()
        self._fatigue_rest_combo.setObjectName("settingsFatigueRestCombo")
        self._fatigue_rest_combo.addItem("Ask me what to do", FATIGUE_REST_PROMPT)
        self._fatigue_rest_combo.addItem("Go to rest automatically", FATIGUE_REST_DYNAMIC)
        self._fatigue_rest_combo.setMinimumWidth(200)
        fatigue_row.addWidget(self._fatigue_rest_combo)
        fatigue_row.addStretch(1)
        layout.addLayout(fatigue_row)

        self._fatigue_rest_combo.currentIndexChanged.connect(self._on_fatigue_rest_changed)

        layout.addWidget(self._make_hline())

        # ── Both Modes section ────────────────────────
        # The manual rest button is shown in both modes, so its unlock
        # gate is not a Fluid setting.  0 disables the gate entirely.
        both_lbl = QLabel("Both Modes")
        both_lbl.setProperty("secondary", True)
        layout.addWidget(both_lbl)

        layout.addLayout(
            self._make_spinbox_row(
                "Rest button unlocks after",
                "rest_button_unlock_min",
                REST_BUTTON_UNLOCK_MIN_M,
                REST_BUTTON_UNLOCK_MAX_M,
                "min",
            )
        )

        return group

    def _make_spinbox_row(
        self, label: str, config_key: str, lo: int, hi: int, suffix: str
    ) -> QHBoxLayout:
        """One integer spinbox row: [label] [spinbox] [unit label] [stretch].

        The unit (e.g. "min" or "blocks") is a plain QLabel placed after
        the spinbox so it sits outside the editable field, keeping the
        input area clean.

        Stores the widget in self._spinboxes[config_key] and wires its
        valueChanged signal to _on_spinbox_changed.
        """
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel(label)
        lbl.setMinimumWidth(190)
        lbl.setProperty("secondary", True)
        row.addWidget(lbl)

        spin = _NoScrollSpinBox()
        spin.setObjectName(f"settingsSpin_{config_key}")
        spin.setRange(lo, hi)
        # No setSuffix — unit label lives outside the editable field.
        spin.setMinimumWidth(80)
        row.addWidget(spin)

        unit_lbl = QLabel(suffix)
        unit_lbl.setProperty("secondary", True)
        row.addWidget(unit_lbl)

        row.addStretch(1)

        self._spinboxes[config_key] = spin
        spin.valueChanged.connect(lambda v, k=config_key: self._on_spinbox_changed(v, k))
        return row

    def _build_threshold_group(self) -> QGroupBox:
        """Threshold group: sliders for Trigger A (head pose) and Away."""
        group = QGroupBox("Thresholds")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self._sliders: dict[str, QSlider] = {}
        self._slider_value_labels: dict[str, QLabel] = {}

        rows = [
            (
                "Trigger A (Head Pose)",
                "head_pose_threshold",
                HEAD_POSE_THRESHOLD_MIN_S,
                HEAD_POSE_THRESHOLD_MAX_S,
            ),
            (
                "Away State",
                "face_lost_threshold",
                FACE_LOST_THRESHOLD_MIN_S,
                FACE_LOST_THRESHOLD_MAX_S,
            ),
        ]

        for display, config_key, lo, hi in rows:
            layout.addLayout(self._make_slider_row(display, config_key, int(lo), int(hi)))

        return group

    def _make_slider_row(self, label: str, config_key: str, lo: int, hi: int) -> QHBoxLayout:
        """One slider row: [label] [min_s] [slider] [max_s] [current_val]."""
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel(label)
        lbl.setMinimumWidth(160)
        lbl.setProperty("secondary", True)
        row.addWidget(lbl)

        lo_lbl = QLabel(f"{lo}s")
        lo_lbl.setProperty("secondary", True)
        lo_lbl.setMinimumWidth(32)
        lo_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lo_lbl)

        slider = _NoScrollSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(lo)
        slider.setMaximum(hi)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        row.addWidget(slider, stretch=1)

        hi_lbl = QLabel(f"{hi}s")
        hi_lbl.setProperty("secondary", True)
        hi_lbl.setMinimumWidth(36)
        row.addWidget(hi_lbl)

        val_lbl = QLabel()
        val_lbl.setMinimumWidth(40)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(val_lbl)

        self._sliders[config_key] = slider
        self._slider_value_labels[config_key] = val_lbl

        # Wire change → config persist + live label update.
        slider.valueChanged.connect(
            lambda v, k=config_key, vl=val_lbl: self._on_slider_changed(v, k, vl)
        )

        return row

    def _build_audio_group(self) -> QGroupBox:
        """Audio group: custom file picker per alert category."""
        group = QGroupBox("Audio")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self._audio_labels: dict[str, QLineEdit] = {}

        rows = [
            ("Distractions", "audio_distraction"),
            ("Fatigue Warning", "audio_fatigue_warning"),
            ("Fatigue Alert", "audio_fatigue"),
        ]

        for display, config_key in rows:
            layout.addLayout(self._make_audio_row(display, config_key))

        return group

    def _make_audio_row(self, label: str, config_key: str) -> QHBoxLayout:
        """One audio row: [label] [path display] [Browse…] [Clear]."""
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel(label)
        lbl.setMinimumWidth(110)
        lbl.setProperty("secondary", True)
        row.addWidget(lbl)

        path_edit = QLineEdit()
        path_edit.setReadOnly(True)
        path_edit.setPlaceholderText("Default system sound")
        row.addWidget(path_edit, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setMinimumWidth(80)
        browse_btn.clicked.connect(
            lambda _, k=config_key, pe=path_edit: self._on_browse_audio(k, pe)
        )
        row.addWidget(browse_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setMinimumWidth(55)
        clear_btn.clicked.connect(lambda _, k=config_key, pe=path_edit: self._on_clear_audio(k, pe))
        row.addWidget(clear_btn)

        self._audio_labels[config_key] = path_edit
        return row

    def _build_camera_group(self) -> QGroupBox:
        """Camera group: device dropdown."""
        group = QGroupBox("Camera")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)

        lbl = QLabel("Input Device:")
        lbl.setProperty("secondary", True)
        layout.addWidget(lbl)

        self._camera_combo = _NoScrollComboBox()
        self._camera_combo.setObjectName("settingsCameraCombo")
        self._camera_combo.setMinimumWidth(280)
        layout.addWidget(self._camera_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMinimumWidth(80)
        refresh_btn.clicked.connect(self._populate_camera_list)
        layout.addWidget(refresh_btn)

        layout.addStretch(1)

        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)

        # Populate immediately.
        self._populate_camera_list()

        return group

    def _build_insights_group(self) -> QGroupBox:
        """Insights group: daily focus goal (hours)."""
        group = QGroupBox("Insights")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)

        lbl = QLabel("Daily Focus Goal:")
        lbl.setProperty("secondary", True)
        layout.addWidget(lbl)

        self._goal_spin = _NoScrollDoubleSpinBox()
        self._goal_spin.setObjectName("settingsGoalSpin")
        self._goal_spin.setRange(0.5, 16.0)
        self._goal_spin.setSingleStep(0.5)
        self._goal_spin.setDecimals(1)
        self._goal_spin.setSuffix(" h")
        self._goal_spin.setMinimumWidth(90)
        layout.addWidget(self._goal_spin)

        layout.addStretch(1)

        self._goal_spin.valueChanged.connect(self._on_goal_changed)

        return group

    def _build_appearance_group(self) -> QGroupBox:
        """Appearance group: Dark / Light theme toggle."""
        group = QGroupBox("Appearance")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)

        lbl = QLabel("Theme:")
        lbl.setProperty("secondary", True)
        layout.addWidget(lbl)

        self._theme_dark_rb = QRadioButton("Dark")
        self._theme_light_rb = QRadioButton("Light")

        self._theme_btn_group = QButtonGroup(self)
        self._theme_btn_group.addButton(self._theme_dark_rb)
        self._theme_btn_group.addButton(self._theme_light_rb)

        layout.addWidget(self._theme_dark_rb)
        layout.addWidget(self._theme_light_rb)
        layout.addStretch(1)

        self._theme_dark_rb.toggled.connect(self._on_theme_toggled)

        return group

    def _build_developer_group(self) -> QGroupBox:
        """Developer / Debug group: overlay toggle + frame-skip."""
        group = QGroupBox("Developer")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # ── Debug overlay checkbox ────────────────────────────────
        overlay_row = QHBoxLayout()
        self._debug_overlay_cb = QCheckBox("Show Debug Overlay on Video Feed")
        self._debug_overlay_cb.setObjectName("settingsDebugOverlayCb")
        self._debug_overlay_cb.setToolTip(
            "Display real-time MediaPipe metrics, engine states, and threshold\n"
            "values on the camera feed.  Shortcut: Ctrl+D.\n\n"
            "When enabled, the video feed remains live during tracking."
        )
        self._debug_overlay_cb.toggled.connect(self._on_debug_overlay_toggled)
        overlay_row.addWidget(self._debug_overlay_cb)
        overlay_row.addStretch(1)

        shortcut_lbl = QLabel("Ctrl+D")
        shortcut_lbl.setProperty("secondary", True)
        overlay_row.addWidget(shortcut_lbl)
        layout.addLayout(overlay_row)

        # Brief description
        desc_lbl = QLabel(
            "  Displays landmarks, timer states, calibration baselines,\n"
            "  and threshold comparisons on the annotated feed."
        )
        desc_lbl.setProperty("secondary", True)
        layout.addWidget(desc_lbl)

        # ── Frame-skip spinner ────────────────────────────────────
        skip_row = QHBoxLayout()
        skip_lbl = QLabel("Overlay frame skip:")
        skip_lbl.setProperty("secondary", True)
        skip_lbl.setMinimumWidth(160)
        skip_row.addWidget(skip_lbl)

        self._debug_frame_skip_spin = QSpinBox()
        self._debug_frame_skip_spin.setObjectName("settingsDebugFrameSkipSpin")
        self._debug_frame_skip_spin.setRange(1, 10)
        self._debug_frame_skip_spin.setValue(1)
        self._debug_frame_skip_spin.setToolTip(
            "1 = draw every frame (default).\n"
            "Higher values skip intermediate frames to reduce CPU usage."
        )
        self._debug_frame_skip_spin.setMinimumWidth(60)
        skip_row.addWidget(self._debug_frame_skip_spin)

        skip_note = QLabel("(1 = every frame)")
        skip_note.setProperty("secondary", True)
        skip_row.addWidget(skip_note)
        skip_row.addStretch(1)
        layout.addLayout(skip_row)

        self._debug_frame_skip_spin.valueChanged.connect(self._on_debug_frame_skip_changed)

        return group

    # ── Helper ─────────────────────────────────────────────────────

    @staticmethod
    def _make_hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    # ── Camera population ──────────────────────────────────────────

    def _populate_camera_list(self) -> None:
        """Scan for camera devices and populate the combo box.

        CameraManager.list_devices() is a static method and is safe to
        call from the GUI thread when no pipeline is running.
        """
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()

        devices = CameraManager.list_devices()

        if not devices:
            self._camera_combo.addItem("No cameras detected", -1)
        else:
            for dev in devices:
                self._camera_combo.addItem(str(dev), dev.index)

        # Restore saved selection.
        saved_index = self._cfg.get_int("camera_index")
        for i in range(self._camera_combo.count()):
            if self._camera_combo.itemData(i) == saved_index:
                self._camera_combo.setCurrentIndex(i)
                break

        self._camera_combo.blockSignals(False)
        log.debug("Camera list populated (%d device(s)).", len(devices))

    # ── Config load ────────────────────────────────────────────────

    def _load_from_config(self) -> None:
        """Read all stored values and apply them to the widgets."""
        cfg = self._cfg

        # ── Alert radios ─────────────────────────────────────────
        for config_key, (_btn_group, radios) in self._alert_groups.items():
            stored = cfg.get_str(config_key)
            for rb in radios:
                if rb.property("_alert_value") == stored:
                    rb.setChecked(True)
                    break

        # ── Desk Writing Mode ─────────────────────────────────────
        self._desk_writing_cb.setChecked(cfg.get_bool("desk_writing_mode"))

        # ── Event review ──────────────────────────────────────────
        self._review_events_cb.setChecked(cfg.get_bool("review_events"))

        # ── Fluid Fatigue Rest Behaviour ─────────────────────────
        fatigue_beh = cfg.get_str("fatigue_rest_behaviour")
        for i in range(self._fatigue_rest_combo.count()):
            if self._fatigue_rest_combo.itemData(i) == fatigue_beh:
                self._fatigue_rest_combo.blockSignals(True)
                self._fatigue_rest_combo.setCurrentIndex(i)
                self._fatigue_rest_combo.blockSignals(False)
                break

        # ── Structured Mode spinboxes ────────────────────────────
        for config_key, spin in self._spinboxes.items():
            val = cfg.get_int(config_key)
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        # ── Sliders ──────────────────────────────────────────────
        for config_key, slider in self._sliders.items():
            val = cfg.get_int(config_key)
            slider.blockSignals(True)
            slider.setValue(val)
            slider.blockSignals(False)
            self._slider_value_labels[config_key].setText(f"{val}s")

        # ── Audio paths ───────────────────────────────────────────
        for config_key, path_edit in self._audio_labels.items():
            stored = cfg.get_str(config_key)
            path_edit.setText(stored)

        # ── Focus goal ────────────────────────────────────────────
        self._goal_spin.blockSignals(True)
        self._goal_spin.setValue(cfg.get_float("daily_focus_goal_h"))
        self._goal_spin.blockSignals(False)

        # ── Theme ─────────────────────────────────────────────────
        stored_theme = cfg.get_str("theme")
        if stored_theme == "light":
            self._theme_light_rb.setChecked(True)
        else:
            self._theme_dark_rb.setChecked(True)

        # ── Debug overlay ─────────────────────────────────
        self._debug_overlay_cb.blockSignals(True)
        self._debug_overlay_cb.setChecked(cfg.get_bool("debug_overlay"))
        self._debug_overlay_cb.blockSignals(False)

        self._debug_frame_skip_spin.blockSignals(True)
        self._debug_frame_skip_spin.setValue(cfg.get_int("debug_overlay_frame_skip"))
        self._debug_frame_skip_spin.blockSignals(False)

    # ── Slot: fluid fatigue rest toggled ───────────────────────────

    def _on_fatigue_rest_changed(self, index: int) -> None:
        value = self._fatigue_rest_combo.itemData(index)
        self._cfg.set("fatigue_rest_behaviour", value)
        log.debug("Fatigue rest config = %s", value)

    # ── Slot: structured mode spinbox changed ─────────────────────────

    def _on_spinbox_changed(self, value: int, config_key: str) -> None:
        """Persist an integer timer setting when its spin box changes."""
        self._cfg.set(config_key, value)
        log.debug("Timer setting: %s = %d", config_key, value)

    # ── Slot: alert radio toggled ──────────────────────────────────

    def _on_alert_toggled(self, button: QRadioButton, checked: bool, config_key: str) -> None:
        if not checked:
            return
        value = button.property("_alert_value")
        if value:
            self._cfg.set(config_key, value)
            log.debug("Alert config: %s = %s", config_key, value)

    # ── Slot: desk writing mode ────────────────────────────────────

    def _on_desk_writing_toggled(self, checked: bool) -> None:
        """Persist and signal the live pipeline."""
        self._cfg.set("desk_writing_mode", checked)
        self.desk_writing_mode_changed.emit(checked)
        log.debug("Desk writing mode: %s", checked)

    # ── Slot: event review ─────────────────────────────────────────

    def _on_review_events_toggled(self, checked: bool) -> None:
        """Persist; MainWindow reads it when each review would open."""
        self._cfg.set("review_events", checked)
        log.debug("Review events: %s", checked)

    # ── Slot: slider changed ───────────────────────────────────────

    def _on_slider_changed(self, value: int, config_key: str, val_label: QLabel) -> None:
        val_label.setText(f"{value}s")
        self._cfg.set(config_key, value)
        log.debug("Threshold: %s = %ds", config_key, value)

    # ── Slot: audio browse / clear ─────────────────────────────────

    def _on_browse_audio(self, config_key: str, path_edit: QLineEdit) -> None:
        start_dir = str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            start_dir,
            "Audio Files (*.wav *.mp3);;All Files (*)",
        )
        if file_path:
            path_edit.setText(file_path)
            self._cfg.set(config_key, file_path)
            log.debug("Audio: %s = %s", config_key, file_path)

    def _on_clear_audio(self, config_key: str, path_edit: QLineEdit) -> None:
        path_edit.clear()
        self._cfg.set(config_key, "")
        log.debug("Audio: %s cleared.", config_key)

    # ── Slot: camera changed ───────────────────────────────────────

    def _on_camera_changed(self, combo_index: int) -> None:
        device_index = self._camera_combo.itemData(combo_index)
        if device_index is not None and device_index >= 0:
            self._cfg.set("camera_index", device_index)
            log.debug("Camera index: %d", device_index)

    # ── Slot: focus goal changed ───────────────────────────────────

    def _on_goal_changed(self, value: float) -> None:
        self._cfg.set("daily_focus_goal_h", value)
        log.debug("Daily focus goal: %.1f h", value)

    # ── Slot: theme toggled ────────────────────────────────────────

    def _on_theme_toggled(self, dark_selected: bool) -> None:
        mode = "dark" if dark_selected else "light"
        self._cfg.set("theme", mode)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)  # type: ignore[arg-type]
        self.theme_changed.emit(mode)
        log.debug("Theme changed to: %s", mode)

    # ── Slot: debug overlay toggled ───────────────────────

    def _on_debug_overlay_toggled(self, checked: bool) -> None:
        """Persist debug overlay enabled state and notify MainWindow."""
        self._cfg.set("debug_overlay", checked)
        self.debug_overlay_changed.emit(checked)
        log.debug("Debug overlay: %s", checked)

    # ── Slot: debug frame skip changed ────────────────────────────

    def _on_debug_frame_skip_changed(self, value: int) -> None:
        """Persist the frame-skip setting (renderer picks it up on next creation)."""
        self._cfg.set("debug_overlay_frame_skip", value)
        log.debug("Debug overlay frame skip: %d", value)

    # ── Public: sync debug checkbox without re-emitting signal ────

    def sync_debug_checkbox(self, enabled: bool) -> None:
        """Update the debug overlay checkbox without emitting debug_overlay_changed.

        Called by MainWindow when Ctrl+D toggles the overlay externally,
        so the Settings UI stays in sync without causing a signal loop.
        """
        self._debug_overlay_cb.blockSignals(True)
        self._debug_overlay_cb.setChecked(enabled)
        self._debug_overlay_cb.blockSignals(False)
