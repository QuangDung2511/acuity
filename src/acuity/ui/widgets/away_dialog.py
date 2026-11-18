# Modal dialog - User Away. Continue / End Session.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   AwayDialog(parent: QWidget) -> QDialog
#
#   Signals:
#     continue_session  – emitted when user clicks "Continue Session"
#     end_session       – emitted when user clicks "End Session"
#
#   Usage:
#     dialog = AwayDialog(parent=self)
#     result = dialog.exec()
#     # result == QDialog.DialogCode.Accepted  → Continue
#     # result == QDialog.DialogCode.Rejected  → End Session
#
# Design intent:
#   "a small, non-intrusive modal pops up:
#    'User Away Detected. Session Paused.'
#    [Continue Session] or [End Session]
#    Clicking Continue → instant silent 10-second re-calibration."
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from acuity.utils.logger import get_logger

log = get_logger(__name__)


class AwayDialog(QDialog):
    """Modal 'User Away Detected' dialog.

    Spec behaviour:
    • Appears when the ABSENT presence state persists beyond the
      configured away threshold.
    • Two actions: Continue (Accepted) → triggers silent re-calibration;
                   End Session (Rejected) → terminates the session.
    • The dialog is NOT closable via the window X button — the user must
      choose one of the two explicit actions.
    """

    continue_session = Signal()
    end_session = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Session Paused")
        # Remove the ? help button; keep title bar but block X close.
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint
        )
        self.setModal(True)
        self.setFixedWidth(360)

        self._build_ui()
        log.debug("AwayDialog created.")

    # ── Layout ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(24, 24, 24, 20)

        # ── Heading ───────────────────────────────────────────────
        heading = QLabel("User Away Detected")
        heading.setProperty("heading", True)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(heading)

        # ── Body text ─────────────────────────────────────────────
        body = QLabel(
            "Session Paused.\n\nYou were away from the camera. "
            "Would you like to continue or end this session?"
        )
        body.setProperty("secondary", True)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(body)

        # ── Buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        end_btn = QPushButton("End Session")
        end_btn.setProperty("role", "danger")
        end_btn.setObjectName("awayEndBtn")
        end_btn.clicked.connect(self._on_end)
        btn_row.addWidget(end_btn)

        continue_btn = QPushButton("Continue Session")
        continue_btn.setProperty("accent", "true")
        continue_btn.setObjectName("awayContinueBtn")
        continue_btn.setDefault(True)
        continue_btn.clicked.connect(self._on_continue)
        btn_row.addWidget(continue_btn)

        root.addLayout(btn_row)

    # ── Slots ──────────────────────────────────────────────────────

    def _on_continue(self) -> None:
        log.info("AwayDialog: user chose Continue Session.")
        self.continue_session.emit()
        self.accept()  # QDialog.DialogCode.Accepted

    def _on_end(self) -> None:
        log.info("AwayDialog: user chose End Session.")
        self.end_session.emit()
        self.reject()  # QDialog.DialogCode.Rejected

    # ── Block Escape / close-button dismissal ──────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        """Prevent Escape from dismissing the dialog without a choice."""
        if event.key() == Qt.Key.Key_Escape:
            return  # swallow — user must press a button
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Block the OS close button — force explicit choice."""
        event.ignore()
