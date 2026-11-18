# Modal dialog - Rest Complete. Start Next Block / End Session.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   RestCompleteDialog(parent: QWidget) -> QDialog
#
#   Usage:
#     dialog = RestCompleteDialog(parent=self)
#     if dialog.exec() == QDialog.DialogCode.Accepted:
#         session_mgr.begin_next_block()
#     else:
#         session_mgr.end_session()
#
# Appears at the end of every rest period (natural completion or skip)
# before the next focus cycle begins.  Forces an explicit user action
# so that the next calibration does not start while the user is still
# off-camera.
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from PySide6.QtCore import Qt
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


class RestCompleteDialog(QDialog):
    """Modal 'Rest Complete' dialog shown when a rest period ends.

    Two actions:
      End Session      (Rejected) → terminate the session.
      Start Next Block (Accepted) → re-calibrate and begin next focus block.

    The dialog cannot be dismissed via Escape or the window close
    button — the user must make an explicit choice.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Rest Complete")
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint
        )
        self.setModal(True)
        self.setFixedWidth(380)

        self._build_ui()
        log.debug("RestCompleteDialog created.")

    # ── Layout ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(24, 24, 24, 20)

        # ── Heading ───────────────────────────────────────────────
        heading = QLabel("Rest Complete")
        heading.setProperty("heading", True)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(heading)

        # ── Body text ─────────────────────────────────────────────
        body = QLabel(
            "Your rest period is over.\n\n"
            "Ready to start the next focus block?\n"
            "A short calibration will run before tracking begins."
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
        end_btn.setObjectName("restCompleteEndBtn")
        end_btn.clicked.connect(self._on_end)
        btn_row.addWidget(end_btn)

        start_btn = QPushButton("Start Next Block")
        start_btn.setProperty("accent", "true")
        start_btn.setObjectName("restCompleteStartBtn")
        start_btn.setDefault(True)
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(start_btn)

        root.addLayout(btn_row)

    # ── Slots ──────────────────────────────────────────────────────

    def _on_end(self) -> None:
        log.info("RestCompleteDialog: user chose End Session.")
        self.reject()  # QDialog.DialogCode.Rejected

    def _on_start(self) -> None:
        log.info("RestCompleteDialog: user chose Start Next Block.")
        self.accept()  # QDialog.DialogCode.Accepted

    # ── Block Escape / close-button dismissal ──────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        """Prevent Escape from dismissing the dialog without a choice."""
        if event.key() == Qt.Key.Key_Escape:
            return  # swallow — user must press a button
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Block the OS close button — force explicit choice."""
        event.ignore()
