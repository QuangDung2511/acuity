# Modal dialog - Fatigue Detected. Continue / Take Rest.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   FatigueRestDialog(rest_duration_s: float, parent: QWidget) -> QDialog
#
#   Usage:
#     dialog = FatigueRestDialog(rest_duration_s=300, parent=self)
#     result = dialog.exec_and_get_result()  # True if Take Rest
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


class FatigueRestDialog(QDialog):
    """Modal 'Fatigue Detected' dialog.

    Appears when fatigue is detected during a Fluid Mode focus block.
    Two actions: Continue Studying (Rejected) -> ignore and continue focus.
                 Take Rest (Accepted) -> transition to rest state.
    """

    def __init__(self, rest_duration_s: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rest_duration_s = rest_duration_s

        self.setWindowTitle("Fatigue Detected")
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint
        )
        self.setModal(True)
        self.setFixedWidth(360)

        self._build_ui()
        log.debug("FatigueRestDialog created.")

    # ── Layout ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(24, 24, 24, 20)

        # ── Heading ───────────────────────────────────────────────
        heading = QLabel("Fatigue Detected")
        heading.setProperty("heading", True)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(heading)

        # ── Body text ─────────────────────────────────────────────
        m, s = divmod(int(self._rest_duration_s), 60)
        body = QLabel(
            "Signs of fatigue detected. Would you like to take a rest?\n\n"
            f"Rest duration: {m} min {s} sec"
        )
        body.setProperty("secondary", True)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(body)

        # ── Buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        continue_btn = QPushButton("Continue Studying")
        continue_btn.setObjectName("fatigueContinueBtn")
        continue_btn.clicked.connect(self._on_continue)
        btn_row.addWidget(continue_btn)

        rest_btn = QPushButton("Take Rest")
        rest_btn.setProperty("accent", "true")
        rest_btn.setObjectName("fatigueRestBtn")
        rest_btn.setDefault(True)
        rest_btn.clicked.connect(self._on_rest)
        btn_row.addWidget(rest_btn)

        root.addLayout(btn_row)

    # ── Public API ─────────────────────────────────────────────────

    def exec_and_get_result(self) -> bool:
        """Execute the dialog and return True if the user wants to take a rest."""
        result = self.exec()
        return result == QDialog.DialogCode.Accepted

    # ── Slots ──────────────────────────────────────────────────────

    def _on_continue(self) -> None:
        log.info("FatigueRestDialog: user chose Continue Studying.")
        self.reject()  # QDialog.DialogCode.Rejected

    def _on_rest(self) -> None:
        log.info("FatigueRestDialog: user chose Take Rest.")
        self.accept()  # QDialog.DialogCode.Accepted

    # ── Block Escape / close-button dismissal ──────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Prevent Escape from dismissing the dialog without a choice."""
        if event.key() == Qt.Key.Key_Escape:
            return  # swallow — user must press a button
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Block the OS close button — force explicit choice."""
        event.ignore()
