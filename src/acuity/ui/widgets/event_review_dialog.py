# Modal dialog - block and session review: mark each event Correct / False alarm.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   EventReviewDialog(events: list[Event], status_text: str, parent)
#   .labels() -> dict[int, str]   event_id → EVENT_LABEL_* for rows the
#                                 user marked; unmarked rows are absent
#
#   Usage:
#     dialog = EventReviewDialog(events, "Block 2 review.")
#     if dialog.exec() == QDialog.DialogCode.Accepted:
#         for event_id, label in dialog.labels().items():
#             queries.set_event_label(event_id, label)
#
# Meant for each rest start (the block just finished) and session end
# (whatever is left).  Short gaps keep the user's memory of each alert
# fresh without interrupting focus.  The labels are the user's ground
# truth: they say whether each alert the live engine raised was real.
# Nothing opens the dialog yet; ticket 04 shows it behind the
# review_events setting.  Labelling is optional
# per row.  An honest "don't remember" is better left blank than
# guessed, so unmarked rows stay NULL and Skip saves nothing.
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from acuity.db.models import Event
from acuity.utils.constants import EVENT_LABEL_CORRECT, EVENT_LABEL_FALSE_ALARM
from acuity.utils.logger import get_logger

log = get_logger(__name__)


class EventReviewDialog(QDialog):
    """Lists a session's events with a Correct / False alarm choice each.

    Save (Accepted) → caller writes ``labels()``.
    Skip (Rejected) → nothing is written; Escape and the close button
                      also skip, since leaving rows unlabelled is safe.
    """

    _MAX_LIST_HEIGHT = 360  # px; ~12 rows before the list scrolls

    def __init__(
        self,
        events: list[Event],
        status_text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Session Alerts")
        self.setModal(True)
        self.setMinimumWidth(480)

        # event_id → (correct button, false-alarm button)
        self._rows: dict[int, tuple[QRadioButton, QRadioButton]] = {}
        self._build_ui(events, status_text)
        log.debug("EventReviewDialog created (%d events).", len(events))

    # ── Layout ─────────────────────────────────────────────────────

    def _build_ui(self, events: list[Event], status_text: str) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(24, 24, 24, 20)

        heading = QLabel("Were these alerts right?")
        heading.setProperty("heading", True)
        root.addWidget(heading)

        intro = QLabel(
            "Mark each event Correct or False alarm. Leave a row blank "
            "if you don't remember. A blank row is better than a guess."
        )
        intro.setProperty("secondary", True)
        intro.setWordWrap(True)
        root.addWidget(intro)

        # ── Event rows (scrolls: a long session can log dozens) ───
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 12, 0)  # clear of the scrollbar
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        for row, event in enumerate(events):
            grid.addWidget(QLabel(self._describe(event)), row, 0)
            correct = QRadioButton("Correct")
            false_alarm = QRadioButton("False alarm")
            group = QButtonGroup(self)  # one exclusive pair per row
            group.addButton(correct)
            group.addButton(false_alarm)
            correct.setChecked(event.label == EVENT_LABEL_CORRECT)
            false_alarm.setChecked(event.label == EVENT_LABEL_FALSE_ALARM)
            grid.addWidget(correct, row, 1)
            grid.addWidget(false_alarm, row, 2)
            self._rows[event.event_id] = (correct, false_alarm)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(grid_host)
        # Let the dialog's own background show through the list.
        scroll.viewport().setAutoFillBackground(False)
        grid_host.setAutoFillBackground(False)
        # A scroll area sizes itself to almost nothing by default; show
        # every row up to a cap, and scroll only past it.
        hint = grid_host.sizeHint()
        scroll.setMinimumHeight(min(hint.height(), self._MAX_LIST_HEIGHT))
        scroll.setMinimumWidth(hint.width() + scroll.verticalScrollBar().sizeHint().width())
        root.addWidget(scroll)

        status = QLabel(status_text)
        status.setProperty("secondary", True)
        status.setWordWrap(True)
        root.addWidget(status)

        # ── Buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        skip_btn = QPushButton("Skip")
        skip_btn.setObjectName("eventReviewSkipBtn")
        skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(skip_btn)

        save_btn = QPushButton("Save Labels")
        save_btn.setProperty("accent", "true")
        save_btn.setObjectName("eventReviewSaveBtn")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        root.addLayout(btn_row)

    @staticmethod
    def _describe(event: Event) -> str:
        clock = datetime.fromisoformat(event.timestamp).strftime("%H:%M:%S")
        reason = f" ({event.reason})" if event.reason else ""
        return f"{clock}   {event.event_type.replace('_', ' ')}{reason}"

    # ── Result ─────────────────────────────────────────────────────

    def labels(self) -> dict[int, str]:
        """Return event_id → label for every row the user marked."""
        result: dict[int, str] = {}
        for event_id, (correct, false_alarm) in self._rows.items():
            if correct.isChecked():
                result[event_id] = EVENT_LABEL_CORRECT
            elif false_alarm.isChecked():
                result[event_id] = EVENT_LABEL_FALSE_ALARM
        return result
