# QPlainTextEdit-based scrolling live event log.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   EventLog()
#   .append_event(text: str)  – prepend HH:MM:SS timestamp and append
#   .clear_log()              – erase all entries
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import datetime

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class EventLog(QWidget):
    """Read-only, auto-scrolling live event log for the Focus tab.

    Layout:
      • "Live Events" heading label
      • QPlainTextEdit (read-only, monospaced, dark background)

    Each call to append_event() prefixes the entry with a wall-clock
    timestamp, for example:
        [14:02:15] Distraction_Pose — 15.2s

    The widget auto-scrolls to the newest entry after each append.
    """

    _TIMESTAMP_FMT = "%H:%M:%S"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── Section label ─────────────────────────────────────────
        heading = QLabel("Live Events")
        heading.setProperty("secondary", True)
        heading.setObjectName("eventLogHeading")
        root.addWidget(heading)

        # ── Log pane ──────────────────────────────────────────────
        self._pane = QPlainTextEdit()
        self._pane.setObjectName("eventLogPane")
        self._pane.setReadOnly(True)
        self._pane.setMaximumBlockCount(500)  # cap memory; oldest lines drop
        self._pane.setPlaceholderText("Events will appear here once a session starts.")
        self._pane.setStyleSheet(
            "QPlainTextEdit#eventLogPane {"
            "  background-color: #1A1A1F;"
            "  color: #E2E2E7;"
            "  border: 1px solid #2C2C34;"
            "  border-radius: 6px;"
            "  padding: 8px;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-size: 12px;"
            "  selection-background-color: #5B8AF5;"
            "}"
        )
        self._pane.setMinimumHeight(120)
        root.addWidget(self._pane)

    # ── Public API ─────────────────────────────────────────────────

    def append_event(self, text: str) -> None:
        """Append a timestamped event line and auto-scroll to bottom.

        Args:
            text: Event description, e.g. "Distraction_Pose — 15.2s".
        """
        ts = datetime.now().strftime(self._TIMESTAMP_FMT)
        line = f"[{ts}] {text}"
        self._pane.appendPlainText(line)
        # Scroll to the bottom so the newest entry is always visible.
        self._pane.moveCursor(QTextCursor.MoveOperation.End)

    def clear_log(self) -> None:
        """Erase all log entries."""
        self._pane.clear()
