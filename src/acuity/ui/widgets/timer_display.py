# Large timer label for the Focus tab.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   TimerDisplay()
#   .set_time(total_seconds: int)  – update the displayed time
#   .set_label(text: str)          – update the mode label above the clock
#   .reset()                       – show "--:--" in idle state
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class TimerDisplay(QWidget):
    """Large timer widget for the Focus tab.

    Layout (top to bottom):
      • mode label  – e.g. "Focus Block 1", "Short Rest", "Calibrating…"
      • clock face  – MM:SS or H:MM:SS in large monospaced text

    The clock face itself is always rendered at a consistent width so
    the layout does not jump as digits change.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ── Mode label ────────────────────────────────────────────
        self._mode_label = QLabel("Ready")
        self._mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mode_label.setProperty("secondary", True)
        self._mode_label.setObjectName("timerModeLabel")
        root.addWidget(self._mode_label)

        # ── Clock face ────────────────────────────────────────────
        self._clock = QLabel("--:--")
        self._clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._clock.setObjectName("timerClock")
        self._clock.setStyleSheet(
            "QLabel#timerClock {"
            "  font-family: 'Segoe UI', 'Inter', monospace;"
            "  font-size: 64px;"
            "  font-weight: 300;"
            "  letter-spacing: 4px;"
            "  color: #E2E2E7;"
            "}"
        )
        root.addWidget(self._clock)

    # ── Public API ─────────────────────────────────────────────────

    def set_time(self, total_seconds: int) -> None:
        """Update the clock face.

        Args:
            total_seconds: Non-negative integer. Formatted as MM:SS for
                           values < 1 hour, otherwise H:MM:SS.
        """
        total_seconds = max(0, int(total_seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            text = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            text = f"{minutes:02d}:{seconds:02d}"
        self._clock.setText(text)

    def set_label(self, text: str) -> None:
        """Update the mode label (e.g. 'Focus Block 2', 'Short Rest')."""
        self._mode_label.setText(text)

    def reset(self) -> None:
        """Return to idle / pre-session state."""
        self._clock.setText("--:--")
        self._mode_label.setText("Ready")
