# Custom QWidget - animated circular Focus Goal gauge.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   CircularProgress(goal_hours: float = 8.0, parent=None)
#   .set_goal(hours: float)            – update the daily goal (hours)
#   .set_value(seconds: float)         – push today's focus seconds;
#                                        arc animates to new position
#   .reset()                           – snap back to 0 (no animation)
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from PySide6.QtCore import (
    Property,  # type: ignore[attr-defined]
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
)
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

# ── Palette (mirrors dark_theme.qss tokens) ───────────────────────
_BG = QColor("#0F0F11")
_TRACK = QColor("#2C2C34")  # border / subtle track ring
_FILL = QColor("#5B8AF5")  # accent — normal progress
_FILL_DONE = QColor("#4CAF82")  # success — goal reached / exceeded
_TEXT_PRI = QColor("#E2E2E7")
_TEXT_SEC = QColor("#888895")

_TRACK_WIDTH = 10  # px — ring stroke width
_FILL_WIDTH = 10  # px — fill arc stroke width
_CAP_STYLE = Qt.PenCapStyle.RoundCap
_START_ANGLE = 90 * 16  # QPainter uses 1/16 degree units; 90° = 12 o'clock
_FULL_SPAN = 360 * 16  # full circle


class CircularProgress(QWidget):
    """Animated arc widget visualising today's focus hours vs. a daily goal.

    The arc sweeps clockwise from 12 o'clock.  It transitions smoothly
    whenever :meth:`set_value` is called, using a 600 ms ease-in-out
    animation on the internal ``_animated_fraction`` property.

    The centre of the widget shows:
      • the formatted current value  (large, primary colour)
      • "/ <goal> h goal"            (small, secondary colour)

    When ``value >= goal`` the fill arc turns green (success token).
    """

    # ── Qt property wired to QPropertyAnimation ───────────────────

    def _get_anim_frac(self) -> float:
        return self.__anim_frac

    def _set_anim_frac(self, value: float) -> None:
        self.__anim_frac = value
        self.update()  # trigger paintEvent

    _animated_fraction = Property(float, fget=_get_anim_frac, fset=_set_anim_frac)

    # ── Construction ──────────────────────────────────────────────

    def __init__(
        self,
        goal_hours: float = 8.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.__anim_frac: float = 0.0  # raw backing store for the Q_PROPERTY
        self._goal_seconds: float = max(0.01, goal_hours * 3600.0)
        self._current_seconds: float = 0.0

        # Animation object — reused for every set_value() call.
        self._animation = QPropertyAnimation(self, b"_animated_fraction", self)
        self._animation.setDuration(600)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # Fixed square size; let the parent layout scale it.
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    # ── Public API ────────────────────────────────────────────────

    def set_goal(self, hours: float) -> None:
        """Update the daily goal in hours.

        The displayed fraction is recalculated immediately.
        """
        self._goal_seconds = max(0.01, hours * 3600.0)
        self._sync_animation(animated=False)

    def set_value(self, seconds: float) -> None:
        """Push today's accumulated focus seconds and animate the arc.

        Args:
            seconds: Non-negative float; values beyond the goal are clamped
                     visually to a full circle but the raw number is shown.
        """
        self._current_seconds = max(0.0, seconds)
        self._sync_animation(animated=True)

    def reset(self) -> None:
        """Snap the arc back to zero without animation (e.g. new day)."""
        self._current_seconds = 0.0
        self._animation.stop()
        self.__anim_frac = 0.0
        self.update()

    # ── Internal helpers ──────────────────────────────────────────

    def _target_fraction(self) -> float:
        """Fraction of goal [0.0 … 1.0], clamped to 1.0 for display."""
        raw = self._current_seconds / self._goal_seconds
        return min(raw, 1.0)

    def _sync_animation(self, *, animated: bool) -> None:
        """Start (or snap) the animation to match the current target."""
        target = self._target_fraction()
        if not animated:
            self._animation.stop()
            self.__anim_frac = target
            self.update()
            return

        self._animation.stop()
        self._animation.setStartValue(self.__anim_frac)
        self._animation.setEndValue(target)
        self._animation.start()

    # ── paintEvent ────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        side = min(self.width(), self.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Centre the square drawing area.
        ox = (self.width() - side) / 2
        oy = (self.height() - side) / 2
        painter.translate(ox, oy)

        margin = _TRACK_WIDTH
        rect = QRectF(margin, margin, side - 2 * margin, side - 2 * margin)

        # ── Track ring ────────────────────────────────────────────
        track_pen = QPen(_TRACK, _TRACK_WIDTH, Qt.PenStyle.SolidLine, _CAP_STYLE)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0, _FULL_SPAN)

        # ── Fill arc ──────────────────────────────────────────────
        fraction = self.__anim_frac
        if fraction > 0.0:
            done = self._current_seconds >= self._goal_seconds
            fill_color = _FILL_DONE if done else _FILL
            fill_pen = QPen(fill_color, _FILL_WIDTH, Qt.PenStyle.SolidLine, _CAP_STYLE)
            painter.setPen(fill_pen)
            span = int(-fraction * _FULL_SPAN)  # negative = clockwise
            painter.drawArc(rect, _START_ANGLE, span)

        # ── Centre text ───────────────────────────────────────────
        self._paint_centre_text(painter, side)

        painter.end()

    def _paint_centre_text(self, painter: QPainter, side: float) -> None:
        """Draw the two-line text in the centre of the ring."""
        cy = side / 2

        # Primary line: current value formatted as H h MM m
        primary_text = self._format_seconds(self._current_seconds)

        # Secondary line: goal
        goal_h = self._goal_seconds / 3600.0
        if goal_h == int(goal_h):
            goal_str = f"/ {int(goal_h)} h goal"
        else:
            goal_str = f"/ {goal_h:.1f} h goal"

        # Primary font — ~22 % of widget side
        pri_size = max(14, int(side * 0.22))
        font = painter.font()
        font.setFamily("Segoe UI")
        font.setPixelSize(pri_size)
        font.setWeight(font.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(_TEXT_PRI))

        line_gap = int(side * 0.06)
        pri_rect = QRectF(0, cy - pri_size - line_gap / 2, side, pri_size * 1.2)
        painter.drawText(pri_rect, Qt.AlignmentFlag.AlignCenter, primary_text)

        # Secondary font
        sec_size = max(10, int(side * 0.13))
        font.setPixelSize(sec_size)
        font.setWeight(font.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QPen(_TEXT_SEC))

        sec_rect = QRectF(0, cy + line_gap / 2, side, sec_size * 1.4)
        painter.drawText(sec_rect, Qt.AlignmentFlag.AlignCenter, goal_str)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format seconds as 'H h MM m' or 'MM m' if under an hour."""
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes = rem // 60
        if hours:
            return f"{hours} h {minutes:02d} m"
        return f"{minutes} m"

    # ── Size hint ─────────────────────────────────────────────────

    def sizeHint(self):  # noqa: N802
        return self.minimumSizeHint()

    def minimumSizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        return QSize(160, 160)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return width
