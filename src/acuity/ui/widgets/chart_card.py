# Chart building blocks for the Insights tab - cards, stat tiles, takeaways.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   palette_for(mode) -> dict            – theme tokens + series / heat ramp
#
#   ChartCard(title, caption, left_label="", bottom_label="", plot_height=220)
#     .plot_item                          – the pyqtgraph PlotItem
#     .clear() / .show_empty(message)
#     .set_legend([(name, colour), ...]) / .set_x_ticks(labels, every=1)
#     .set_hover(fn(x, y) -> str | None) / .apply_theme(palette)
#
#   draw_bars / draw_stacked_bars / draw_line / draw_heatmap
#   draw_hline / draw_vline                – marks onto a ChartCard
#
#   StatTile(title)       .set_value(value, sub="") / .apply_theme(palette)
#   TakeawayList()        .set_items(items, range_label) / .apply_theme(palette)
#
# Design:
#   • Cards live inside a QScrollArea, so a plot never takes the mouse
#     wheel (pyqtgraph would zoom instead of the page scrolling) and has
#     no pan / zoom / context menu.  These are read-only figures.
#   • The plot area has a fixed height that includes its axis band, so a
#     card never grows a nested scrollbar.
#   • Legends are a rich-text QLabel row rather than pyqtgraph's
#     LegendItem: the swatch carries the series colour, the name stays in
#     text ink, and nothing floats over the data.
#   • Series colours are validated categorical slots (blue, orange)
#     stepped separately per theme; dark on #1A1A1F passes every check.
#     Series charts always carry a legend
#     and hover text, so colour never carries the meaning alone.  Status
#     colours (success / danger) are reserved for good-vs-bad meaning in
#     TakeawayList.
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from acuity.ui.styles.theme import DARK, LIGHT

# ── Chart palette ─────────────────────────────────────────────────
# Categorical order: head pose, window.  Colour follows the entity, so
# the same event type is the same colour on every chart.
_SERIES: dict[str, tuple[str, str]] = {
    "dark": ("#3987e5", "#d95926"),
    "light": ("#2a78d6", "#eb6834"),
}

# Sequential blue for the heatmap: "none" recedes into the surface,
# "most" is the strongest step for that theme.
_HEAT_RAMP: dict[str, tuple[str, ...]] = {
    "dark": ("#22262e", "#184f95", "#3987e5", "#9ec5f4"),
    "light": ("#f3f7fd", "#9ec5f4", "#3987e5", "#184f95"),
}

_BAR_WIDTH = 0.8  # fraction of a slot; the rest is the gap between bars
_EMPTY_TAKEAWAYS = (
    "No takeaways yet — this range needs more focus time before Acuity will draw conclusions."
)


def palette_for(mode: str) -> dict:
    """Theme tokens plus chart-only keys ``series`` and ``heat``."""
    key = "light" if mode == "light" else "dark"
    base = LIGHT if key == "light" else DARK
    return {**base, "series": _SERIES[key], "heat": _HEAT_RAMP[key]}


def _card_style(object_name: str, pal: dict) -> str:
    # Scoped by object name so the frame style does not cascade to children.
    return (
        f"#{object_name} {{ background-color: {pal['surface']}; "
        f"border: 1px solid {pal['border']}; border-radius: 6px; }}"
    )


# ════════════════════════════════════════════════════════════════
#  PLOT WIDGET
# ════════════════════════════════════════════════════════════════


class StaticPlotWidget(pg.PlotWidget):
    """Read-only PlotWidget: no pan/zoom/menu, wheel passes to the page."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        item = self.getPlotItem()
        item.setMouseEnabled(x=False, y=False)
        item.setMenuEnabled(False)
        item.hideButtons()
        item.getViewBox().setBorder(None)
        self._hover: Callable[[float, float], str | None] | None = None
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def set_hover(self, fn: Callable[[float, float], str | None] | None) -> None:
        self._hover = fn

    def wheelEvent(self, event) -> None:  # noqa: N802
        # Ignored so the enclosing QScrollArea scrolls instead of the plot zooming.
        event.ignore()

    def leaveEvent(self, event) -> None:  # noqa: N802
        QToolTip.hideText()
        super().leaveEvent(event)

    def _on_mouse_moved(self, pos) -> None:
        if self._hover is None:
            return
        view_box = self.getPlotItem().getViewBox()
        if not view_box.sceneBoundingRect().contains(pos):
            QToolTip.hideText()
            return
        point = view_box.mapSceneToView(pos)
        text = self._hover(point.x(), point.y())
        if text:
            QToolTip.showText(QCursor.pos(), text, self)
        else:
            QToolTip.hideText()


# ════════════════════════════════════════════════════════════════
#  CHART CARD
# ════════════════════════════════════════════════════════════════


class ChartCard(QFrame):
    """Titled card: caption, optional legend row, plot or empty state."""

    def __init__(
        self,
        title: str,
        caption: str,
        left_label: str = "",
        bottom_label: str = "",
        plot_height: int = 220,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("insightsChartCard")
        self._axis_labels = {"left": left_label, "bottom": bottom_label}

        col = QVBoxLayout(self)
        col.setContentsMargins(14, 12, 14, 12)
        col.setSpacing(4)

        self._title = QLabel(title)
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        col.addWidget(self._title)

        self._caption = QLabel(caption)
        self._caption.setProperty("secondary", True)
        self._caption.setWordWrap(True)
        col.addWidget(self._caption)

        self._legend = QLabel()
        self._legend.setProperty("secondary", True)
        self._legend.setTextFormat(Qt.TextFormat.RichText)
        self._legend.hide()
        col.addWidget(self._legend)

        self._plot = StaticPlotWidget()
        self._empty = QLabel()
        self._empty.setProperty("secondary", True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._plot)
        self._stack.addWidget(self._empty)
        self._stack.setFixedHeight(plot_height)
        col.addWidget(self._stack)

        self.apply_theme(palette_for("dark"))

    # ── Accessors ─────────────────────────────────────────────────

    @property
    def plot_item(self) -> pg.PlotItem:
        return self._plot.getPlotItem()

    @property
    def plot_widget(self) -> StaticPlotWidget:
        return self._plot

    def is_empty(self) -> bool:
        return self._stack.currentWidget() is self._empty

    # ── State ─────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all marks, ticks, legend and hover; show the plot."""
        item = self.plot_item
        item.clear()
        item.getAxis("bottom").setTicks(None)
        item.getAxis("left").setTicks(None)
        self._plot.set_hover(None)
        self._legend.hide()
        self._stack.setCurrentWidget(self._plot)

    def show_empty(self, message: str) -> None:
        self.clear()
        self._empty.setText(message)
        self._stack.setCurrentWidget(self._empty)

    def set_legend(self, entries: Sequence[tuple[str, str]]) -> None:
        html = "&nbsp;&nbsp;&nbsp;".join(
            f'<span style="color:{colour}">&#9632;</span>&nbsp;{name}' for name, colour in entries
        )
        self._legend.setText(html)
        self._legend.show()

    def set_x_ticks(self, labels: Sequence[str], every: int = 1) -> None:
        ticks = [(i, label) for i, label in enumerate(labels) if i % every == 0]
        self.plot_item.getAxis("bottom").setTicks([ticks])

    def set_hover(self, fn: Callable[[float, float], str | None]) -> None:
        self._plot.set_hover(fn)

    def apply_theme(self, pal: dict) -> None:
        self.setStyleSheet(_card_style("insightsChartCard", pal))
        self._plot.setBackground(pal["surface"])
        item = self.plot_item
        for name in ("left", "bottom"):
            axis = item.getAxis(name)
            # Axes draw the grid; behind the ViewBox so gridlines never cross marks.
            axis.setZValue(-1000)
            axis.setPen(pg.mkPen(pal["border"]))
            axis.setTextPen(pg.mkPen(pal["text_secondary"]))
            if self._axis_labels[name]:
                item.setLabel(name, self._axis_labels[name], color=pal["text_secondary"])
        item.showGrid(x=False, y=True, alpha=0.35)


# ════════════════════════════════════════════════════════════════
#  MARKS
# ════════════════════════════════════════════════════════════════


def _fit_x(card: ChartCard, n: int) -> None:
    card.plot_item.setXRange(-0.6, max(n, 1) - 0.4, padding=0)


def _fit_y(card: ChartCard, top: float) -> None:
    card.plot_item.setYRange(0, top * 1.15 if top > 0 else 1.0, padding=0)


def draw_bars(card: ChartCard, heights: Sequence[float], colour: str) -> None:
    """One series of bars at x = 0 … n-1."""
    n = len(heights)
    card.plot_item.addItem(
        pg.BarGraphItem(
            x=list(range(n)),
            height=list(heights),
            width=_BAR_WIDTH,
            brush=colour,
            pen=pg.mkPen(None),
        )
    )
    _fit_x(card, n)
    _fit_y(card, max(heights, default=0.0))


def draw_stacked_bars(
    card: ChartCard,
    layers: Sequence[tuple[Sequence[float | None], str]],
    gap_colour: str,
) -> None:
    """Stacked bars, first layer on the bottom; ``None`` stacks as 0.

    Segments are outlined in the surface colour so adjacent fills read
    as separate marks without a drawn border.
    """
    n = len(layers[0][0]) if layers else 0
    base = [0.0] * n
    for values, colour in layers:
        heights = [v or 0.0 for v in values]
        card.plot_item.addItem(
            pg.BarGraphItem(
                x=list(range(n)),
                height=heights,
                y0=list(base),
                width=_BAR_WIDTH,
                brush=colour,
                pen=pg.mkPen(gap_colour, width=1.5),
            )
        )
        base = [b + h for b, h in zip(base, heights)]
    _fit_x(card, n)
    _fit_y(card, max(base, default=0.0))


def draw_line(
    card: ChartCard,
    values: Sequence[float | None],
    colour: str,
    surface: str,
) -> None:
    """Line with markers; ``None`` breaks the line (a gap, not a zero)."""
    xs = np.arange(len(values), dtype=float)
    ys = np.array([np.nan if v is None else v for v in values], dtype=float)
    card.plot_item.addItem(
        pg.PlotDataItem(
            xs,
            ys,
            pen=pg.mkPen(colour, width=2),
            connect="finite",
        )
    )
    finite = ~np.isnan(ys)
    if finite.any():
        # Markers keep an isolated point visible between two gaps.
        card.plot_item.addItem(
            pg.ScatterPlotItem(
                x=xs[finite],
                y=ys[finite],
                size=8,
                brush=pg.mkBrush(colour),
                pen=pg.mkPen(surface, width=2),
            )
        )
    _fit_x(card, len(values))
    _fit_y(card, float(np.nanmax(ys)) if finite.any() else 0.0)


def draw_heatmap(
    card: ChartCard,
    grid: Sequence[Sequence[float]],
    ramp: Sequence[str],
    row_labels: Sequence[str],
    col_every: int = 3,
) -> float:
    """Rows × columns heatmap (row 0 at the top).  Returns the max cell."""
    arr = np.array(grid, dtype=float).T  # ImageItem indexes [x, y]
    top = float(arr.max()) if arr.size else 0.0
    colour_map = pg.ColorMap(
        np.linspace(0.0, 1.0, len(ramp)),
        [pg.mkColor(c) for c in ramp],
    )
    image = pg.ImageItem(arr)
    image.setLookupTable(colour_map.getLookupTable(0.0, 1.0, 256))
    image.setLevels((0.0, top if top > 0 else 1.0))
    image.setRect(0, 0, arr.shape[0], arr.shape[1])

    item = card.plot_item
    item.addItem(image)
    item.getViewBox().invertY(True)
    item.showGrid(x=False, y=False)
    item.setXRange(0, arr.shape[0], padding=0)
    item.setYRange(0, arr.shape[1], padding=0)
    item.getAxis("bottom").setTicks(
        [[(col + 0.5, f"{col:02d}") for col in range(0, arr.shape[0], col_every)]]
    )
    left = item.getAxis("left")
    left.setTicks([[(row + 0.5, label) for row, label in enumerate(row_labels)]])
    # Fixed width: text ticks set after the axis first sized itself were
    # otherwise clipped to nothing on the first draw.
    left.setWidth(44)
    return top


def draw_hline(card: ChartCard, y: float, colour: str, label: str) -> None:
    card.plot_item.addItem(
        pg.InfiniteLine(
            pos=y,
            angle=0,
            pen=pg.mkPen(colour, width=1),
            label=label,
            labelOpts={"position": 0.97, "color": colour, "anchors": [(1, 1), (1, 1)]},
        )
    )


def draw_vline(card: ChartCard, x: float, colour: str, label: str) -> None:
    card.plot_item.addItem(
        pg.InfiniteLine(
            pos=x,
            angle=90,
            pen=pg.mkPen(colour, width=1),
            label=label,
            labelOpts={"position": 0.92, "color": colour, "anchors": [(0, 0), (0, 0)]},
        )
    )


# ════════════════════════════════════════════════════════════════
#  STAT TILE
# ════════════════════════════════════════════════════════════════


class StatTile(QFrame):
    """Small headline figure: title, value, optional sub-line."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("insightsStatTile")

        col = QVBoxLayout(self)
        col.setContentsMargins(12, 10, 12, 10)
        col.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setProperty("secondary", True)
        col.addWidget(title_lbl)

        self._value = QLabel("—")
        font = self._value.font()
        font.setPixelSize(20)
        font.setBold(True)
        self._value.setFont(font)
        col.addWidget(self._value)

        self._sub = QLabel()
        self._sub.setProperty("secondary", True)
        self._sub.setWordWrap(True)
        self._sub.hide()
        col.addWidget(self._sub)

        self.apply_theme(palette_for("dark"))

    def set_value(self, value: str, sub: str = "") -> None:
        self._value.setText(value)
        self._sub.setText(sub)
        self._sub.setVisible(bool(sub))

    def value_text(self) -> str:
        return self._value.text()

    def apply_theme(self, pal: dict) -> None:
        self.setStyleSheet(_card_style("insightsStatTile", pal))


# ════════════════════════════════════════════════════════════════
#  TAKEAWAY LIST
# ════════════════════════════════════════════════════════════════


class TakeawayList(QWidget):
    """Sidebar list of evidence-gated findings (see insights_metrics)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        self._heading = QLabel("Takeaways")
        self._heading.setProperty("heading", True)
        col.addWidget(self._heading)

        self._rows = QVBoxLayout()
        self._rows.setSpacing(10)
        col.addLayout(self._rows)
        col.addStretch(1)

        self._items: list = []
        self._palette = palette_for("dark")
        self._render()

    def set_items(self, items: Sequence, range_label: str) -> None:
        self._items = list(items)
        self._heading.setText(f"Takeaways · {range_label}")
        self._render()

    def item_texts(self) -> list[str]:
        return [t.text for t in self._items]

    def apply_theme(self, pal: dict) -> None:
        self._palette = pal
        self._render()

    def _render(self) -> None:
        while self._rows.count():
            widget = self._rows.takeAt(0).widget()
            if widget is not None:
                # Hide first: deleteLater waits for the event loop, and a
                # detached row keeps painting over the new ones until then.
                widget.hide()
                widget.deleteLater()

        if not self._items:
            empty = QLabel(_EMPTY_TAKEAWAYS)
            empty.setProperty("secondary", True)
            empty.setWordWrap(True)
            self._rows.addWidget(empty)
            return

        pal = self._palette
        tone_colour = {
            "good": pal["success"],
            "bad": pal["danger"],
        }
        for item in self._items:
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(8)

            glyph = QLabel({"up": "▲", "down": "▼"}.get(item.direction, "●"))
            glyph.setFixedWidth(14)
            glyph.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            glyph.setStyleSheet(f"color: {tone_colour.get(item.tone, pal['text_secondary'])};")
            line.addWidget(glyph)

            text = QLabel(item.text)
            text.setWordWrap(True)
            line.addWidget(text, stretch=1)

            self._rows.addWidget(row)
