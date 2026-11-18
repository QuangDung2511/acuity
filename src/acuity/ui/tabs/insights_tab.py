# Insights tab - range-scoped focus, block, fatigue and distraction analytics.
# ─────────────────────────────────────────────────────────────────
# Public API (consumed by MainWindow):
#
#   InsightsTab()
#   .refresh()          – reload everything from the DB and redraw
#   .set_range(key)     – "today" | "7d" | "30d" | "all"; refreshes
#   .set_theme(mode)    – "dark" | "light"; re-themes and redraws
#
# Layout (follows docs/ui_sketches/insights_tab.png):
#   ┌──────────────────────────────────────┬──────────────────────┐
#   │ [Today] [7 days] [30 days] [All]     │  Daily Progress      │
#   │ ┌ scroll ──────────────────────────┐ │  [CircularProgress]  │
#   │ │ Focus             tiles + 2 charts│ │  Yesterday  X h MM m │
#   │ │ Sessions & blocks tiles + 1 chart │ │  Streak     N days   │
#   │ │ Fatigue           tiles + 2 charts│ │  ──────────────────  │
#   │ │ Distraction       tiles + 2 charts│ │  Takeaways · range   │
#   │ │ Your ratings      tiles           │ │                      │
#   │ └───────────────────────────────────┘ │                      │
#   └──────────────────────────────────────┴──────────────────────┘
#
#   The range row sits above the scroll area and scopes every chart,
#   tile and takeaway.  The gauge, Yesterday and Streak stay on today.
#
# Honesty rules (thresholds live in src/utils/constants.py, INSIGHTS_*):
#   • Only closed blocks ≥ INSIGHTS_MIN_BLOCK_S count, and only events
#     inside them — the running block joins once it ends.
#   • Events are compared per focus hour, never as raw counts; periods
#     with too little focus are drawn as gaps, not spikes.
#   • Events.duration is threshold time, not episode length, so nothing
#     here plots "time spent distracted".
#   • Takeaways only state what clears an evidence gate.
#   • A detector's precision is shown only once the user has rated
#     INSIGHTS_PRECISION_MIN_LABELS of its events (review_events setting).
#
# Data comes exclusively from acuity.db.queries; all aggregation lives in
# acuity.core.insights_metrics so it can be tested without Qt.
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import date, timedelta

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import acuity.core.insights_metrics as metrics
import acuity.db.queries as queries
from acuity.ui.widgets.chart_card import (
    ChartCard,
    StatTile,
    TakeawayList,
    draw_bars,
    draw_heatmap,
    draw_hline,
    draw_line,
    draw_stacked_bars,
    draw_vline,
    palette_for,
)
from acuity.ui.widgets.circular_progress import CircularProgress
from acuity.utils.config import ConfigManager
from acuity.utils.constants import (
    EVENT_AWAY,
    EVENT_DISTRACTION_POSE,
    EVENT_DISTRACTION_WINDOW,
    EVENT_FATIGUE,
    EVENT_FATIGUE_WARNING,
    INSIGHTS_HIST_BIN_MIN,
    INSIGHTS_HIST_MAX_MIN,
    INSIGHTS_MIN_BLOCK_S,
    INSIGHTS_MIN_RATE_FOCUS_S,
    INSIGHTS_PRECISION_MIN_LABELS,
    INSIGHTS_RANGE_7D,
    INSIGHTS_RANGE_30D,
    INSIGHTS_RANGE_ALL,
    INSIGHTS_RANGE_DEFAULT,
    INSIGHTS_RANGE_TODAY,
    SESSION_MODE_FLUID,
    SESSION_MODE_STRUCTURED,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)

_RANGE_BUTTONS: tuple[tuple[str, str], ...] = (
    (INSIGHTS_RANGE_TODAY, "Today"),
    (INSIGHTS_RANGE_7D, "7 days"),
    (INSIGHTS_RANGE_30D, "30 days"),
    (INSIGHTS_RANGE_ALL, "All"),
)

# (event type, legend name, palette slot).  Slot order is the validated
# categorical order, and colour follows the event type on every chart.
_BEHAVIOUR_SERIES: tuple[tuple[str, str, int], ...] = (
    (EVENT_DISTRACTION_POSE, "Head pose", 0),
    (EVENT_DISTRACTION_WINDOW, "Window", 1),
)

# Detectors the user can rate in the review dialog (Away is presence
# bookkeeping, not a judgement, so it is never reviewed).
_RATED_DETECTORS: tuple[tuple[str, str], ...] = (
    (EVENT_FATIGUE, "Fatigue alerts"),
    (EVENT_FATIGUE_WARNING, "Fatigue warnings"),
    (EVENT_DISTRACTION_POSE, "Head pose"),
    (EVENT_DISTRACTION_WINDOW, "Window"),
)

_MIN_RATE_MIN = INSIGHTS_MIN_RATE_FOCUS_S // 60
_NO_FOCUS = "No focus recorded in this range."
_THIN_FOCUS = (
    f"No period in this range has {_MIN_RATE_MIN} min of focus, so no rate can be drawn yet."
)


class InsightsTab(QWidget):
    """Insights tab — Tab 1 of the main window.

    Left: a range selector over a scrollable column of five sections
    (Focus, Sessions & blocks, Fatigue, Distraction, Your ratings), each
    a row of stat tiles, most with one or two charts.

    Right: today's focus-goal gauge, Yesterday, Streak, and the
    evidence-gated takeaways for the selected range.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = ConfigManager()
        self._range_key = INSIGHTS_RANGE_DEFAULT
        self._palette = palette_for("dark")
        self._build_ui()
        self.set_theme(self._cfg.get_str("theme"))  # themes, then refreshes

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        root.addWidget(self._make_chart_column(), stretch=3)
        root.addWidget(self._make_stats_panel(), stretch=1)

    # ── Left column ───────────────────────────────────────────────

    def _make_chart_column(self) -> QWidget:
        """Range row above a scroll area holding the five sections."""
        column = QWidget()
        column.setObjectName("insightsChartColumn")
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(12)

        col.addLayout(self._make_range_row())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        col.addWidget(scroll, stretch=1)

        content = QWidget()
        scroll.setWidget(content)
        body = QVBoxLayout(content)
        # Right margin keeps the vertical scrollbar clear of the chart edges.
        body.setContentsMargins(0, 0, 14, 8)
        body.setSpacing(28)
        body.setAlignment(Qt.AlignmentFlag.AlignTop)

        body.addWidget(self._make_focus_section())
        body.addWidget(self._make_blocks_section())
        body.addWidget(self._make_fatigue_section())
        body.addWidget(self._make_distraction_section())
        body.addWidget(self._make_ratings_section())
        return column

    def _make_range_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._range_group = QButtonGroup(self)
        self._range_group.setExclusive(True)
        self._range_buttons: dict[str, QPushButton] = {}

        for key, label in _RANGE_BUTTONS:
            btn = QPushButton(label)
            btn.setObjectName(f"insightsRange_{key}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, k=key: self.set_range(k))
            self._range_group.addButton(btn)
            self._range_buttons[key] = btn
            row.addWidget(btn)

        row.addStretch(1)
        return row

    @staticmethod
    def _make_section(title: str, caption: str) -> tuple[QWidget, QVBoxLayout]:
        section = QWidget()
        col = QVBoxLayout(section)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        heading = QLabel(title)
        heading.setProperty("heading", True)
        col.addWidget(heading)

        sub = QLabel(caption)
        sub.setProperty("secondary", True)
        sub.setWordWrap(True)
        col.addWidget(sub)
        return section, col

    @staticmethod
    def _add_tiles(layout: QVBoxLayout, titles: Sequence[str]) -> list[StatTile]:
        row = QHBoxLayout()
        row.setSpacing(10)
        tiles = [StatTile(title) for title in titles]
        for tile in tiles:
            row.addWidget(tile, stretch=1)
        layout.addLayout(row)
        return tiles

    def _make_focus_section(self) -> QWidget:
        section, col = self._make_section(
            "Focus",
            "How much you focused, and when in the week you do it.",
        )
        (
            self._tile_focus_total,
            self._tile_focus_avg,
            self._tile_focus_days,
            self._tile_focus_best,
        ) = self._add_tiles(
            col,
            ("Focus time", "Per active day", "Active days", "Best day"),
        )
        self._focus_card = ChartCard(
            "Focus time",
            "Minutes of focus in each period. On daily views the line marks your goal.",
            left_label="Minutes",
        )
        self._heatmap_card = ChartCard(
            "When you focus",
            "Total focus minutes in this range by weekday and hour of day.",
            plot_height=240,
        )
        col.addWidget(self._focus_card)
        col.addWidget(self._heatmap_card)
        return section

    def _make_blocks_section(self) -> QWidget:
        section, col = self._make_section(
            "Sessions & blocks",
            "How long your blocks run, and whether rests happen.",
        )
        (
            self._tile_sessions,
            self._tile_block_median,
            self._tile_rest_taken,
            self._tile_rest_skipped,
        ) = self._add_tiles(
            col,
            ("Sessions", "Median block", "Rest taken", "Rests skipped"),
        )
        self._blocks_card = ChartCard(
            "Block length",
            "How long focus blocks ran before ending. The marker is your Structured block length.",
            left_label="Blocks",
            bottom_label="Minutes",
        )
        col.addWidget(self._blocks_card)
        return section

    def _make_fatigue_section(self) -> QWidget:
        section, col = self._make_section(
            "Fatigue",
            "How soon fatigue shows up in a block, and whether that is changing.",
        )
        (self._tile_fatigue_rate, self._tile_microsleeps, self._tile_warnings) = self._add_tiles(
            col,
            ("Alerts / focus hour", "Microsleeps", "Warnings"),
        )
        self._onset_card = ChartCard(
            "Fatigue onset",
            "Minutes from block start to each fatigue alert (includes calibration and pauses).",
            left_label="Alerts",
            bottom_label="Minutes into block",
        )
        self._fatigue_trend_card = ChartCard(
            "Fatigue over time",
            f"Fatigue alerts per focus hour. Gaps mark periods with under "
            f"{_MIN_RATE_MIN} min of focus.",
            left_label="Per focus hour",
        )
        col.addWidget(self._onset_card)
        col.addWidget(self._fatigue_trend_card)
        return section

    def _make_distraction_section(self) -> QWidget:
        section, col = self._make_section(
            "Distraction",
            "Rates are per focus hour, so long and short days compare fairly.",
        )
        (self._tile_pose_rate, self._tile_window_rate, self._tile_away) = self._add_tiles(
            col,
            ("Head pose / focus h", "Window / focus h", "Away periods"),
        )
        self._rate_card = ChartCard(
            "Distraction over time",
            f"Events per focus hour. Gaps mark periods with under {_MIN_RATE_MIN} min of focus.",
            left_label="Per focus hour",
        )
        self._when_card = ChartCard(
            "Where in the block",
            "Minutes from block start to each event.",
            left_label="Events",
            bottom_label="Minutes into block",
        )
        col.addWidget(self._rate_card)
        col.addWidget(self._when_card)
        return section

    def _make_ratings_section(self) -> QWidget:
        section, col = self._make_section(
            "Your ratings",
            "How often each detector was right for you, from the alerts you rated. "
            f"A figure appears once a detector has {INSIGHTS_PRECISION_MIN_LABELS} "
            'ratings. Turn on "Ask me to rate alerts at each rest" in Settings to '
            "rate them.",
        )
        tiles = self._add_tiles(col, [name for _, name in _RATED_DETECTORS])
        self._precision_tiles = {
            event_type: tile for (event_type, _), tile in zip(_RATED_DETECTORS, tiles)
        }
        return section

    # ── Right column ──────────────────────────────────────────────

    def _make_stats_panel(self) -> QFrame:
        """Right panel: today's gauge, Yesterday, Streak, then takeaways."""
        panel = QFrame()
        panel.setObjectName("insightsStatsPanel")
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setMinimumWidth(260)

        col = QVBoxLayout(panel)
        col.setContentsMargins(16, 20, 16, 20)
        col.setSpacing(8)

        heading = QLabel("Daily Progress")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setProperty("heading", True)
        col.addWidget(heading)

        # Max 200×200 so it never dominates; hasHeightForWidth keeps it square.
        self._gauge = CircularProgress(
            goal_hours=self._cfg.get_float("daily_focus_goal_h"),
        )
        self._gauge.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        self._gauge.setMinimumSize(140, 140)
        self._gauge.setMaximumSize(200, 200)
        col.addWidget(self._gauge, alignment=Qt.AlignmentFlag.AlignCenter)

        col.addSpacing(8)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        col.addWidget(sep)
        col.addSpacing(4)

        self._yesterday_label = self._make_stat_row(col, "Yesterday")
        self._streak_label = self._make_stat_row(col, "Streak")

        col.addSpacing(6)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        col.addWidget(sep2)
        col.addSpacing(8)

        # Own scroll area so a long list never squeezes the gauge.
        self._takeaways = TakeawayList()
        takeaway_scroll = QScrollArea()
        takeaway_scroll.setWidgetResizable(True)
        takeaway_scroll.setFrameShape(QFrame.Shape.NoFrame)
        takeaway_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        takeaway_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        takeaway_scroll.viewport().setAutoFillBackground(False)
        takeaway_scroll.setWidget(self._takeaways)
        col.addWidget(takeaway_scroll, stretch=1)
        return panel

    @staticmethod
    def _make_stat_row(layout: QVBoxLayout, title: str) -> QLabel:
        """Add a two-column stat row to *layout*; return the value QLabel."""
        row = QHBoxLayout()
        row.setSpacing(8)

        name_lbl = QLabel(title)
        name_lbl.setProperty("secondary", True)
        row.addWidget(name_lbl)

        row.addStretch(1)

        value_lbl = QLabel("—")
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value_lbl)

        layout.addLayout(row)
        return value_lbl

    # ── Public API ────────────────────────────────────────────────

    @property
    def range_key(self) -> str:
        return self._range_key

    def set_range(self, key: str) -> None:
        """Select a range ("today" / "7d" / "30d" / "all") and redraw."""
        if key not in self._range_buttons:
            raise ValueError(f"Unknown insights range: {key!r}")
        self._range_key = key
        for k, btn in self._range_buttons.items():
            btn.setChecked(k == key)
            btn.setProperty("accent", k == key)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.refresh()

    def set_theme(self, mode: str) -> None:
        """Re-theme every chart for "dark" / "light" and redraw."""
        self._palette = palette_for(mode)
        pg.setConfigOption("background", self._palette["surface"])
        pg.setConfigOption("foreground", self._palette["text_secondary"])
        for card in self.findChildren(ChartCard):
            card.apply_theme(self._palette)
        for tile in self.findChildren(StatTile):
            tile.apply_theme(self._palette)
        self._takeaways.apply_theme(self._palette)
        self.set_range(self._range_key)  # also syncs the range buttons

    def refresh(self) -> None:
        """Reload all data from the DB and repaint every widget."""
        today = date.today()
        self._refresh_daily_panel(today)

        key = self._range_key
        earliest = None
        if key == INSIGHTS_RANGE_ALL:
            try:
                earliest = queries.get_earliest_block_date(INSIGHTS_MIN_BLOCK_S)
            except Exception:
                log.exception("Failed to load earliest block date")
        start, end, prev_start, prev_end = metrics.range_bounds(key, today, earliest)

        blocks = _load(
            "focus blocks", queries.get_focus_blocks_in_range, start, end, INSIGHTS_MIN_BLOCK_S
        )
        events = _load(
            "events", queries.get_events_in_range_with_offset, start, end, INSIGHTS_MIN_BLOCK_S
        )
        rests = _load("rests", queries.get_rests_in_range, start, end)

        prev_blocks: list[dict] | None = None
        prev_events: list[dict] | None = None
        if prev_start is not None and prev_end is not None:
            prev_blocks = _load(
                "previous focus blocks",
                queries.get_focus_blocks_in_range,
                prev_start,
                prev_end,
                INSIGHTS_MIN_BLOCK_S,
            )
            prev_events = _load(
                "previous events",
                queries.get_events_in_range_with_offset,
                prev_start,
                prev_end,
                INSIGHTS_MIN_BLOCK_S,
            )

        gran = metrics.granularity_for(key)
        axis = metrics.bucket_axis(start, end, gran)
        focus_s = metrics.focus_seconds_by_bucket(blocks, axis, gran)
        summary = metrics.summarise(blocks, events)
        labels = [label for _, label in axis]
        hover_labels = _hover_labels(axis, gran)

        self._draw_focus_section(summary, blocks, focus_s, labels, hover_labels, gran)
        self._draw_blocks_section(summary, blocks, rests)
        self._draw_fatigue_section(summary, events, axis, gran, focus_s, labels, hover_labels)
        self._draw_distraction_section(summary, events, axis, gran, focus_s, labels, hover_labels)
        self._draw_ratings_section(events)

        self._takeaways.set_items(
            metrics.build_takeaways(
                key,
                blocks,
                events,
                prev_blocks,
                prev_events,
                structured_work_min=self._cfg.get_int("structured_work_min"),
            ),
            metrics.range_label(key),
        )

    # ── Daily panel ───────────────────────────────────────────────

    def _refresh_daily_panel(self, today: date) -> None:
        goal_h = self._cfg.get_float("daily_focus_goal_h")

        try:
            today_seconds = queries.get_daily_focus_total(today)
        except Exception:
            log.exception("Failed to load today's focus total")
            today_seconds = 0.0
        self._gauge.set_goal(goal_h)
        self._gauge.set_value(today_seconds)

        try:
            yest_seconds = queries.get_daily_focus_total(today - timedelta(days=1))
        except Exception:
            log.exception("Failed to load yesterday's focus total")
            yest_seconds = 0.0
        self._yesterday_label.setText(_fmt_seconds(yest_seconds))

        try:
            streak = queries.get_focus_streak()
        except Exception:
            log.exception("Failed to load focus streak")
            streak = 0
        self._streak_label.setText(f"{streak} day{'s' if streak != 1 else ''}")

    # ── Section drawing ───────────────────────────────────────────

    def _draw_focus_section(
        self,
        summary: metrics.RangeSummary,
        blocks: Sequence[dict],
        focus_s: Sequence[float],
        labels: Sequence[str],
        hover_labels: Sequence[str],
        gran: str,
    ) -> None:
        pal = self._palette

        self._tile_focus_total.set_value(_fmt_seconds(summary.focus_s))
        self._tile_focus_avg.set_value(
            _fmt_seconds(summary.focus_s / summary.active_days) if summary.active_days else "—"
        )
        self._tile_focus_days.set_value(str(summary.active_days))
        if summary.best_day is not None:
            best_day, best_s = summary.best_day
            self._tile_focus_best.set_value(_fmt_seconds(best_s), best_day.strftime("%a %m/%d"))
        else:
            self._tile_focus_best.set_value("—")

        card = self._focus_card
        if summary.focus_s <= 0:
            card.show_empty(_NO_FOCUS)
        else:
            card.clear()
            minutes = [s / 60.0 for s in focus_s]
            draw_bars(card, minutes, pal["series"][0])
            if gran == metrics.GRAN_DAY:
                goal_min = self._cfg.get_float("daily_focus_goal_h") * 60.0
                draw_hline(card, goal_min, pal["text_secondary"], "Goal")
                card.plot_item.setYRange(0, max(max(minutes), goal_min) * 1.15, padding=0)
            card.set_x_ticks(labels, _tick_every(len(labels), gran))
            card.set_hover(
                _indexed_hover(
                    len(minutes),
                    lambda i: f"{hover_labels[i]}: {_fmt_seconds(minutes[i] * 60)} of focus",
                )
            )

        heat = self._heatmap_card
        if gran == metrics.GRAN_HOUR:
            heat.show_empty(
                "The weekday × hour pattern needs more than one day — choose 7 days or longer."
            )
        elif not blocks:
            heat.show_empty(_NO_FOCUS)
        else:
            heat.clear()
            grid = metrics.focus_heatmap(blocks)
            top = draw_heatmap(heat, grid, pal["heat"], metrics.WEEKDAY_LABELS)
            heat.set_legend(
                [
                    ("less", pal["heat"][1]),
                    (f"more (busiest slot {top:.0f} min)", pal["heat"][-1]),
                ]
            )

            def heat_hover(x: float, y: float) -> str | None:
                col, row = int(math.floor(x)), int(math.floor(y))
                if not (0 <= row < 7 and 0 <= col < 24):
                    return None
                return (
                    f"{metrics.WEEKDAY_LABELS[row]} {col:02d}:00–{(col + 1) % 24:02d}:00: "
                    f"{grid[row][col]:.0f} min"
                )

            heat.set_hover(heat_hover)

    def _draw_blocks_section(
        self,
        summary: metrics.RangeSummary,
        blocks: Sequence[dict],
        rests: Sequence[dict],
    ) -> None:
        pal = self._palette

        modes = summary.sessions_by_mode
        mode_parts = [
            f"{modes[mode]} {mode}"
            for mode in (SESSION_MODE_STRUCTURED, SESSION_MODE_FLUID)
            if modes.get(mode)
        ]
        self._tile_sessions.set_value(str(summary.session_count), " · ".join(mode_parts))
        self._tile_block_median.set_value(
            _fmt_seconds(summary.median_block_s) if summary.median_block_s is not None else "—"
        )
        ratio = metrics.rest_taken_ratio(rests)
        if ratio is None:
            self._tile_rest_taken.set_value("—")
        else:
            self._tile_rest_taken.set_value(f"{ratio * 100:.0f}%", "median share of planned rest")
        skipped = sum(1 for r in rests if r["skipped"])
        self._tile_rest_skipped.set_value(f"{skipped} of {len(rests)}" if rests else "—")

        card = self._blocks_card
        if not blocks:
            card.show_empty(_NO_FOCUS)
            return
        card.clear()
        counts = metrics.histogram([b["duration"] / 60.0 for b in blocks])
        draw_bars(card, counts, pal["series"][0])
        card.set_x_ticks(metrics.histogram_labels())
        work_min = self._cfg.get_int("structured_work_min")
        if work_min < INSIGHTS_HIST_MAX_MIN:
            # Minute v sits at x = v / bin - 0.5 (bar i spans bin i's edges).
            draw_vline(
                card,
                work_min / INSIGHTS_HIST_BIN_MIN - 0.5,
                pal["text_secondary"],
                f"{work_min} min",
            )
        card.set_hover(
            _indexed_hover(
                len(counts),
                lambda i: f"{_bin_label(i, len(counts))}: {_plural(counts[i], 'block')}",
            )
        )

    def _draw_fatigue_section(
        self,
        summary: metrics.RangeSummary,
        events: Sequence[dict],
        axis: metrics.Axis,
        gran: str,
        focus_s: Sequence[float],
        labels: Sequence[str],
        hover_labels: Sequence[str],
    ) -> None:
        pal = self._palette

        self._tile_fatigue_rate.set_value(_fmt_rate(summary.per_focus_hour(EVENT_FATIGUE)))
        reasons = metrics.reason_counts(events, EVENT_FATIGUE)
        total = sum(reasons.values())
        sub = f"of {_plural(total, 'alert')}" if total else ""
        if reasons.get(None):
            sub += f" · {reasons[None]} unrecorded"
        self._tile_microsleeps.set_value(str(reasons.get("microsleep", 0)), sub)
        self._tile_warnings.set_value(str(summary.event_counts.get(EVENT_FATIGUE_WARNING, 0)))

        onset_card = self._onset_card
        onsets = metrics.onset_minutes(events, EVENT_FATIGUE)
        if not onsets:
            onset_card.show_empty("No fatigue alerts in this range.")
        else:
            onset_card.clear()
            counts = metrics.histogram(onsets)
            draw_bars(onset_card, counts, pal["series"][0])
            onset_card.set_x_ticks(metrics.histogram_labels())
            onset_card.set_hover(
                _indexed_hover(
                    len(counts),
                    lambda i: f"{_bin_label(i, len(counts))}: {_plural(counts[i], 'alert')}",
                )
            )

        trend = self._fatigue_trend_card
        alert_counts = metrics.event_counts_by_bucket(events, EVENT_FATIGUE, axis, gran)
        rates = metrics.rates_per_focus_hour(alert_counts, focus_s)
        if all(rate is None for rate in rates):
            trend.show_empty(_THIN_FOCUS if summary.focus_s > 0 else _NO_FOCUS)
        else:
            trend.clear()
            draw_line(trend, rates, pal["series"][0], pal["surface"])
            trend.set_x_ticks(labels, _tick_every(len(labels), gran))

            def trend_text(i: int) -> str:
                if rates[i] is None:
                    return f"{hover_labels[i]}: under {_MIN_RATE_MIN} min of focus"
                return (
                    f"{hover_labels[i]}: {rates[i]:.1f} per focus hour "
                    f"({_plural(alert_counts[i], 'alert')})"
                )

            trend.set_hover(_indexed_hover(len(rates), trend_text))

    def _draw_distraction_section(
        self,
        summary: metrics.RangeSummary,
        events: Sequence[dict],
        axis: metrics.Axis,
        gran: str,
        focus_s: Sequence[float],
        labels: Sequence[str],
        hover_labels: Sequence[str],
    ) -> None:
        pal = self._palette

        rate_tiles = (self._tile_pose_rate, self._tile_window_rate)
        for tile, (event_type, _, _) in zip(rate_tiles, _BEHAVIOUR_SERIES):
            tile.set_value(
                _fmt_rate(summary.per_focus_hour(event_type)),
                _plural(summary.event_counts.get(event_type, 0), "event"),
            )
        self._tile_away.set_value(str(summary.event_counts.get(EVENT_AWAY, 0)))

        legend = [(name, pal["series"][slot]) for _, name, slot in _BEHAVIOUR_SERIES]

        # ── Rate over time (stacked) ──────────────────────────────
        rate_card = self._rate_card
        per_type = {
            event_type: metrics.rates_per_focus_hour(
                metrics.event_counts_by_bucket(events, event_type, axis, gran),
                focus_s,
            )
            for event_type, _, _ in _BEHAVIOUR_SERIES
        }
        # Every type shares the focus denominator, so gaps coincide.
        gaps = per_type[_BEHAVIOUR_SERIES[0][0]]
        if all(rate is None for rate in gaps):
            rate_card.show_empty(_THIN_FOCUS if summary.focus_s > 0 else _NO_FOCUS)
        else:
            rate_card.clear()
            draw_stacked_bars(
                rate_card,
                [(per_type[et], pal["series"][slot]) for et, _, slot in _BEHAVIOUR_SERIES],
                gap_colour=pal["surface"],
            )
            rate_card.set_legend(legend)
            rate_card.set_x_ticks(labels, _tick_every(len(labels), gran))

            def rate_text(i: int) -> str:
                if gaps[i] is None:
                    return f"{hover_labels[i]}: under {_MIN_RATE_MIN} min of focus"
                values = " · ".join(
                    f"{name} {per_type[et][i]:.1f}" for et, name, _ in _BEHAVIOUR_SERIES
                )
                return f"{hover_labels[i]}: {values} per focus hour"

            rate_card.set_hover(_indexed_hover(len(gaps), rate_text))

        # ── Where in the block (stacked histogram) ────────────────
        when_card = self._when_card
        hists = {
            event_type: metrics.histogram(metrics.onset_minutes(events, event_type))
            for event_type, _, _ in _BEHAVIOUR_SERIES
        }
        if not any(sum(h) for h in hists.values()):
            when_card.show_empty("No distraction events in this range.")
        else:
            when_card.clear()
            draw_stacked_bars(
                when_card,
                [(hists[et], pal["series"][slot]) for et, _, slot in _BEHAVIOUR_SERIES],
                gap_colour=pal["surface"],
            )
            when_card.set_legend(legend)
            when_card.set_x_ticks(metrics.histogram_labels())
            n_bins = len(hists[_BEHAVIOUR_SERIES[0][0]])
            when_card.set_hover(
                _indexed_hover(
                    n_bins,
                    lambda i: (
                        f"{_bin_label(i, n_bins)}: "
                        + " · ".join(f"{name} {hists[et][i]}" for et, name, _ in _BEHAVIOUR_SERIES)
                    ),
                )
            )

    def _draw_ratings_section(self, events: Sequence[dict]) -> None:
        for event_type, tile in self._precision_tiles.items():
            correct, labelled = metrics.label_counts(events, event_type)
            value = metrics.precision(events, event_type)
            if value is not None:
                tile.set_value(f"{value * 100:.0f}%", f"{correct} of {labelled} correct")
            elif labelled:
                tile.set_value("—", f"{labelled} of {INSIGHTS_PRECISION_MIN_LABELS} ratings")
            else:
                tile.set_value("—", "no ratings")


# ── Module-level helpers ──────────────────────────────────────────


def _load(what: str, fn: Callable[..., list[dict]], *args) -> list[dict]:
    """Run one query; on failure log it and return [] so the tab still draws."""
    try:
        return fn(*args)
    except Exception:
        log.exception("Failed to load %s", what)
        return []


def _fmt_seconds(seconds: float) -> str:
    """Format seconds as 'H h MM m' or 'MM m' if under one hour."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours} h {minutes:02d} m"
    return f"{minutes} m"


def _fmt_rate(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:.1f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _tick_every(n: int, gran: str) -> int:
    """Label spacing so the x-axis stays legible (≈ 10 labels max)."""
    if gran == metrics.GRAN_HOUR:
        return 3
    return max(1, math.ceil(n / 10))


def _bin_label(i: int, n_bins: int) -> str:
    if i == n_bins - 1:
        return f"{INSIGHTS_HIST_MAX_MIN}+ min"
    return f"{i * INSIGHTS_HIST_BIN_MIN}–{(i + 1) * INSIGHTS_HIST_BIN_MIN} min"


def _hover_labels(axis: metrics.Axis, gran: str) -> list[str]:
    if gran == metrics.GRAN_HOUR:
        return [f"{key:%H}:00–{(key.hour + 1) % 24:02d}:00" for key, _ in axis]
    if gran == metrics.GRAN_DAY:
        return [key.strftime("%a %m/%d") for key, _ in axis]
    return [f"Week of {key:%m/%d}" for key, _ in axis]


def _indexed_hover(
    n: int,
    text_for: Callable[[int], str],
) -> Callable[[float, float], str | None]:
    """Hover callback that snaps x to the nearest bar / point index."""

    def hover(x: float, _y: float) -> str | None:
        i = int(round(x))
        return text_for(i) if 0 <= i < n else None

    return hover
