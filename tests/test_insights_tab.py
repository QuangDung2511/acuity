# Smoke tests for the Insights tab widget.
# ─────────────────────────────────────────────────────────────────
# The numbers are tested in test_insights_metrics.py; these check the
# widget wiring: every range draws, empty data shows empty states, the
# theme switch works, a chart never swallows the scroll wheel, and the
# ratings tiles show precision only from 5 ratings.
#
# Runs headless (QT_QPA_PLATFORM=offscreen) against an isolated
# DatabaseManager under ACUITY_DATA_DIR (tmp_path, set by conftest) with
# ConfigManager mocked.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent

import acuity.db.queries as queries
from acuity.db.database import DatabaseManager
from acuity.ui.tabs.insights_tab import InsightsTab
from acuity.utils.constants import (
    EVENT_DISTRACTION_POSE,
    EVENT_DISTRACTION_WINDOW,
    EVENT_LABEL_CORRECT,
    EVENT_LABEL_FALSE_ALARM,
)


@pytest.fixture()
def db():
    manager = DatabaseManager()
    manager.connect()
    with patch("acuity.db.queries.get_db", return_value=manager):
        yield manager
    manager.close()


@pytest.fixture()
def cfg():
    values = {"theme": "dark", "daily_focus_goal_h": 4.0, "structured_work_min": 25}
    mock = MagicMock()
    mock.get_str.side_effect = values.__getitem__
    mock.get_float.side_effect = values.__getitem__
    mock.get_int.side_effect = values.__getitem__
    with patch("acuity.ui.tabs.insights_tab.ConfigManager", return_value=mock):
        yield mock


def _seed(days_ago: int, hour: int, duration_s: float, label: str | None = None) -> None:
    start = datetime.combine(date.today() - timedelta(days=days_ago), time(hour))
    sid = queries.create_session(start_time=start.isoformat())
    bid = queries.start_focus_block(sid, 1, start_time=start.isoformat())
    queries.end_focus_block(bid, duration_s)
    event_id = queries.insert_event(
        sid,
        bid,
        EVENT_DISTRACTION_POSE,
        10.0,
        timestamp=(start + timedelta(minutes=5)).isoformat(),
    )
    if label is not None:
        queries.set_event_label(event_id, label)
    queries.end_session(sid, duration_s)


@pytest.fixture()
def tab(qtbot, db, cfg):
    widget = InsightsTab()
    qtbot.addWidget(widget)
    return widget


def test_empty_database_shows_empty_states(tab):
    assert tab.range_key == "7d"
    assert tab._focus_card.is_empty()
    assert tab._blocks_card.is_empty()
    assert tab._rate_card.is_empty()
    assert tab._takeaways.item_texts() == []


def test_every_range_draws_seeded_data(qtbot, db, cfg):
    for days_ago in (1, 2, 3):
        _seed(days_ago, 15, 1500)
    widget = InsightsTab()
    qtbot.addWidget(widget)

    for key in ("today", "7d", "30d", "all"):
        widget.set_range(key)
        assert widget.range_key == key

    widget.set_range("7d")
    assert not widget._focus_card.is_empty()
    assert not widget._heatmap_card.is_empty()
    assert not widget._blocks_card.is_empty()
    assert widget._tile_focus_total.value_text() == "1 h 15 m"
    assert widget._tile_focus_days.value_text() == "3"
    assert any("peak focus window" in t for t in widget._takeaways.item_texts())

    widget.set_range("today")
    assert widget._heatmap_card.is_empty()  # a single day has no weekly pattern


def test_hover_text_resolves(qtbot, db, cfg):
    _seed(1, 15, 1500)
    widget = InsightsTab()
    qtbot.addWidget(widget)
    hover = widget._blocks_card.plot_widget._hover
    assert hover is not None
    assert hover(5.0, 0.0) == "25–30 min: 1 block"
    assert hover(99.0, 0.0) is None


def test_light_theme_redraws(tab):
    tab.set_theme("light")
    assert tab._palette["surface"] == "#FFFFFF"
    tab.set_theme("dark")
    assert tab._palette["surface"] == "#1A1A1F"


def test_unknown_range_raises(tab):
    with pytest.raises(ValueError):
        tab.set_range("90d")


def test_wheel_over_chart_is_not_consumed(tab):
    """A chart inside the scroll area must pass the wheel on, not zoom."""
    plot = tab._focus_card.plot_widget
    before = plot.getPlotItem().getViewBox().viewRange()
    event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    event.accept()
    plot.wheelEvent(event)
    assert not event.isAccepted()
    assert plot.getPlotItem().getViewBox().viewRange() == before


def test_ratings_tile_shows_precision_from_five_ratings(qtbot, db, cfg):
    """4 correct + 1 false alarm on head pose reads 80 %; the window
    detector, never rated, shows no figure."""
    for days_ago, label in enumerate([EVENT_LABEL_CORRECT] * 4 + [EVENT_LABEL_FALSE_ALARM]):
        _seed(days_ago + 1, 15, 1500, label=label)
    widget = InsightsTab()
    qtbot.addWidget(widget)

    pose = widget._precision_tiles[EVENT_DISTRACTION_POSE]
    assert pose.value_text() == "80%"
    assert widget._precision_tiles[EVENT_DISTRACTION_WINDOW].value_text() == "—"


def test_ratings_tile_withholds_precision_below_five(qtbot, db, cfg):
    """Four ratings is too few: one more would move the figure 20+ points,
    so the tile shows no figure rather than a misleading 100 %."""
    for days_ago in range(1, 5):
        _seed(days_ago, 15, 1500, label=EVENT_LABEL_CORRECT)
    widget = InsightsTab()
    qtbot.addWidget(widget)
    assert widget._precision_tiles[EVENT_DISTRACTION_POSE].value_text() == "—"
