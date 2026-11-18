# Smoke test for the main window under PySide6.
# ─────────────────────────────────────────────────────────────────
# Builds the real MainWindow offscreen: every tab, widget, style sheet
# and signal connection it makes at start-up.  A PySide6 port break in
# the UI layer (a renamed enum, a Property, a signal signature) fails
# here instead of only at a manual launch.  Session behaviour is
# covered by test_session_e2e.py.  The sound player is stubbed (see
# conftest.silent_sound_player).
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QTabWidget

import acuity.db.database as database
from acuity.ui.main_window import MainWindow
from acuity.ui.styles.theme import apply_theme


@pytest.fixture()
def window(qtbot, monkeypatch, silent_sound_player):
    monkeypatch.setattr(database, "_instance", None)
    apply_theme(QApplication.instance(), mode="dark")
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    database.get_db().close()


def test_window_opens_on_focus_with_insights_and_settings(window):
    """The user lands on Focus; Insights and Settings sit alongside."""
    tabs = window.centralWidget()
    assert isinstance(tabs, QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Focus", "Insights", "Settings"]
    assert tabs.currentIndex() == 0


def test_closing_the_window_hides_it_to_the_tray(window):
    """Close (X) must not end a running session: the app lives in the tray."""
    window.close()
    assert not window.isVisible()
