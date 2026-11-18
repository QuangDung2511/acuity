# Shared test fixtures, and the guard that tests never write into the repo.
# ─────────────────────────────────────────────────────────────────
# Every test runs with ACUITY_DATA_DIR pointing at its own tmp_path, so
# no test can create a database or data file in the repo or in the
# user's real app-data folder, even one that forgets to ask for it.
#
# User settings (ConfigManager → QSettings INI) are redirected the same
# way, so a test that builds the real pipeline or UI neither reads the
# developer's settings (debug overlay on, a different camera) nor
# writes over them.
#
# The repo-write guard enforces the first promise: when the run ends,
# any git-ignored file in the repo created or modified during the run
# (a log, database or CSV some code wrote project-relative) fails it.
# Only ignored files count, so editing source files while the suite
# runs in the background does not trip it.
#
# Thread: GUI (pytest's main thread).
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

_REPO = Path(__file__).resolve().parents[1]
# Ignored folders that legitimately change during a run, or that tests
# never touch but a developer may edit meanwhile (.scratch tickets).
_UNWATCHED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".import_linter_cache",
    ".scratch",
}
_run_started = 0.0


def pytest_configure(config):
    # Runs before any test module is imported, so import-time writes count.
    global _run_started
    _run_started = time.time()


def _ignored_files_written_since(start: float) -> list[str]:
    written = []
    for dirpath, dirnames, filenames in os.walk(_REPO):
        dirnames[:] = [
            d for d in dirnames if d not in _UNWATCHED_DIRS and not d.endswith(".egg-info")
        ]
        for name in filenames:
            path = Path(dirpath, name)
            if path.stat().st_mtime >= start:
                written.append(path.relative_to(_REPO).as_posix())
    if not written:
        return []
    result = subprocess.run(
        ["git", "-C", str(_REPO), "check-ignore", "--stdin"],
        input="\n".join(written),
        capture_output=True,
        text=True,
    )
    # Exit 1 means none are ignored; 128 means git failed (no repo, no git).
    return sorted(result.stdout.split()) if result.returncode == 0 else []


def pytest_sessionfinish(session, exitstatus):
    written = _ignored_files_written_since(_run_started)
    if not written:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.ensure_newline()
        reporter.write_sep("=", "tests wrote runtime data into the repo", red=True)
        for rel in written:
            reporter.write_line(f"  {rel}")
        reporter.write_line(
            "Write under acuity.system.paths.data_dir() (tmp_path in tests), "
            "never relative to the repo."
        )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """The per-test data folder (ACUITY_DATA_DIR)."""
    from acuity.system.paths import DATA_DIR_ENV

    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture()
def silent_sound_player(monkeypatch):
    """Replace AlertManager's SoundPlayer with a recording stub.

    For tests that build a MainWindow: QtMultimedia can hang on some
    machines, so only test_alert_manager.py (its own time-limited CI
    step) creates a real QMediaPlayer.
    """
    from unittest.mock import MagicMock

    import acuity.core.alert_manager as alert_manager

    monkeypatch.setattr(alert_manager, "SoundPlayer", MagicMock())


@pytest.fixture(autouse=True)
def settings_dir(tmp_path):
    """The per-test folder that holds ConfigManager's INI file."""
    folder = tmp_path / "settings"
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(folder))
    return folder
