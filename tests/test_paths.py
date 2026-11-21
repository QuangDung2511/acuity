# Tests for the paths adapter (data_dir, log_dir).
# ─────────────────────────────────────────────────────────────────
# Every file Acuity writes lands under these two folders, so they decide
# whether user data ends up in the right per-user place on each OS and
# whether tests and development stay out of it (ACUITY_DATA_DIR).
# Expected locations come from each OS's documented app-data folder,
# not from QStandardPaths, so the tests can disagree with the code.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
from pathlib import Path

from acuity.system import paths


def _platform_app_data() -> Path:
    """The OS's per-user app-data folder, from its own documentation."""
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def test_override_redirects_both_folders(tmp_path, monkeypatch):
    """ACUITY_DATA_DIR is how tests and development keep every write,
    the log included, out of the user's real data folder."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))

    assert paths.data_dir() == tmp_path
    assert paths.log_dir() == tmp_path / "logs"


def test_without_override_folders_sit_under_platform_app_data(monkeypatch):
    """An installed app must keep its data in the OS's per-user app-data
    folder, under the organisation and application name ("Acuity")."""
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    expected = _platform_app_data() / "Acuity" / "Acuity"

    assert paths.data_dir() == expected
    assert paths.log_dir() == expected / "logs"
