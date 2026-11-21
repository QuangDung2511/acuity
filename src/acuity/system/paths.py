# Where Acuity keeps its per-user data (database, logs, debug CSVs).
# ─────────────────────────────────────────────────────────────────
# Responsibilities:
#   • data_dir(): the user data folder.  ACUITY_DATA_DIR overrides it, so
#     development and tests never touch the user's real data.
#   • log_dir(): the log folder (app log and debug CSVs), <data_dir>/logs,
#     so the same override redirects it.
#
# Thread: any.  data_dir() writes process-global Qt state (the
# organisation and application name), but always the same constant
# values, so concurrent calls agree.
#
# Design:
#   • One implementation for every OS: QStandardPaths already maps
#     AppDataLocation to each platform's per-user folder.  So, unlike the
#     other adapters, there is no create_*() factory: there is no
#     OS-specific code for one to choose.
#   • AppDataLocation is built from the application's organisation and
#     name.  Unset, Qt returns the bare per-user folder (%APPDATA% on
#     Windows), so data_dir() sets them first, to the values app.main()
#     sets (APP_ORG and APP_NAME, both "Acuity").  That gives
#     %APPDATA%\Acuity\Acuity, ~/Library/Application Support/Acuity/Acuity
#     or ~/.local/share/Acuity/Acuity.  The doubled name is Qt's
#     convention and sits next to QSettings' Acuity/Acuity.ini on Windows.
#   • Logs live under the data folder, not a separate OS log location:
#     Qt has no log location on all three OSes, and one folder keeps
#     everything a user might delete or send in a bug report together.
#   • Folders are not created here; the caller that writes creates them.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QStandardPaths

from acuity.utils.constants import APP_NAME, APP_ORG

DATA_DIR_ENV = "ACUITY_DATA_DIR"


def data_dir() -> Path:
    """Return the per-user data folder, or ``$ACUITY_DATA_DIR`` if set."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override)
    QCoreApplication.setOrganizationName(APP_ORG)
    QCoreApplication.setApplicationName(APP_NAME)
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(location)


def log_dir() -> Path:
    """Return the log folder: ``logs`` under :func:`data_dir`."""
    return data_dir() / "logs"
