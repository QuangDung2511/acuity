# Where Acuity keeps its per-user data (the database).
# ─────────────────────────────────────────────────────────────────
# Responsibilities:
#   • data_dir(): the user data folder.  ACUITY_DATA_DIR overrides it, so
#     development and tests never touch the user's real data.
#
# Thread: any.  data_dir() writes process-global Qt state (the
# organisation and application name), but always the same constant
# values, so concurrent calls agree.
#
# Design:
#   • Minimal, Windows-tested adapter (ticket 02).  Ticket 06 adds
#     log_dir() and the create_*() factory the package header describes,
#     and confirms macOS and Linux.
#   • AppDataLocation is built from the application's organisation and
#     name.  Unset, Qt returns the bare %APPDATA% folder, so data_dir()
#     sets them first, to the values the original app set at startup
#     (APP_ORG and APP_NAME, both "Acuity").  On Windows that gives
#     %APPDATA%\Acuity\Acuity, next to QSettings' %APPDATA%\Acuity\Acuity.ini.
#   • The folder is not created here; the caller that writes creates it.
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
