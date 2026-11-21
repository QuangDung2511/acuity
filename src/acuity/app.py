# Entry point - bootstraps QApplication, SystemTray, MainWindow.
# ─────────────────────────────────────────────────────────────────
# Deliverable:
#   • Creates QApplication
#   • Initialises logging & database
#   • Creates MainWindow (which owns AcuityTrayIcon internally)
#   • Applies dark theme
#   • Runs the Qt event loop
#
# Thread: GUI (the main thread; it owns the Qt event loop).
#
# Installed as the `acuity` console command ([project.scripts] in
# pyproject.toml), which replaces the original run.py launcher.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from acuity.db.database import get_db
from acuity.system import paths
from acuity.ui.main_window import MainWindow
from acuity.ui.styles.theme import apply_theme
from acuity.utils.config import ConfigManager
from acuity.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def main() -> int:
    """Application entry point.

    Returns the process exit code (0 = success).
    """
    # 1. Logging  (get_logger already attached the console handler at
    #    import; this adds the log file in the user log folder)
    setup_logging(paths.log_dir())
    log.info("═══ Acuity starting ═══")

    # 2. Qt application instance  (must exist before any QWidget)
    app = QApplication(sys.argv)
    app.setApplicationName("Acuity")
    app.setOrganizationName("Acuity")

    # Prevent the app from quitting when the last *visible* window is
    # closed — the tray icon keeps the session alive.
    app.setQuitOnLastWindowClosed(False)

    # 3. User configuration  (QSettings — requires QApplication)
    config = ConfigManager()
    log.info("Config loaded from %s", config.settings_path())
    log.debug("Current settings: %s", config.as_dict())

    # 4. Database  (auto-creates schema on first connect)
    db = get_db()
    db.connect()
    log.info("Database ready at %s", db.path)

    # 5. Theme
    apply_theme(app, mode="dark")

    # 6. Main window (owns AcuityTrayIcon internally)
    window = MainWindow()
    window.show()
    log.info("Main window and system tray displayed.")

    # 7. Run the event loop
    log.info("Entering Qt event loop.")
    exit_code = app.exec()

    # 8. Cleanup
    db.close()
    log.info("═══ Acuity shut down (exit code %d) ═══", exit_code)
    return exit_code
