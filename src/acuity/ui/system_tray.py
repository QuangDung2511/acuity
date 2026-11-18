# QSystemTrayIcon - tray icon, context menu (Show / Quit).
# ─────────────────────────────────────────────────────────────────
# Public API:
#   AcuityTrayIcon(parent: QMainWindow) -> QSystemTrayIcon
#     .show_window()   – re-show and raise the main window
#     .notify(title, message, icon_type)
#                      – OS-level toast (wraps showMessage)
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMainWindow, QMenu, QSystemTrayIcon

from acuity.utils.constants import APP_NAME
from acuity.utils.logger import get_logger

log = get_logger(__name__)


class AcuityTrayIcon(QSystemTrayIcon):
    """System-tray icon for Acuity.

    Responsibilities:
    - Sit in the OS system tray while the app is running.
    - Provide a right-click context menu: Show | ── | Quit.
    - Expose notify() for OS-level toast messages.
    - Single-click on icon restores the main window.
    """

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(parent=window)

        self._window = window

        # ── Icon ──────────────────────────────────────────────────
        # Use Qt's built-in "application" standard icon as a fallback;
        # replace with a custom icon file when assets/icons/acuity.ico exists.
        icon = QIcon.fromTheme("applications-education")
        if icon.isNull():
            # Absolute fallback: use Qt's ComputerIcon standard pixmap
            from PySide6.QtWidgets import QApplication, QStyle

            icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setIcon(icon)
        self.setToolTip(APP_NAME)

        # ── Context menu ──────────────────────────────────────────
        menu = QMenu()

        show_action = QAction("Show", menu)
        show_action.triggered.connect(self.show_window)
        menu.addAction(show_action)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._on_quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)

        # ── Activation: single-click restores window ──────────────
        self.activated.connect(self._on_activated)

        log.debug("AcuityTrayIcon initialised.")

    # ── Public helpers ─────────────────────────────────────────────

    def show_window(self) -> None:
        """Restore and raise the main window from the tray."""
        self._window.showNormal()
        self._window.raise_()
        self._window.activateWindow()

    def notify(
        self,
        title: str,
        message: str,
        icon_type: QSystemTrayIcon.MessageIcon = QSystemTrayIcon.MessageIcon.Information,
        duration_ms: int = 4000,
    ) -> None:
        """Show an OS-level toast notification.

        Uses QSystemTrayIcon.showMessage, which on Windows delegates to
        the native WinRT/Shell notification API.

        Args:
            title:       Notification heading.
            message:     Notification body text.
            icon_type:   Information | Warning | Critical | NoIcon.
            duration_ms: Hint to the OS (Windows typically ignores this).
        """
        if not self.isSystemTrayAvailable():
            log.warning("System tray not available; toast skipped: %s – %s", title, message)
            return
        self.showMessage(title, message, icon_type, duration_ms)
        log.debug("Toast sent: %r / %r", title, message)

    # ── Private slots ──────────────────────────────────────────────

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Single-click on the tray icon restores the window."""
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_window()

    def _on_quit(self) -> None:
        """Fully terminate the application from the tray menu.

        Releases resources (camera, DB) by letting main.py's cleanup
        run through the normal QApplication.quit() path.
        """
        log.info("Quit requested from system tray.")
        from PySide6.QtWidgets import QApplication

        QApplication.quit()
