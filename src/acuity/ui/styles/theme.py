# Dark/Light theme QSS generator and palette constants.
# ─────────────────────────────────────────────────────────────────
# Public API:
#   DARK   – palette dict for the dark theme
#   LIGHT  – palette dict for the light theme
#   apply_theme(app: QApplication, mode: str) -> None
#       mode: "dark" | "light"
#
# Thread: GUI.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Literal

from PySide6.QtWidgets import QApplication

# ── Palette token definitions ─────────────────────────────────────
# Each palette is a plain dict; keys map 1-to-1 to the CSS custom-
# property names used as comments inside the QSS files.
# All colours are sRGB hex strings.

DARK: dict[str, str] = {
    # Surfaces
    "bg": "#0F0F11",  # window / base background
    "surface": "#1A1A1F",  # cards, panels, tab content
    "surface_alt": "#232329",  # slightly lifted (hover rows, etc.)
    "border": "#2C2C34",  # subtle dividers
    # Text
    "text_primary": "#E2E2E7",  # body text
    "text_secondary": "#888895",  # captions, placeholders
    "text_disabled": "#4A4A55",  # disabled labels / buttons
    # Accent (calm focus-blue)
    "accent": "#5B8AF5",  # primary interactive colour
    "accent_hover": "#7AA3FF",  # button/tab hover
    "accent_pressed": "#4571D8",  # button press
    # Semantic
    "success": "#4CAF82",  # session running / on-track
    "warning": "#E0A843",  # mild alert
    "danger": "#E05C5C",  # distraction / fatigue alert
    # Scrollbar
    "scroll_handle": "#3A3A45",
    "scroll_track": "#1A1A1F",
}

LIGHT: dict[str, str] = {
    # Surfaces
    "bg": "#F5F5F7",
    "surface": "#FFFFFF",
    "surface_alt": "#EBEBEF",
    "border": "#D1D1DA",
    # Text
    "text_primary": "#1A1A1F",
    "text_secondary": "#6B6B78",
    "text_disabled": "#ABABBB",
    # Accent (same hue, slightly darker for contrast on white)
    "accent": "#4A7AE8",
    "accent_hover": "#3665CC",
    "accent_pressed": "#2A55B8",
    # Semantic
    "success": "#2E9E66",
    "warning": "#C68A20",
    "danger": "#C84040",
    # Scrollbar
    "scroll_handle": "#C0C0CC",
    "scroll_track": "#EBEBEF",
}

# ── QSS file paths ────────────────────────────────────────────────
_STYLES_DIR = Path(__file__).parent
_QSS_FILES: dict[str, Path] = {
    "dark": _STYLES_DIR / "dark_theme.qss",
    "light": _STYLES_DIR / "light_theme.qss",
}


def apply_theme(app: QApplication, mode: Literal["dark", "light"] = "dark") -> None:
    """Load the QSS for *mode* and apply it to *app*.

    Raises FileNotFoundError if the QSS file for the requested mode is
    missing — fail loud (universal_rules Rule 12).
    """
    qss_path = _QSS_FILES.get(mode)
    if qss_path is None:
        raise ValueError(f"Unknown theme mode: {mode!r}. Expected 'dark' or 'light'.")
    if not qss_path.exists():
        raise FileNotFoundError(f"QSS file not found: {qss_path}")

    qss = qss_path.read_text(encoding="utf-8")
    app.setStyleSheet(qss)
