# Python logging config (console, plus a file once the app names a folder).
# ─────────────────────────────────────────────────────────────────
# Dual-sink setup:
#   1. StreamHandler       → stderr / console (INFO+), on first use
#   2. RotatingFileHandler → <log_dir>/acuity.log (DEBUG+), only once
#      app.main() calls setup_logging(log_dir)
#
# Thread: any.  The stdlib logging handlers are thread-safe.
#
# Design:
#   • Importing a module never writes a file.  The folder comes from
#     acuity.system.paths, which uses Qt; this module cannot import it,
#     because the pure core modules import the logger and must stay
#     Qt-free.  So the app passes the folder in.
#
# Usage in any module:
#   from acuity.utils.logger import get_logger
#   log = get_logger(__name__)
#   log.info("Session started")
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from acuity.utils.constants import LOG_BACKUP_COUNT, LOG_FILENAME, LOG_MAX_BYTES

# ── Formatting ──────────────────────────────────────────────────
_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour codes to log-level names for console output."""

    _COLOURS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname}{self._RESET}"
        return super().format(record)


def setup_logging(log_dir: Path | None = None) -> None:
    """Configure the root logger.

    The console handler is attached on the first call.  A rotating file
    handler in *log_dir* is attached the first time a *log_dir* is given.
    Safe to call multiple times.
    """
    root = logging.getLogger()

    # ── Console handler ─────────────────────────────────────────
    # Guard against duplicate handler attachment
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_ColourFormatter(_LOG_FMT, datefmt=_LOG_DATE_FMT))
        root.addHandler(console_handler)

    # ── File handler (rotating) ─────────────────────────────────
    if log_dir is None or any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=str(log_dir / LOG_FILENAME),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATE_FMT))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for *name* (typically ``__name__``).

    Automatically calls :func:`setup_logging` on first use so
    callers don't need to worry about initialisation order.
    """
    setup_logging()
    return logging.getLogger(name)
