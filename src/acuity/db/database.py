# SQLite connection manager (singleton, thread-safe).
# ─────────────────────────────────────────────────────────────────
# Provides a single DatabaseManager accessed via get_db().
#
# Thread: any.  The GUI thread and the CV worker share one connection.
#
# Design decisions:
#   • check_same_thread=False  – the CV pipeline runs on a QThread
#     and must INSERT events while the GUI thread SELECTs for
#     Insights.  SQLite in WAL mode supports this safely.
#   • Auto-schema creation     – tables are created on first connect
#     so there is zero manual setup for end users.
#   • Connection is opened once and reused for the lifetime of the
#     application; closed explicitly on shutdown via close().
#   • The file lives in the per-user data folder from system.paths
#     (ACUITY_DATA_DIR overrides it), resolved when the manager is
#     created, so tests can point it at a tmp path.
#   • No migrations: this schema started fresh with the open-source
#     port, so every column is in its CREATE TABLE.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from acuity.system.paths import data_dir
from acuity.utils.constants import DB_FILENAME
from acuity.utils.logger import get_logger

log = get_logger(__name__)

# ── SQL: Schema Definition ─────────────────────────────────────
# Tables: Sessions → FocusBlocks → Events
#                  → Rests

_CREATE_SESSIONS_TABLE = """\
CREATE TABLE IF NOT EXISTS Sessions (
    session_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time       TEXT    NOT NULL,   -- ISO-8601 timestamp
    end_time         TEXT,               -- NULL while session is active
    mode             TEXT    NOT NULL,   -- 'Structured' | 'Fluid'
    total_focus_time REAL    DEFAULT 0,  -- seconds of actual focus
    total_rest_time  REAL    DEFAULT 0   -- seconds of actual rest
);
"""

_CREATE_FOCUS_BLOCKS_TABLE = """\
CREATE TABLE IF NOT EXISTS FocusBlocks (
    block_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL,
    block_number INTEGER NOT NULL,       -- 1-based within the session
    start_time   TEXT    NOT NULL,       -- ISO-8601 timestamp
    end_time     TEXT,                   -- NULL while block is active
    duration     REAL    DEFAULT 0,     -- seconds (active focus, excludes pauses)
    FOREIGN KEY (session_id) REFERENCES Sessions(session_id)
        ON DELETE CASCADE
);
"""

_CREATE_RESTS_TABLE = """\
CREATE TABLE IF NOT EXISTS Rests (
    rest_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL,
    block_id         INTEGER NOT NULL,   -- the focus block this rest follows
    rest_type        TEXT    NOT NULL,   -- 'Short' | 'Long' | 'Fluid'
    start_time       TEXT    NOT NULL,   -- ISO-8601 timestamp
    end_time         TEXT,               -- NULL while resting
    planned_duration REAL    NOT NULL,   -- seconds (what was prescribed)
    actual_duration  REAL    DEFAULT 0,  -- seconds (what actually happened)
    skipped          INTEGER DEFAULT 0,  -- 1 if user clicked Skip
    FOREIGN KEY (session_id) REFERENCES Sessions(session_id)
        ON DELETE CASCADE,
    FOREIGN KEY (block_id)   REFERENCES FocusBlocks(block_id)
        ON DELETE CASCADE
);
"""

_CREATE_EVENTS_TABLE = """\
CREATE TABLE IF NOT EXISTS Events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,         -- denormalised for fast session queries
    block_id   INTEGER NOT NULL,         -- for per-block analytics
    timestamp  TEXT    NOT NULL,          -- ISO-8601 timestamp
    event_type TEXT    NOT NULL,          -- see constants.EVENT_TYPES
    duration   REAL    DEFAULT 0,        -- seconds
    reason     TEXT,                     -- engine reason; NULL = none given
    label      TEXT,                     -- user's verdict; NULL = unlabelled
    FOREIGN KEY (session_id) REFERENCES Sessions(session_id)
        ON DELETE CASCADE,
    FOREIGN KEY (block_id)   REFERENCES FocusBlocks(block_id)
        ON DELETE CASCADE
);
"""

# ── Indexes ─────────────────────────────────────────────────────
_CREATE_INDEXES = (
    # FocusBlocks by session (list blocks in a session)
    "CREATE INDEX IF NOT EXISTS idx_focusblocks_session ON FocusBlocks(session_id);",
    # Rests by session (rest analytics per session)
    "CREATE INDEX IF NOT EXISTS idx_rests_session ON Rests(session_id);",
    # Rests by block (find the rest that followed a given block)
    "CREATE INDEX IF NOT EXISTS idx_rests_block ON Rests(block_id);",
    # Events by session (live event log, session-level queries)
    "CREATE INDEX IF NOT EXISTS idx_events_session ON Events(session_id);",
    # Events by block (per-block analytics)
    "CREATE INDEX IF NOT EXISTS idx_events_block ON Events(block_id);",
    # Events by timestamp (date-range filtering on Insights tab)
    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON Events(timestamp);",
)


class DatabaseManager:
    """Thread-safe SQLite connection manager with auto-schema creation.

    Intended to be used as a singleton via :func:`get_db`.  The file is
    ``DB_FILENAME`` in :func:`acuity.system.paths.data_dir`.
    """

    def __init__(self) -> None:
        self._db_path = data_dir() / DB_FILENAME
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # ── Connection lifecycle ────────────────────────────────────

    def connect(self) -> sqlite3.Connection:
        """Open (or return the existing) database connection.

        On first call the data directory is created, WAL mode is
        enabled, foreign-key enforcement is turned on, and all
        schema tables are created if they do not already exist.
        """
        if self._conn is not None:
            return self._conn

        with self._lock:
            # Double-check after acquiring lock
            if self._conn is not None:
                return self._conn

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Opening database at %s", self._db_path)

            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            # Return rows as sqlite3.Row for dict-like access
            self._conn.row_factory = sqlite3.Row

            # Enable WAL for concurrent reads/writes across threads
            self._conn.execute("PRAGMA journal_mode=WAL;")
            # Enforce foreign-key constraints
            self._conn.execute("PRAGMA foreign_keys=ON;")

            self._create_schema()
            log.info("Database ready (WAL mode, FK enforced).")

        return self._conn

    def close(self) -> None:
        """Close the database connection and release resources."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    log.info("Database connection closed.")
                except sqlite3.Error as exc:
                    log.error("Error closing database: %s", exc)
                finally:
                    self._conn = None

    # ── Schema bootstrap ────────────────────────────────────────

    def _create_schema(self) -> None:
        """Execute all CREATE TABLE / INDEX statements."""
        assert self._conn is not None
        with self._conn:
            self._conn.execute(_CREATE_SESSIONS_TABLE)
            self._conn.execute(_CREATE_FOCUS_BLOCKS_TABLE)
            self._conn.execute(_CREATE_RESTS_TABLE)
            self._conn.execute(_CREATE_EVENTS_TABLE)
            for idx_sql in _CREATE_INDEXES:
                self._conn.execute(idx_sql)
        log.debug("Schema verified / created.")

    # ── Convenience ─────────────────────────────────────────────

    @property
    def connection(self) -> sqlite3.Connection:
        """Shortcut that auto-connects on first access."""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    @property
    def path(self) -> Path:
        """Return the resolved database file path."""
        return self._db_path

    def __repr__(self) -> str:
        state = "connected" if self._conn else "disconnected"
        return f"<DatabaseManager path={self._db_path!s} {state}>"


# ── Module-level singleton ──────────────────────────────────────
_instance: DatabaseManager | None = None
_instance_lock = threading.Lock()


def get_db() -> DatabaseManager:
    """Return the application-wide DatabaseManager singleton.

    Thread-safe; the instance is created on first call and reused
    for the entire application lifetime.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DatabaseManager()
    return _instance
