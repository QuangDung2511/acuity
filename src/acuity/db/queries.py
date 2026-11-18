# All INSERT / SELECT queries (sessions, focus blocks, rests, events, aggregations).
# ─────────────────────────────────────────────────────────────────
# Every public function acquires the connection from DatabaseManager
# and returns typed model objects or plain dicts for aggregations.
#
# Thread: any (see database.py).  Each function is atomic within a single
# ``with conn:`` block.  WAL mode in database.py ensures the
# CV-pipeline QThread can INSERT while the GUI thread SELECTs.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import date, datetime, timedelta

from acuity.db.database import get_db
from acuity.db.models import Event, FocusBlock, Rest, Session
from acuity.utils.constants import (
    EVENT_LABEL_CORRECT,
    EVENT_LABEL_FALSE_ALARM,
    SESSION_MODE_DEFAULT,
)
from acuity.utils.logger import get_logger

log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════
#  SESSION QUERIES
# ════════════════════════════════════════════════════════════════


def create_session(
    mode: str | None = None,
    start_time: str | None = None,
) -> int:
    """Insert a new session and return its ``session_id``.

    Parameters
    ----------
    mode : str, optional
        ``'Structured'`` or ``'Fluid'``.  Defaults to
        ``SESSION_MODE_DEFAULT`` (Structured).
    start_time : str, optional
        ISO-8601 timestamp.  Defaults to *now* if omitted.
    """
    if mode is None:
        mode = SESSION_MODE_DEFAULT
    if start_time is None:
        start_time = datetime.now().isoformat(timespec="seconds")

    conn = get_db().connection
    with conn:
        cursor = conn.execute(
            "INSERT INTO Sessions "
            "(start_time, end_time, mode, total_focus_time, total_rest_time) "
            "VALUES (?, NULL, ?, 0, 0);",
            (start_time, mode),
        )
    session_id = cursor.lastrowid
    log.info("Session created: id=%d, mode=%s", session_id, mode)
    return session_id


def end_session(
    session_id: int,
    total_focus_time: float,
    total_rest_time: float = 0.0,
) -> None:
    """Finalise an active session.

    Parameters
    ----------
    session_id : int
        ID of the session to close.
    total_focus_time : float
        Accumulated focus time in seconds.
    total_rest_time : float
        Accumulated rest time in seconds.
    """
    end_time = datetime.now().isoformat(timespec="seconds")
    conn = get_db().connection
    with conn:
        conn.execute(
            "UPDATE Sessions "
            "SET end_time = ?, total_focus_time = ?, total_rest_time = ? "
            "WHERE session_id = ?;",
            (end_time, total_focus_time, total_rest_time, session_id),
        )
    log.info(
        "Session ended: id=%d, focus=%.1fs, rest=%.1fs",
        session_id,
        total_focus_time,
        total_rest_time,
    )


def get_session(session_id: int) -> Session | None:
    """Fetch a single session by ID, or ``None`` if not found."""
    row = (
        get_db()
        .connection.execute(
            "SELECT * FROM Sessions WHERE session_id = ?;",
            (session_id,),
        )
        .fetchone()
    )
    return Session.from_row(row) if row else None


def get_active_session() -> Session | None:
    """Return the currently active (un-ended) session, if any."""
    row = (
        get_db()
        .connection.execute(
            "SELECT * FROM Sessions WHERE end_time IS NULL ORDER BY session_id DESC LIMIT 1;",
        )
        .fetchone()
    )
    return Session.from_row(row) if row else None


def get_sessions_by_date(target_date: date | str) -> list[Session]:
    """Return all sessions whose *start_time* falls on *target_date*."""
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM Sessions WHERE date(start_time) = ? ORDER BY start_time;",
            (target_date,),
        )
        .fetchall()
    )
    return [Session.from_row(r) for r in rows]


def get_sessions_in_range(
    start_date: date | str,
    end_date: date | str,
) -> list[Session]:
    """Return all sessions within a date range (inclusive)."""
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM Sessions WHERE date(start_time) BETWEEN ? AND ? ORDER BY start_time;",
            (start_date, end_date),
        )
        .fetchall()
    )
    return [Session.from_row(r) for r in rows]


def get_all_sessions() -> list[Session]:
    """Return every session ordered by most recent first."""
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM Sessions ORDER BY start_time DESC;",
        )
        .fetchall()
    )
    return [Session.from_row(r) for r in rows]


# ════════════════════════════════════════════════════════════════
#  FOCUS BLOCK QUERIES
# ════════════════════════════════════════════════════════════════


def start_focus_block(
    session_id: int,
    block_number: int,
    start_time: str | None = None,
) -> int:
    """Insert a new focus block and return its ``block_id``.

    Parameters
    ----------
    session_id : int
        Parent session.
    block_number : int
        1-based ordinal within the session.
    start_time : str, optional
        ISO-8601 timestamp.  Defaults to *now*.
    """
    if start_time is None:
        start_time = datetime.now().isoformat(timespec="seconds")

    conn = get_db().connection
    with conn:
        cursor = conn.execute(
            "INSERT INTO FocusBlocks "
            "(session_id, block_number, start_time, end_time, duration) "
            "VALUES (?, ?, ?, NULL, 0);",
            (session_id, block_number, start_time),
        )
    block_id = cursor.lastrowid
    log.info(
        "Focus block started: id=%d, session=%d, block=#%d",
        block_id,
        session_id,
        block_number,
    )
    return block_id


def end_focus_block(block_id: int, duration: float) -> None:
    """Finalise a focus block with its active duration.

    Parameters
    ----------
    block_id : int
        ID of the block to close.
    duration : float
        Active focus time in seconds (excludes paused time).
    """
    end_time = datetime.now().isoformat(timespec="seconds")
    conn = get_db().connection
    with conn:
        conn.execute(
            "UPDATE FocusBlocks SET end_time = ?, duration = ? WHERE block_id = ?;",
            (end_time, duration, block_id),
        )
    log.info("Focus block ended: id=%d, duration=%.1fs", block_id, duration)


def get_blocks_for_session(session_id: int) -> list[FocusBlock]:
    """Return all focus blocks in a session, ordered by block_number."""
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM FocusBlocks WHERE session_id = ? ORDER BY block_number;",
            (session_id,),
        )
        .fetchall()
    )
    return [FocusBlock.from_row(r) for r in rows]


def get_current_block(session_id: int) -> FocusBlock | None:
    """Return the active (un-ended) focus block for a session."""
    row = (
        get_db()
        .connection.execute(
            "SELECT * FROM FocusBlocks "
            "WHERE session_id = ? AND end_time IS NULL "
            "ORDER BY block_id DESC LIMIT 1;",
            (session_id,),
        )
        .fetchone()
    )
    return FocusBlock.from_row(row) if row else None


def get_focus_block_count(session_id: int) -> int:
    """Return the number of completed focus blocks in a session."""
    row = (
        get_db()
        .connection.execute(
            "SELECT COUNT(*) AS cnt FROM FocusBlocks "
            "WHERE session_id = ? AND end_time IS NOT NULL;",
            (session_id,),
        )
        .fetchone()
    )
    return int(row["cnt"])


# ════════════════════════════════════════════════════════════════
#  REST QUERIES
# ════════════════════════════════════════════════════════════════


def start_rest(
    session_id: int,
    block_id: int,
    rest_type: str,
    planned_duration: float,
    start_time: str | None = None,
) -> int:
    """Insert a new rest period and return its ``rest_id``.

    Parameters
    ----------
    session_id : int
        Parent session.
    block_id : int
        The focus block this rest follows.
    rest_type : str
        ``'Short'``, ``'Long'``, or ``'Fluid'``.
    planned_duration : float
        Prescribed rest duration in seconds.
    start_time : str, optional
        ISO-8601 timestamp.  Defaults to *now*.
    """
    if start_time is None:
        start_time = datetime.now().isoformat(timespec="seconds")

    conn = get_db().connection
    with conn:
        cursor = conn.execute(
            "INSERT INTO Rests "
            "(session_id, block_id, rest_type, start_time, end_time, "
            " planned_duration, actual_duration, skipped) "
            "VALUES (?, ?, ?, ?, NULL, ?, 0, 0);",
            (session_id, block_id, rest_type, start_time, planned_duration),
        )
    rest_id = cursor.lastrowid
    log.info(
        "Rest started: id=%d, type=%s, planned=%.0fs",
        rest_id,
        rest_type,
        planned_duration,
    )
    return rest_id


def end_rest(rest_id: int, actual_duration: float) -> None:
    """Finalise a rest period that completed naturally.

    Parameters
    ----------
    rest_id : int
        ID of the rest to close.
    actual_duration : float
        Actual elapsed rest time in seconds.
    """
    end_time = datetime.now().isoformat(timespec="seconds")
    conn = get_db().connection
    with conn:
        conn.execute(
            "UPDATE Rests SET end_time = ?, actual_duration = ? WHERE rest_id = ?;",
            (end_time, actual_duration, rest_id),
        )
    log.info("Rest ended: id=%d, actual=%.1fs", rest_id, actual_duration)


def skip_rest(rest_id: int, actual_duration: float) -> None:
    """Mark a rest as skipped, recording how long the user actually rested.

    Parameters
    ----------
    rest_id : int
        ID of the rest being skipped.
    actual_duration : float
        Seconds the user rested before clicking Skip.
        0 if they skipped immediately.
    """
    end_time = datetime.now().isoformat(timespec="seconds")
    conn = get_db().connection
    with conn:
        conn.execute(
            "UPDATE Rests SET end_time = ?, actual_duration = ?, skipped = 1 WHERE rest_id = ?;",
            (end_time, actual_duration, rest_id),
        )
    log.info(
        "Rest skipped: id=%d, actual=%.1fs",
        rest_id,
        actual_duration,
    )


def get_rests_for_session(session_id: int) -> list[Rest]:
    """Return all rests in a session, ordered chronologically."""
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM Rests WHERE session_id = ? ORDER BY start_time;",
            (session_id,),
        )
        .fetchall()
    )
    return [Rest.from_row(r) for r in rows]


def get_active_rest() -> Rest | None:
    """Return the currently active rest (end_time IS NULL), if any."""
    row = (
        get_db()
        .connection.execute(
            "SELECT * FROM Rests WHERE end_time IS NULL ORDER BY rest_id DESC LIMIT 1;",
        )
        .fetchone()
    )
    return Rest.from_row(row) if row else None


# ════════════════════════════════════════════════════════════════
#  EVENT QUERIES
# ════════════════════════════════════════════════════════════════


def insert_event(
    session_id: int,
    block_id: int,
    event_type: str,
    duration: float = 0.0,
    timestamp: str | None = None,
    reason: str | None = None,
) -> int:
    """Insert an event and return its ``event_id``.

    Parameters
    ----------
    session_id : int
        Parent session (denormalised for fast queries).
    block_id : int
        Focus block where this event occurred.
    event_type : str
        One of ``constants.EVENT_TYPES``.
    duration : float
        Duration in seconds.
    timestamp : str, optional
        ISO-8601 timestamp.  Defaults to *now*.
    reason : str, optional
        Engine reason (``'microsleep'``, ``'sleep'``, …).  ``None``
        when the engine gives none.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    conn = get_db().connection
    with conn:
        cursor = conn.execute(
            "INSERT INTO Events "
            "(session_id, block_id, timestamp, event_type, duration, reason) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (session_id, block_id, timestamp, event_type, duration, reason),
        )
    event_id = cursor.lastrowid
    log.debug(
        "Event logged: id=%d session=%d block=%d type=%s dur=%.1fs reason=%s",
        event_id,
        session_id,
        block_id,
        event_type,
        duration,
        reason,
    )
    return event_id


def set_event_label(event_id: int, label: str | None) -> None:
    """Record the user's verdict on an event (``None`` clears it)."""
    if label not in (EVENT_LABEL_CORRECT, EVENT_LABEL_FALSE_ALARM, None):
        raise ValueError(f"Unknown event label: {label!r}")
    conn = get_db().connection
    with conn:
        conn.execute(
            "UPDATE Events SET label = ? WHERE event_id = ?;",
            (label, event_id),
        )
    log.debug("Event %d labelled %s", event_id, label)


def get_events_for_session(session_id: int) -> list[Event]:
    """Return all events in a session, ordered chronologically."""
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM Events WHERE session_id = ? ORDER BY timestamp;",
            (session_id,),
        )
        .fetchall()
    )
    return [Event.from_row(r) for r in rows]


def get_events_for_block(block_id: int) -> list[Event]:
    """Return all events in a specific focus block."""
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM Events WHERE block_id = ? ORDER BY timestamp;",
            (block_id,),
        )
        .fetchall()
    )
    return [Event.from_row(r) for r in rows]


def get_events_by_type(
    session_id: int,
    event_type: str,
) -> list[Event]:
    """Return events of a specific type within a session."""
    rows = (
        get_db()
        .connection.execute(
            "SELECT * FROM Events WHERE session_id = ? AND event_type = ? ORDER BY timestamp;",
            (session_id, event_type),
        )
        .fetchall()
    )
    return [Event.from_row(r) for r in rows]


def get_events_in_range(
    start_date: date | str,
    end_date: date | str,
    event_type: str | None = None,
) -> list[Event]:
    """Return events within a date range, optionally filtered by type."""
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()

    if event_type:
        rows = (
            get_db()
            .connection.execute(
                "SELECT * FROM Events "
                "WHERE date(timestamp) BETWEEN ? AND ? "
                "  AND event_type = ? "
                "ORDER BY timestamp;",
                (start_date, end_date, event_type),
            )
            .fetchall()
        )
    else:
        rows = (
            get_db()
            .connection.execute(
                "SELECT * FROM Events WHERE date(timestamp) BETWEEN ? AND ? ORDER BY timestamp;",
                (start_date, end_date),
            )
            .fetchall()
        )
    return [Event.from_row(r) for r in rows]


# ════════════════════════════════════════════════════════════════
#  AGGREGATION QUERIES  (Insights Tab)
# ════════════════════════════════════════════════════════════════

# ── Focus Aggregations ──────────────────────────────────────────


def get_daily_focus_total(target_date: date | str) -> float:
    """Return total focus time (seconds) for a given date.

    Sums closed focus blocks rather than ``Sessions.total_focus_time``.
    A session's total is written only when the session ends, so one the
    app never closed (killed, crashed) reports 0 even though its
    completed blocks were persisted (7 such sessions in the private
    corpus, n=1).  Summing blocks also lets the total advance at
    each block end instead of only at session end.
    """
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    row = (
        get_db()
        .connection.execute(
            "SELECT COALESCE(SUM(duration), 0) AS total "
            "FROM FocusBlocks "
            "WHERE end_time IS NOT NULL AND date(start_time) = ?;",
            (target_date,),
        )
        .fetchone()
    )
    return float(row["total"])


def get_daily_rest_total(target_date: date | str) -> float:
    """Return total actual rest time (seconds) for a given date."""
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    row = (
        get_db()
        .connection.execute(
            "SELECT COALESCE(SUM(actual_duration), 0) AS total "
            "FROM Rests WHERE date(start_time) = ?;",
            (target_date,),
        )
        .fetchone()
    )
    return float(row["total"])


def get_focus_rest_ratio(target_date: date | str) -> float:
    """Return ``focus / rest`` ratio for a given date.

    Returns 0.0 if no rest was taken (avoids division by zero).
    """
    focus = get_daily_focus_total(target_date)
    rest = get_daily_rest_total(target_date)
    return focus / rest if rest > 0 else 0.0


# ── Event Aggregations ─────────────────────────────────────────


def get_daily_event_counts(target_date: date | str) -> dict[str, int]:
    """Return ``{event_type: count}`` for a given date."""
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT event_type, COUNT(*) AS cnt FROM Events "
            "WHERE date(timestamp) = ? GROUP BY event_type;",
            (target_date,),
        )
        .fetchall()
    )
    return {r["event_type"]: r["cnt"] for r in rows}


def get_event_counts_in_range(
    start_date: date | str,
    end_date: date | str,
) -> dict[str, int]:
    """Return ``{event_type: count}`` aggregated over a date range."""
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT event_type, COUNT(*) AS cnt FROM Events "
            "WHERE date(timestamp) BETWEEN ? AND ? "
            "GROUP BY event_type;",
            (start_date, end_date),
        )
        .fetchall()
    )
    return {r["event_type"]: r["cnt"] for r in rows}


def get_daily_event_timeline(target_date: date | str) -> list[dict]:
    """Return events grouped by hour for a given date.

    Returns ``[{'hour': int, 'event_type': str, 'count': int}, ...]``
    """
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT CAST(strftime('%%H', timestamp) AS INTEGER) AS hour, "
            "       event_type, COUNT(*) AS cnt "
            "FROM Events WHERE date(timestamp) = ? "
            "GROUP BY hour, event_type ORDER BY hour;",
            (target_date,),
        )
        .fetchall()
    )
    return [{"hour": r["hour"], "event_type": r["event_type"], "count": r["cnt"]} for r in rows]


# ── Per-Block Analytics ────────────────────────────────────────


def get_events_per_block_position(
    start_date: date | str,
    end_date: date | str,
) -> list[dict]:
    """Return event counts grouped by block_number across sessions.

    Answers: *"Do I get more distracted in block 1 vs block 4?"*

    Returns ``[{'block_number': int, 'event_type': str, 'count': int}, ...]``
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT fb.block_number, e.event_type, COUNT(*) AS cnt "
            "FROM Events e "
            "JOIN FocusBlocks fb ON e.block_id = fb.block_id "
            "WHERE date(e.timestamp) BETWEEN ? AND ? "
            "GROUP BY fb.block_number, e.event_type "
            "ORDER BY fb.block_number;",
            (start_date, end_date),
        )
        .fetchall()
    )
    return [
        {"block_number": r["block_number"], "event_type": r["event_type"], "count": r["cnt"]}
        for r in rows
    ]


def get_block_duration_trend(
    start_date: date | str,
    end_date: date | str,
) -> list[dict]:
    """Return average focus duration per block position.

    Answers: *"Does my focus duration drop off in later blocks?"*

    Returns ``[{'block_number': int, 'avg_duration': float}, ...]``
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT block_number, AVG(duration) AS avg_dur "
            "FROM FocusBlocks "
            "WHERE end_time IS NOT NULL "
            "  AND date(start_time) BETWEEN ? AND ? "
            "GROUP BY block_number "
            "ORDER BY block_number;",
            (start_date, end_date),
        )
        .fetchall()
    )
    return [
        {"block_number": r["block_number"], "avg_duration": round(float(r["avg_dur"]), 1)}
        for r in rows
    ]


# ── Fatigue Analytics ──────────────────────────────────────────


def get_fatigue_onset_times(
    start_date: date | str,
    end_date: date | str,
) -> list[dict]:
    """Return fatigue events with elapsed time into their focus block.

    Returns ``[{'date': str, 'block_number': int,
    'fatigue_time': str, 'elapsed_min': float}, ...]``
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT date(e.timestamp) AS event_date, "
            "       fb.block_number, "
            "       e.timestamp AS fatigue_time, "
            "       (julianday(e.timestamp) - julianday(fb.start_time)) "
            "           * 1440 AS elapsed_min "
            "FROM Events e "
            "JOIN FocusBlocks fb ON e.block_id = fb.block_id "
            "WHERE e.event_type = 'Fatigue' "
            "  AND date(e.timestamp) BETWEEN ? AND ? "
            "ORDER BY e.timestamp;",
            (start_date, end_date),
        )
        .fetchall()
    )
    return [
        {
            "date": r["event_date"],
            "block_number": r["block_number"],
            "fatigue_time": r["fatigue_time"],
            "elapsed_min": round(float(r["elapsed_min"]), 1),
        }
        for r in rows
    ]


# ── Rest Analytics ─────────────────────────────────────────────


def get_rest_compliance_rate(
    start_date: date | str,
    end_date: date | str,
) -> float:
    """Return average rest compliance (actual/planned) over a date range.

    1.0 = user always takes the full prescribed rest.
    0.0 = no rests taken or all skipped immediately.
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    row = (
        get_db()
        .connection.execute(
            "SELECT AVG(CASE WHEN planned_duration > 0 "
            "   THEN actual_duration * 1.0 / planned_duration "
            "   ELSE 0 END) AS compliance "
            "FROM Rests "
            "WHERE date(start_time) BETWEEN ? AND ?;",
            (start_date, end_date),
        )
        .fetchone()
    )
    val = row["compliance"]
    return round(float(val), 3) if val is not None else 0.0


# ── Streak Calculation ──────────────────────────────────────────


def get_focus_streak() -> int:
    """Return the current consecutive-day focus streak.

    A day counts if it has at least one completed session.
    The streak is counted backwards from today; if today has no
    session yet, the streak starts from yesterday.
    """
    conn = get_db().connection
    rows = conn.execute(
        "SELECT DISTINCT date(start_time) AS d "
        "FROM Sessions WHERE end_time IS NOT NULL "
        "ORDER BY d DESC;",
    ).fetchall()

    if not rows:
        return 0

    dates = [date.fromisoformat(r["d"]) for r in rows]
    today = date.today()

    # If most recent session is older than yesterday, streak is broken
    if dates[0] < today - timedelta(days=1):
        return 0

    streak = 0
    expected = dates[0]  # start from the most recent day with a session

    for d in dates:
        if d == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif d < expected:
            break
    return streak


# ── Range Analytics (Insights tab) ─────────────────────────────
# Plain-dict rows consumed by acuity.core.insights_metrics.  Blocks and
# events share one filter: closed blocks lasting at least *min_block_s*.
# An unclosed block (app killed mid-block, or the block still running)
# has duration 0, so counting its events would inflate every
# per-focus-hour rate; they join the analytics once the block ends.


def get_focus_blocks_in_range(
    start_date: date | str,
    end_date: date | str,
    min_block_s: float,
) -> list[dict]:
    """Return closed focus blocks starting within a date range (inclusive).

    Returns ``[{'block_id', 'session_id', 'block_number', 'start_time',
    'duration', 'mode'}, ...]`` ordered by start time.
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT fb.block_id, fb.session_id, fb.block_number, "
            "       fb.start_time, fb.duration, s.mode "
            "FROM FocusBlocks fb "
            "JOIN Sessions s ON s.session_id = fb.session_id "
            "WHERE fb.end_time IS NOT NULL AND fb.duration >= ? "
            "  AND date(fb.start_time) BETWEEN ? AND ? "
            "ORDER BY fb.start_time;",
            (min_block_s, start_date, end_date),
        )
        .fetchall()
    )
    return [dict(r) for r in rows]


def get_events_in_range_with_offset(
    start_date: date | str,
    end_date: date | str,
    min_block_s: float,
) -> list[dict]:
    """Return events in a date range with minutes since their block started.

    Only events inside blocks that pass the same filter as
    :func:`get_focus_blocks_in_range`.  ``elapsed_min`` is wall clock
    from block start, so it includes calibration and any pauses.

    Returns ``[{'event_id', 'block_id', 'timestamp', 'event_type',
    'reason', 'label', 'elapsed_min'}, ...]`` ordered by timestamp.
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT e.event_id, e.block_id, e.timestamp, e.event_type, e.reason, e.label, "
            "       (julianday(e.timestamp) - julianday(fb.start_time)) "
            "           * 1440 AS elapsed_min "
            "FROM Events e "
            "JOIN FocusBlocks fb ON e.block_id = fb.block_id "
            "WHERE fb.end_time IS NOT NULL AND fb.duration >= ? "
            "  AND date(e.timestamp) BETWEEN ? AND ? "
            "ORDER BY e.timestamp;",
            (min_block_s, start_date, end_date),
        )
        .fetchall()
    )
    return [dict(r) for r in rows]


def get_rests_in_range(
    start_date: date | str,
    end_date: date | str,
) -> list[dict]:
    """Return closed rests starting within a date range (inclusive).

    Returns ``[{'rest_id', 'rest_type', 'start_time', 'planned_duration',
    'actual_duration', 'skipped'}, ...]`` ordered by start time.
    """
    if isinstance(start_date, date):
        start_date = start_date.isoformat()
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    rows = (
        get_db()
        .connection.execute(
            "SELECT rest_id, rest_type, start_time, planned_duration, "
            "       actual_duration, skipped "
            "FROM Rests "
            "WHERE end_time IS NOT NULL AND date(start_time) BETWEEN ? AND ? "
            "ORDER BY start_time;",
            (start_date, end_date),
        )
        .fetchall()
    )
    return [dict(r) for r in rows]


def get_earliest_block_date(min_block_s: float) -> date | None:
    """Return the date of the first closed block ≥ *min_block_s*, if any."""
    row = (
        get_db()
        .connection.execute(
            "SELECT MIN(date(start_time)) AS d FROM FocusBlocks "
            "WHERE end_time IS NOT NULL AND duration >= ?;",
            (min_block_s,),
        )
        .fetchone()
    )
    return date.fromisoformat(row["d"]) if row["d"] else None


# ── Housekeeping ────────────────────────────────────────────────


def delete_sessions_before(cutoff_date: date | str) -> int:
    """Delete sessions (and cascading children) older than *cutoff_date*.

    Returns the number of deleted session rows.
    """
    if isinstance(cutoff_date, date):
        cutoff_date = cutoff_date.isoformat()
    conn = get_db().connection
    with conn:
        cursor = conn.execute(
            "DELETE FROM Sessions WHERE date(start_time) < ?;",
            (cutoff_date,),
        )
    deleted = cursor.rowcount
    log.info("Housekeeping: deleted %d sessions before %s", deleted, cutoff_date)
    return deleted


def get_session_count() -> int:
    """Return the total number of sessions in the database."""
    row = (
        get_db()
        .connection.execute(
            "SELECT COUNT(*) AS cnt FROM Sessions;",
        )
        .fetchone()
    )
    return int(row["cnt"])


def get_event_count() -> int:
    """Return the total number of events in the database."""
    row = (
        get_db()
        .connection.execute(
            "SELECT COUNT(*) AS cnt FROM Events;",
        )
        .fetchone()
    )
    return int(row["cnt"])
