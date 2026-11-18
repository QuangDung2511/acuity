# ORM-style dataclasses for Sessions, FocusBlocks, Rests, and Events.
# ─────────────────────────────────────────────────────────────────
# Each dataclass mirrors its SQLite table 1:1.
#
#   Session    ↔  Sessions table
#   FocusBlock ↔  FocusBlocks table
#   Rest       ↔  Rests table
#   Event      ↔  Events table
#
# Factory methods:
#   • from_row(sqlite3.Row) → hydrate from a SELECT result
#   • to_insert_tuple()     → values tuple for parameterised INSERT
#
# Thread: any.  Plain data, no I/O.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ════════════════════════════════════════════════════════════════
#  SESSION
# ════════════════════════════════════════════════════════════════


@dataclass
class Session:
    """Represents a single study/work session (the entire sitting).

    A session spans from when the user clicks Start to when they
    click End.  It contains multiple :class:`FocusBlock` and
    :class:`Rest` periods.

    Attributes
    ----------
    session_id : int | None
        Auto-assigned by SQLite; ``None`` before INSERT.
    start_time : str
        ISO-8601 timestamp when the session began.
    end_time : str | None
        ISO-8601 timestamp when the session ended,
        or ``None`` while still active.
    mode : str
        ``'Structured'`` (Pomodoro) or ``'Fluid'`` (Stopwatch).
    total_focus_time : float
        Accumulated focus duration in **seconds** across all blocks.
    total_rest_time : float
        Accumulated rest duration in **seconds** across all rests.
    """

    session_id: int | None = field(default=None)
    start_time: str = field(default="")
    end_time: str | None = field(default=None)
    mode: str = field(default="")
    total_focus_time: float = field(default=0.0)
    total_rest_time: float = field(default=0.0)

    # ── Factories ───────────────────────────────────────────────

    @classmethod
    def from_row(cls, row) -> Session:
        """Create a Session from a ``sqlite3.Row`` or dict-like mapping."""
        return cls(
            session_id=row["session_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            mode=row["mode"],
            total_focus_time=float(row["total_focus_time"] or 0.0),
            total_rest_time=float(row["total_rest_time"] or 0.0),
        )

    # ── Serialisation ───────────────────────────────────────────

    def to_insert_tuple(self) -> tuple:
        """Return values for a parameterised INSERT (excludes session_id).

        Matches: ``INSERT INTO Sessions
        (start_time, end_time, mode, total_focus_time, total_rest_time)``
        """
        return (
            self.start_time,
            self.end_time,
            self.mode,
            self.total_focus_time,
            self.total_rest_time,
        )

    # ── Helpers ─────────────────────────────────────────────────

    @property
    def start_datetime(self) -> datetime:
        """Parse *start_time* into a :class:`datetime` object."""
        return datetime.fromisoformat(self.start_time)

    @property
    def end_datetime(self) -> datetime | None:
        """Parse *end_time* into a :class:`datetime`, or ``None``."""
        return datetime.fromisoformat(self.end_time) if self.end_time else None

    @property
    def is_active(self) -> bool:
        """Return True if the session has not been ended yet."""
        return self.end_time is None

    def __repr__(self) -> str:
        status = "active" if self.is_active else "ended"
        return (
            f"<Session id={self.session_id} mode={self.mode} "
            f"{status} focus={self.total_focus_time:.0f}s "
            f"rest={self.total_rest_time:.0f}s>"
        )


# ════════════════════════════════════════════════════════════════
#  FOCUS BLOCK
# ════════════════════════════════════════════════════════════════


@dataclass
class FocusBlock:
    """Represents a single focus period within a session.

    In Structured Mode a session contains multiple blocks separated
    by rests.  In Fluid Mode each continuous focus period until
    fatigue or user pause is one block.

    Attributes
    ----------
    block_id : int | None
        Auto-assigned by SQLite; ``None`` before INSERT.
    session_id : int
        Foreign key linking to the parent :class:`Session`.
    block_number : int
        1-based ordinal within the session (1st block = 1, etc.).
    start_time : str
        ISO-8601 timestamp when the block began.
    end_time : str | None
        ISO-8601 timestamp when the block ended,
        or ``None`` while still active.
    duration : float
        Active focus duration in **seconds** (excludes any paused
        time — paused time is implicitly ``end - start - duration``).
    """

    block_id: int | None = field(default=None)
    session_id: int = field(default=0)
    block_number: int = field(default=0)
    start_time: str = field(default="")
    end_time: str | None = field(default=None)
    duration: float = field(default=0.0)

    # ── Factories ───────────────────────────────────────────────

    @classmethod
    def from_row(cls, row) -> FocusBlock:
        """Create a FocusBlock from a ``sqlite3.Row`` or dict-like mapping."""
        return cls(
            block_id=row["block_id"],
            session_id=row["session_id"],
            block_number=row["block_number"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            duration=float(row["duration"] or 0.0),
        )

    # ── Serialisation ───────────────────────────────────────────

    def to_insert_tuple(self) -> tuple:
        """Return values for a parameterised INSERT (excludes block_id).

        Matches: ``INSERT INTO FocusBlocks
        (session_id, block_number, start_time, end_time, duration)``
        """
        return (
            self.session_id,
            self.block_number,
            self.start_time,
            self.end_time,
            self.duration,
        )

    # ── Helpers ─────────────────────────────────────────────────

    @property
    def start_datetime(self) -> datetime:
        """Parse *start_time* into a :class:`datetime` object."""
        return datetime.fromisoformat(self.start_time)

    @property
    def end_datetime(self) -> datetime | None:
        """Parse *end_time* into a :class:`datetime`, or ``None``."""
        return datetime.fromisoformat(self.end_time) if self.end_time else None

    @property
    def is_active(self) -> bool:
        """Return True if this focus block is still running."""
        return self.end_time is None

    def __repr__(self) -> str:
        status = "active" if self.is_active else f"{self.duration:.0f}s"
        return (
            f"<FocusBlock id={self.block_id} session={self.session_id} "
            f"#{self.block_number} {status}>"
        )


# ════════════════════════════════════════════════════════════════
#  REST
# ════════════════════════════════════════════════════════════════


@dataclass
class Rest:
    """Represents a rest period following a focus block.

    Attributes
    ----------
    rest_id : int | None
        Auto-assigned by SQLite; ``None`` before INSERT.
    session_id : int
        Foreign key linking to the parent :class:`Session`.
    block_id : int
        Foreign key to the :class:`FocusBlock` this rest follows.
    rest_type : str
        ``'Short'`` (Structured), ``'Long'`` (Structured, after
        every N blocks), or ``'Fluid'`` (dynamic rest).
    start_time : str
        ISO-8601 timestamp when the rest began.
    end_time : str | None
        ISO-8601 timestamp when the rest ended,
        or ``None`` while still resting.
    planned_duration : float
        Prescribed rest duration in **seconds**.
    actual_duration : float
        Actual rest duration in **seconds**.  Always reflects
        real elapsed time, even if the user skipped early.
    skipped : bool
        ``True`` if the user clicked Skip before the planned
        duration elapsed.
    """

    rest_id: int | None = field(default=None)
    session_id: int = field(default=0)
    block_id: int = field(default=0)
    rest_type: str = field(default="")
    start_time: str = field(default="")
    end_time: str | None = field(default=None)
    planned_duration: float = field(default=0.0)
    actual_duration: float = field(default=0.0)
    skipped: bool = field(default=False)

    # ── Factories ───────────────────────────────────────────────

    @classmethod
    def from_row(cls, row) -> Rest:
        """Create a Rest from a ``sqlite3.Row`` or dict-like mapping."""
        return cls(
            rest_id=row["rest_id"],
            session_id=row["session_id"],
            block_id=row["block_id"],
            rest_type=row["rest_type"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            planned_duration=float(row["planned_duration"] or 0.0),
            actual_duration=float(row["actual_duration"] or 0.0),
            skipped=bool(row["skipped"]),
        )

    # ── Serialisation ───────────────────────────────────────────

    def to_insert_tuple(self) -> tuple:
        """Return values for a parameterised INSERT (excludes rest_id).

        Matches: ``INSERT INTO Rests
        (session_id, block_id, rest_type, start_time, end_time,
         planned_duration, actual_duration, skipped)``
        """
        return (
            self.session_id,
            self.block_id,
            self.rest_type,
            self.start_time,
            self.end_time,
            self.planned_duration,
            self.actual_duration,
            int(self.skipped),
        )

    # ── Helpers ─────────────────────────────────────────────────

    @property
    def start_datetime(self) -> datetime:
        """Parse *start_time* into a :class:`datetime` object."""
        return datetime.fromisoformat(self.start_time)

    @property
    def end_datetime(self) -> datetime | None:
        """Parse *end_time* into a :class:`datetime`, or ``None``."""
        return datetime.fromisoformat(self.end_time) if self.end_time else None

    @property
    def is_active(self) -> bool:
        """Return True if this rest period is still running."""
        return self.end_time is None

    @property
    def compliance_ratio(self) -> float:
        """Return ``actual / planned`` (1.0 = full rest taken).

        Returns 0.0 if ``planned_duration`` is zero (shouldn't happen
        in practice, but avoids division by zero).
        """
        if self.planned_duration <= 0:
            return 0.0
        return self.actual_duration / self.planned_duration

    def __repr__(self) -> str:
        status = "active" if self.is_active else f"{self.actual_duration:.0f}s"
        skip = " SKIPPED" if self.skipped else ""
        return f"<Rest id={self.rest_id} type={self.rest_type} {status}{skip}>"


# ════════════════════════════════════════════════════════════════
#  EVENT
# ════════════════════════════════════════════════════════════════


@dataclass
class Event:
    """Represents a single tracked event within a focus block.

    Events only occur during focus blocks, never during rest.

    Attributes
    ----------
    event_id : int | None
        Auto-assigned by SQLite; ``None`` before INSERT.
    session_id : int
        Foreign key linking to the parent :class:`Session`.
        Denormalised for fast session-level queries.
    block_id : int
        Foreign key linking to the :class:`FocusBlock` where
        this event occurred.  Enables per-block analytics.
    timestamp : str
        ISO-8601 timestamp of when the event was detected.
    event_type : str
        One of the ``constants.EVENT_TYPES``:
        ``Distraction_Pose``, ``Fatigue``, ``Fatigue_Warning``,
        ``Away``.
    duration : float
        Duration of the event in **seconds**.
    reason : str | None
        Why the engine fired (e.g. ``'microsleep'``).  ``None`` for
        event types whose engine gives no reason (distraction, away).
    label : str | None
        The user's verdict from the post-session review:
        ``EVENT_LABEL_CORRECT`` or ``EVENT_LABEL_FALSE_ALARM``.
        ``None`` until labelled.  Not part of the INSERT — labels are
        written afterwards by ``queries.set_event_label``.
    """

    event_id: int | None = field(default=None)
    session_id: int = field(default=0)
    block_id: int = field(default=0)
    timestamp: str = field(default="")
    event_type: str = field(default="")
    duration: float = field(default=0.0)
    reason: str | None = field(default=None)
    label: str | None = field(default=None)

    # ── Factories ───────────────────────────────────────────────

    @classmethod
    def from_row(cls, row) -> Event:
        """Create an Event from a ``sqlite3.Row`` or dict-like mapping."""
        return cls(
            event_id=row["event_id"],
            session_id=row["session_id"],
            block_id=row["block_id"],
            timestamp=row["timestamp"],
            event_type=row["event_type"],
            duration=float(row["duration"] or 0.0),
            reason=row["reason"],
            label=row["label"],
        )

    # ── Serialisation ───────────────────────────────────────────

    def to_insert_tuple(self) -> tuple:
        """Return values for a parameterised INSERT (excludes event_id).

        Matches: ``INSERT INTO Events
        (session_id, block_id, timestamp, event_type, duration, reason)``
        """
        return (
            self.session_id,
            self.block_id,
            self.timestamp,
            self.event_type,
            self.duration,
            self.reason,
        )

    # ── Helpers ─────────────────────────────────────────────────

    @property
    def timestamp_datetime(self) -> datetime:
        """Parse *timestamp* into a :class:`datetime` object."""
        return datetime.fromisoformat(self.timestamp)

    def __repr__(self) -> str:
        return (
            f"<Event id={self.event_id} session={self.session_id} "
            f"block={self.block_id} type={self.event_type} "
            f"dur={self.duration:.1f}s>"
        )
