# Pure aggregation for the Insights tab - ranges, buckets, rates, takeaways, precision.
# ─────────────────────────────────────────────────────────────────
# Responsibilities:
#   • Turn a range key ("today" / "7d" / "30d" / "all") into date bounds
#     plus the equal-length comparison window immediately before it.
#   • Lay focus blocks and events onto a time axis whose granularity
#     follows the range (hour / day / ISO week), keeping empty buckets
#     so gaps stay visible.
#   • Normalise event counts per focus hour.  Raw counts mostly measure
#     how long you worked (2.4 – 27.6 events per focus hour across days),
#     so no chart or takeaway compares raw counts across periods.
#   • Build the sidebar takeaways.  Every rule sits behind an evidence
#     gate (INSIGHTS_TAKEAWAY_* in constants) and says nothing rather
#     than overclaim.
#   • Per-detector precision from the user's review labels, shown only
#     from INSIGHTS_PRECISION_MIN_LABELS labels of that event type.
#
# Usage:
#   start, end, prev_start, prev_end = range_bounds("7d", date.today())
#   gran  = granularity_for("7d")
#   axis  = bucket_axis(start, end, gran)
#   focus = focus_seconds_by_bucket(blocks, axis, gran)
#   items = build_takeaways("7d", blocks, events, prev_blocks,
#                           prev_events, structured_work_min=25)
#
# Inputs are the plain dicts returned by the acuity.db.queries range
# functions:
#   block  {block_id, session_id, block_number, start_time, duration, mode}
#   event  {event_id, block_id, timestamp, event_type, reason, elapsed_min,
#           label}
#   rest   {rest_type, start_time, planned_duration, actual_duration, skipped}
#
# Thread: pure/any.  No Qt and no DB imports, so every number the tab
# shows is unit-testable without a QApplication or a database.
#
# Design:
#   A block's *active* duration is laid out contiguously from its start
#   time when spread across hours.  Pauses inside a block are not
#   stored, so this is the best placement the schema allows; it only
#   shifts minutes between neighbouring hours, never changes totals.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import statistics
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from acuity.utils.constants import (
    EVENT_DISTRACTION_POSE,
    EVENT_DISTRACTION_WINDOW,
    EVENT_FATIGUE,
    EVENT_LABEL_CORRECT,
    EVENT_LABEL_FALSE_ALARM,
    INSIGHTS_HIST_BIN_MIN,
    INSIGHTS_HIST_MAX_MIN,
    INSIGHTS_MIN_RATE_FOCUS_S,
    INSIGHTS_PRECISION_MIN_LABELS,
    INSIGHTS_RANGE_7D,
    INSIGHTS_RANGE_30D,
    INSIGHTS_RANGE_ALL,
    INSIGHTS_RANGE_TODAY,
    INSIGHTS_TAKEAWAY_MAX,
    INSIGHTS_TAKEAWAY_MAX_RATE_ITEMS,
    INSIGHTS_TAKEAWAY_MIN_ACTIVE_DAYS,
    INSIGHTS_TAKEAWAY_MIN_EVENTS,
    INSIGHTS_TAKEAWAY_MIN_FOCUS_CHANGE,
    INSIGHTS_TAKEAWAY_MIN_FOCUS_S,
    INSIGHTS_TAKEAWAY_MIN_ONSETS,
    INSIGHTS_TAKEAWAY_MIN_RATE_CHANGE,
    SESSION_MODE_STRUCTURED,
)

# ── Granularity ───────────────────────────────────────────────────
GRAN_HOUR = "hour"
GRAN_DAY = "day"
GRAN_WEEK = "week"

_RANGE_DAYS: dict[str, int] = {
    INSIGHTS_RANGE_TODAY: 1,
    INSIGHTS_RANGE_7D: 7,
    INSIGHTS_RANGE_30D: 30,
}

_RANGE_LABELS: dict[str, str] = {
    INSIGHTS_RANGE_TODAY: "today",
    INSIGHTS_RANGE_7D: "7 days",
    INSIGHTS_RANGE_30D: "30 days",
    INSIGHTS_RANGE_ALL: "all time",
}

_COMPARISON_LABELS: dict[str, str] = {
    INSIGHTS_RANGE_TODAY: "yesterday",
    INSIGHTS_RANGE_7D: "the previous 7 days",
    INSIGHTS_RANGE_30D: "the previous 30 days",
}

WEEKDAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

AxisKey = datetime | date
Axis = list[tuple[AxisKey, str]]


# ════════════════════════════════════════════════════════════════
#  RANGES & AXES
# ════════════════════════════════════════════════════════════════


def range_bounds(
    key: str,
    today: date,
    earliest: date | None = None,
) -> tuple[date, date, date | None, date | None]:
    """Return ``(start, end, prev_start, prev_end)`` for a range key.

    The previous window has the same length and ends the day before
    *start*.  ``"all"`` starts at *earliest* (the first recorded block)
    and has no previous window.  An unknown key raises ``KeyError``.
    """
    if key == INSIGHTS_RANGE_ALL:
        start = min(earliest, today) if earliest is not None else today
        return start, today, None, None
    days = _RANGE_DAYS[key]
    start = today - timedelta(days=days - 1)
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    return start, today, prev_start, prev_end


def granularity_for(key: str) -> str:
    """Bucket size for a range: hour for today, week for all, else day."""
    if key == INSIGHTS_RANGE_TODAY:
        return GRAN_HOUR
    if key == INSIGHTS_RANGE_ALL:
        return GRAN_WEEK
    return GRAN_DAY


def range_label(key: str) -> str:
    """Short human label for a range key ("7 days")."""
    return _RANGE_LABELS[key]


def comparison_label(key: str) -> str | None:
    """How takeaways name the previous window, or ``None`` for "all"."""
    return _COMPARISON_LABELS.get(key)


def bucket_key(moment: datetime, gran: str) -> AxisKey:
    """Map a moment onto its bucket key for *gran*."""
    if gran == GRAN_HOUR:
        return moment.replace(minute=0, second=0, microsecond=0)
    if gran == GRAN_DAY:
        return moment.date()
    return moment.date() - timedelta(days=moment.weekday())


def bucket_axis(start: date, end: date, gran: str) -> Axis:
    """Every bucket between *start* and *end* inclusive, empty ones too."""
    axis: Axis = []
    if gran == GRAN_HOUR:
        day = start
        while day <= end:
            for hour in range(24):
                axis.append((datetime(day.year, day.month, day.day, hour), f"{hour:02d}:00"))
            day += timedelta(days=1)
    elif gran == GRAN_DAY:
        day = start
        while day <= end:
            axis.append((day, day.strftime("%m/%d")))
            day += timedelta(days=1)
    else:
        week = start - timedelta(days=start.weekday())
        while week <= end:
            axis.append((week, week.strftime("%m/%d")))
            week += timedelta(days=7)
    return axis


# ════════════════════════════════════════════════════════════════
#  FOCUS
# ════════════════════════════════════════════════════════════════


def _spread(start: datetime, seconds: float) -> Iterator[tuple[datetime, float]]:
    """Yield ``(hour_start, seconds)`` chunks of a block laid out from *start*."""
    remaining = max(0.0, float(seconds))
    cursor = start
    while remaining > 1e-9:
        hour_start = cursor.replace(minute=0, second=0, microsecond=0)
        next_hour = hour_start + timedelta(hours=1)
        chunk = min(remaining, (next_hour - cursor).total_seconds())
        yield hour_start, chunk
        remaining -= chunk
        cursor = next_hour


def focus_seconds_by_bucket(
    blocks: Iterable[dict],
    axis: Axis,
    gran: str,
) -> list[float]:
    """Focus seconds per bucket; time falling outside the axis is dropped."""
    index = {key: i for i, (key, _) in enumerate(axis)}
    out = [0.0] * len(axis)
    for block in blocks:
        start = datetime.fromisoformat(block["start_time"])
        for hour_start, secs in _spread(start, block["duration"]):
            i = index.get(bucket_key(hour_start, gran))
            if i is not None:
                out[i] += secs
    return out


def focus_heatmap(blocks: Iterable[dict]) -> list[list[float]]:
    """Focus minutes as ``grid[weekday][hour]`` (Monday = 0)."""
    grid = [[0.0] * 24 for _ in range(7)]
    for block in blocks:
        start = datetime.fromisoformat(block["start_time"])
        for hour_start, secs in _spread(start, block["duration"]):
            grid[hour_start.weekday()][hour_start.hour] += secs / 60.0
    return grid


# ════════════════════════════════════════════════════════════════
#  EVENTS
# ════════════════════════════════════════════════════════════════


def event_counts_by_bucket(
    events: Iterable[dict],
    event_type: str,
    axis: Axis,
    gran: str,
) -> list[int]:
    """Count of *event_type* per bucket."""
    index = {key: i for i, (key, _) in enumerate(axis)}
    out = [0] * len(axis)
    for event in events:
        if event["event_type"] != event_type:
            continue
        i = index.get(bucket_key(datetime.fromisoformat(event["timestamp"]), gran))
        if i is not None:
            out[i] += 1
    return out


def rates_per_focus_hour(
    counts: Sequence[int],
    focus_s: Sequence[float],
    min_focus_s: float = INSIGHTS_MIN_RATE_FOCUS_S,
) -> list[float | None]:
    """Events per focus hour per bucket; ``None`` where focus is too thin."""
    return [
        (n / (secs / 3600.0)) if secs >= min_focus_s else None for n, secs in zip(counts, focus_s)
    ]


def reason_counts(events: Iterable[dict], event_type: str) -> dict[str | None, int]:
    """``{reason: count}`` for one event type; ``None`` = unrecorded."""
    out: dict[str | None, int] = {}
    for event in events:
        if event["event_type"] == event_type:
            reason = event.get("reason") or None
            out[reason] = out.get(reason, 0) + 1
    return out


def onset_minutes(events: Iterable[dict], event_type: str) -> list[float]:
    """Minutes since block start for each *event_type* event."""
    return [
        float(e["elapsed_min"])
        for e in events
        if e["event_type"] == event_type and e.get("elapsed_min") is not None
    ]


def label_counts(events: Iterable[dict], event_type: str) -> tuple[int, int]:
    """``(correct, labelled)`` for one event type; unrated events are skipped."""
    correct = labelled = 0
    for event in events:
        if event["event_type"] != event_type:
            continue
        label = event.get("label")
        if label in (EVENT_LABEL_CORRECT, EVENT_LABEL_FALSE_ALARM):
            labelled += 1
            correct += label == EVENT_LABEL_CORRECT
    return correct, labelled


def precision(events: Iterable[dict], event_type: str) -> float | None:
    """Share of rated *event_type* events the user marked correct.

    ``None`` below INSIGHTS_PRECISION_MIN_LABELS ratings, where a single
    rating would swing the figure too far to mean anything.
    """
    correct, labelled = label_counts(events, event_type)
    if labelled < INSIGHTS_PRECISION_MIN_LABELS:
        return None
    return correct / labelled


# ════════════════════════════════════════════════════════════════
#  DISTRIBUTIONS & RESTS
# ════════════════════════════════════════════════════════════════


def histogram(
    values: Iterable[float],
    bin_w: float = INSIGHTS_HIST_BIN_MIN,
    max_v: float = INSIGHTS_HIST_MAX_MIN,
) -> list[int]:
    """Counts per ``bin_w`` bin from 0; values ≥ *max_v* share the last bin."""
    n_bins = int(max_v // bin_w) + 1
    counts = [0] * n_bins
    for value in values:
        i = int(max(0.0, value) // bin_w)
        counts[min(i, n_bins - 1)] += 1
    return counts


def histogram_labels(
    bin_w: float = INSIGHTS_HIST_BIN_MIN,
    max_v: float = INSIGHTS_HIST_MAX_MIN,
) -> list[str]:
    """Left-edge labels matching :func:`histogram`; the last reads "60+"."""
    n_bins = int(max_v // bin_w) + 1
    labels = [f"{int(i * bin_w)}" for i in range(n_bins - 1)]
    labels.append(f"{int(max_v)}+")
    return labels


def rest_taken_ratio(rests: Iterable[dict]) -> float | None:
    """Median of ``min(actual / planned, 1)``; ``None`` with no rests.

    Capped at 1 because a skipped rest's actual duration runs until the
    user returns: 6 rests measured 1.5–74× planned (371 min on a 5 min
    rest), which made the old uncapped mean meaningless.
    """
    ratios = [
        min(r["actual_duration"] / r["planned_duration"], 1.0)
        for r in rests
        if r["planned_duration"] > 0
    ]
    return statistics.median(ratios) if ratios else None


# ════════════════════════════════════════════════════════════════
#  RANGE SUMMARY
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RangeSummary:
    """Headline figures for one date window."""

    focus_s: float
    active_days: int
    best_day: tuple[date, float] | None  # (day, focus seconds)
    session_count: int
    median_block_s: float | None
    sessions_by_mode: dict[str, int]
    event_counts: dict[str, int]

    def per_focus_hour(self, event_type: str) -> float | None:
        """Rate of *event_type* per focus hour, ``None`` below the focus floor."""
        if self.focus_s < INSIGHTS_MIN_RATE_FOCUS_S:
            return None
        return self.event_counts.get(event_type, 0) / (self.focus_s / 3600.0)


def summarise(blocks: Sequence[dict], events: Iterable[dict]) -> RangeSummary:
    """Compute a :class:`RangeSummary` from range-query rows."""
    by_day: dict[date, float] = {}
    modes: dict[str, set[int]] = {}
    for block in blocks:
        day = datetime.fromisoformat(block["start_time"]).date()
        by_day[day] = by_day.get(day, 0.0) + block["duration"]
        modes.setdefault(block.get("mode") or "", set()).add(block["session_id"])

    counts: dict[str, int] = {}
    for event in events:
        counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1

    durations = [b["duration"] for b in blocks]
    best = max(by_day.items(), key=lambda kv: kv[1]) if by_day else None
    return RangeSummary(
        focus_s=sum(durations),
        active_days=len(by_day),
        best_day=best,
        session_count=len({b["session_id"] for b in blocks}),
        median_block_s=statistics.median(durations) if durations else None,
        sessions_by_mode={mode: len(ids) for mode, ids in modes.items()},
        event_counts=counts,
    )


# ════════════════════════════════════════════════════════════════
#  TAKEAWAYS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Takeaway:
    """One sidebar finding.

    *tone* is ``"good"`` / ``"bad"`` only when better-or-worse is
    unambiguous (more focus, fewer events per focus hour); *direction*
    drives the ▲ / ▼ glyph so colour never carries the meaning alone.
    """

    text: str
    tone: str = "neutral"  # "good" | "bad" | "neutral"
    direction: str | None = None  # "up" | "down" | None


# Rate-change subjects in tie-break order (stable sort keeps this order).
_RATE_SUBJECTS: tuple[tuple[str, str], ...] = (
    (EVENT_DISTRACTION_POSE, "Head-pose distractions"),
    (EVENT_DISTRACTION_WINDOW, "Window distractions"),
    (EVENT_FATIGUE, "Fatigue alerts"),
)


def build_takeaways(
    range_key: str,
    cur_blocks: Sequence[dict],
    cur_events: Sequence[dict],
    prev_blocks: Sequence[dict] | None,
    prev_events: Sequence[dict] | None,
    structured_work_min: int,
) -> list[Takeaway]:
    """Evidence-gated findings for the selected range, highest priority first.

    Order: event-rate changes (≤ INSIGHTS_TAKEAWAY_MAX_RATE_ITEMS),
    focus change, fatigue onset, peak focus window.
    Capped at INSIGHTS_TAKEAWAY_MAX.  Returns ``[]`` when nothing
    clears its gate — the caller shows an empty-state sentence.
    """
    items: list[Takeaway] = []
    cur = summarise(cur_blocks, cur_events)
    prev_label = comparison_label(range_key)

    if prev_label is not None and prev_blocks is not None:
        prev = summarise(prev_blocks, prev_events or [])
        items.extend(_rate_changes(cur, prev, prev_label))
        # Today is still in progress; comparing its running total against
        # all of yesterday would read as "down" every morning.  Rates are
        # normalised, so they are safe to compare mid-day.
        if range_key != INSIGHTS_RANGE_TODAY:
            focus_item = _focus_change(cur, prev, prev_label)
            if focus_item is not None:
                items.append(focus_item)

    for item in (
        _fatigue_onset(cur_blocks, cur_events, structured_work_min),
        _peak_window(cur_blocks, cur.active_days),
    ):
        if item is not None:
            items.append(item)

    return items[:INSIGHTS_TAKEAWAY_MAX]


def _rate_changes(
    cur: RangeSummary,
    prev: RangeSummary,
    prev_label: str,
) -> list[Takeaway]:
    if cur.focus_s < INSIGHTS_TAKEAWAY_MIN_FOCUS_S or prev.focus_s < INSIGHTS_TAKEAWAY_MIN_FOCUS_S:
        return []

    found: list[tuple[float, Takeaway]] = []
    for event_type, subject in _RATE_SUBJECTS:
        n_cur = cur.event_counts.get(event_type, 0)
        n_prev = prev.event_counts.get(event_type, 0)
        # A zero baseline has no ratio, and a handful of events is noise.
        if n_prev == 0 or n_cur + n_prev < INSIGHTS_TAKEAWAY_MIN_EVENTS:
            continue
        rate_cur = n_cur / (cur.focus_s / 3600.0)
        rate_prev = n_prev / (prev.focus_s / 3600.0)
        change = (rate_cur - rate_prev) / rate_prev
        if abs(change) < INSIGHTS_TAKEAWAY_MIN_RATE_CHANGE:
            continue
        fewer = change < 0
        text = (
            f"{subject}: {round(abs(change) * 100)}% "
            f"{'fewer' if fewer else 'more'} per focus hour than {prev_label}."
        )
        found.append(
            (
                abs(change),
                Takeaway(text, "good" if fewer else "bad", "down" if fewer else "up"),
            )
        )

    found.sort(key=lambda pair: pair[0], reverse=True)
    return [t for _, t in found[:INSIGHTS_TAKEAWAY_MAX_RATE_ITEMS]]


def _focus_change(
    cur: RangeSummary,
    prev: RangeSummary,
    prev_label: str,
) -> Takeaway | None:
    if cur.focus_s < INSIGHTS_TAKEAWAY_MIN_FOCUS_S or prev.focus_s < INSIGHTS_TAKEAWAY_MIN_FOCUS_S:
        return None
    change = (cur.focus_s - prev.focus_s) / prev.focus_s
    if abs(change) < INSIGHTS_TAKEAWAY_MIN_FOCUS_CHANGE:
        return None
    up = change > 0
    text = (
        f"Focus {'up' if up else 'down'} {round(abs(change) * 100)}% vs "
        f"{prev_label} ({cur.focus_s / 3600:.1f} h vs {prev.focus_s / 3600:.1f} h)."
    )
    return Takeaway(text, "good" if up else "bad", "up" if up else "down")


def _fatigue_onset(
    blocks: Sequence[dict],
    events: Sequence[dict],
    structured_work_min: int,
) -> Takeaway | None:
    onsets = onset_minutes(events, EVENT_FATIGUE)
    if len(onsets) < INSIGHTS_TAKEAWAY_MIN_ONSETS:
        return None
    median = statistics.median(onsets)
    text = f"Fatigue alerts typically come ~{median:.0f} min into a block"
    has_structured = any(b.get("mode") == SESSION_MODE_STRUCTURED for b in blocks)
    if has_structured and median < structured_work_min:
        text += f", before your {structured_work_min}-min Structured blocks end"
    return Takeaway(text + ".")


def _peak_window(blocks: Sequence[dict], active_days: int) -> Takeaway | None:
    if active_days < INSIGHTS_TAKEAWAY_MIN_ACTIVE_DAYS:
        return None
    grid = focus_heatmap(blocks)
    by_hour = [sum(grid[day][hour] for day in range(7)) for hour in range(24)]
    total = sum(by_hour)
    if total <= 0:
        return None
    best = max(range(23), key=lambda h: by_hour[h] + by_hour[h + 1])
    share = (by_hour[best] + by_hour[best + 1]) / total
    # Trim an empty edge hour: when all the focus sits in one hour, the
    # first of two tied windows would otherwise report the empty hour
    # before it ("14:00–16:00" for focus that only ever happened at 15:00).
    first, last = best, best + 1
    if by_hour[first] <= 0:
        first = last
    elif by_hour[last] <= 0:
        last = first
    return Takeaway(
        f"Your peak focus window is {first:02d}:00–{last + 1:02d}:00 "
        f"({share * 100:.0f}% of focus time)."
    )
