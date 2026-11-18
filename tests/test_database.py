# Tests for database module.
# ─────────────────────────────────────────────────────────────────
# Covers where the database lives, its schema, the event-label setter,
# and the queries the Insights tab depends on.  Every test runs against
# a fresh DatabaseManager under ACUITY_DATA_DIR (a tmp path set by
# conftest.py), patched into acuity.db.queries, so no test touches a real
# database or writes into the repo.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

import acuity.db.queries as queries
from acuity.db.database import DatabaseManager
from acuity.utils.constants import EVENT_LABEL_FALSE_ALARM

DAY = date(2015, 9, 9)


@pytest.fixture()
def db():
    manager = DatabaseManager()
    manager.connect()
    with patch("acuity.db.queries.get_db", return_value=manager):
        yield manager
    manager.close()


def _closed_block(session_id: int, number: int, start: str, duration: float) -> int:
    block_id = queries.start_focus_block(session_id, number, start_time=start)
    queries.end_focus_block(block_id, duration)
    return block_id


# ════════════════════════════════════════════════════════════════
#  LOCATION & SCHEMA
# ════════════════════════════════════════════════════════════════


class TestLocation:
    def test_database_is_created_under_acuity_data_dir(self, data_dir):
        """ACUITY_DATA_DIR is how development and tests keep the database
        out of the user's real data folder and out of the repo."""
        manager = DatabaseManager()
        manager.connect()
        try:
            assert (data_dir / "acuity.db").is_file()
        finally:
            manager.close()


class TestSchema:
    def test_fresh_schema_has_reason_and_label(self, db):
        """The database starts fresh, so both columns come from CREATE TABLE
        rather than from an in-place migration."""
        columns = {r[1] for r in db.connection.execute("PRAGMA table_info(Events);")}
        assert {"reason", "label"} <= columns

    def test_reopening_an_existing_database_keeps_its_rows(self, db):
        """Every app start runs the schema bootstrap on the user's existing
        file; it must leave their history intact."""
        sid = queries.create_session(start_time="2015-09-09T10:00:00")
        db.close()

        reopened = DatabaseManager()
        reopened.connect()
        try:
            with patch("acuity.db.queries.get_db", return_value=reopened):
                assert queries.get_session(sid) is not None
        finally:
            reopened.close()

    def test_insert_event_round_trips_reason(self, db):
        sid = queries.create_session(start_time="2015-09-09T10:00:00")
        bid = queries.start_focus_block(sid, 1, start_time="2015-09-09T10:00:00")
        queries.insert_event(
            sid, bid, "Fatigue", 1.5, timestamp="2015-09-09T10:05:00", reason="microsleep"
        )
        queries.insert_event(sid, bid, "Distraction_Pose", 5.0, timestamp="2015-09-09T10:06:00")

        events = queries.get_events_for_session(sid)
        assert [(e.event_type, e.reason) for e in events] == [
            ("Fatigue", "microsleep"),
            ("Distraction_Pose", None),
        ]


class TestEventLabel:
    def test_set_event_label_round_trips_and_clears(self, db):
        sid = queries.create_session(start_time="2015-10-22T10:00:00")
        bid = queries.start_focus_block(sid, 1, start_time="2015-10-22T10:00:00")
        first = queries.insert_event(
            sid, bid, "Fatigue", 1.5, timestamp="2015-10-22T10:05:00", reason="microsleep"
        )
        second = queries.insert_event(
            sid, bid, "Distraction_Pose", 10.0, timestamp="2015-10-22T10:06:00"
        )

        queries.set_event_label(first, "false_alarm")
        queries.set_event_label(second, "correct")
        assert [e.label for e in queries.get_events_for_session(sid)] == [
            "false_alarm",
            "correct",
        ]

        queries.set_event_label(second, None)
        assert queries.get_events_for_session(sid)[1].label is None

    def test_set_event_label_rejects_unknown_label(self, db):
        with pytest.raises(ValueError):
            queries.set_event_label(1, "maybe")


# ════════════════════════════════════════════════════════════════
#  INSIGHTS QUERIES
# ════════════════════════════════════════════════════════════════


class TestFocusTotals:
    def test_daily_total_counts_blocks_of_unclosed_session(self, db):
        """A session the app never closed keeps total_focus_time = 0, but
        its finished blocks were persisted and must still count."""
        sid = queries.create_session(start_time="2015-09-09T15:00:00")
        _closed_block(sid, 1, "2015-09-09T15:00:00", 1500)
        _closed_block(sid, 2, "2015-09-09T15:30:00", 1200)
        queries.start_focus_block(sid, 3, start_time="2015-09-09T16:00:00")  # never closed

        assert queries.get_session(sid).total_focus_time == 0
        assert queries.get_daily_focus_total(DAY) == 2700


class TestRangeQueries:
    def test_blocks_filter_short_and_unclosed(self, db):
        sid = queries.create_session(mode="Fluid", start_time="2015-09-09T09:00:00")
        _closed_block(sid, 1, "2015-09-09T09:00:00", 30)  # too short
        kept = _closed_block(sid, 2, "2015-09-09T09:10:00", 600)
        queries.start_focus_block(sid, 3, start_time="2015-09-09T09:30:00")
        _closed_block(sid, 4, "2015-09-12T09:00:00", 600)  # outside range

        rows = queries.get_focus_blocks_in_range(DAY, DAY, 60)
        assert [(r["block_id"], r["duration"], r["mode"]) for r in rows] == [
            (kept, 600, "Fluid"),
        ]

    def test_events_with_offset_and_block_filter(self, db):
        sid = queries.create_session(start_time="2015-09-09T10:00:00")
        good = _closed_block(sid, 1, "2015-09-09T10:00:00", 1500)
        short = _closed_block(sid, 2, "2015-09-09T11:00:00", 30)
        queries.insert_event(
            sid, good, "Fatigue", 1.0, timestamp="2015-09-09T10:16:00", reason="microsleep"
        )
        queries.insert_event(sid, short, "Distraction_Pose", 10.0, timestamp="2015-09-09T11:00:20")

        rows = queries.get_events_in_range_with_offset(DAY, DAY, 60)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "Fatigue"
        assert rows[0]["reason"] == "microsleep"
        assert rows[0]["elapsed_min"] == pytest.approx(16.0, abs=0.01)

    def test_events_in_range_carry_the_review_label(self, db):
        """Insights computes per-detector precision from these rows."""
        sid = queries.create_session(start_time="2015-09-09T10:00:00")
        bid = _closed_block(sid, 1, "2015-09-09T10:00:00", 1500)
        rated = queries.insert_event(sid, bid, "Fatigue", 1.0, timestamp="2015-09-09T10:16:00")
        queries.insert_event(sid, bid, "Fatigue", 1.0, timestamp="2015-09-09T10:20:00")
        queries.set_event_label(rated, EVENT_LABEL_FALSE_ALARM)

        rows = queries.get_events_in_range_with_offset(DAY, DAY, 60)
        assert [r["label"] for r in rows] == [EVENT_LABEL_FALSE_ALARM, None]

    def test_rests_only_closed(self, db):
        sid = queries.create_session(start_time="2015-09-09T10:00:00")
        bid = _closed_block(sid, 1, "2015-09-09T10:00:00", 1500)
        done = queries.start_rest(sid, bid, "Short", 300, start_time="2015-09-09T10:25:00")
        queries.skip_rest(done, 120)
        queries.start_rest(sid, bid, "Short", 300, start_time="2015-09-09T11:00:00")

        rows = queries.get_rests_in_range(DAY, DAY)
        assert [(r["rest_id"], r["actual_duration"], r["skipped"]) for r in rows] == [
            (done, 120, 1),
        ]

    def test_earliest_block_date(self, db):
        assert queries.get_earliest_block_date(60) is None
        sid = queries.create_session(start_time="2015-08-01T10:00:00")
        _closed_block(sid, 1, "2015-08-01T10:00:00", 30)
        _closed_block(sid, 2, "2015-08-03T10:00:00", 600)
        assert queries.get_earliest_block_date(60) == date(2015, 8, 3)
