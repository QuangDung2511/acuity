# Tests for acuity.core.insights_metrics - ranges, buckets, rates, takeaways, precision.
# ─────────────────────────────────────────────────────────────────
# Pure logic, no Qt and no DB.  The takeaway cases are a worked
# example shaped on the comparison the evidence gates were set from
# (last 7 days vs the 7 before, private corpus, n=1), with the event
# types the port keeps:
#   focus      4.59 h vs 3.55 h
#   head pose  16 vs 23 events  → must fire (−46% per focus hour)
#   fatigue     5 vs 1          → must be withheld (6 events)
#   window      3 vs 0          → must be withheld (zero baseline)
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from datetime import date, datetime

import pytest

import acuity.core.insights_metrics as m
from acuity.utils.constants import (
    EVENT_DISTRACTION_POSE,
    EVENT_DISTRACTION_WINDOW,
    EVENT_FATIGUE,
    EVENT_LABEL_CORRECT,
    EVENT_LABEL_FALSE_ALARM,
    INSIGHTS_TAKEAWAY_MAX,
    SESSION_MODE_FLUID,
    SESSION_MODE_STRUCTURED,
)

TODAY = date(2015, 9, 10)


def _block(
    start: str, duration_s: float, mode: str = SESSION_MODE_STRUCTURED, session_id: int = 1
) -> dict:
    return {
        "block_id": 0,
        "session_id": session_id,
        "block_number": 1,
        "start_time": start,
        "duration": duration_s,
        "mode": mode,
    }


def _events(
    event_type: str,
    n: int,
    timestamp: str,
    elapsed: float = 10.0,
    reason: str | None = None,
    label: str | None = None,
) -> list[dict]:
    return [
        {
            "event_id": i,
            "block_id": 0,
            "timestamp": timestamp,
            "event_type": event_type,
            "reason": reason,
            "elapsed_min": elapsed,
            "label": label,
        }
        for i in range(n)
    ]


# ════════════════════════════════════════════════════════════════
#  RANGES & AXES
# ════════════════════════════════════════════════════════════════


class TestRanges:
    def test_seven_days_and_previous_window(self):
        assert m.range_bounds("7d", TODAY) == (
            date(2015, 9, 4),
            TODAY,
            date(2015, 8, 28),
            date(2015, 9, 3),
        )

    def test_today_compares_with_yesterday(self):
        assert m.range_bounds("today", TODAY) == (
            TODAY,
            TODAY,
            date(2015, 9, 9),
            date(2015, 9, 9),
        )

    def test_all_starts_at_earliest_and_has_no_previous(self):
        assert m.range_bounds("all", TODAY, date(2015, 6, 14)) == (
            date(2015, 6, 14),
            TODAY,
            None,
            None,
        )

    def test_unknown_key_fails_loud(self):
        with pytest.raises(KeyError):
            m.range_bounds("90d", TODAY)

    def test_granularity(self):
        assert m.granularity_for("today") == m.GRAN_HOUR
        assert m.granularity_for("7d") == m.GRAN_DAY
        assert m.granularity_for("30d") == m.GRAN_DAY
        assert m.granularity_for("all") == m.GRAN_WEEK


class TestAxis:
    def test_day_axis_keeps_empty_days(self):
        axis = m.bucket_axis(date(2015, 9, 4), TODAY, m.GRAN_DAY)
        assert len(axis) == 7
        assert axis[0] == (date(2015, 9, 4), "09/04")

    def test_hour_axis_has_24_buckets(self):
        axis = m.bucket_axis(TODAY, TODAY, m.GRAN_HOUR)
        assert len(axis) == 24
        assert axis[15] == (datetime(2015, 9, 10, 15), "15:00")

    def test_week_axis_is_monday_aligned(self):
        axis = m.bucket_axis(date(2015, 9, 2), TODAY, m.GRAN_WEEK)  # Wed → Thu
        assert [key for key, _ in axis] == [date(2015, 8, 31), date(2015, 9, 7)]


# ════════════════════════════════════════════════════════════════
#  FOCUS & EVENTS
# ════════════════════════════════════════════════════════════════


class TestFocusSpread:
    def test_block_across_midnight_splits_between_days(self):
        axis = m.bucket_axis(date(2015, 9, 9), TODAY, m.GRAN_DAY)
        blocks = [_block("2015-09-09T23:30:00", 3600)]
        assert m.focus_seconds_by_bucket(blocks, axis, m.GRAN_DAY) == [1800.0, 1800.0]

    def test_time_outside_axis_is_dropped(self):
        axis = m.bucket_axis(TODAY, TODAY, m.GRAN_DAY)
        blocks = [_block("2015-09-09T23:30:00", 3600)]
        assert m.focus_seconds_by_bucket(blocks, axis, m.GRAN_DAY) == [1800.0]

    def test_heatmap_spreads_across_hours(self):
        grid = m.focus_heatmap([_block("2015-09-10T10:45:00", 1800)])  # Thursday
        assert grid[3][10] == pytest.approx(15.0)
        assert grid[3][11] == pytest.approx(15.0)
        assert sum(map(sum, grid)) == pytest.approx(30.0)


class TestRates:
    def test_thin_focus_is_a_gap_not_a_spike(self):
        # Worst real bucket: 19 events in 1.5 min of focus would plot as ~760/h.
        rates = m.rates_per_focus_hour([19, 8], [90, 2 * 3600])
        assert rates == [None, pytest.approx(4.0)]

    def test_rate_drawn_at_the_floor(self):
        assert m.rates_per_focus_hour([1], [600]) == [pytest.approx(6.0)]
        assert m.rates_per_focus_hour([1], [599]) == [None]

    def test_event_counts_by_bucket(self):
        axis = m.bucket_axis(date(2015, 9, 9), TODAY, m.GRAN_DAY)
        events = _events(EVENT_DISTRACTION_POSE, 2, "2015-09-10T15:00:00") + _events(
            EVENT_FATIGUE, 1, "2015-09-09T15:00:00"
        )
        assert m.event_counts_by_bucket(events, EVENT_DISTRACTION_POSE, axis, m.GRAN_DAY) == [0, 2]

    def test_reason_counts_treats_empty_as_unrecorded(self):
        events = (
            _events(EVENT_FATIGUE, 2, "2015-09-10T15:00:00", reason="microsleep")
            + _events(EVENT_FATIGUE, 1, "2015-09-10T15:00:00", reason="")
            + _events(EVENT_FATIGUE, 1, "2015-09-10T15:00:00")
        )
        assert m.reason_counts(events, EVENT_FATIGUE) == {"microsleep": 2, None: 2}


class TestDistributions:
    def test_histogram_overflow_and_clamp(self):
        counts = m.histogram([-1, 0, 4.9, 5, 59.9, 60, 104], bin_w=5, max_v=60)
        assert len(counts) == 13
        assert counts[0] == 3  # -1 clamps to 0
        assert counts[1] == 1
        assert counts[11] == 1
        assert counts[12] == 2  # 60 and 104 share the overflow bin

    def test_histogram_labels(self):
        labels = m.histogram_labels(bin_w=5, max_v=60)
        assert labels[0] == "0" and labels[-1] == "60+" and len(labels) == 13

    def test_rest_ratio_capped_at_one(self):
        rests = [
            {"planned_duration": 300, "actual_duration": 22290},  # 371.5 min skipped rest
            {"planned_duration": 300, "actual_duration": 150},
            {"planned_duration": 0, "actual_duration": 99},  # ignored
        ]
        assert m.rest_taken_ratio(rests) == pytest.approx(0.75)

    def test_rest_ratio_none_without_rests(self):
        assert m.rest_taken_ratio([]) is None


class TestSummary:
    def test_summary_fields(self):
        blocks = [
            _block("2015-09-09T10:00:00", 1500, SESSION_MODE_STRUCTURED, session_id=1),
            _block("2015-09-09T11:00:00", 1500, SESSION_MODE_STRUCTURED, session_id=1),
            _block("2015-09-10T15:00:00", 600, SESSION_MODE_FLUID, session_id=2),
        ]
        s = m.summarise(blocks, _events(EVENT_DISTRACTION_POSE, 3, "2015-09-10T15:05:00"))
        assert s.focus_s == 3600
        assert s.active_days == 2
        assert s.best_day == (date(2015, 9, 9), 3000)
        assert s.session_count == 2
        assert s.median_block_s == 1500
        assert s.sessions_by_mode == {SESSION_MODE_STRUCTURED: 1, SESSION_MODE_FLUID: 1}
        assert s.per_focus_hour(EVENT_DISTRACTION_POSE) == pytest.approx(3.0)

    def test_rate_withheld_below_focus_floor(self):
        s = m.summarise(
            [_block("2015-09-10T15:00:00", 120)],
            _events(EVENT_DISTRACTION_POSE, 3, "2015-09-10T15:01:00"),
        )
        assert s.per_focus_hour(EVENT_DISTRACTION_POSE) is None


# ════════════════════════════════════════════════════════════════
#  TAKEAWAYS
# ════════════════════════════════════════════════════════════════


def _worked_periods(extra_cur_events: list[dict] | None = None):
    """The worked last-7-vs-previous-7 example in the module header."""
    cur_blocks = [
        _block("2015-09-05T15:00:00", 4.59 * 3600 / 3, session_id=1),
        _block("2015-09-07T15:00:00", 4.59 * 3600 / 3, session_id=2),
        _block("2015-09-09T15:00:00", 4.59 * 3600 / 3, session_id=3),
    ]
    prev_blocks = [_block("2015-09-01T15:00:00", 3.55 * 3600, session_id=4)]
    # The five fatigue alerts carry onsets with a 16 min median, so the
    # onset takeaway also fires (it ranks after the comparisons).
    fatigue = [
        e
        for minute in (10, 12, 16, 20, 40)
        for e in _events(EVENT_FATIGUE, 1, "2015-09-09T15:30:00", elapsed=minute)
    ]
    cur_events = (
        _events(EVENT_DISTRACTION_POSE, 16, "2015-09-09T15:30:00")
        + fatigue
        + _events(EVENT_DISTRACTION_WINDOW, 3, "2015-09-09T15:30:00")
        + (extra_cur_events or [])
    )
    prev_events = _events(EVENT_DISTRACTION_POSE, 23, "2015-09-01T15:30:00") + _events(
        EVENT_FATIGUE, 1, "2015-09-01T15:30:00"
    )
    return cur_blocks, cur_events, prev_blocks, prev_events


def _texts(items) -> list[str]:
    return [t.text for t in items]


class TestTakeawayGates:
    def test_head_pose_drop_fires_with_correct_percentage(self):
        # (16 / 4.59 h) / (23 / 3.55 h) − 1 = 3.49 / 6.48 − 1 = −0.46
        items = m.build_takeaways("7d", *_worked_periods(), structured_work_min=25)
        pose = [t for t in items if t.text.startswith("Head-pose distractions")]
        assert len(pose) == 1
        assert pose[0].text == (
            "Head-pose distractions: 46% fewer per focus hour than the previous 7 days."
        )
        assert (pose[0].tone, pose[0].direction) == ("good", "down")

    def test_small_counts_and_zero_baseline_withheld(self):
        texts = " ".join(_texts(m.build_takeaways("7d", *_worked_periods(), 25)))
        assert "Fatigue alerts:" not in texts  # 5 vs 1 — 6 events
        assert "Window" not in texts  # 3 vs 0 — no baseline

    def test_focus_change(self):
        items = m.build_takeaways("7d", *_worked_periods(), structured_work_min=25)
        focus = [t for t in items if t.text.startswith("Focus")]
        assert len(focus) == 1
        assert focus[0].text.startswith("Focus up 29% vs the previous 7 days")
        assert (focus[0].tone, focus[0].direction) == ("good", "up")

    def test_all_range_makes_no_comparisons(self):
        cur_blocks, cur_events, _, _ = _worked_periods()
        items = m.build_takeaways("all", cur_blocks, cur_events, None, None, 25)
        assert not any(t.direction for t in items)

    def test_today_skips_total_focus_comparison(self):
        cur_blocks, cur_events, prev_blocks, prev_events = _worked_periods()
        items = m.build_takeaways("today", cur_blocks, cur_events, prev_blocks, prev_events, 25)
        assert not any(t.text.startswith("Focus") for t in items)

    def test_rate_change_needs_an_hour_of_focus_each_side(self):
        cur_blocks = [_block("2015-09-09T15:00:00", 1800)]
        prev_blocks = [_block("2015-09-01T15:00:00", 3.55 * 3600)]
        cur_events = _events(EVENT_DISTRACTION_POSE, 16, "2015-09-09T15:10:00")
        prev_events = _events(EVENT_DISTRACTION_POSE, 23, "2015-09-01T15:10:00")
        items = m.build_takeaways("7d", cur_blocks, cur_events, prev_blocks, prev_events, 25)
        assert not any(t.direction for t in items)


class TestTakeawayRules:
    def test_onset_clause_only_when_before_block_length(self):
        blocks = [_block("2015-09-09T15:00:00", 3600)]
        events = [
            e
            for minute in (10, 12, 16, 20, 40)
            for e in _events(EVENT_FATIGUE, 1, "2015-09-09T15:30:00", elapsed=minute)
        ]
        early = m.build_takeaways("all", blocks, events, None, None, structured_work_min=25)
        assert _texts(early) == [
            "Fatigue alerts typically come ~16 min into a block, "
            "before your 25-min Structured blocks end."
        ]
        late = m.build_takeaways("all", blocks, events, None, None, structured_work_min=10)
        assert _texts(late) == ["Fatigue alerts typically come ~16 min into a block."]

    def test_onset_needs_enough_alerts(self):
        blocks = [_block("2015-09-09T15:00:00", 3600)]
        events = _events(EVENT_FATIGUE, 4, "2015-09-09T15:30:00")
        assert m.build_takeaways("all", blocks, events, None, None, 25) == []

    def test_onset_clause_omitted_without_structured_blocks(self):
        blocks = [_block("2015-09-09T15:00:00", 3600, SESSION_MODE_FLUID)]
        events = _events(EVENT_FATIGUE, 5, "2015-09-09T15:30:00", elapsed=10)
        assert _texts(m.build_takeaways("all", blocks, events, None, None, 25)) == [
            "Fatigue alerts typically come ~10 min into a block."
        ]

    def test_peak_window_needs_active_days(self):
        two_days = [_block("2015-09-08T15:00:00", 3600), _block("2015-09-09T15:00:00", 3600)]
        assert m.build_takeaways("all", two_days, [], None, None, 25) == []

        three_days = two_days + [_block("2015-09-10T09:00:00", 1800)]
        # All 15:00 focus sits in one hour, so the empty neighbour is trimmed.
        assert _texts(m.build_takeaways("all", three_days, [], None, None, 25)) == [
            "Your peak focus window is 15:00–16:00 (80% of focus time)."
        ]

    def test_peak_window_spans_two_hours_when_both_hold_focus(self):
        blocks = [_block(f"2015-09-0{d}T15:30:00", 3600) for d in (7, 8, 9)]
        assert _texts(m.build_takeaways("all", blocks, [], None, None, 25)) == [
            "Your peak focus window is 15:00–17:00 (100% of focus time)."
        ]

    def test_cap_and_priority_order(self):
        # Window 12 vs 4: (12 / 4.59 h) / (4 / 3.55 h) − 1 = +1.32
        window_spike = _events(EVENT_DISTRACTION_WINDOW, 9, "2015-09-09T15:30:00")
        cur_blocks, cur_events, prev_blocks, prev_events = _worked_periods(window_spike)
        prev_events = prev_events + _events(EVENT_DISTRACTION_WINDOW, 4, "2015-09-01T15:30:00")

        items = m.build_takeaways(
            "7d", cur_blocks, cur_events, prev_blocks, prev_events, structured_work_min=25
        )
        texts = _texts(items)
        assert len(items) == INSIGHTS_TAKEAWAY_MAX
        assert texts[0].startswith("Window distractions: 132% more")  # largest change first
        assert texts[1].startswith("Head-pose distractions: 46% fewer")
        assert texts[2].startswith("Focus up 29%")
        assert texts[3].startswith("Fatigue alerts typically")  # peak window cut

    def test_nothing_qualifies(self):
        assert m.build_takeaways("7d", [], [], [], [], 25) == []


# ════════════════════════════════════════════════════════════════
#  PRECISION (from review labels)
# ════════════════════════════════════════════════════════════════

_TS = "2015-09-09T15:30:00"


class TestPrecision:
    """Users who rate alerts at each rest see how often each detector is right
    for them.  Below 5 labels one rating swings the figure by 20+ points, so
    no figure is shown at all."""

    def test_precision_appears_at_five_labels(self):
        # Worked example: 4 correct + 1 false alarm → 4 / 5 = 80 %.
        # The two unrated events count neither way.
        events = (
            _events(EVENT_DISTRACTION_POSE, 4, _TS, label=EVENT_LABEL_CORRECT)
            + _events(EVENT_DISTRACTION_POSE, 1, _TS, label=EVENT_LABEL_FALSE_ALARM)
            + _events(EVENT_DISTRACTION_POSE, 2, _TS)
        )
        assert m.label_counts(events, EVENT_DISTRACTION_POSE) == (4, 5)
        assert m.precision(events, EVENT_DISTRACTION_POSE) == pytest.approx(0.8)

    def test_precision_absent_below_five_labels(self):
        # 3 correct + 1 false alarm is 4 labels: withheld, however many
        # unrated events there are.
        events = (
            _events(EVENT_FATIGUE, 3, _TS, label=EVENT_LABEL_CORRECT)
            + _events(EVENT_FATIGUE, 1, _TS, label=EVENT_LABEL_FALSE_ALARM)
            + _events(EVENT_FATIGUE, 10, _TS)
        )
        assert m.label_counts(events, EVENT_FATIGUE) == (3, 4)
        assert m.precision(events, EVENT_FATIGUE) is None

    def test_each_detector_counts_only_its_own_labels(self):
        # Window: 5 false alarms → 0 %.  Head pose has only 4 labels, so the
        # window's labels must not lift it over the gate.
        events = _events(EVENT_DISTRACTION_WINDOW, 5, _TS, label=EVENT_LABEL_FALSE_ALARM) + _events(
            EVENT_DISTRACTION_POSE, 4, _TS, label=EVENT_LABEL_CORRECT
        )
        assert m.precision(events, EVENT_DISTRACTION_WINDOW) == 0.0
        assert m.precision(events, EVENT_DISTRACTION_POSE) is None
