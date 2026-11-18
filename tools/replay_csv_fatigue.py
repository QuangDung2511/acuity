# Replays logged debug CSVs through the real FatigueEngine.
# ─────────────────────────────────────────────────────────────────
# Thread: standalone CLI (no threads, no Qt).  It is the fatigue invariant's
# gate: `--json` output over the private corpus must stay byte-identical
# to the evaluated build's.  Its `_REQUIRED` column list (plus
# blink_score, yawn_threshold, fatigue_score, timestamp_wall) is the
# debug-CSV contract the data logger must keep.  Design notes are in
# the docstring below.
# ─────────────────────────────────────────────────────────────────
r"""Replay logged debug CSVs through the real FatigueEngine.

Complements the original build's video replay script.  That needs a
video; the corpus in data/logs/ is CSV only.  Every per-frame input the fatigue
engine consumes is logged, so the engine can be re-driven from the CSV
and its *scored* output compared across code or constant variants.

Usage:
    .venv\Scripts\python.exe tools/replay_csv_fatigue.py [CSV ...]
        (default: every data/logs/session_*.csv with the required columns)
    --set NAME=VALUE   override a module-level name in fatigue_engine
                       before replay (repeatable), e.g.
                       --set EAR_YAW_RATE_SUSTAIN_S=0.15
    --validate         report max |replayed - logged| fatigue_score per
                       session (only meaningful against the engine version
                       that wrote the log)
    --json OUT         dump per-session events for diffing two runs

Design notes
------------
- time.monotonic inside fatigue_engine is patched to the logged
  timestamp_mono, so every seconds-based window sees the timing the
  session actually had.
- The engine is constructed with a dummy baseline and then given the
  logged blink_threshold / yawn_threshold, which are what the
  calibration produced.  Baseline pitch is recovered from pitch and
  pitch_delta (= |pitch - baseline|): of the two candidates per frame,
  the one that stays constant across the session is the baseline.
- blink_score is logged per frame only by newer sessions, and is passed
  through when present.  Older sessions have no column, so
  it is passed as None and the blink-confirm gate fails open: microsleep
  and sleep counts are exact for sessions that log it and an upper bound
  for those that don't.  Warning / alert-tier counts are valid for both.
- --episodes prints every eyes-shut episode of at least 1.0 s with its
  bilateral share and blink median — the distribution SLEEP_CLOSURE_S is
  chosen from.
- The replay is NOT sample-exact against the log.  DataLogger stamps
  timestamp_mono a few ms *after* the engine ran, so a frame landing
  within milliseconds of a settle-window deadline can resolve the other
  way, and the difference compounds through blink / PERCLOS state
  (in one logged session the first divergence fell on a settle deadline,
  with a max |score diff| of 0.39 later; its logged events still
  reproduce exactly, private corpus, n=1).  Compare replay against replay — variant vs
  current code — never replay against the logged score.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import acuity.core.fatigue_engine as fe_mod
from acuity.core.face_mesh import FaceResult
from acuity.core.presence import PresenceState

_REQUIRED = [
    "timestamp_mono",
    "presence",
    "face_detected",
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "pitch",
    "yaw",
    "blink_threshold",
    "pitch_delta",
]


def _baseline_pitch(rows: list[dict]) -> float:
    votes: Counter = Counter()
    for r in rows:
        if r["face_detected"] != "1":
            continue
        p, d = float(r["pitch"]), float(r["pitch_delta"])
        votes[round(p + d, 1)] += 1
        votes[round(p - d, 1)] += 1
    return votes.most_common(1)[0][0] if votes else 0.0


def replay(path: str) -> dict:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    events: list[tuple[str, str, str]] = []
    wall = [""]
    clock = [float(rows[0]["timestamp_mono"])]
    fe_mod.time = SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0])

    engine = fe_mod.FatigueEngine(
        baseline_ear=0.3,
        callback=lambda e: events.append((wall[0], e.event_type, e.reason)),
    )
    engine._blink_threshold = float(rows[0]["blink_threshold"])
    if rows[0].get("yawn_threshold"):
        engine._yawn_threshold = float(rows[0]["yawn_threshold"])
    base_pitch = _baseline_pitch(rows)

    max_diff = 0.0
    supp_time = obs_time = 0.0
    supp_onsets = 0
    prev_t = None
    prev_supp = False
    ratio_by_bucket: dict[int, list[float]] = defaultdict(list)
    pause_hist: deque[tuple[float, float, bool]] = deque()
    pause_sum = tot_sum = 0.0
    episodes: list[tuple] = []
    episode = None
    context_withheld = 0
    prev_ctx = False

    for r in rows:
        t = float(r["timestamp_mono"])
        clock[0] = t
        wall[0] = r["timestamp_wall"][:19]
        face_ok = r["face_detected"] == "1"
        face = FaceResult(
            ear_left=float(r["ear_left"]),
            ear_right=float(r["ear_right"]),
            ear_avg=float(r["ear_avg"]),
            mar=float(r["mar"]),
            pitch=float(r["pitch"]),
            yaw=float(r["yaw"]),
            roll=0.0,
            face_detected=face_ok,
            blink_score=float(r["blink_score"]) if r.get("blink_score") else None,
        )
        presence = PresenceState[r["presence"]]
        engine.update(
            face, presence, head_pitch=face.pitch, baseline_pitch=base_pitch, mar=face.mar
        )
        if engine.debug_microsleep_withheld == "context" and not prev_ctx:
            context_withheld += 1
        prev_ctx = engine.debug_microsleep_withheld == "context"

        # Episode tracking: snapshot while shut, emit when the episode ends.
        start = engine._eyes_closed_start
        if start is not None and engine._eyes_open_since is None:
            episode = (
                start,
                t,
                engine.debug_episode_bilateral_share,
                engine.debug_episode_blink_confirm if engine._episode_blink_scores else None,
                wall[0],
            )
        if episode is not None and start != episode[0]:
            if episode[1] - episode[0] >= 1.0:
                episodes.append(
                    (
                        episode[4],
                        round(episode[1] - episode[0], 2),
                        round(episode[2], 2),
                        None if episode[3] is None else round(episode[3], 3),
                    )
                )
            episode = None
        if r.get("fatigue_score"):
            max_diff = max(max_diff, abs(engine.fatigue_score - float(r["fatigue_score"])))

        dt = 0.0 if prev_t is None else min(t - prev_t, 0.5)
        prev_t = t
        paused = presence != PresenceState.PRESENT or engine.debug_ear_suppressed
        if face_ok and presence == PresenceState.PRESENT:
            obs_time += dt
            if engine.debug_ear_suppressed:
                supp_time += dt
                if not prev_supp:
                    supp_onsets += 1
        prev_supp = engine.debug_ear_suppressed

        # Blink-rate bias probe: measured rate / baseline, bucketed by the
        # share of the trailing 60 s that was paused.
        pause_hist.append((t, dt, paused))
        tot_sum += dt
        pause_sum += dt if paused else 0.0
        while pause_hist and pause_hist[0][0] < t - 60.0:
            _, d, p = pause_hist.popleft()
            tot_sum -= d
            pause_sum -= d if p else 0.0
        bb = engine.baseline_blink_rate
        if (
            not paused
            and bb
            and bb >= 5.0
            and tot_sum > 55.0
            and engine._windows_since_baseline >= 1
        ):
            ratio_by_bucket[min(int(pause_sum / tot_sum * 10), 6)].append(engine.blink_rate / bb)

    hours = (float(rows[-1]["timestamp_mono"]) - float(rows[0]["timestamp_mono"])) / 3600
    return {
        "events": events,
        "hours": hours,
        "max_score_diff": max_diff,
        "supp_time": supp_time,
        "obs_time": obs_time,
        "supp_onsets": supp_onsets,
        "ratio_by_bucket": {k: v for k, v in ratio_by_bucket.items()},
        "episodes": episodes,
        "context_withheld": context_withheld,
        "has_blink_score": bool(rows[0].get("blink_score")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--episodes", action="store_true")
    args = ap.parse_args()
    # The engine logs every blink at DEBUG; over the corpus that is millions
    # of lines and dominates the runtime.
    logging.disable(logging.CRITICAL)

    for spec in args.set:
        name, value = spec.split("=", 1)
        setattr(fe_mod, name, type(getattr(fe_mod, name))(value))

    files = args.csvs or sorted(glob.glob("data/logs/session_*.csv"))
    usable = []
    for f in files:
        with open(f, newline="") as fh:
            header = next(csv.reader(fh), [])
        if (
            all(c in header for c in _REQUIRED)
            and Path(f).stat().st_size > len(",".join(header)) + 2
        ):
            usable.append(f)

    totals = Counter()
    hours = supp = obs = onsets = 0.0
    buckets: dict[int, list[float]] = defaultdict(list)
    per_session = {}
    for f in usable:
        res = replay(f)
        name = Path(f).stem
        per_session[name] = res["events"]
        hours += res["hours"]
        supp += res["supp_time"]
        obs += res["obs_time"]
        onsets += res["supp_onsets"]
        for k, v in res["ratio_by_bucket"].items():
            buckets[k].extend(v)
        for _, _, reason in res["events"]:
            totals[reason] += 1
        totals["context_withheld"] += res["context_withheld"]
        if args.validate:
            print(f"  {name}: max |score diff| {res['max_score_diff']:.4f}  events {res['events']}")
        if args.episodes:
            tag = "" if res["has_blink_score"] else "  (no blink_score)"
            for ep in res["episodes"]:
                print(f"  EP {name} {ep[0]} dur={ep[1]:.2f}s bilat={ep[2]:.2f} blink={ep[3]}{tag}")

    print(f"{len(usable)} sessions, {hours:.1f} h")
    print(
        f"EAR suppressed {supp / obs * 100:.1f}% of observed face time, "
        f"{onsets / hours:.0f} onsets/h"
    )
    for reason in (
        "warning_threshold",
        "alert_threshold",
        "microsleep",
        "sleep",
        "context_withheld",
    ):
        print(f"  {reason:18s} {totals[reason]:4d}  ({totals[reason] / hours:.2f}/h)")
    print("  paused share of last 60 s -> median blink_rate/baseline (n)")
    for k in sorted(buckets):
        print(f"    {k * 10:2d}%+  {statistics.median(buckets[k]):.2f}  ({len(buckets[k])})")
    if args.json:
        Path(args.json).write_text(json.dumps(per_session, indent=1))


if __name__ == "__main__":
    main()
