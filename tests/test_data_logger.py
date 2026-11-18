# Tests for the per-frame debug CSV (DataLogger).
# ─────────────────────────────────────────────────────────────────
# The debug CSVs are the fatigue corpus: tools/replay_csv_fatigue.py
# re-drives the fatigue engine from them to prove the engine still
# behaves as evaluated.  These tests hold the logger to that contract
# and to the data-folder rule (nothing written outside data_dir()).
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from acuity.core.data_logger import DataLogger
from acuity.core.debug_snapshot import DebugSnapshot

_REPLAY_TOOL = Path(__file__).resolve().parents[1] / "tools" / "replay_csv_fatigue.py"

# Read by the replay tool when present (blink_score, yawn_threshold,
# fatigue_score) or for event timestamps (timestamp_wall), on top of its
# _REQUIRED list, so the logger must keep writing them.
_REPLAY_OPTIONAL = ["blink_score", "yawn_threshold", "fatigue_score", "timestamp_wall"]


def _replay_required_columns() -> list[str]:
    """The replay tool's own _REQUIRED list, loaded from the tool itself."""
    spec = importlib.util.spec_from_file_location("replay_csv_fatigue", _REPLAY_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module._REQUIRED)


def _header_on_disk(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_csv_header_has_every_column_the_replay_tool_reads(data_dir):
    """A renamed or dropped column silently removes a session from the
    replay gate (the tool skips CSVs that lack a required column), so
    the fatigue invariant would go unchecked on new recordings."""
    logger = DataLogger()
    logger.close()

    header = _header_on_disk(logger.path)
    missing = [c for c in _replay_required_columns() + _REPLAY_OPTIONAL if c not in header]
    assert missing == []


def test_csv_is_written_under_the_data_folder(data_dir):
    """Runtime data goes to the per-user data folder (ACUITY_DATA_DIR in
    tests), never into the repo or the working directory."""
    with DataLogger() as logger:
        logger.log(DebugSnapshot(), event_fired=True, event_type="Fatigue", event_reason="x")

    assert logger.path.parent == (data_dir / "logs").resolve()
    with logger.path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + one frame
