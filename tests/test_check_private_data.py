# Tests for tools/check_private_data.py, the private-data guard.
# ─────────────────────────────────────────────────────────────────
# Seam: the script's command line, run inside a throwaway git repo,
# observed through its exit code and output. That is exactly how CI
# and the pre-commit hook call it, so nothing here reaches inside.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_private_data.py"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _track(repo: Path, *paths: str) -> None:
    """Create each file and stage it (staged files are what a commit would add)."""
    for rel in paths:
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "--", *paths], check=True)


def _run_guard(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_SCRIPT)], cwd=repo, capture_output=True, text=True)


def test_tracked_database_fails_and_is_named(repo: Path) -> None:
    _track(repo, "src/app.py", "backup.db")

    result = _run_guard(repo)

    assert result.returncode == 1
    assert "backup.db" in result.stderr
    assert "src/app.py" not in result.stderr


# Each of these leaked, or would have, from the original repo.
@pytest.mark.parametrize(
    "path",
    [
        "acuity.db-wal",
        "acuity.db-shm",
        "acuity.db-journal",
        "acuity.db.bak",  # hand-made backup copy
        "replay.csv",
        "tests/fixtures/session.mp4",
        "tests/fixtures/SESSION.MP4",  # Windows git does not care about case
        "tests/fixtures/private/labels.json",  # anything in the private fixtures dir
        "acuity.log",
        "acuity.log.1",  # size-rotated log
        "acuity.log.2031-01-31",  # time-rotated log
        "acuity.log.gz",  # compressed log
        "data/notes.txt",  # anything under the runtime data dir
    ],
)
def test_each_private_kind_fails(repo: Path, path: str) -> None:
    _track(repo, path)

    result = _run_guard(repo)

    assert result.returncode == 1
    assert path in result.stderr


def test_source_that_only_looks_similar_passes(repo: Path) -> None:
    _track(
        repo,
        "src/acuity/db/database.py",  # the db package, not a database
        "src/acuity/core/data_logger.py",
        "docs/logging.md",
        "tools/replay_csv_fatigue.py",
    )

    assert _run_guard(repo).returncode == 0
