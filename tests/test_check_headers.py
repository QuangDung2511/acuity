# Tests for tools/check_headers.py, the module-header check (rule 4).
# ─────────────────────────────────────────────────────────────────
# Seam: the script's command line, run from a throwaway project root,
# observed through its exit code and output. That is how CI and the
# pre-commit hook call it.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_headers.py"
_RULE = "# " + "─" * 65

_GOOD_MODULE = f"""# Pure scoring helpers.
{_RULE}
# Thread: pure.
{_RULE}
x = 1
"""


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT)], cwd=root, capture_output=True, text=True, encoding="utf-8"
    )


def test_a_complete_header_passes(tmp_path: Path) -> None:
    _write(tmp_path, "src/acuity/core/scoring.py", _GOOD_MODULE)

    assert _run(tmp_path).returncode == 0


def test_a_module_without_a_thread_line_fails_and_is_named(tmp_path: Path) -> None:
    _write(tmp_path, "src/acuity/core/scoring.py", _GOOD_MODULE.replace("# Thread: pure.", "#"))
    _write(tmp_path, "src/acuity/core/fine.py", _GOOD_MODULE)

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "src/acuity/core/scoring.py" in result.stderr
    assert "Thread" in result.stderr
    assert "fine.py" not in result.stderr


def test_packages_and_tests_state_no_thread(tmp_path: Path) -> None:
    # A package __init__ states its layer's import rule instead, and a
    # test file has no thread of its own.
    no_thread = f"# Purpose.\n{_RULE}\n# May import: utils.\n{_RULE}\n"
    _write(tmp_path, "src/acuity/core/__init__.py", no_thread)
    _write(tmp_path, "tests/test_scoring.py", no_thread)

    assert _run(tmp_path).returncode == 0


def test_a_file_that_opens_with_a_docstring_fails(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_scoring.py", '"""Tests for scoring."""\n')

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "line 1" in result.stderr


def test_an_unclosed_box_fails(tmp_path: Path) -> None:
    _write(tmp_path, "tools/replay.py", f"# Replay tool.\n{_RULE}\n# Thread: CLI.\nimport sys\n")

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "not closed" in result.stderr
