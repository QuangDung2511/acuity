# Module-header check: the mechanical half of CODING_STANDARDS.md rule 4.
# ─────────────────────────────────────────────────────────────────
# What it checks, for every .py file under src/acuity, tools and tests:
#   • line 1 is a "# <purpose>" line and line 2 a "# ───" box rule;
#   • a later "# ───" rule closes the box, with only "#" lines between;
#   • modules under src/ and tools/, except a package __init__.py,
#     have a "# Thread: ..." line in the box (worker, GUI, pure/any or
#     standalone CLI).
# Whether the purpose, thread and rationale are right stays a review call.
#
# Usage (from the repo root):
#   python tools/check_headers.py     # exit 0 clean, 1 if any are found
#
# Thread: standalone CLI (no threads, no Qt). Stdlib only, like the
# private-data guard, so it runs before (or without) installing the package.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

_ROOTS = ("src/acuity", "tools", "tests")
_NEEDS_THREAD = ("src/", "tools/")
_RULE = "# ─"


def problem(rel: str, text: str) -> str | None:
    """Return what is wrong with the header of the file at *rel*, or None."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# ") or lines[0].startswith(_RULE):
        return "line 1 must be a '# <purpose>' line"
    if len(lines) < 2 or not lines[1].startswith(_RULE):
        return "line 2 must be a '# ───' box rule"
    box = []
    for line in lines[2:]:
        if not line.startswith("#"):
            break
        if line.startswith(_RULE):
            needs_thread = rel.startswith(_NEEDS_THREAD) and not rel.endswith("__init__.py")
            if needs_thread and not any(b.startswith("# Thread:") for b in box):
                return "the header box has no '# Thread: ...' line"
            return None
        box.append(line)
    return "the header box is not closed by a second '# ───' rule"


def main() -> int:
    root = Path.cwd()
    found = []
    for top in _ROOTS:
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            msg = problem(rel, path.read_text(encoding="utf-8"))
            if msg:
                found.append(f"  {rel}: {msg}")
    if not found:
        return 0
    print("Module headers that break CODING_STANDARDS.md rule 4:", file=sys.stderr)
    print("\n".join(found), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
