# Private-data guard: fail if git tracks any runtime or recorded data.
# ─────────────────────────────────────────────────────────────────
# Why it exists:
#   The project this repo was ported from committed and pushed its
#   SQLite database, WAL/SHM files, rotated logs and a replay CSV.
#   This makes that mistake fail at commit time (pre-commit hook) and
#   again in CI, instead of relying on .gitignore alone, which `git
#   add -f` or a missing pattern silently defeats.
#
# What counts:
#   Every path in the git index (`git ls-files`), so files staged for
#   the commit being made are caught too. Untracked files are ignored.
#
# Usage (from the repo root):
#   python tools/check_private_data.py     # exit 0 clean, 1 if found
#
# Thread: standalone CLI (no threads, no Qt). Stdlib only, so it runs
# before (or without) installing the package.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
import subprocess
import sys

# The single definition of "private data"; the docs point here.
#   *.db plus any sidecar or copy (x.db-wal, x.db.bak), debug CSVs,
#   recorded video, logs plus any rotation or compression (x.log.1,
#   x.log.2031-01-31, x.log.gz), the runtime data/ dir and the private
#   fixtures dir. Case-insensitive, because git on Windows is.
_PRIVATE_PATH = re.compile(
    r"\.(db|csv|mp4)($|[.-])|\.log($|\.)|^data/|^tests/fixtures/private/",
    re.IGNORECASE,
)


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout.split("\0")
    found = [p for p in tracked if p and _PRIVATE_PATH.search(p)]
    if not found:
        return 0
    print("Private data is tracked by git and must not be committed:", file=sys.stderr)
    for path in found:
        print(f"  {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
