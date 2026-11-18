#!/usr/bin/env bash
# Claude Code SessionStart hook: put the project venv first on PATH.
# ─────────────────────────────────────────────────────────────────
# Agent shells start without the venv active, so `python`, `ruff`,
# `lint-imports` and the pre-commit hooks (language: system) resolve to
# the wrong interpreter or to nothing. Lines appended to $CLAUDE_ENV_FILE
# are sourced before every Bash tool command for the rest of the session.
#
# Thread: standalone script, run once per session start, /clear and resume.
# ─────────────────────────────────────────────────────────────────
[ -n "$CLAUDE_ENV_FILE" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-$PWD}"
# Git Bash on Windows: c:\Users\... must become /c/Users/... for PATH.
if command -v cygpath >/dev/null 2>&1; then
  root="$(cygpath -u "$root")"
fi

for bin in "$root/.venv/Scripts" "$root/.venv/bin"; do
  if [ -d "$bin" ]; then
    # Single quotes keep $PATH literal, so it expands when sourced.
    printf 'export PATH="%s:$PATH"\n' "$bin" >> "$CLAUDE_ENV_FILE"
    printf 'export VIRTUAL_ENV="%s"\n' "${bin%/*}" >> "$CLAUDE_ENV_FILE"
    exit 0
  fi
done
