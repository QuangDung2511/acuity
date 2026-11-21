# CLAUDE.md

Acuity: a desktop app (PySide6) that watches the user through the webcam during focus sessions and detects fatigue and distraction. Everything runs locally: no network calls, no cloud inference, nothing leaves the machine.

## Commands

```
.venv\Scripts\python.exe -m pip install -e ".[dev]"   # once
pre-commit install                                     # once, venv active
python -m pytest -q                                    # tests
ruff check && ruff format --check && lint-imports      # lint and layering
python tools/check_private_data.py                     # private-data guard
python tools/check_headers.py                          # module headers
```

CI (`.github/workflows/ci.yml`) runs the same checks on Windows, macOS and Ubuntu. Run Qt tests headless with `QT_QPA_PLATFORM=offscreen`. Integration tests that need the private video fixture (`tests/fixtures/private/`) skip themselves when it is absent.

## Architecture

- **Layers**, one-directional: `acuity.ui` → `acuity.core` → (`acuity.db`, `acuity.utils`, `acuity.system`). `system` holds the platform adapters and never imports `core` or `ui`. `lint-imports` enforces this, plus a no-Qt contract on the pure core modules listed in `pyproject.toml`.
- **Two threads.** The camera/CV pipeline (`core/cv_pipeline.py`) is a `QThread` worker: camera → face mesh → pose → presence → engines. Everything else (session manager, timers, alerts, UI) lives on the GUI thread. They talk only through Qt signals; the GUI side may set plain attributes the worker reads once per frame.
- **Session state machine** (`core/session_manager.py`): IDLE → CALIBRATING → FOCUS ⇄ REST → … → ENDED, with AWAY (face lost) and PAUSED (manual) reachable from FOCUS. Every focus block opens with a 10 s calibration and builds fresh engines; the camera is released during REST and AWAY.
- **Engines are pure logic**, constructed from that block's `CalibrationResult`. Thresholds are relative to the user's own baseline wherever the signal allows; the few absolute ones (e.g. the blink-confirmation and PERCLOS gates) say so in their comments.
- **Fatigue model**: a composite score from four signals (PERCLOS, blink duration, yawning, blink frequency) drives a warning tier and an alert tier. Microsleep and sleep are separate eye-closure paths behind a chain of gates. The reasoning behind each gate and threshold is in `docs/decisions/`.

## The fatigue invariant

`fatigue_engine.py` must stay **behaviour-identical** to the evaluated build. The replay gate checks it: replay the private debug-CSV corpus through the engine and require byte-identical JSON against the stored baseline. The corpus is private, so the gate runs only on the maintainer's machine; instructions and the baseline live in the gitignored `.scratch/open-source-port/replay-gate/README.md`.

- Comment edits to `fatigue_engine.py` are fine; logic edits are not.
- Re-run the replay gate after any change that touches the fatigue path (the engine, its constants, calibration, face mesh, or the replay tool), and report the result.

## Standards

Review rules live in `CODING_STANDARDS.md`. Read it before writing code that will be reviewed.

## Privacy

- Never commit runtime data. `tools/check_private_data.py` defines what counts, and fails the commit if you do.
- Evidence from the private corpus is summarised: a measurement marked "(private corpus, n=1)". Keep event IDs, session filenames, dates and personal details out of code, comments, docs and commit messages.
- Long tuning histories belong in `docs/decisions/`; the code comment carries a short summary and a pointer.

## Git workflow

Before starting a ticket, committing, pushing or opening a PR, read `docs/agents/git-workflow.md`: one branch and one PR per ticket, merged into a protected `main` once CI passes on all three OSes.

## Agent skills

### Issue tracker

Local markdown under `.scratch/<feature>/` (gitignored, never pushed). See `docs/agents/issue-tracker.md`.

### Triage labels

The five default roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), written as each ticket's `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` at the root, ADRs in **`docs/decisions/`** (not `docs/adr/`; this overrides skill defaults). See `docs/agents/domain.md`.
