# Coding standards

The judgement rules a reviewer checks a diff against. Anything a tool enforces is not repeated here:

| Enforced by | What |
|---|---|
| `ruff format` / `ruff check` | formatting, import order, unused code, common bug patterns |
| `lint-imports` | layering `ui → core → (db, utils, system)`; `system` never imports `core` or `ui` |
| `tools/check_private_data.py` | no private data in git (the script defines what counts) |
| `tools/check_headers.py` | module header shape: purpose line, box rule, `# Thread:` line (rule 4) |
| `tests/conftest.py` | tests write no runtime data into the repo |

If a rule below can be made mechanical, make it a check and delete it from here.

## Evidence

1. **Every threshold constant carries an evidence comment.** Either the measurement it rests on (distribution, sample size, why it moved), or the literal phrase *unmeasured default, tune from labels*. Changing a threshold means updating its evidence, not just the number.
2. **Evidence from the private corpus is summarised, never referenced.** Write "median 0.21 over 35 sessions (private corpus, n=1)", not an event ID, session filename or date. Requirement tags from the predecessor's spec (`FR-xx`, `NFR-xx`, `§n`) point at documents this repo does not have; write the plain-language intent instead. Long tuning histories go in `docs/decisions/`, and the comment on the constant keeps a short summary plus a pointer to the ADR. Exception: comments carried over verbatim by a port *import* commit (rule 14) are scrubbed in the following cuts commit; flag only references a diff *adds*.

## Time

3. **Time is wall-clock seconds, never frame counts.** No "every Nth frame" and no thresholds in frames. Detection must not change when the capture rate does.

## Modules

4. **Module headers tell the truth.** `tools/check_headers.py` enforces the shape. The reviewer checks the content: the purpose line matches what the module does, the `# Thread:` line names the thread it really runs on (worker, GUI, pure/any, or standalone CLI), and the box carries the design rationale a reader would otherwise have to rediscover. A package `__init__.py` states its layer's import rule instead of a thread.
5. **Pure core modules stay pure.** Engines, presence, calibration, focus rules: no Qt and no I/O. (The ported engines read `time.monotonic()` themselves. New modules such as the window engine take `now` as an argument, which makes them testable with irregular time steps.) import-linter enforces the Qt part for modules listed in the `forbidden` contract in `pyproject.toml`; a ticket that adds a pure module adds it there.
6. **`debug_*` properties are read-only observation points** for the debug overlay and CSV. Control flow never reads them.
7. **Head-pose angles are compared with `wrap_deg`**, never by plain subtraction. Euler angles wrap at ±180°.
8. **Platform-specific code lives in `acuity.system`** behind a `create_*()` factory. Only the factory imports the OS-specific module.

## Change discipline

9. **Surgical changes.** A diff touches only what its ticket needs. No drive-by refactors, renames or reformatting of neighbouring code. Flag cleanups separately instead.
10. **Simplest thing that works.** No parameters, hooks or abstractions for a need the ticket does not have.
11. **Surface conflicts, don't blend them.** If new code meets two contradictory patterns, follow one, say why in the PR or ticket, and flag the other for cleanup.
12. **Fail loud.** A skipped test, an unhandled platform, or an unavailable adapter is reported (log, `available`/`reason`, test skip reason), never silently ignored.

## Tests

13. **Tests sit at agreed seams and assert behaviour,** not internals. Each test's name or docstring says *why* the behaviour matters, and expected values come from an independent source (a worked example, the spec, a measurement), not recomputed the way the code computes them.

## Porting from the original codebase (Phase 1 only)

14. **Two commits per port ticket:**
    1. *Import*: copy the files verbatim, then apply only mechanical transforms, in this order: `src.` → `acuity.`, PyQt6 → PySide6 (`pyqtSignal` → `Signal`, `pyqtSlot` → `Slot`), `ruff check --fix`, `ruff format`, then `ruff check --add-noqa` so any finding ruff cannot fix is suppressed rather than hand-edited. No hand edits at all. The commit message records the exact commands, so a reviewer can re-run them instead of reading the diff.
    2. *Cuts*: every removal and behavioural edit, including resolving the `# noqa` markers the import added. This is the commit to review line by line.
15. **The fatigue invariant holds** (CLAUDE.md): no logic edit to `fatigue_engine.py`, and a ticket that touches the fatigue path reports a byte-identical replay gate.
