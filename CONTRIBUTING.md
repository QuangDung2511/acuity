# Contributing

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The hooks run the tools from your venv, so commit with it active.

## Checks

Before each commit (pre-commit): private-data guard, `ruff format`, `ruff check`, `lint-imports`.
Before each push (pre-push): `pytest -q -x`.
CI runs the same checks plus the full test suite on Windows, macOS and Ubuntu.

Run Qt tests headless with `QT_QPA_PLATFORM=offscreen`.

## Rules

- Read [CODING_STANDARDS.md](CODING_STANDARDS.md). Reviews check against it.
- Never commit recordings, databases, logs or debug CSVs. The guard will refuse them.
- Detection thresholds are tuned on one person and one camera. If you change one, put the measurement behind it in the constant's comment, or mark it *unmeasured default, tune from labels* (see CODING_STANDARDS.md rule 1).
