# Testing Guide

## Scope

- `tests/e2e/` contains deterministic browser E2E tests for the Dash reserving UI.
- The tests run against a temporary config/session setup and do not modify your normal `sessions/` files.

## Prerequisites

```bash
uv pip install -r requirements.txt
uv run python -m playwright install chromium
```

## Run tests

- Full dashboard E2E suite:

```bash
uv run pytest tests/e2e -m e2e -q
```

- Single test:

```bash
uv run pytest tests/e2e/test_dashboard_e2e.py::test_drop_updates_emergence_and_results -q
```

## Current E2E coverage

- Drop selection in chainladder heatmap updates both emergence and results plots.
- Editing Bornhuetter-Ferguson apriori values updates results output.

## Failure artifacts

- On E2E test failure, Playwright artifacts are written to `tests/artifacts/e2e/`:
  - `<test_name>.png` full-page screenshot
  - `<test_name>.zip` Playwright trace

Open traces with:

```bash
uv run python -m playwright show-trace tests/artifacts/e2e/<test_name>.zip
```

## AI workflow recommendation

- After any code change touching `source/app.py`, `source/dashboard.py`, `source/reserving.py`, or `source/triangle.py`, run at least the impacted E2E test(s).
- If the impacted set is unclear, run the full `tests/e2e` suite.
- If a test fails, include artifact paths from `tests/artifacts/e2e/` in the report and summarize the failing step.
