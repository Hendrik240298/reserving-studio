# Testing Guide

## Scope

- `tests/e2e/` contains deterministic browser E2E tests for the Dash reserving UI.
- `tests/unit/` contains fast unit tests for app input workflow, domain adapters, interactive session payloads, and plotting/presentation helpers.
- The tests run against a temporary config/session setup and do not modify your normal `sessions/` files.

## Prerequisites

```bash
uv pip install -r requirements.txt
uv run python -m playwright install chromium
```

## Run tests

- Full unit suite:

```bash
uv run python -m pytest tests/unit -q
```

- Single unit test:

```bash
uv run python -m pytest tests/unit/test_plot_builders.py::test_plot_triangle_heatmap_clean_returns_figure_and_payload -q
```

- Full dashboard E2E suite:

```bash
uv run python -m pytest tests/e2e -m e2e -q
```

- Single test:

```bash
uv run python -m pytest tests/e2e/test_dashboard_e2e.py::test_drop_updates_emergence_and_results -q
```

## Current E2E coverage

- Drop selection in chainladder heatmap updates both emergence and results plots.
- Editing Bornhuetter-Ferguson apriori values updates results output.
- Results method selection styling/behavior remains stable across table state changes.

## Current unit coverage

- `ClaimsCollection` validation behavior, including explicit `id` requirement for claim-level methods.
- `PremiumRepository` normalization for UWY-level premium and cumulative premium triangle input.
- App input pipeline wiring (`build_sample_triangle`) through claims/premium repositories.
- Interactive session payload contracts (`ParamsStoreSnapshot`, `ResultsStoreSnapshot`, controller finalize flow).
- Heatmap core cache key determinism and change detection.
- Numeric formatting and customdata shape generation.
- Heatmap core payload and dropped-cell masking behavior.
- Figure contract checks for data triangle table, emergence chart, results table, and clean heatmap rendering.

## Scripted interactive workflow check

- To validate the script-driven workflow (manual read -> GUI -> finalize -> script continuation):

```bash
uv run python examples/run_quarterly_interactive.py
```

- In the app, open Results and click **Finalize & Continue**.
- The script should resume and print finalized segment, selected methods by UWY, and top rows of numeric results.

## Failure artifacts

- On E2E test failure, Playwright artifacts are written to `tests/artifacts/e2e/`:
  - `<test_name>.png` full-page screenshot
  - `<test_name>.zip` Playwright trace

Open traces with:

```bash
uv run python -m playwright show-trace tests/artifacts/e2e/<test_name>.zip
```

## AI workflow recommendation

- After any code change touching `source/app.py`, `source/dashboard.py`, `source/reserving.py`, `source/triangle.py`, or `source/presentation/*.py`, run the relevant unit tests first (`tests/unit`).
- After any code change touching interactive workflow files (`source/app.py`, `source/dashboard.py`, `source/reserving.py`, `source/triangle.py`), run at least the impacted E2E test(s).
- If the impacted set is unclear, run the full `tests/e2e` suite.
- If a test fails, include artifact paths from `tests/artifacts/e2e/` in the report and summarize the failing step.
