# Actuary Workflow Guide

This is the practical end-to-end workflow for using Reserving Studio as an actuarial analysis tool.

For first-time setup on your own data (including roadblocks and fixes), see `docs/start-with-your-data.md`.

## Workflow overview

1. Load config and active session.
2. Load claims and premium inputs.
3. Build reserving triangle.
4. Run Chainladder + Bornhuetter-Ferguson.
5. Interact in Dash UI and iterate assumptions.
6. Finalize and pass outputs back to script (optional).

## Input modes

### A) Default local run (sample-driven)

- Run `uv run python -m source.app`.
- Uses sample quarterly claims and local premium CSV.
- Best for smoke testing, demos, and learning the controls.

### B) Scripted workflow (recommended for repeatable analyses)

- Build claims/premium dataframes in your script.
- Start interactive session and finalize from UI.
- Use returned payload for reporting/ETL.

Entrypoints are in `source/app.py`:

- `build_workflow_from_dataframes(...)`
- `run_interactive_session(...)`

#### Minimal script template for your own reserving run

```python
from source.app import (
    build_workflow_from_dataframes,
    create_interactive_session_controller,
    run_interactive_session,
)
from source.config_manager import ConfigManager
import pandas as pd


def main() -> None:
    config = ConfigManager.from_yaml("config.yml")

    claims_df = pd.read_csv("your_claims.csv")
    premium_df = pd.read_csv("your_premium.csv")

    reserving = build_workflow_from_dataframes(
        claims_df=claims_df,
        premium_df=premium_df,
        config=config,
    )
    controller = create_interactive_session_controller()

    finalized = run_interactive_session(
        reserving,
        config=config,
        controller=controller,
        port=8050,
        timeout_seconds=None,
    )

    print(finalized.results_df.head())


if __name__ == "__main__":
    main()
```

Use `examples/run_quarterly_interactive.py`, `examples/run_clrd_interactive.py`, and `examples/run_sql_interactive.py` as production-ready patterns.

#### What to consider when setting up your own script

- **Canonical columns:** claims must resolve to `id`, `uw_year`, `period`, `paid`, `outstanding`; premium must resolve to a supported schema from `docs/config-practical-reference.md`.
- **Dates and periods:** keep `uw_year` and `period` parseable; mixed or ambiguous formats can cause coercion/validation failures.
- **Granularity intent:** choose `quarterly` vs `yearly` in config up front, because it changes aggregation and reserve dynamics.
- **Segment/session isolation:** use a unique `segment` (and ideally `session.path`) per portfolio or scenario to avoid carrying assumptions between runs.
- **Finalize contract:** downstream reporting should consume `finalized.results_df` plus method metadata from `finalized.params_store` for auditability.
- **Reproducibility:** persist the config file and session YAML used in the run together with exported results.

### C) Config-driven SQL/CSV input

- Define sources under `workflow.input` in YAML.
- Claims and premium can each come from `sql` or `csv`.
- Use column mapping if source names differ from canonical fields.

## Practical reserving loop

1. Start with default assumptions.
2. Drop questionable link ratios in Chainladder tab.
3. Adjust tail curve and fit/attachment assumptions.
4. Set BF apriori factors by UWY.
5. In Results, choose selected method by UWY (`chainladder` or `bornhuetter_ferguson`).
6. Validate IBNR and loss ratio behavior against expectations.
7. Finalize and persist.

## What is persisted

Session YAML stores operational assumptions, including:

- `average`
- `tail_curve`
- `drops`
- `tail_attachment_age`
- `tail_fit_period`
- `bf_apriori_by_uwy`
- `selected_ultimate_by_uwy`

Files are segment-scoped in `sessions/` (for example `sessions/quarterly.yml`).

## Finalized outputs (script mode)

On finalization, you receive:

- `params_store`: normalized run parameters from the UI.
- `results_store`: rendered and tabular result payload.
- `results_df`: numeric dataframe for downstream calculations and reporting.

This keeps the actuarial judgment in UI while preserving machine-usable outputs.
