# Start Reserving With Your Own Data

This is the end-to-end guide for a first-time user who wants to run Reserving Studio on real portfolio data.

It covers:

- how to prepare claims and premium inputs
- how to configure CSV or SQL reads
- how to run the app and iterate assumptions
- what common roadblocks look like and how to fix them

## 1) Prerequisites

From repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run app (default sample) once to confirm environment:

```bash
uv run python -m source.app
```

Open `http://127.0.0.1:8050`.

## 2) Decide your input path

Use one of these:

1. **CSV config path** (best first real-data run)
2. **SQL config path** (production-like data extraction)
3. **Python script path** (you already have dataframes in memory)

If this is your first run, start with CSV config path.

## 3) Data preparation (most important step)

### Claims data: required canonical columns

Your claims dataset must resolve to these columns:

- `id`: claim identifier (string or numeric)
- `uw_year`: underwriting year (or value coercible to year)
- `period`: valuation period/date
- `paid`: paid movement amount
- `outstanding`: outstanding movement amount

If your source uses different names, map with `column_map` in config.

### Premium data: supported canonical schemas

Premium input can be supplied in one of three schemas:

1. `UnderwritingYear`, `Premium`
2. `origin`, `development`, `Premium_selected`
3. `uw_year`, `period`, `Premium_selected`

For new projects, schema (3) is usually easiest and clearest.

Premium shape recommendations:

- If possible, provide premium as one row per UWY (`Premium_selected` at the selected/latest valuation). This is the safest with the current pipeline.
- If you provide premium by multiple development periods, keep the semantics consistent (incremental vs cumulative) and verify latest diagonal values in the Results tab.

### Date and period formatting rules

For best reliability:

- `uw_year`: use a 4-digit year or a date in that year (`2020`, `2020-01-01`)
- `period` (claims): use parseable dates (for example quarter-end dates)
- `period` (premium): use parseable dates, quarter labels, or numeric development lags (months)

Recommended pattern for quarterly reserving:

- `uw_year` at year start (`YYYY-01-01`)
- `period` at quarter end (`YYYY-03-31`, `YYYY-06-30`, `YYYY-09-30`, `YYYY-12-31`)

### Data quality checks before first run

At minimum, check:

- no missing required columns
- numeric fields are numeric (`paid`, `outstanding`, `Premium_selected`)
- no unparseable `uw_year` / `period`
- no duplicate premium rows for same `uw_year` + `period`

Minimal pre-check script:

```python
import pandas as pd

claims = pd.read_csv("data/claims.csv")
premium = pd.read_csv("data/premium.csv")

assert {"id", "uw_year", "period", "paid", "outstanding"}.issubset(claims.columns)
assert {"uw_year", "period", "Premium_selected"}.issubset(premium.columns)

claims["paid"] = pd.to_numeric(claims["paid"], errors="raise")
claims["outstanding"] = pd.to_numeric(claims["outstanding"], errors="raise")
premium["Premium_selected"] = pd.to_numeric(premium["Premium_selected"], errors="raise")

pd.to_datetime(claims["uw_year"], errors="raise")
pd.to_datetime(claims["period"], errors="raise")
pd.to_datetime(premium["uw_year"], errors="raise")
pd.to_datetime(premium["period"], errors="raise")
```

## 4) Configure a first real-data run (CSV path)

Create a config file (for example `examples/config_my_portfolio.yml`):

```yaml
paths:
  results: results/
  plots: plots/
  data: data/
  sessions: sessions/

first date: 2018
last date: "December 2024"
segment: my_portfolio_q
granularity: quarterly

workflow:
  input:
    claims:
      source: csv
      path: data/claims.csv
      # only if needed
      # column_map:
      #   claim_id: id
      #   underwriting_year: uw_year
      #   valuation_date: period
      #   paid_movement: paid
      #   reserve_movement: outstanding
    premium:
      source: csv
      path: data/premium.csv
      # only if needed
      # column_map:
      #   underwriting_year: uw_year
      #   valuation_date: period
      #   earned_premium: Premium_selected

session:
  path: sessions/my_portfolio_q.yml
```

Run app with this config:

```bash
RESERVING_CONFIG=examples/config_my_portfolio.yml uv run python -m source.app
```

## 5) Configure SQL input (template path)

Use `examples/config_sql_template.yml` as baseline.

Important fields:

- `workflow.input.sql.driver`
- `workflow.input.sql.server`
- `workflow.input.sql.database`
- `workflow.input.claims.query_file`
- `workflow.input.premium.query_file`
- optional `column_map` for claims and premium

Run:

```bash
uv run python examples/run_sql_interactive.py
```

## 6) Python-script path (best for custom ETL)

If you already load/clean data in Python, use this pattern:

```python
from source.app import (
    build_workflow_from_dataframes,
    create_interactive_session_controller,
    run_interactive_session,
)
from source.config_manager import ConfigManager

config = ConfigManager.from_yaml("examples/config_my_portfolio.yml")

# claims_df and premium_df prepared by your own ETL
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

results_df = finalized.results_df
params = finalized.params_store
```

## 7) First-session workflow in the UI

Recommended sequence:

1. **Data tab**: verify triangle shape and values look reasonable.
2. **Chainladder tab**: apply drops where link ratios are clearly non-representative.
3. **Chainladder tab**: review average and tail settings.
4. **Bornhuetter-Ferguson tab**: set apriori by UWY if needed.
5. **Results tab**: compare CL vs BF and set selected method by UWY.
6. **Finalize & Continue** (when script-driven) to pass output to downstream code.

## 8) Roadblocks and fixes

### "Claims dataframe is missing required columns"

Cause: claims input does not resolve to canonical claims fields.

Fix:

- add missing columns in source data, or
- map source names with `workflow.input.claims.column_map`

### "Premium dataframe must contain either ..."

Cause: premium input schema is unsupported.

Fix:

- reshape data to one of supported premium schemas, or
- map your columns to canonical names before load

### "invalid 'period' values"

Cause: non-parseable period values.

Fix:

- use explicit date strings (`YYYY-MM-DD`) or valid quarter labels
- avoid mixed free-text period values

### Duplicate premium rows by (`uw_year`, `period`)

Cause: same key appears multiple times after mapping.

Fix:

- aggregate premium to one row per key before loading
- ensure ETL grouping keys match reserving granularity

### SQL connection/import errors

Cause: ODBC driver mismatch or missing `pyodbc`.

Fix:

- verify ODBC driver installed and matches config
- install `pyodbc` in active environment
- test SQL query directly before app run

### Session assumptions bleeding between runs

Cause: sharing same segment/session path across different portfolios.

Fix:

- assign unique `segment` and `session.path` per portfolio/scenario

## 9) Reproducibility checklist (recommended)

For each finalized reserving run, store together:

- input extract reference (CSV snapshot or SQL query + params)
- config YAML used for the run
- session YAML used/generated
- finalized `results_df` export
- selected method metadata from `params_store`

This gives a clean audit trail from assumptions to reported figures.

## 10) Where to go next

- `docs/actuary-quickstart.md` for fast local startup
- `docs/actuary-workflow.md` for workflow context
- `docs/config-practical-reference.md` for YAML options
- `docs/practical-notes.md` for operational notes
