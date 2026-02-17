# Data Handling Deep Dive

This document explains the internal data pipeline from raw inputs to reserving outputs.

## 1) Canonical internal contracts

### Claims canonical contract

- `id`
- `uw_year`
- `period`
- `paid`
- `outstanding`

All claims paths are normalized to this shape before triangle construction.

### Premium canonical contract

- `uw_year`
- `period`
- `Premium_selected`

Then enriched to include:

- `GWP`
- `EPI`
- `GWP_Forecast`

by mirroring `Premium_selected`.

## 2) Ingestion flow by component

### Claims

`source/claims_repository.py`:

1. load input from dataframe/CSV/SQL
2. apply optional `column_map`
3. normalize blanks to null-equivalent
4. coerce numeric movement columns
5. enforce required columns
6. normalize IDs and dates
7. apply corrective business rules

Corrective rules include:

- fill missing claim IDs (structured fallback)
- split IDs when one claim spans multiple UWY values
- remove net-zero claims
- correct period values earlier than UWY
- adjust implausible `loss_year` (when present)

### Premium

`source/premium_input_repository.py` + `source/premium_repository.py`:

1. load input from dataframe/CSV/SQL
2. apply optional `column_map`
3. detect schema variant
4. coerce years/periods/numerics
5. convert cumulative premium triangles to incremental where needed
6. validate uniqueness of (`uw_year`, `period`)
7. finalize reserving schema

## 3) Temporal coercion logic

### Claims temporal parsing (high level)

- `uw_year`: accepts year-like values and converts to start-of-year timestamp
- `period`: accepts datetimes, quarter labels (`YYYYQn`), `YYYYMM`, parseable date strings

Invalid coercion raises explicit `ValueError`.

### Premium temporal parsing (high level)

- `uw_year`: year-like values -> start-of-year timestamp
- `period`:
  - parseable datetime
  - quarter label
  - numeric month lag from UWY when schema is development-based

Non-positive numeric lags are rejected.

## 4) Aggregation and granularity

`source/example_workflow.py` exposes `transform_inputs_granularity(...)`:

- `quarterly` => 3-month step
- `yearly` => 12-month step

Both claims and premium are aggregated to consistent step indices and converted back to period timestamps.

## 5) Triangle assembly behavior

`source/triangle.py` builds one `cl.Triangle` containing:

- `incurred`
- `outstanding`
- `paid`
- `Premium_selected`

Implementation shape:

- claims rows carry claim values and zero premium
- premium rows carry premium and zero claim values
- rows are concatenated and passed into chainladder constructor

The triangle is stored as cumulative internally.

## 6) Reserving output assembly

`source/reserving.py` produces a combined results dataframe containing:

- `cl_ultimate`, `cl_loss_ratio`
- `bf_ultimate`, `bf_loss_ratio`
- `incurred`, `Premium`
- selected `ultimate` based on final/mapped method
- `selected_method`

This table is the primary numeric output consumed by UI tables and script finalization.

## 7) UI payload data structures

`source/interactive_session.py` defines typed snapshots:

- `ParamsStoreSnapshot`
- `ResultsStoreSnapshot`
- `FinalizePayload`

`FinalizePayload.results_df` is the machine-readable numeric output for downstream ETL/reporting.

## 8) Data handling pitfalls to watch during extension

- introducing new input schema without adding canonical mapping
- relying on ambiguous period strings
- silently allowing duplicated premium keys
- forgetting to update cache-key construction when adding new model parameters
- bypassing `ConfigManager` and causing session drift across tabs/processes
