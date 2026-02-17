# Troubleshooting Index

Use this as a fast lookup: error/symptom -> likely cause -> fix -> technical location.

| Error / Symptom | Likely Cause | Recommended Fix | Technical Location |
|---|---|---|---|
| `Claims dataframe is missing required columns` | Claims input does not map to canonical claims schema | Add missing fields or configure `workflow.input.claims.column_map` | `source/claims_collection.py`, `source/claims_repository.py` |
| `Premium dataframe must contain either ...` | Premium schema not one of accepted variants | Reshape premium to supported schema or map columns | `source/premium_repository.py` |
| `Premium dataframe contains duplicate ('uw_year', 'period') rows` | Duplicate premium keys after mapping/aggregation | Aggregate premium to one row per key before load | `source/premium_repository.py` |
| `Premium dataframe contains invalid 'uw_year' values` | Non-coercible year values | Use 4-digit year or parseable date values | `source/premium_repository.py` |
| `Premium dataframe contains invalid 'period' values` | Period labels not parseable as date/quarter/dev lag | Use date, quarter labels, or valid month-lag integers | `source/premium_repository.py` |
| `Premium dataframe contains non-positive numeric 'period' values` | Numeric dev lags include 0/negative values | Use strictly positive month-lag integers | `source/premium_repository.py` |
| `No dataframe provided to load premium data from` | Missing dataframe in constructor call | Validate integration path and pass premium dataframe | `source/premium_repository.py`, `source/premium_input_repository.py` |
| `No dataframe provided to load claims data from` | Missing claims dataframe in constructor call | Validate integration path and pass claims dataframe | `source/claims_repository.py` |
| `... query file not found` | SQL query path invalid | Fix `query_file` path in config | `source/claims_repository.py`, `source/premium_input_repository.py` |
| `pyodbc is required for SQL input source` | Missing SQL driver dependency in environment | Install `pyodbc`; validate ODBC driver in config | `source/claims_repository.py`, `source/premium_input_repository.py` |
| `average must be 'volume' or 'simple'` | Unsupported development averaging value | Use only `volume` or `simple` | `source/reserving.py` |
| `curve is ... but has to be ...` | Unsupported tail curve value | Use `exponential`, `inverse_power`, or `weibull` | `source/reserving.py` |
| `apriori must be >= 0` (or not numeric) | Invalid BF apriori input | Supply numeric non-negative factors | `source/reserving.py`, `source/services/params_service.py` |
| `Missing BF apriori factors for origins` | BF mapping does not cover all required origins | Provide complete per-UWY mapping | `source/reserving.py` |
| `Results not available. Call reserve() first` | Accessing outputs before model execution | Ensure recalculation/reserve runs before reading outputs | `source/reserving.py` |
| App appears to reuse stale assumptions | Session file reused across different portfolios/scenarios | Use unique `segment` and `session.path` per scenario | `source/config_manager.py`, `config.yml` |
| Other tabs do not update after change | Browser sync message not emitted/received or stale version ignored | Verify `sync_version` increments and same segment is used | `source/services/session_sync_service.py`, `assets/tab_sync.js`, `docs/cross-tab-sync.md` |
| Session YAML parse errors on startup | Corrupted YAML file | App auto-quarantines file; recreate assumptions | `source/config_manager.py` |
| Interactive script waits forever | Finalize action not triggered in UI | Click Finalize or set timeout for `run_interactive_session` | `source/app.py`, `source/interactive_session.py` |
| `Interactive session did not finalize before timeout` | Timeout too short for review cycle | Increase `timeout_seconds` or set `None` | `source/app.py` |

## Quick diagnostic sequence

1. Confirm config path: `RESERVING_CONFIG` or `config.yml`.
2. Validate claims/premium schemas and date coercion with a pre-check script.
3. Confirm session isolation (`segment`, `session.path`).
4. Re-run app and inspect exact raised message.
5. Jump to referenced module in this table.
