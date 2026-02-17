# Technical Architecture Guide

This document is the developer-oriented reference for the local `source/` application.

It explains:

- module and class responsibilities
- important functions and entrypoints
- end-to-end data handling
- session/sync/caching behavior

## 1) Runtime modes and entrypoints

Primary entrypoint: `source/app.py`.

Supported execution modes:

1. **Default local app mode**
   - command: `uv run python -m source.app`
   - flow: sample data -> reserving objects -> Dash UI
2. **Scripted interactive mode**
   - build `Reserving` from custom dataframes
   - run UI in background thread via `run_interactive_session(...)`
   - consume finalized payload in script
3. **Config-driven input mode**
   - load claims/premium from CSV or SQL through `source/input_loader.py`
   - typically used by `examples/run_*_interactive.py`

## 2) Module map (local app)

- `source/app.py`
  - orchestration helpers and executable `main()`
  - builds workflow and launches dashboard
- `source/config_manager.py`
  - YAML config/session interface (single persistence boundary)
  - session versioning and atomic writes
- `source/input_loader.py`
  - config-driven routing to claims/premium repositories
- `source/claims_repository.py`
  - claims ingestion from dataframe/CSV/SQL and normalization/validation
- `source/premium_input_repository.py`
  - raw premium ingestion from dataframe/CSV/SQL
- `source/premium_repository.py`
  - premium canonicalization for reserving triangle consumption
- `source/claims_collection.py`
  - claims domain wrapper with lightweight validation and accessors
- `source/triangle.py`
  - creation of chainladder `Triangle` from claims + premium
- `source/reserving.py`
  - reserving engine wrapper (development, tail, BF, result assembly)
- `source/dashboard.py`
  - Dash UI composition and callback wiring
- `source/interactive_session.py`
  - typed snapshot/finalize payload contracts for script-driven workflows
- `source/presentation/plot_builders.py`
  - figure building and heatmap-core rendering helpers
- `source/services/`
  - callback logic split into focused services:
    - `ParamsService`
    - `ReservingService`
    - `SessionSyncService`
    - `CacheService`

## 3) Core classes and why they matter

### `ConfigManager` (`source/config_manager.py`)

Most important because it is the only YAML/session boundary.

Key methods:

- `from_yaml(...)`: load base config
- `load_session()`: load per-segment session settings
- `save_session_with_version(...)`: persist and increment `sync_version`
- `get_workflow_input()`: returns custom input config branch
- `get_session_path()`: active session file location

Technical notes:

- session writes are atomic (`.tmp` + replace)
- corrupted session YAML is quarantined to `.corrupt.<timestamp>`
- sync version is monotonic per session file

### `ClaimsRepository` (`source/claims_repository.py`)

Main claims ingestion and sanitation layer.

Key methods:

- `from_dataframe(...)`, `from_csv(...)`, `from_sql(...)`
- `get_claims_df()`
- internal cleaning sequence in constructor:
  - required column validation
  - missing ID repair (`fix_movements_wo_claim_id`)
  - date coercion (`update_date_format`)
  - net-zero claim filter
  - claim ID split for multiple UWY
  - loss-year correction and period-before-UWY handling

### `PremiumRepository` (`source/premium_repository.py`)

Main premium canonicalization layer for downstream triangle usage.

Accepted source schemas:

1. `UnderwritingYear`, `Premium`
2. `origin`, `development`, `Premium_selected`
3. `uw_year`, `period`, `Premium_selected`

Key methods:

- `from_dataframe(...)`, `from_sql(...)`
- `get_premium()`
- normalization path decides schema-specific converter and then finalizes with:
  - `uw_year`, `period`, `Premium_selected`
  - replicated columns `GWP`, `EPI`, `GWP_Forecast`

### `Triangle` (`source/triangle.py`)

Domain adapter from claims + premium dataframes to chainladder `Triangle`.

Key method:

- `Triangle.from_claims(claims, premium)`

Behavior:

- appends premium rows to claims-style rows
- builds `cl.Triangle(..., cumulative=False)` and immediately stores cumulative form (`incr_to_cum()`)
- exposes `get_triangle()` for reserving engine

### `Reserving` (`source/reserving.py`)

Primary actuarial engine wrapper.

Key configuration methods:

- `set_development(average, drop, drop_valuation)`
- `set_tail(curve, attachment_age, projection_period, fit_period)`
- `set_bornhuetter_ferguson(apriori)`

Key execution method:

- `reserve(final_ultimate, selected_ultimate_by_uwy)`

Key outputs:

- `get_results()` -> results dataframe
- `get_emergence_pattern()` -> actual vs expected emergence
- `get_triangle_heatmap_data()` -> link ratios + ldf/tail + base triangle data

### `Dashboard` (`source/dashboard.py`)

UI composition + callback coordination layer.

Notable internal responsibilities:

- load defaults from session at startup
- maintain figure/data caches
- delegate callback work to services (`ParamsService`, `ReservingService`, `SessionSyncService`)
- publish finalize payload when running in interactive script mode

### Services (`source/services/*.py`)

- `ParamsService`
  - normalizes drop/tail/BF/method UI state
  - converts UI structures to engine-ready types
- `ReservingService`
  - applies recalculation and builds payloads
  - separates model cache key and display cache key concerns
- `SessionSyncService`
  - persists local updates and sets sync metadata
  - applies incoming sync payload behavior
- `CacheService`
  - LRU-like dictionary cache with deep-copy semantics
  - deterministic cache key construction

## 4) Important functions in `source/app.py`

- `load_config()`
  - resolves `RESERVING_CONFIG` env var fallback to `config.yml`
- `build_workflow_from_dataframes(claims_df, premium_df, config=None)`
  - preferred API for custom ETL integrations
- `build_workflow_from_collections(claims, premium, config=None)`
  - lower-level domain-object variant
- `build_sample_triangle()`
  - default sample bootstrap path
- `build_reserving(triangle, config=None)`
  - applies session/default parameters and executes first reserve
- `run_interactive_session(...)`
  - launches UI on thread and blocks until finalize/fail/cancel
- `wait_for_finalize(...)`
  - synchronization boundary for script-driven workflows
- `launch_dashboard(...)`
  - non-blocking utility for direct dashboard launch

## 5) End-to-end data handling

### A) Input acquisition

1. `ConfigManager` loads YAML.
2. `input_loader.load_inputs_from_config(...)` chooses:
   - example data path (`source/example_workflow.py`) when no `workflow.input`
   - or claims/premium repositories for CSV/SQL.

### B) Input normalization

Claims path (`ClaimsRepository`):

- resolves column mapping
- validates required columns
- coerces `uw_year`, `period`
- enforces claim-level consistency and removes problematic records

Premium path (`PremiumInputRepository` -> `PremiumRepository`):

- resolves column mapping
- accepts one of three schemas
- coerces types/dates, computes incremental premium if needed
- validates uniqueness on (`uw_year`, `period`)

### C) Granularity transformation

`source/example_workflow.py` provides `transform_inputs_granularity(...)` for quarterly/yearly aggregation by step size.

This transforms both claims and premium to aligned step-based valuation dates.

### D) Triangle construction

`Triangle.from_claims(...)` merges normalized claims and premium into one reserving-ready dataset and creates a chainladder triangle.

### E) Model execution

`ReservingService.apply_recalculation(...)` applies UI/session params to `Reserving`:

1. `set_development(...)`
2. `set_tail(...)`
3. `set_bornhuetter_ferguson(...)`
4. `reserve(...)`

Results and emergence data are extracted after each recalculation.

### F) Presentation payload

`ReservingService.build_results_payload(...)` composes:

- triangle figure payload
- emergence figure payload
- rendered results table rows
- display metadata for drops/tail/method selections

`source/presentation/plot_builders.py` provides core figure generation and heatmap enrichment (`customdata`, dropped-cell overlays, tail/fit highlights).

## 6) Session persistence and cross-tab sync

- Local UI changes persist through `ConfigManager.save_session_with_version(...)`.
- `sync_version` is included in payloads to prevent stale overwrite.
- browser tab event bridge lives in `assets/tab_sync.js`.
- technical details: `docs/cross-tab-sync.md`.

## 7) Caching model

Three cache key layers are used:

1. **Model cache key**: assumptions impacting recalculation output.
2. **Results cache key**: model key + per-UWY selected method mapping.
3. **Visual cache key**: assumptions impacting figure shape/styling overlays.

Caches are bounded in-memory dicts managed by `CacheService`.

## 8) Extension points (recommended)

- Add new input source types in `source/input_loader.py` and repository constructors.
- Add new UI parameters via:
  - `ParamsService` normalization
  - `ReservingService` cache-key and recalculation wiring
  - `ConfigManager` session persistence payload
- Add new result visualizations in `source/presentation/plot_builders.py` and wire in `Dashboard`.

## 9) Typical failure surfaces

- schema mismatches (claims/premium canonical fields)
- period coercion errors (non-parseable date/development labels)
- duplicate premium keys (`uw_year`, `period`)
- missing BF apriori mapping for selected origins
- inconsistent session reuse across segments
