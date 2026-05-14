# Results Tab Change Impact Map

Use this page before implementing any new option in the Results tab (new selectable method, new derived metric, new result column behavior).

It gives you:

- a compact overview of the end-to-end flow
- a reusable impact-checklist template so you do not miss a layer

## 1) Overall overview (one-screen mental model)

The Results tab is the end of a state pipeline with two core stores.

`UI inputs -> params-store -> ReservingService recalculation -> results-store -> results table/figures -> optional finalize payload`

### Main runtime components

| Layer | Responsibility | Main files |
|---|---|---|
| Orchestration | Build reserving workflow, launch Dash, optional interactive finalize | `source/app.py` |
| UI + callbacks | Reduce UI events into params, trigger recalc, hydrate table/plots, finalize action | `source/dashboard.py` |
| Param normalization | Keep params-store shape valid and deterministic | `source/services/params_service.py` |
| Recalculation + payload | Apply assumptions to model, build figures + table rows, use caches | `source/services/reserving_service.py` |
| Session + sync | Persist session YAML and version cross-tab updates | `source/config_manager.py`, `source/services/session_sync_service.py` |
| Actuarial engine | Chainladder/BF execution, result DataFrame construction | `source/reserving.py` |
| Interactive contract | Typed snapshots + finalize payload returned to Python script | `source/interactive_session.py` |

### Data objects to keep in sync

| Object | Lives in | Meaning |
|---|---|---|
| `params-store` | Dash store (`source/dashboard.py`) | User intent (drops, average, tail, BF apriori, selected method by UWY) |
| `results-store` | Dash store (`source/dashboard.py`) | Computed outputs (table rows, figures, metadata, sync version) |
| `df_results` | `Reserving` (`source/reserving.py`) | Canonical numeric output used to build table rows |
| `FinalizePayload` | `source/interactive_session.py` | Script-facing payload after "Finalize & Continue" |

## 2) Change impact map template (copy before each change)

Copy this table into your task notes and fill it out before coding.

| Step | Check | Typical files | Status | Notes |
|---|---|---|---|---|
| 1 | Define behavior precisely (calculation rule, selection rule, fallback rule). | design note / issue | TODO | |
| 2 | Confirm UI interaction model (dropdown, clickable cell, toggle, read-only column). | `source/dashboard.py` | TODO | |
| 3 | Extend allowed option values in all validators/normalizers. | `source/services/params_service.py`, `source/app.py`, `source/interactive_session.py`, `source/reserving.py` | TODO | |
| 4 | Add/update params-store keys if new state is needed. | `source/dashboard.py`, `source/services/params_service.py` | TODO | |
| 5 | Wire callback reducer updates (`_reduce_params`) for new trigger(s). | `source/dashboard.py` | TODO | |
| 6 | Update engine output columns or selection logic. | `source/reserving.py` | TODO | |
| 7 | Ensure recalculation service includes new parameter in model/results cache keys where required. | `source/services/reserving_service.py`, `source/services/cache_service.py` | TODO | |
| 8 | Update payload-building for Results tab rows/fields. | `source/services/reserving_service.py`, `source/dashboard.py` | TODO | |
| 9 | Update table columns, selection/highlight logic, and view modes (absolute/relative). | `source/dashboard.py` | TODO | |
| 10 | Update session persistence (save/load) and sync propagation payload. | `source/services/session_sync_service.py`, `source/config_manager.py` | TODO | |
| 11 | Update finalize snapshots/contracts if new fields must be returned to scripts. | `source/interactive_session.py`, `source/dashboard.py` | TODO | |
| 12 | Add/adjust tests (unit first, E2E if UI behavior changes). | `tests/unit`, `tests/e2e` | TODO | |
| 13 | Add docs updates (workflow map or architecture guide if scope changed). | `docs/*.md`, `README.md` | TODO | |

## 3) Fast pre-implementation questions

Use these questions to avoid rework:

1. Is this a new method choice, a new metric display, or both?
2. Does it change only rendering, or also actuarial selection logic in `Reserving.reserve(...)`?
3. Does it require persistence across restarts and tab sync?
4. Does it affect cache key identity (model outputs or only display projection)?
5. Must finalized script payload include this new choice/value?

## 4) Minimum verification checklist

- Change in Results tab updates immediately without callback errors.
- Selection survives refresh/restart (session YAML roundtrip).
- Cross-tab behavior remains monotonic via `sync_version`.
- Finalize returns expected data in `FinalizePayload.results_df` and snapshot fields.
- No regression in existing CL/BF selection behavior.
