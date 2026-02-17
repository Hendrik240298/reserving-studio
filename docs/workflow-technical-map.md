# Workflow Technical Map

Use this page when you know the actuarial workflow step you care about and want to jump to the technical implementation quickly.

## 1) Load configuration and session

- **What actuary sees:** selected segment, saved assumptions restored on app start.
- **Technical entrypoint:** `source/app.py` (`load_config`, `build_reserving`).
- **Core implementation:** `source/config_manager.py` (`from_yaml`, `load_session`, `save_session_with_version`).
- **Deep-dive docs:** `docs/technical-architecture.md`, `docs/data-handling-deep-dive.md`.

## 2) Read claims and premium inputs

- **What actuary sees:** app starts from sample, CSV, or SQL-backed data.
- **Technical entrypoint:** `source/input_loader.py` (`load_inputs_from_config`).
- **Claims pipeline:** `source/claims_repository.py`.
- **Premium pipeline:** `source/premium_input_repository.py`, `source/premium_repository.py`.
- **Where mapping is handled:** `workflow.input.<claims|premium>.column_map` in config.

## 3) Normalize and validate data

- **What actuary sees:** data accepted or explicit validation errors.
- **Technical logic:**
  - claims coercion/cleanup in `source/claims_repository.py`
  - premium schema normalization in `source/premium_repository.py`
  - granularity transform in `source/example_workflow.py` (`transform_inputs_granularity`).
- **Deep-dive docs:** `docs/data-handling-deep-dive.md`.

## 4) Build reserving triangle

- **What actuary sees:** Data tab triangles and downstream model inputs.
- **Technical core:** `source/triangle.py` (`Triangle.from_claims`, `_create_triangle`, `get_triangle`).
- **Important behavior:** claims rows and premium rows are merged into one chainladder triangle structure.

## 5) Set Chainladder assumptions (drops, average, tail)

- **What actuary sees:** Chainladder tab controls and live recalculation.
- **Technical UI-state normalization:** `source/services/params_service.py`.
- **Technical model application:** `source/services/reserving_service.py` (`apply_recalculation`).
- **Engine methods:** `source/reserving.py` (`set_development`, `set_tail`).

## 6) Set Bornhuetter-Ferguson apriori

- **What actuary sees:** BF tab factor edits by UWY.
- **Technical state handling:** `source/services/params_service.py` (`build_bf_apriori_rows`, `bf_rows_to_mapping`).
- **Engine methods:** `source/reserving.py` (`set_bornhuetter_ferguson`, `_apply_bf_apriori_to_exposure`).

## 7) Select final method by UWY

- **What actuary sees:** per-UWY selection (`chainladder` vs `bornhuetter_ferguson`) in Results tab.
- **Technical state handling:** `source/services/params_service.py` (`build_selected_ultimate_by_uwy`).
- **Technical payload projection:** `source/services/reserving_service.py` (`_apply_selected_ultimate_to_results`).
- **Engine-level selection:** `source/reserving.py` (`reserve`).

## 8) Render results and visuals

- **What actuary sees:** heatmap, emergence, result table.
- **Dashboard wiring:** `source/dashboard.py`.
- **Figure builders:** `source/presentation/plot_builders.py`.
- **Heatmap internals:** `build_heatmap_core`, `plot_triangle_heatmap_clean`.

## 9) Persist assumptions and sync tabs

- **What actuary sees:** assumptions remain after refresh/restart; multiple tabs stay aligned.
- **Session persistence:** `source/config_manager.py`.
- **Sync orchestration:** `source/services/session_sync_service.py`.
- **Browser bridge:** `assets/tab_sync.js`.
- **Deep-dive docs:** `docs/cross-tab-sync.md`.

## 10) Finalize and return results to script

- **What actuary sees:** "Finalize & Continue" returns control to script.
- **Contract types:** `source/interactive_session.py` (`FinalizePayload`, snapshots).
- **Runtime flow:** `source/app.py` (`run_interactive_session`, `wait_for_finalize`).

## 11) Fast navigation by intent

- **I need data issues explained:** `docs/data-handling-deep-dive.md`.
- **I need module/class architecture:** `docs/technical-architecture.md`.
- **I need user onboarding on own data:** `docs/start-with-your-data.md`.
- **I need sync behavior details:** `docs/cross-tab-sync.md`.
