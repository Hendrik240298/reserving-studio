# API Reference (Local `source/` App)

This is a practical API index for developers and technical actuaries.

It focuses on the most important callable interfaces used in reserving workflows.

## `source/app.py`

### Workflow bootstrap

- `load_config() -> ConfigManager | None`
- `build_sample_triangle() -> Triangle`
- `build_reserving(triangle: Triangle, config: ConfigManager | None = None) -> Reserving`

### Custom data integrations

- `build_workflow_from_dataframes(claims_df: pd.DataFrame, premium_df: pd.DataFrame, *, config: ConfigManager | None = None) -> Reserving`
- `build_workflow_from_collections(claims: ClaimsCollection, premium: PremiumRepository, *, config: ConfigManager | None = None) -> Reserving`

### Interactive session runtime

- `create_interactive_session_controller() -> InteractiveSessionController`
- `launch_dashboard(reserving: Reserving, *, config: ConfigManager | None = None, controller: InteractiveSessionController | None = None, debug: bool = False, port: int = 8050) -> Dashboard`
- `wait_for_finalize(controller: InteractiveSessionController, *, timeout_seconds: float | None = None) -> FinalizePayload`
- `run_interactive_session(reserving: Reserving, *, config: ConfigManager | None = None, controller: InteractiveSessionController | None = None, port: int = 8050, timeout_seconds: float | None = None, debug: bool = False) -> FinalizePayload`

## `source/config_manager.py`

### Construction and config access

- `ConfigManager.from_yaml(file_name) -> ConfigManager`
- `get_segment() -> str`
- `get_granularity() -> str`
- `get_workflow_dataset() -> str`
- `get_workflow_clrd_lob() -> str`
- `get_workflow_quarterly_premium_csv() -> str`
- `get_workflow_input() -> dict`

### Session persistence and sync versioning

- `get_session_path() -> Path`
- `load_session() -> dict`
- `save_session(data: dict) -> None`
- `get_sync_version() -> int`
- `save_session_with_version(data: dict) -> int`

## `source/claims_repository.py`

### Main construction methods

- `ClaimsRepository.from_dataframe(config_manager, dataframe, column_map=None) -> ClaimsRepository`
- `ClaimsRepository.from_csv(config_manager, *, csv_path: Path, column_map=None) -> ClaimsRepository`
- `ClaimsRepository.from_sql(config_manager, *, query_path: Path, sql_settings: dict, params=None, column_map=None) -> ClaimsRepository`

### Output

- `get_claims_df() -> pd.DataFrame`

## `source/premium_repository.py`

### Main construction methods

- `PremiumRepository.from_dataframe(config_manager, dataframe) -> PremiumRepository`
- `PremiumRepository.from_sql(config_manager, *, query_path: Path, sql_settings: dict, params=None, column_map=None) -> PremiumRepository`

### Output and update

- `get_premium() -> pd.DataFrame`
- `set_premium(df: pd.DataFrame) -> None`

## `source/claims_collection.py`

- `ClaimsCollection(df: pd.DataFrame)`
- `iter_claims()`
- `get_claim_amounts() -> pd.Series`
- `to_dataframe() -> pd.DataFrame`

## `source/triangle.py`

- `Triangle(data: pd.DataFrame)`
- `Triangle.from_claims(claims: ClaimsCollection, premium: PremiumRepository) -> Triangle`
- `get_triangle(type: str = "incurred")`
- `calc_ave(type: str = "incurred")`
- `calc_coeff_of_var(type: str = "incurred")`

## `source/reserving.py`

### Parameterization

- `set_development(average: str = "volume", drop: list | None = None, drop_valuation: list | None = None)`
- `set_tail(curve: str = "weibull", attachment_age: int | None = None, projection_period: int | None = None, fit_period: tuple[int, int | None] | None = None)`
- `set_bornhuetter_ferguson(apriori: float | dict[str, float] = 0.6)`

### Execution and outputs

- `reserve(final_ultimate: Literal["chainladder", "bornhuetter_ferguson"] = "chainladder", selected_ultimate_by_uwy: dict[str, str] | None = None)`
- `get_results() -> pd.DataFrame`
- `get_emergence_pattern() -> pd.DataFrame`
- `get_triangle_heatmap_data() -> dict`

## `source/interactive_session.py`

### Contracts

- `ParamsStoreSnapshot`
- `ResultsStoreSnapshot`
- `FinalizePayload`
- `SelectionMethod = Literal["chainladder", "bornhuetter_ferguson"]`

### Controller

- `InteractiveSessionController.publish_latest(...)`
- `InteractiveSessionController.finalize(payload: FinalizePayload)`
- `InteractiveSessionController.fail(message: str)`
- `InteractiveSessionController.cancel()`
- `utc_now() -> datetime`

## `source/services/params_service.py`

Important helpers:

- `build_params_state(...) -> ParamsState`
- `load_session_params_state(...) -> ParamsState`
- `toggle_drop(...) -> list[list[str | int]]`
- `drops_to_tuples(...) -> list[tuple[str, int]] | None`
- `derive_tail_fit_period(selection) -> tuple[int, int | None] | None`
- `build_bf_apriori_rows(raw=None) -> list[dict[str, object]]`
- `bf_rows_to_mapping(rows) -> dict[str, float]`

## `source/services/reserving_service.py`

Primary runtime methods:

- `get_or_build_results_payload(...) -> dict`
- `apply_recalculation(...) -> None`
- `build_results_payload(...) -> dict`

## `source/services/session_sync_service.py`

- `apply_sync_source_payload(...) -> tuple[ResultsPayload, dict | None]`
- `apply_local_source_payload(...) -> tuple[ResultsPayload, dict | None]`

## `source/services/cache_service.py`

- `get(cache, key) -> dict | None`
- `set(cache, key, value) -> None`
- `get_or_build(cache, cache_key, label, builder) -> dict`
- `build_model_cache_key(...) -> str`
- `build_results_cache_key(...) -> str`
- `build_visual_cache_key(...) -> str`

## `source/presentation/plot_builders.py`

Primary public functions (also exported via `source/presentation/__init__.py`):

- `plot_data_triangle_table(...) -> go.Figure`
- `plot_emergence(...) -> go.Figure`
- `plot_reserving_results_table(...) -> go.Figure`
- `build_heatmap_core(...) -> dict`
- `build_heatmap_core_cache_key(...) -> str`
- `plot_triangle_heatmap_clean(...) -> tuple[go.Figure, dict]`
