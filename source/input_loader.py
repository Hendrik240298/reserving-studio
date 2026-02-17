from __future__ import annotations

from pathlib import Path

import pandas as pd

from source.claims_repository import ClaimsRepository
from source.config_manager import ConfigManager
from source.example_workflow import build_example_inputs, transform_inputs_granularity
from source.premium_input_repository import PremiumInputRepository


def load_inputs_from_config(
    config: ConfigManager,
    *,
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workflow_input = config.get_workflow_input()
    if not workflow_input:
        return build_example_inputs(config, repo_root=repo_root)

    sql_settings = workflow_input.get("sql", {})
    claims_cfg = workflow_input.get("claims", {})
    premium_cfg = workflow_input.get("premium", {})

    claims_df = _load_claims_dataset(
        config,
        claims_cfg,
        sql_settings=sql_settings,
        repo_root=repo_root,
    )
    premium_df = _load_premium_dataset(
        config,
        premium_cfg,
        sql_settings=sql_settings,
        repo_root=repo_root,
    )

    transformed_claims_df, transformed_premium_df = transform_inputs_granularity(
        claims_df,
        premium_df,
        granularity=config.get_granularity(),
    )
    transformed_claims_df.attrs["values_are_cumulative"] = bool(
        claims_df.attrs.get("values_are_cumulative", False)
    )
    return transformed_claims_df, transformed_premium_df


def _load_claims_dataset(
    config: ConfigManager,
    dataset_cfg: dict,
    *,
    sql_settings: dict,
    repo_root: Path,
) -> pd.DataFrame:
    dataset_name = "claims"
    source = str(dataset_cfg.get("source", "")).strip().lower()
    column_map = _extract_column_map(dataset_cfg, dataset_name=dataset_name)

    if source == "csv":
        path_value = dataset_cfg.get("path")
        if not path_value:
            raise ValueError(
                f"workflow.input.{dataset_name}.path is required for csv source"
            )
        csv_path = _resolve_path(Path(str(path_value)), repo_root=repo_root)
        repository = ClaimsRepository.from_csv(
            config,
            csv_path=csv_path,
            column_map=column_map,
        )
        claims_df = repository.get_claims_df()
        claims_df.attrs["values_are_cumulative"] = _read_values_are_cumulative(
            dataset_cfg,
            dataset_name=dataset_name,
        )
        return claims_df

    if source == "sql":
        query_file = dataset_cfg.get("query_file")
        if not query_file:
            raise ValueError(
                f"workflow.input.{dataset_name}.query_file is required for sql source"
            )
        params = dataset_cfg.get("params")
        if params is None:
            params = []
        if not isinstance(params, list):
            raise ValueError(
                f"workflow.input.{dataset_name}.params must be a list when provided"
            )
        query_path = _resolve_path(Path(str(query_file)), repo_root=repo_root)
        repository = ClaimsRepository.from_sql(
            config,
            query_path=query_path,
            params=params,
            sql_settings=sql_settings,
            column_map=column_map,
        )
        claims_df = repository.get_claims_df()
        claims_df.attrs["values_are_cumulative"] = _read_values_are_cumulative(
            dataset_cfg,
            dataset_name=dataset_name,
        )
        return claims_df

    raise ValueError(
        f"Unsupported source '{source}' for workflow.input.{dataset_name}. "
        "Use 'sql' or 'csv'."
    )


def _load_premium_dataset(
    config: ConfigManager,
    dataset_cfg: dict,
    *,
    sql_settings: dict,
    repo_root: Path,
) -> pd.DataFrame:
    dataset_name = "premium"
    source = str(dataset_cfg.get("source", "")).strip().lower()
    column_map = _extract_column_map(dataset_cfg, dataset_name=dataset_name)

    if source == "csv":
        path_value = dataset_cfg.get("path")
        if not path_value:
            raise ValueError(
                f"workflow.input.{dataset_name}.path is required for csv source"
            )
        csv_path = _resolve_path(Path(str(path_value)), repo_root=repo_root)
        repository = PremiumInputRepository.from_csv(
            config,
            csv_path=csv_path,
            column_map=column_map,
        )
        return repository.get_premium_df()

    if source == "sql":
        query_file = dataset_cfg.get("query_file")
        if not query_file:
            raise ValueError(
                f"workflow.input.{dataset_name}.query_file is required for sql source"
            )
        params = dataset_cfg.get("params")
        if params is None:
            params = []
        if not isinstance(params, list):
            raise ValueError(
                f"workflow.input.{dataset_name}.params must be a list when provided"
            )
        query_path = _resolve_path(Path(str(query_file)), repo_root=repo_root)
        repository = PremiumInputRepository.from_sql(
            config,
            query_path=query_path,
            params=params,
            sql_settings=sql_settings,
            column_map=column_map,
        )
        return repository.get_premium_df()

    raise ValueError(
        f"Unsupported source '{source}' for workflow.input.{dataset_name}. "
        "Use 'sql' or 'csv'."
    )


def _extract_column_map(
    dataset_cfg: dict, *, dataset_name: str
) -> dict[str, str] | None:
    column_map = dataset_cfg.get("column_map")
    if column_map is None:
        return None
    if not isinstance(column_map, dict):
        raise ValueError(f"workflow.input.{dataset_name}.column_map must be a mapping")
    return {str(key): str(value) for key, value in column_map.items()}


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def _read_values_are_cumulative(dataset_cfg: dict, *, dataset_name: str) -> bool:
    raw_value = dataset_cfg.get("values_are_cumulative", False)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(
        f"workflow.input.{dataset_name}.values_are_cumulative must be boolean"
    )
