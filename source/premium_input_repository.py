from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from source.claims_repository import _build_sql_connection
from source.config_manager import ConfigManager


DEFAULT_PREMIUM_COLUMN_MAP = {
    "uw_year": "uw_year",
    "period": "period",
    "Premium_selected": "Premium_selected",
}


class PremiumInputRepository:
    def __init__(self, dataframe: pd.DataFrame, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager
        self._df = dataframe.copy()

    @classmethod
    def from_dataframe(
        cls,
        config_manager: ConfigManager,
        dataframe: pd.DataFrame,
        column_map: dict[str, str] | None = None,
    ) -> "PremiumInputRepository":
        if dataframe is None:
            raise ValueError("No dataframe provided to load premium data from.")
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Premium input must be a pandas DataFrame.")

        df_premium = dataframe.copy()
        if column_map:
            df_premium = df_premium.rename(columns=column_map)
        df_premium = df_premium.rename(columns=DEFAULT_PREMIUM_COLUMN_MAP)
        df_premium = df_premium.replace("", np.nan)
        df_premium = df_premium.replace(" ", np.nan)
        return cls(df_premium, config_manager)

    @classmethod
    def from_csv(
        cls,
        config_manager: ConfigManager,
        *,
        csv_path: Path,
        column_map: dict[str, str] | None = None,
    ) -> "PremiumInputRepository":
        if not csv_path.exists():
            raise FileNotFoundError(f"Premium CSV not found at {csv_path}")
        dataframe = pd.read_csv(csv_path)
        return cls.from_dataframe(config_manager, dataframe, column_map=column_map)

    @classmethod
    def from_sql(
        cls,
        config_manager: ConfigManager,
        *,
        query_path: Path,
        sql_settings: dict,
        params: list[object] | None = None,
        column_map: dict[str, str] | None = None,
    ) -> "PremiumInputRepository":
        if not query_path.exists():
            raise FileNotFoundError(f"Premium SQL query file not found at {query_path}")
        query = query_path.read_text(encoding="utf-8")

        connection_string, timeout_seconds = _build_sql_connection(sql_settings)

        try:
            import pyodbc
        except ImportError as exc:
            raise ImportError(
                "pyodbc is required for SQL input source. Install it in your environment."
            ) from exc

        connection = pyodbc.connect(connection_string, timeout=timeout_seconds)
        try:
            dataframe = pd.read_sql(query, connection, params=params or [])
        finally:
            connection.close()

        return cls.from_dataframe(config_manager, dataframe, column_map=column_map)

    def get_premium_df(self) -> pd.DataFrame:
        return self._df.copy()
