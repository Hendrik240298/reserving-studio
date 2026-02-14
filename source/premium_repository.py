from __future__ import annotations

import logging

import pandas as pd

from source.config_manager import ConfigManager


class PremiumRepository:
    def __init__(
        self,
        config_manager: ConfigManager | None,
        dataframe: pd.DataFrame | None = None,
    ) -> None:
        self.config_manager = config_manager
        if dataframe is None:
            raise ValueError("PremiumRepository requires a dataframe input.")
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("PremiumRepository requires a pandas DataFrame input.")
        self._df_premium = self._normalize_dataframe(dataframe)

    @classmethod
    def from_dataframe(
        cls,
        config_manager: ConfigManager | None,
        dataframe: pd.DataFrame | None,
    ) -> "PremiumRepository":
        if dataframe is None:
            raise ValueError("No dataframe provided to load premium data from.")
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Premium input must be a pandas DataFrame.")
        return cls(config_manager, dataframe)

    def _normalize_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df_premium = dataframe.copy()

        if {"UnderwritingYear", "Premium"}.issubset(df_premium.columns):
            normalized = self._normalize_uwy_level_premium(df_premium)
        elif {"origin", "development", "Premium_selected"}.issubset(df_premium.columns):
            normalized = self._normalize_cumulative_triangle_premium(df_premium)
        elif {"uw_year", "period", "Premium_selected"}.issubset(df_premium.columns):
            normalized = self._normalize_canonical_premium(df_premium)
        else:
            raise ValueError(
                "Premium dataframe must contain either {'UnderwritingYear', 'Premium'} "
                "or {'origin', 'development', 'Premium_selected'} "
                "or {'uw_year', 'period', 'Premium_selected'} columns."
            )

        duplicate_mask = normalized.duplicated(subset=["uw_year", "period"], keep=False)
        if duplicate_mask.any():
            duplicate_rows = normalized.loc[duplicate_mask, ["uw_year", "period"]]
            raise ValueError(
                "Premium dataframe contains duplicate ('uw_year', 'period') rows: "
                f"{duplicate_rows.to_dict(orient='records')[:5]}"
            )

        return normalized

    def _normalize_uwy_level_premium(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df_premium = dataframe.rename(
            columns={
                "UnderwritingYear": "uw_year",
                "Premium": "Premium_selected",
            }
        ).copy()
        df_premium = df_premium[["uw_year", "Premium_selected"]].copy()

        df_premium["uw_year"] = self._coerce_uw_year(df_premium["uw_year"])
        df_premium["period"] = df_premium["uw_year"] + pd.offsets.QuarterEnd(0)
        df_premium["Premium_selected"] = self._coerce_numeric(
            df_premium["Premium_selected"],
            "Premium_selected",
        )

        return self._finalize_schema(pd.DataFrame(df_premium))

    def _normalize_cumulative_triangle_premium(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        df_premium = dataframe.copy()
        df_premium["uw_year"] = self._coerce_uw_year(df_premium["origin"])
        df_premium["period"] = self._coerce_quarter_period(df_premium["development"])
        df_premium["Premium_selected"] = self._coerce_numeric(
            df_premium["Premium_selected"],
            "Premium_selected",
        )

        df_premium = pd.DataFrame(
            df_premium[["uw_year", "period", "Premium_selected"]].copy()
        )
        df_premium = df_premium.sort_values(by=["uw_year", "period"]).reset_index(
            drop=True
        )
        df_premium["Premium_selected"] = (
            df_premium.groupby("uw_year")["Premium_selected"]
            .diff()
            .fillna(df_premium["Premium_selected"])
        )
        df_premium["Premium_selected"] = df_premium["Premium_selected"].clip(lower=0.0)

        return self._finalize_schema(pd.DataFrame(df_premium))

    def _normalize_canonical_premium(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df_premium = dataframe[["uw_year", "period", "Premium_selected"]].copy()
        df_premium["uw_year"] = self._coerce_uw_year(df_premium["uw_year"])
        df_premium["period"] = self._coerce_quarter_period(df_premium["period"])
        df_premium["Premium_selected"] = self._coerce_numeric(
            df_premium["Premium_selected"],
            "Premium_selected",
        )
        return self._finalize_schema(pd.DataFrame(df_premium))

    def _finalize_schema(self, dataframe: object):
        df_premium = pd.DataFrame(dataframe)[
            ["uw_year", "period", "Premium_selected"]
        ].copy()
        for column in ["GWP", "EPI", "GWP_Forecast"]:
            df_premium[column] = df_premium["Premium_selected"]
        return pd.DataFrame(
            df_premium[
                [
                    "uw_year",
                    "period",
                    "GWP",
                    "EPI",
                    "GWP_Forecast",
                    "Premium_selected",
                ]
            ].copy()
        )

    def _coerce_uw_year(self, values: object) -> pd.Series:
        series = pd.Series(values)

        if isinstance(series.dtype, pd.PeriodDtype):
            coerced = series.dt.to_timestamp("Y")
        elif pd.api.types.is_datetime64_any_dtype(series):
            coerced = series.dt.to_period("Y").dt.to_timestamp("Y")
        else:
            string_values = series.astype(str)
            parsed = pd.to_datetime(string_values.str[:4], format="%Y", errors="coerce")
            coerced = pd.Series(parsed, index=series.index)

        if coerced.isna().any():
            raise ValueError("Premium dataframe contains invalid 'uw_year' values.")
        return coerced

    def _coerce_quarter_period(self, values: object) -> pd.Series:
        series = pd.Series(values)

        if isinstance(series.dtype, pd.PeriodDtype):
            coerced = series.dt.to_timestamp("Q")
        elif pd.api.types.is_datetime64_any_dtype(series):
            coerced = series.dt.to_period("Q").dt.to_timestamp("Q")
        else:
            string_values = series.astype(str)
            try:
                quarter_index = pd.PeriodIndex(string_values, freq="Q")
            except Exception as exc:
                raise ValueError(
                    "Premium dataframe contains invalid quarterly 'period' or 'development' values."
                ) from exc
            coerced = pd.Series(quarter_index.to_timestamp("Q"), index=series.index)

        if coerced.isna().any():
            raise ValueError("Premium dataframe contains invalid 'period' values.")
        return coerced

    def _coerce_numeric(self, values: object, column: str) -> pd.Series:
        series = pd.Series(values)
        coerced = pd.to_numeric(series, errors="coerce")
        coerced_series = pd.Series(coerced, index=series.index)
        if coerced_series.isna().any():
            raise ValueError(
                f"Premium dataframe contains invalid numeric values in '{column}'."
            )
        return coerced_series.astype("float64")

    def get_premium(self) -> pd.DataFrame:
        return self._df_premium.copy()

    def set_premium(self, df: pd.DataFrame) -> None:
        self._df_premium = self._normalize_dataframe(df)

    def update_date_format(self) -> None:
        logging.info(self._df_premium)
        self._df_premium = self._normalize_dataframe(self._df_premium)
        logging.info(self._df_premium)
