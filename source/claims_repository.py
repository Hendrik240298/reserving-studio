from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from source.config_manager import ConfigManager


DEFAULT_CLAIMS_COLUMN_MAP = {
    "id": "id",
    "uw_year": "uw_year",
    "dev_period": "dev_period",
    "period": "period",
    "paid": "paid",
    "outstanding": "outstanding",
    "loss_year": "loss_year",
    "accept_id": "accept_id",
    "loss_name": "loss_name",
}


class ClaimsRepository:
    def __init__(self, dataframe: pd.DataFrame, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager
        self.df_claims = dataframe.copy()
        self._validate_required_columns()
        self.fix_movements_wo_claim_id()
        self.update_date_format()
        self.filter_net_zero_claims()
        self.correct_multiple_uwy_per_claim_id()
        self.correct_wrong_loss_year()
        self.correct_period_before_uwy()

    @classmethod
    def from_dataframe(
        cls,
        config_manager: ConfigManager,
        dataframe: pd.DataFrame,
        column_map: dict[str, str] | None = None,
    ) -> "ClaimsRepository":
        if dataframe is None:
            raise ValueError("No dataframe provided to load claims data from.")
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Claims input must be a pandas DataFrame.")

        df_claims = dataframe.copy()
        if column_map:
            df_claims = df_claims.rename(columns=column_map)
        df_claims = df_claims.rename(columns=DEFAULT_CLAIMS_COLUMN_MAP)
        df_claims = df_claims.replace("", np.nan)
        df_claims = df_claims.replace(" ", np.nan)

        if "paid" in df_claims.columns:
            df_claims["paid"] = pd.to_numeric(
                df_claims["paid"], errors="coerce"
            ).fillna(0.0)
        if "outstanding" in df_claims.columns:
            df_claims["outstanding"] = pd.to_numeric(
                df_claims["outstanding"], errors="coerce"
            ).fillna(0.0)

        return cls(df_claims, config_manager)

    @classmethod
    def from_csv(
        cls,
        config_manager: ConfigManager,
        *,
        csv_path: Path,
        column_map: dict[str, str] | None = None,
    ) -> "ClaimsRepository":
        if not csv_path.exists():
            raise FileNotFoundError(f"Claims CSV not found at {csv_path}")
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
    ) -> "ClaimsRepository":
        if not query_path.exists():
            raise FileNotFoundError(f"Claims SQL query file not found at {query_path}")
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

    def get_claims_df(self) -> pd.DataFrame:
        return self.df_claims.copy()

    def _validate_required_columns(self) -> None:
        required = {"id", "uw_year", "period", "paid", "outstanding"}
        missing = required - set(self.df_claims.columns)
        if missing:
            raise ValueError(f"Claims dataframe is missing required columns: {missing}")

    def fix_movements_wo_claim_id(self) -> None:
        mask = self.df_claims["id"].isna()
        if not mask.any():
            return

        if {"accept_id", "loss_name", "loss_year"}.issubset(self.df_claims.columns):
            group = (
                self.df_claims[mask]
                .copy()
                .groupby(["accept_id", "loss_name", "loss_year"])
                .ngroup()
            )
            accept_ids = self.df_claims.loc[mask, "accept_id"].astype(str).values
            group_nums = group.astype(str).str.zfill(7).values
            self.df_claims.loc[mask, "id"] = accept_ids + "_" + group_nums
            return

        fallback_ids = [f"missing_id_{idx}" for idx in self.df_claims.index[mask]]
        self.df_claims.loc[mask, "id"] = fallback_ids

    def update_date_format(self) -> None:
        self.df_claims["uw_year"] = _coerce_uw_year(self.df_claims["uw_year"])
        self.df_claims["period"] = _coerce_period(self.df_claims["period"])

    def correct_multiple_uwy_per_claim_id(self) -> None:
        id_uwy_counts = self.df_claims.groupby("id")["uw_year"].nunique()
        duplicate_ids = id_uwy_counts[id_uwy_counts > 1].index

        if len(duplicate_ids) == 0:
            return

        mask = self.df_claims["id"].isin(duplicate_ids)
        self.df_claims.loc[mask, "id"] = (
            self.df_claims.loc[mask, "id"].astype(str)
            + "_"
            + self.df_claims.loc[mask, "uw_year"].dt.year.astype(str)
        )

    def correct_period_before_uwy(self) -> None:
        incurred = self.df_claims["paid"] + self.df_claims["outstanding"]
        positive_mask = (self.df_claims["uw_year"] > self.df_claims["period"]) & (
            incurred > 0
        )
        if positive_mask.any():
            self.df_claims.loc[positive_mask, "period"] = self.df_claims.loc[
                positive_mask, "uw_year"
            ]

        non_positive_mask = (self.df_claims["uw_year"] > self.df_claims["period"]) & (
            incurred <= 0
        )
        if non_positive_mask.any():
            self.df_claims = self.df_claims[~non_positive_mask].copy()

    def filter_net_zero_claims(self) -> None:
        claim = self.df_claims.groupby("id")[["paid", "outstanding"]].sum()
        claim["incurred"] = claim["paid"] + claim["outstanding"]
        zero_claims = claim[claim["incurred"] == 0].index
        if len(zero_claims) > 0:
            self.df_claims = self.df_claims[
                ~self.df_claims["id"].isin(zero_claims)
            ].copy()

    def correct_wrong_loss_year(self) -> None:
        if "loss_year" not in self.df_claims.columns:
            return

        coerced = pd.to_numeric(self.df_claims["loss_year"], errors="coerce")
        self.df_claims["loss_year"] = coerced
        lower = self.config_manager.get_first_UWY()
        upper = int(self.config_manager.get_latest_period() / 100)

        mask = (self.df_claims["loss_year"] < lower) | (
            self.df_claims["loss_year"] > upper
        )
        claim_ids = self.df_claims.loc[mask, "id"].unique()

        for claim_id in claim_ids:
            claim = self.df_claims[self.df_claims["id"] == claim_id]
            claim_incurred = claim["paid"] + claim["outstanding"]
            active_periods = claim.loc[claim_incurred > 0, "period"]
            if len(active_periods) == 0:
                continue
            corrected_loss_year = int(active_periods.min().year)
            self.df_claims.loc[self.df_claims["id"] == claim_id, "loss_year"] = (
                corrected_loss_year
            )


def _coerce_uw_year(values: pd.Series) -> pd.Series:
    series = pd.Series(values)
    if isinstance(series.dtype, pd.PeriodDtype):
        uwy = series.dt.to_timestamp(how="start")
    elif pd.api.types.is_datetime64_any_dtype(series):
        uwy = (
            pd.to_datetime(series, errors="coerce")
            .dt.to_period("Y")
            .dt.to_timestamp(how="start")
        )
    else:
        text = series.astype(str).str.extract(r"(\d{4})", expand=False)
        uwy = pd.to_datetime(text, format="%Y", errors="coerce")

    if uwy.isna().any():
        raise ValueError("Claims dataframe contains invalid 'uw_year' values.")
    return pd.Series(uwy, index=series.index)


def _coerce_period(values: pd.Series) -> pd.Series:
    series = pd.Series(values)
    if isinstance(series.dtype, pd.PeriodDtype):
        coerced = series.dt.to_timestamp(how="end").dt.normalize()
    elif pd.api.types.is_datetime64_any_dtype(series):
        coerced = pd.to_datetime(series, errors="coerce")
    else:
        text = series.astype(str).str.strip()
        quarter_pattern = re.compile(r"^(\d{4})Q([1-4])$")
        yyyymm_pattern = re.compile(r"^\d{6}$")

        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        quarter_mask = text.str.match(quarter_pattern)
        if quarter_mask.any():
            quarter_index = pd.PeriodIndex(text[quarter_mask], freq="Q")
            parsed.loc[quarter_mask] = quarter_index.to_timestamp("Q")

        month_mask = text.str.match(yyyymm_pattern)
        if month_mask.any():
            parsed.loc[month_mask] = pd.to_datetime(
                text[month_mask],
                format="%Y%m",
                errors="coerce",
            ) + pd.offsets.MonthEnd(0)

        other_mask = ~(quarter_mask | month_mask)
        if other_mask.any():
            parsed.loc[other_mask] = pd.to_datetime(text[other_mask], errors="coerce")

        coerced = parsed

    if pd.Series(coerced).isna().any():
        raise ValueError("Claims dataframe contains invalid 'period' values.")
    return pd.Series(coerced, index=series.index)


def _build_sql_connection(sql_settings: dict) -> tuple[str, int]:
    driver = str(sql_settings.get("driver", "")).strip()
    server = str(sql_settings.get("server", "")).strip()
    database = str(sql_settings.get("database", "")).strip()
    trusted = _to_bool(
        sql_settings.get("trusted_connection", True),
        key_name="workflow.input.sql.trusted_connection",
    )
    timeout_seconds_raw = sql_settings.get("timeout_seconds", 30)

    if not driver:
        raise ValueError("workflow.input.sql.driver is required for SQL input source")
    if not server:
        raise ValueError("workflow.input.sql.server is required for SQL input source")
    if not database:
        raise ValueError("workflow.input.sql.database is required for SQL input source")

    try:
        timeout_seconds = int(timeout_seconds_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "workflow.input.sql.timeout_seconds must be an integer"
        ) from exc

    parts = [
        f"DRIVER={{{driver}}}",
        f"SERVER={server}",
        f"DATABASE={database}",
        f"Trusted_Connection={'yes' if trusted else 'no'}",
    ]

    encrypt = sql_settings.get("encrypt")
    if encrypt is not None:
        encrypt_flag = (
            "yes" if _to_bool(encrypt, key_name="workflow.input.sql.encrypt") else "no"
        )
        parts.append(f"Encrypt={encrypt_flag}")

    trust_server_certificate = sql_settings.get("trust_server_certificate")
    if trust_server_certificate is not None:
        trust_flag = (
            "yes"
            if _to_bool(
                trust_server_certificate,
                key_name="workflow.input.sql.trust_server_certificate",
            )
            else "no"
        )
        parts.append(f"TrustServerCertificate={trust_flag}")

    return ";".join(parts), timeout_seconds


def _to_bool(value: object, *, key_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{key_name} must be a boolean value")
