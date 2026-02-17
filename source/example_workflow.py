from __future__ import annotations

from pathlib import Path

import chainladder as cl
import pandas as pd

from source.config_manager import ConfigManager
from source.premium_repository import PremiumRepository


def build_example_inputs(
    config: ConfigManager,
    *,
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = config.get_workflow_dataset()
    if dataset == "quarterly":
        claims_df = _load_quarterly_claims_df()
        premium_df = _load_quarterly_premium_df(config, repo_root=repo_root)
    elif dataset == "clrd":
        claims_df = _load_clrd_claims_df(config)
        premium_df = _load_clrd_premium_df(config)
    else:
        raise ValueError(
            f"Unsupported workflow dataset '{dataset}'. Use 'quarterly' or 'clrd'."
        )

    return transform_inputs_granularity(
        claims_df,
        premium_df,
        granularity=config.get_granularity(),
    )


def transform_inputs_granularity(
    claims_df: pd.DataFrame,
    premium_df: pd.DataFrame,
    *,
    granularity: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if granularity not in {"quarterly", "yearly"}:
        raise ValueError(
            f"Unsupported granularity '{granularity}'. Use 'quarterly' or 'yearly'."
        )

    months_per_step = 3 if granularity == "quarterly" else 12
    values_are_cumulative = bool(claims_df.attrs.get("values_are_cumulative", False))

    normalized_claims = _normalize_claims(claims_df)
    normalized_premium = _normalize_premium(premium_df)

    claims_out = _aggregate_claims_by_step(
        normalized_claims,
        months_per_step,
        values_are_cumulative=values_are_cumulative,
    )
    premium_out = _aggregate_premium_by_step(normalized_premium, months_per_step)
    claims_out.attrs["values_are_cumulative"] = values_are_cumulative
    return claims_out, premium_out


def _load_quarterly_claims_df() -> pd.DataFrame:
    if cl.__file__ is None:
        raise ValueError("chainladder package path not available")

    csv_path = Path(cl.__file__).parent / "utils" / "data" / "quarterly.csv"
    dataframe = pd.read_csv(csv_path)

    dataframe = dataframe.rename(columns={"origin": "uw_year", "development": "period"})
    if "incurred" not in dataframe.columns:
        raise ValueError("quarterly.csv missing required 'incurred' column")

    dataframe["uw_year"] = pd.to_datetime(
        dataframe["uw_year"].astype(str).str[:4] + "-01-01",
        errors="raise",
    )
    dataframe["period"] = pd.PeriodIndex(
        dataframe["period"].astype(str),
        freq="Q",
    ).to_timestamp("Q")

    dataframe["incurred"] = pd.Series(
        pd.to_numeric(dataframe["incurred"], errors="coerce"),
        index=dataframe.index,
    ).fillna(0.0)
    dataframe["paid"] = _to_incremental_by_origin(
        dataframe,
        origin_col="uw_year",
        period_col="period",
        value_col="incurred",
    )
    dataframe["outstanding"] = 0.0
    dataframe["id"] = [f"quarterly_{i}" for i in range(len(dataframe))]

    return pd.DataFrame(
        dataframe[["id", "uw_year", "period", "paid", "outstanding"]].copy()
    )


def _load_quarterly_premium_df(
    config: ConfigManager, *, repo_root: Path
) -> pd.DataFrame:
    premium_path = repo_root / config.get_workflow_quarterly_premium_csv()
    if not premium_path.exists():
        raise FileNotFoundError(f"Premium CSV not found at {premium_path}")
    return pd.read_csv(premium_path)


def _load_clrd_portfolio_df(config: ConfigManager) -> pd.DataFrame:
    dataframe = cl.load_sample("clrd").to_frame(origin_as_datetime=False).reset_index()

    required_columns = {
        "origin",
        "development",
        "IncurLoss",
        "CumPaidLoss",
        "EarnedPremDIR",
    }
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"clrd sample is missing required columns: {missing}")

    if "LOB" not in dataframe.columns:
        raise ValueError("clrd sample is missing required 'LOB' column")

    clrd_lob = config.get_workflow_clrd_lob()
    filtered = dataframe[dataframe["LOB"].astype(str).str.lower() == clrd_lob].copy()
    if filtered.empty:
        raise ValueError(f"No CLRD rows found for LOB '{clrd_lob}'.")

    portfolio = (
        filtered[["origin", "development", "IncurLoss", "CumPaidLoss", "EarnedPremDIR"]]
        .groupby(["origin", "development"], as_index=False)
        .sum(numeric_only=True)
    )
    return pd.DataFrame(portfolio)


def _load_clrd_claims_df(config: ConfigManager) -> pd.DataFrame:
    portfolio = _load_clrd_portfolio_df(config)
    claims = portfolio.copy()

    claims["uw_year"] = pd.to_datetime(
        claims["origin"].astype(str) + "-01-01",
        errors="raise",
    )
    base_month = claims["uw_year"].dt.to_period("Y").dt.asfreq("M", how="start")
    claims["period"] = (
        base_month + (claims["development"].astype(int) - 1)
    ).dt.to_timestamp("M")

    claims["paid_cumulative"] = pd.Series(
        pd.to_numeric(claims["CumPaidLoss"], errors="coerce"),
        index=claims.index,
    )
    claims["paid"] = _to_incremental_by_origin(
        claims,
        origin_col="origin",
        period_col="development",
        value_col="paid_cumulative",
    )
    claims["outstanding"] = 0.0
    claims["id"] = [f"clrd_{i}" for i in range(len(claims))]

    return pd.DataFrame(
        claims[["id", "uw_year", "period", "paid", "outstanding"]].copy()
    )


def _load_clrd_premium_df(config: ConfigManager) -> pd.DataFrame:
    portfolio = _load_clrd_portfolio_df(config)
    grouped = pd.DataFrame(
        portfolio[["origin", "EarnedPremDIR"]]
        .groupby("origin", as_index=False)
        .max(numeric_only=True)
    )
    premium = pd.DataFrame(
        {
            "uw_year": grouped["origin"],
            "Premium_selected": grouped["EarnedPremDIR"],
        }
    )
    premium["uw_year"] = pd.to_datetime(
        premium["uw_year"].astype(str) + "-01-01",
        errors="raise",
    )
    premium["period"] = 12
    premium["Premium_selected"] = pd.Series(
        pd.to_numeric(premium["Premium_selected"], errors="coerce"),
        index=premium.index,
    ).fillna(0.0)
    return pd.DataFrame(premium[["uw_year", "period", "Premium_selected"]].copy())


def _to_incremental_by_origin(
    dataframe: pd.DataFrame,
    *,
    origin_col: str,
    period_col: str,
    value_col: str,
) -> pd.Series:
    ordered = dataframe.sort_values([origin_col, period_col]).copy()
    values = pd.Series(ordered[value_col], index=ordered.index)
    origins = pd.Series(ordered[origin_col], index=ordered.index)
    incremental = pd.Series(values.groupby(origins).diff(), index=ordered.index)
    incremental = incremental.fillna(values).fillna(0.0)
    return pd.Series(incremental.reindex(dataframe.index), index=dataframe.index)


def _normalize_claims(claims_df: pd.DataFrame) -> pd.DataFrame:
    claims = claims_df.copy()
    claims["uw_year"] = pd.to_datetime(claims["uw_year"], errors="coerce")
    claims["period"] = pd.to_datetime(claims["period"], errors="coerce")
    claims["paid"] = pd.Series(
        pd.to_numeric(claims["paid"], errors="coerce"),
        index=claims.index,
    ).fillna(0.0)
    claims["outstanding"] = pd.Series(
        pd.to_numeric(claims["outstanding"], errors="coerce"),
        index=claims.index,
    ).fillna(0.0)
    claims["id"] = [f"claim_{i}" for i in range(len(claims))]
    return pd.DataFrame(
        claims[["id", "uw_year", "period", "paid", "outstanding"]].copy()
    )


def _normalize_premium(premium_df: pd.DataFrame) -> pd.DataFrame:
    normalized = PremiumRepository.from_dataframe(
        config_manager=None,
        dataframe=premium_df,
    ).get_premium()
    premium = normalized[["uw_year", "period", "Premium_selected"]].copy()
    premium["uw_year"] = pd.to_datetime(premium["uw_year"], errors="coerce")
    premium["period"] = pd.to_datetime(premium["period"], errors="coerce")
    premium["Premium_selected"] = pd.Series(
        pd.to_numeric(premium["Premium_selected"], errors="coerce"),
        index=premium.index,
    ).fillna(0.0)
    return premium


def _aggregate_claims_by_step(
    claims_df: pd.DataFrame,
    months_per_step: int,
    *,
    values_are_cumulative: bool,
) -> pd.DataFrame:
    claims = claims_df.copy()
    claims = claims.dropna(subset=["uw_year", "period"])
    claims["step"] = _step_index(claims["uw_year"], claims["period"], months_per_step)

    if values_are_cumulative:
        grouped_by_period = (
            claims.groupby(["uw_year", "period", "step"], as_index=False)[
                ["paid", "outstanding"]
            ]
            .sum(numeric_only=True)
            .sort_values(["uw_year", "step", "period"])
            .reset_index(drop=True)
        )
        grouped = (
            grouped_by_period.groupby(["uw_year", "step"], as_index=False)
            .tail(1)
            .sort_values(["uw_year", "step"])
            .reset_index(drop=True)
        )
    else:
        grouped = (
            claims.groupby(["uw_year", "step"], as_index=False)[["paid", "outstanding"]]
            .sum(numeric_only=True)
            .sort_values(["uw_year", "step"])
            .reset_index(drop=True)
        )

    grouped["period"] = _step_to_period(
        grouped["uw_year"], grouped["step"], months_per_step
    )
    grouped["id"] = [f"claim_{i}" for i in range(len(grouped))]
    return pd.DataFrame(
        grouped[["id", "uw_year", "period", "paid", "outstanding"]].copy()
    )


def _aggregate_premium_by_step(
    premium_df: pd.DataFrame,
    months_per_step: int,
) -> pd.DataFrame:
    premium = premium_df.copy()
    premium = premium.dropna(subset=["uw_year", "period"])
    premium["step"] = _step_index(
        premium["uw_year"],
        premium["period"],
        months_per_step,
    )
    grouped = (
        premium.groupby(["uw_year", "step"], as_index=False)["Premium_selected"]
        .sum(numeric_only=True)
        .sort_values(["uw_year", "step"])
        .reset_index(drop=True)
    )
    grouped["period"] = _step_to_period(
        grouped["uw_year"], grouped["step"], months_per_step
    )
    return pd.DataFrame(grouped[["uw_year", "period", "Premium_selected"]].copy())


def _step_index(
    uw_year: pd.Series,
    period: pd.Series,
    months_per_step: int,
) -> pd.Series:
    uwy = pd.to_datetime(uw_year, errors="coerce")
    valuation = pd.to_datetime(period, errors="coerce")
    months = (
        (valuation.dt.year - uwy.dt.year) * 12 + (valuation.dt.month - uwy.dt.month) + 1
    )
    step = ((months - 1) // months_per_step) + 1
    return pd.Series(step.astype("int64"), index=uw_year.index)


def _step_to_period(
    uw_year: pd.Series,
    step: pd.Series,
    months_per_step: int,
) -> pd.Series:
    uwy = pd.to_datetime(uw_year, errors="coerce")
    base_month = uwy.dt.to_period("Y").dt.asfreq("M", how="start")
    step_series = pd.Series(pd.to_numeric(step, errors="coerce"), index=step.index)
    month_index = (step_series.astype("int64") * months_per_step) - 1
    period = (base_month + month_index).dt.to_timestamp("M")
    return pd.Series(period, index=uw_year.index)
