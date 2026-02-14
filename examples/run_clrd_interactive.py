from __future__ import annotations

from pathlib import Path
import sys

import chainladder as cl
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import (
    build_workflow_from_dataframes,
    create_interactive_session_controller,
    run_interactive_session,
)
from source.config_manager import ConfigManager


CLRD_LOB = "comauto"


def _load_clrd_config() -> ConfigManager:
    config_path = REPO_ROOT / "examples" / "config_clrd.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"CLRD config not found at {config_path}")
    return ConfigManager.from_yaml(config_path)


def _load_clrd_portfolio_df() -> pd.DataFrame:
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

    filtered = dataframe[dataframe["LOB"].astype(str).str.lower() == CLRD_LOB].copy()
    if filtered.empty:
        raise ValueError(f"No CLRD rows found for LOB '{CLRD_LOB}'.")

    portfolio = (
        filtered[["origin", "development", "IncurLoss", "CumPaidLoss", "EarnedPremDIR"]]
        .groupby(["origin", "development"], as_index=False)
        .sum(numeric_only=True)
    )
    return pd.DataFrame(portfolio)


def _to_incremental_by_origin(dataframe: pd.DataFrame, column: str) -> pd.Series:
    ordered = dataframe.sort_values(["origin", "development"]).copy()
    values = pd.Series(ordered[column], index=ordered.index)
    origins = pd.Series(ordered["origin"], index=ordered.index)
    incremental = pd.Series(
        values.groupby(origins).diff(),
        index=ordered.index,
    )
    incremental = incremental.fillna(values).fillna(0.0)
    return pd.Series(incremental.reindex(dataframe.index), index=dataframe.index)


def _load_clrd_claims_df() -> pd.DataFrame:
    portfolio = _load_clrd_portfolio_df()
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

    claims["incurred"] = _to_incremental_by_origin(claims, "paid_cumulative")
    claims["paid"] = _to_incremental_by_origin(claims, "paid_cumulative")
    claims["outstanding"] = 0.0
    claims["id"] = [f"clrd_{i}" for i in range(len(claims))]

    return pd.DataFrame(
        claims[["id", "uw_year", "period", "paid", "outstanding"]].copy()
    )


def _load_clrd_premium_df() -> pd.DataFrame:
    portfolio = _load_clrd_portfolio_df()
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
        pd.to_numeric(
            premium["Premium_selected"],
            errors="coerce",
        ),
        index=premium.index,
    ).fillna(0.0)
    premium = pd.DataFrame(premium[["uw_year", "period", "Premium_selected"]].copy())
    return premium


def main() -> None:
    config = _load_clrd_config()
    claims_df = _load_clrd_claims_df()
    premium_df = _load_clrd_premium_df()

    reserving = build_workflow_from_dataframes(
        claims_df=claims_df,
        premium_df=premium_df,
        config=config,
    )
    controller = create_interactive_session_controller()

    print("Dashboard starting at http://127.0.0.1:8050")
    print("Use the UI, then click 'Finalize & Continue' in the Results tab.")

    finalized = run_interactive_session(
        reserving,
        config=config,
        controller=controller,
        port=8050,
        timeout_seconds=None,
    )

    print("\nInteractive session finalized.")
    print(f"Segment: {finalized.segment}")
    print(f"Finalized at: {finalized.finalized_at_utc.isoformat()}")
    print("Selected methods by UWY:")
    for uwy, method in finalized.params_store.selected_ultimate_by_uwy.items():
        print(f"  {uwy}: {method}")

    print("\nTop 5 rows of finalized numeric results:")
    print(finalized.results_df.head())


if __name__ == "__main__":
    main()
