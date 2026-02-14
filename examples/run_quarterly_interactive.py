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
    load_config,
    run_interactive_session,
)


def _load_quarterly_claims_df() -> pd.DataFrame:
    if cl.__file__ is None:
        raise ValueError("chainladder package path not available")

    csv_path = Path(cl.__file__).parent / "utils" / "data" / "quarterly.csv"
    dataframe = pd.read_csv(csv_path)

    dataframe = dataframe.rename(
        columns={
            "origin": "uw_year",
            "development": "period",
        }
    )
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

    paid = pd.Series(pd.to_numeric(dataframe["incurred"], errors="coerce"))
    dataframe["paid"] = paid.fillna(0.0)
    dataframe["outstanding"] = 0.0
    dataframe["id"] = [f"quarterly_{i}" for i in range(len(dataframe))]

    return pd.DataFrame(
        dataframe[["id", "uw_year", "period", "paid", "outstanding"]].copy()
    )


def _load_quarterly_premium_df() -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[1]
    premium_path = repo_root / "data" / "quarterly_premium.csv"
    if not premium_path.exists():
        raise FileNotFoundError(f"Premium CSV not found at {premium_path}")
    return pd.read_csv(premium_path)


def main() -> None:
    config = load_config()
    claims_df = _load_quarterly_claims_df()
    premium_df = _load_quarterly_premium_df()

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
