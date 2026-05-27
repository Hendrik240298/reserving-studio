from __future__ import annotations

from pathlib import Path
import sys

import chainladder as cl
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config


def test_quarterly_granular_claims_reconcile_to_quarterly_sample() -> None:
    config = ConfigManager.from_yaml(REPO_ROOT / "examples" / "config_quarterly.yml")

    claims_df, _ = load_inputs_from_config(config, repo_root=REPO_ROOT)

    actual = claims_df.copy()
    actual["uw_year"] = pd.to_datetime(actual["uw_year"], errors="raise").dt.year
    actual["period"] = pd.to_datetime(actual["period"], errors="raise")
    actual["paid"] = pd.to_numeric(actual["paid"], errors="raise")
    actual["outstanding"] = pd.to_numeric(actual["outstanding"], errors="raise")
    actual["incurred"] = actual["paid"] + actual["outstanding"]
    actual = actual.groupby(["uw_year", "period"], as_index=False)[
        ["paid", "outstanding", "incurred"]
    ].sum(numeric_only=True)

    if cl.__file__ is None:
        raise AssertionError("chainladder package path not available")
    sample_path = Path(cl.__file__).parent / "utils" / "data" / "quarterly.csv"
    expected = pd.read_csv(sample_path)
    expected["uw_year"] = expected["origin"].astype(int)
    expected["period"] = pd.PeriodIndex(expected["development"], freq="Q").to_timestamp(
        "Q"
    )
    expected["paid"] = pd.Series(
        pd.to_numeric(expected["paid"], errors="coerce"), index=expected.index
    ).fillna(0.0)
    expected["incurred"] = pd.Series(
        pd.to_numeric(expected["incurred"], errors="coerce"), index=expected.index
    ).fillna(0.0)
    expected["paid"] = expected[["paid", "incurred"]].min(axis=1)
    expected["outstanding"] = expected["incurred"] - expected["paid"]
    expected = expected[["uw_year", "period", "paid", "outstanding", "incurred"]].copy()

    pd.testing.assert_frame_equal(
        actual.sort_values(["uw_year", "period"]).reset_index(drop=True),
        expected.sort_values(["uw_year", "period"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_quarterly_granular_claims_metadata_is_stable_and_useful() -> None:
    dataset_path = REPO_ROOT / "data" / "quarterly_granular_claims.csv"
    claims_df = pd.read_csv(dataset_path)

    required_columns = {
        "id",
        "accept_id",
        "business_id",
        "InsuredName",
        "LossName",
        "loss_name",
        "loss_year",
        "uw_year",
        "dev_period",
        "period",
        "paid",
        "outstanding",
        "incurred",
        "paid_movement",
        "outstanding_movement",
        "incurred_movement",
        "claim_status",
    }
    assert required_columns.issubset(claims_df.columns)

    assert 3 <= claims_df["business_id"].nunique() <= 5
    assert (claims_df["outstanding"] >= 0).all()
    assert (claims_df["paid"] <= claims_df["incurred"]).all()
    assert ((claims_df["paid"] + claims_df["outstanding"]) == claims_df["incurred"]).all()

    grouped = claims_df.groupby("id")
    assert (grouped["InsuredName"].nunique() == 1).all()
    assert (grouped["LossName"].nunique() == 1).all()
    assert (grouped["loss_name"].nunique() == 1).all()
    assert (grouped["business_id"].nunique() == 1).all()
    assert (grouped["uw_year"].nunique() == 1).all()

    claim_counts = claims_df.groupby("uw_year")["id"].nunique()
    assert claim_counts.loc[1996:2004].min() >= 89
    assert claim_counts.loc[2005] < claim_counts.loc[2004]
    assert claim_counts.loc[2006] < claim_counts.loc[2005]
