from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.example_workflow import transform_inputs_granularity


def test_transform_inputs_granularity_yearly_aggregates_claims_and_premium() -> None:
    claims_df = pd.DataFrame(
        {
            "id": ["c1", "c2", "c3", "c4"],
            "uw_year": [
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
            ],
            "period": [
                pd.Timestamp("2000-03-31"),
                pd.Timestamp("2000-06-30"),
                pd.Timestamp("2000-09-30"),
                pd.Timestamp("2000-12-31"),
            ],
            "paid": [10.0, 20.0, 30.0, 40.0],
            "outstanding": [1.0, 2.0, 3.0, 4.0],
        }
    )
    premium_df = pd.DataFrame(
        {
            "uw_year": [
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
            ],
            "period": [
                pd.Timestamp("2000-03-31"),
                pd.Timestamp("2000-06-30"),
                pd.Timestamp("2000-09-30"),
                pd.Timestamp("2000-12-31"),
            ],
            "Premium_selected": [100.0, 100.0, 100.0, 100.0],
        }
    )

    claims_out, premium_out = transform_inputs_granularity(
        claims_df,
        premium_df,
        granularity="yearly",
    )

    assert len(claims_out) == 1
    assert float(claims_out.loc[0, "paid"]) == 100.0
    assert float(claims_out.loc[0, "outstanding"]) == 10.0
    assert claims_out.loc[0, "period"] == pd.Timestamp("2000-12-31")

    assert len(premium_out) == 1
    assert float(premium_out.loc[0, "Premium_selected"]) == 400.0
    assert premium_out.loc[0, "period"] == pd.Timestamp("2000-12-31")


def test_transform_inputs_granularity_rejects_unknown_value() -> None:
    claims_df = pd.DataFrame(
        {
            "id": ["c1"],
            "uw_year": [pd.Timestamp("2000-01-01")],
            "period": [pd.Timestamp("2000-03-31")],
            "paid": [10.0],
            "outstanding": [1.0],
        }
    )
    premium_df = pd.DataFrame(
        {
            "uw_year": [pd.Timestamp("2000-01-01")],
            "period": [pd.Timestamp("2000-03-31")],
            "Premium_selected": [100.0],
        }
    )

    try:
        transform_inputs_granularity(
            claims_df,
            premium_df,
            granularity="monthly",
        )
    except ValueError as exc:
        assert "Unsupported granularity" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported granularity")


def test_transform_inputs_granularity_yearly_keeps_cumulative_claim_levels() -> None:
    claims_df = pd.DataFrame(
        {
            "id": ["c1", "c2", "c3", "c4"],
            "uw_year": [
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
            ],
            "period": [
                pd.Timestamp("2000-03-31"),
                pd.Timestamp("2000-06-30"),
                pd.Timestamp("2000-09-30"),
                pd.Timestamp("2000-12-31"),
            ],
            "paid": [10.0, 20.0, 30.0, 40.0],
            "outstanding": [100.0, 90.0, 80.0, 70.0],
        }
    )
    claims_df.attrs["values_are_cumulative"] = True

    premium_df = pd.DataFrame(
        {
            "uw_year": [pd.Timestamp("2000-01-01")],
            "period": [pd.Timestamp("2000-12-31")],
            "Premium_selected": [400.0],
        }
    )

    claims_out, _ = transform_inputs_granularity(
        claims_df,
        premium_df,
        granularity="yearly",
    )

    assert len(claims_out) == 1
    assert float(claims_out.loc[0, "paid"]) == 40.0
    assert float(claims_out.loc[0, "outstanding"]) == 70.0
    assert claims_out.attrs.get("values_are_cumulative") is True
