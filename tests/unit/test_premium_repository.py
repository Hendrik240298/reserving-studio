from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.premium_repository import PremiumRepository


def test_from_dataframe_accepts_uwy_level_premium() -> None:
    dataframe = pd.DataFrame(
        {
            "UnderwritingYear": [2000, 2001],
            "Premium": [1000.0, 1200.0],
        }
    )

    repository = PremiumRepository.from_dataframe(
        config_manager=None, dataframe=dataframe
    )
    premium = repository.get_premium()

    assert set(premium.columns) == {
        "uw_year",
        "period",
        "GWP",
        "EPI",
        "GWP_Forecast",
        "Premium_selected",
    }
    assert float(premium.iloc[0]["Premium_selected"]) == 1000.0
    assert float(premium.iloc[1]["Premium_selected"]) == 1200.0


def test_from_dataframe_accepts_cumulative_triangle_premium() -> None:
    dataframe = pd.DataFrame(
        {
            "origin": [1995, 1995, 1996],
            "development": ["1995Q1", "1995Q2", "1996Q1"],
            "Premium_selected": [100.0, 160.0, 80.0],
        }
    )

    repository = PremiumRepository.from_dataframe(
        config_manager=None, dataframe=dataframe
    )
    premium = repository.get_premium()

    uwy_text = pd.to_datetime(premium["uw_year"], errors="raise").astype(str)
    premium_1995 = premium[uwy_text.str[:4] == "1995"].copy()
    premium_1995 = premium_1995.assign(
        _period_sort=pd.to_datetime(premium_1995["period"], errors="raise")
    ).sort_values(by=["_period_sort"])
    values_1995 = premium_1995["Premium_selected"].tolist()

    assert values_1995 == [100.0, 60.0]


def test_from_dataframe_accepts_annual_month_lag_periods() -> None:
    dataframe = pd.DataFrame(
        {
            "uw_year": [
                pd.Timestamp("1995-01-01"),
                pd.Timestamp("1995-01-01"),
                pd.Timestamp("1996-01-01"),
            ],
            "period": [12, 24, 12],
            "Premium_selected": [100.0, 110.0, 90.0],
        }
    )

    repository = PremiumRepository.from_dataframe(
        config_manager=None,
        dataframe=dataframe,
    )
    premium = repository.get_premium().sort_values(["uw_year", "period"])

    uwy_text = pd.to_datetime(premium["uw_year"], errors="raise").astype(str)
    premium_1995 = premium[uwy_text.str[:4] == "1995"]
    premium_1995_period_text = pd.to_datetime(
        premium_1995["period"], errors="raise"
    ).astype(str)
    assert [text[5:7] for text in premium_1995_period_text.tolist()] == ["12", "12"]
    assert [text[:4] for text in premium_1995_period_text.tolist()] == ["1995", "1996"]


def test_from_dataframe_accepts_cumulative_triangle_with_annual_lags() -> None:
    dataframe = pd.DataFrame(
        {
            "origin": [1995, 1995, 1996],
            "development": [12, 24, 12],
            "Premium_selected": [120.0, 200.0, 100.0],
        }
    )

    repository = PremiumRepository.from_dataframe(
        config_manager=None,
        dataframe=dataframe,
    )
    premium = repository.get_premium().sort_values(["uw_year", "period"])

    uwy_text = pd.to_datetime(premium["uw_year"], errors="raise").astype(str)
    premium_1995 = premium[uwy_text.str[:4] == "1995"]
    premium_1995_period_text = pd.to_datetime(
        premium_1995["period"], errors="raise"
    ).astype(str)
    assert premium_1995["Premium_selected"].tolist() == [120.0, 80.0]
    assert [text[:4] for text in premium_1995_period_text.tolist()] == ["1995", "1996"]
    assert [text[5:7] for text in premium_1995_period_text.tolist()] == ["12", "12"]
