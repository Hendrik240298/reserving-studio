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

    premium_1995 = premium[premium["uw_year"].dt.year == 1995].sort_values("period")
    values_1995 = premium_1995["Premium_selected"].tolist()

    assert values_1995 == [100.0, 60.0]
