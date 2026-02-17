from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.claims_collection import ClaimsCollection
from source.premium_repository import PremiumRepository
from source.triangle import Triangle


def test_triangle_respects_cumulative_claims_input() -> None:
    claims_df = pd.DataFrame(
        {
            "id": ["c1", "c1"],
            "uw_year": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "period": [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30")],
            "paid": [100.0, 150.0],
            "outstanding": [50.0, 40.0],
        }
    )
    premium_df = pd.DataFrame(
        {
            "uw_year": [pd.Timestamp("2020-01-01")],
            "period": [pd.Timestamp("2020-03-31")],
            "Premium_selected": [200.0],
        }
    )

    claims = ClaimsCollection(claims_df, values_are_cumulative=True)
    premium = PremiumRepository.from_dataframe(
        config_manager=None, dataframe=premium_df
    )

    triangle = Triangle.from_claims(claims, premium)

    incurred = triangle.get_triangle("incurred")["incurred"].to_frame()
    paid = triangle.get_triangle("paid")["paid"].to_frame()

    assert float(incurred.iloc[0, 0]) == 150.0
    assert float(incurred.iloc[0, 1]) == 190.0
    assert float(paid.iloc[0, 0]) == 100.0
    assert float(paid.iloc[0, 1]) == 150.0
