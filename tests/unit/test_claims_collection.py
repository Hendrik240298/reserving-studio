from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.claims_collection import ClaimsCollection


def test_claim_level_methods_require_id_column() -> None:
    dataframe = pd.DataFrame(
        {
            "uw_year": [pd.Timestamp("2000-01-01")],
            "period": [pd.Timestamp("2000-03-31")],
            "paid": [100.0],
            "outstanding": [50.0],
        }
    )
    claims = ClaimsCollection(dataframe)

    with pytest.raises(ValueError, match="requires an 'id' column"):
        list(claims.iter_claims())

    with pytest.raises(ValueError, match="requires an 'id' column"):
        claims.get_claim_amounts()


def test_claim_amounts_sum_incurred_by_id() -> None:
    dataframe = pd.DataFrame(
        {
            "id": ["a", "a", "b"],
            "uw_year": [
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2001-01-01"),
            ],
            "period": [
                pd.Timestamp("2000-03-31"),
                pd.Timestamp("2000-06-30"),
                pd.Timestamp("2001-03-31"),
            ],
            "paid": [100.0, 50.0, 25.0],
            "outstanding": [20.0, 30.0, 5.0],
        }
    )
    claims = ClaimsCollection(dataframe)

    amounts = claims.get_claim_amounts()

    assert float(amounts.loc["a"]) == 200.0
    assert float(amounts.loc["b"]) == 30.0
