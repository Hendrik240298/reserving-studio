from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.claims_repository import ClaimsRepository
from source.config_manager import ConfigManager


def _build_config(tmp_path: Path) -> ConfigManager:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  results: results/",
                "  plots: plots/",
                "  data: data/",
                "  sessions: sessions/",
                "first date: 2018",
                'last date: "December 2024"',
                "segment: test",
                "granularity: quarterly",
                "session:",
                "  path: sessions/test.yml",
            ]
        ),
        encoding="utf-8",
    )
    return ConfigManager.from_yaml(config_path)


def test_claims_repository_standardizes_and_cleans_dataframe(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    dataframe = pd.DataFrame(
        {
            "id": ["A", "A", "B", "ZERO", None],
            "uw_year": [2020, 2021, 2020, 2020, 2020],
            "dev_period": [3, 3, 6, 3, 3],
            "period": ["201912", "2021Q2", "202006", "202003", "202003"],
            "paid": [100.0, 50.0, 10.0, 0.0, 5.0],
            "outstanding": [0.0, 0.0, 5.0, 0.0, 2.0],
            "loss_year": [2000, 2030, 2020, 2020, 2020],
            "accept_id": ["X", "X", "Y", "Z", "ACC"],
            "loss_name": ["L", "L", "M", "N", "O"],
        }
    )

    repository = ClaimsRepository.from_dataframe(config, dataframe)
    claims_df = repository.get_claims_df()

    assert {"id", "uw_year", "period", "paid", "outstanding"}.issubset(
        claims_df.columns
    )
    assert claims_df["id"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(claims_df["uw_year"])
    assert pd.api.types.is_datetime64_any_dtype(claims_df["period"])
    assert "ZERO" not in claims_df["id"].astype(str).tolist()

    corrected_period = claims_df.loc[
        claims_df["id"].astype(str).str.startswith("A_2020"), "period"
    ]
    assert not corrected_period.empty
    assert corrected_period.iloc[0] == pd.Timestamp("2020-01-01")


def test_claims_repository_requires_standardized_columns(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    dataframe = pd.DataFrame(
        {
            "uw_year": [2020],
            "period": ["202003"],
            "paid": [10.0],
        }
    )

    try:
        ClaimsRepository.from_dataframe(config, dataframe)
    except ValueError as exc:
        assert "required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing required claims columns")
