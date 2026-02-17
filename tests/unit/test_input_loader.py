from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config


def test_load_inputs_from_config_csv_yearly(tmp_path: Path) -> None:
    claims_path = tmp_path / "claims.csv"
    premium_path = tmp_path / "premium.csv"
    config_path = tmp_path / "config.yml"

    claims = pd.DataFrame(
        {
            "id": ["c1", "c2", "c3", "c4"],
            "uw_year": [
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
            ],
            "period": [
                "2000-03-31",
                "2000-06-30",
                "2000-09-30",
                "2000-12-31",
            ],
            "paid": [10.0, 20.0, 30.0, 40.0],
            "outstanding": [1.0, 2.0, 3.0, 4.0],
        }
    )
    claims.to_csv(claims_path, index=False)

    premium = pd.DataFrame(
        {
            "uw_year": ["2000", "2000", "2000", "2000"],
            "period": ["2000Q1", "2000Q2", "2000Q3", "2000Q4"],
            "Premium_selected": [100.0, 100.0, 100.0, 100.0],
        }
    )
    premium.to_csv(premium_path, index=False)

    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  results: results/",
                "  plots: plots/",
                "  data: data/",
                "  sessions: sessions/",
                "first date: 1900",
                'last date: "December 2000"',
                "segment: test_csv",
                "granularity: yearly",
                "workflow:",
                "  input:",
                "    claims:",
                "      source: csv",
                "      path: claims.csv",
                "    premium:",
                "      source: csv",
                "      path: premium.csv",
                "session:",
                "  path: sessions/test_csv.yml",
            ]
        ),
        encoding="utf-8",
    )

    config = ConfigManager.from_yaml(config_path)
    claims_df, premium_df = load_inputs_from_config(config, repo_root=tmp_path)

    assert len(claims_df) == 1
    assert float(claims_df.loc[0, "paid"]) == 100.0
    assert float(claims_df.loc[0, "outstanding"]) == 10.0

    assert len(premium_df) == 1
    assert float(premium_df.loc[0, "Premium_selected"]) == 400.0


def test_load_inputs_from_config_sql_requires_driver(tmp_path: Path) -> None:
    claims_sql = tmp_path / "claims.sql"
    premium_sql = tmp_path / "premium.sql"
    config_path = tmp_path / "config.yml"

    claims_sql.write_text("SELECT 1 AS paid", encoding="utf-8")
    premium_sql.write_text("SELECT 1 AS Premium_selected", encoding="utf-8")

    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  results: results/",
                "  plots: plots/",
                "  data: data/",
                "  sessions: sessions/",
                "first date: 1900",
                'last date: "December 2000"',
                "segment: test_sql",
                "granularity: quarterly",
                "workflow:",
                "  input:",
                "    sql:",
                "      server: localhost",
                "      database: reserving",
                "      trusted_connection: true",
                "      timeout_seconds: 30",
                "    claims:",
                "      source: sql",
                "      query_file: claims.sql",
                "      params: []",
                "    premium:",
                "      source: sql",
                "      query_file: premium.sql",
                "      params: []",
                "session:",
                "  path: sessions/test_sql.yml",
            ]
        ),
        encoding="utf-8",
    )

    config = ConfigManager.from_yaml(config_path)

    try:
        load_inputs_from_config(config, repo_root=tmp_path)
    except ValueError as exc:
        assert "workflow.input.sql.driver" in str(exc)
    else:
        raise AssertionError("Expected ValueError when SQL driver is missing")
