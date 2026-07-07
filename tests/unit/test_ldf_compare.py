from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.ldf_compare import (  # noqa: E402
    ScenarioRun,
    build_ldf_table,
    load_scenarios,
    normalize_drop,
)


def test_normalize_drop_uses_pair_contract() -> None:
    assert normalize_drop([["2000", 6], ["2002", 9]]) == [("2000", 6), ("2002", 9)]


def test_load_scenario_json_uses_reserving_settings_shape() -> None:
    scenarios = load_scenarios(
        scenarios_file=None,
        scenario_json=[
            'drop_2000_2002_6:{"development":{"drop":[["2000",6],["2002",6]]}}'
        ],
    )

    assert scenarios["drop_2000_2002_6"]["development"]["drop"] == [
        ("2000", 6),
        ("2002", 6),
    ]


def test_load_scenarios_file(tmp_path: Path) -> None:
    scenarios_file = tmp_path / "scenarios.yml"
    scenarios_file.write_text(
        "\n".join(
            [
                "scenarios:",
                "  simple_average:",
                "    development:",
                "      average: simple",
                "  drop_2002_6:",
                "    development:",
                "      drop:",
                "        - [\"2002\", 6]",
            ]
        ),
        encoding="utf-8",
    )

    scenarios = load_scenarios(scenarios_file=scenarios_file, scenario_json=[])

    assert scenarios["simple_average"]["development"]["average"] == "simple"
    assert scenarios["drop_2002_6"]["development"]["drop"] == [("2002", 6)]


def test_build_ldf_table() -> None:
    baseline = _run(
        "default",
        ldf=pd.Series({"3-6": 1.20, "6-9": 1.10}),
    )
    scenario = _run(
        "drop_2002_6",
        ldf=pd.Series({"3-6": 1.25, "6-9": 1.08}),
    )

    ldf_table = build_ldf_table(baseline, [scenario])

    assert list(ldf_table.columns) == [
        "age_label",
        "start_month",
        "default_ldf",
        "drop_2002_6_ldf",
        "drop_2002_6_delta",
        "drop_2002_6_delta_pct",
    ]
    assert round(float(ldf_table.loc[0, "drop_2002_6_delta"]), 6) == 0.05


def _run(name: str, *, ldf: pd.Series) -> ScenarioRun:
    return ScenarioRun(
        name=name,
        settings={},
        effective_settings={},
        ldf=ldf,
        warnings=[],
    )
