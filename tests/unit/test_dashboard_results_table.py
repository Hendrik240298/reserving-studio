from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.dashboard import Dashboard


def _dashboard_stub() -> Dashboard:
    return Dashboard.__new__(Dashboard)


def _sample_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "incurred": [1200.0, 1600.0],
            "Premium": [2000.0, 2500.0],
            "cl_ultimate": [1800.0, 2100.0],
            "bf_ultimate": [1750.0, 2200.0],
            "ultimate": [1800.0, 2200.0],
        },
        index=["2001", "2002"],
    )


def _width_by_column(style_cell_conditional: list[dict]) -> dict[str, str]:
    output: dict[str, str] = {}
    for rule in style_cell_conditional:
        if not isinstance(rule, dict):
            continue
        condition = rule.get("if")
        if not isinstance(condition, dict):
            continue
        column_id = condition.get("column_id")
        width = rule.get("width")
        if isinstance(column_id, str) and isinstance(width, str):
            output[column_id] = width
    return output


def test_results_table_columns_toggle_between_absolute_and_relative() -> None:
    dashboard = _dashboard_stub()

    absolute_columns = dashboard._build_results_table_columns("absolute")
    relative_columns = dashboard._build_results_table_columns("relative")

    absolute_ids = [column["id"] for column in absolute_columns]
    relative_ids = [column["id"] for column in relative_columns]

    assert "cl_ultimate_display" in absolute_ids
    assert "bf_ultimate_display" in absolute_ids
    assert "ultimate_display" in absolute_ids
    assert "cl_loss_ratio_display" not in absolute_ids
    assert "bf_loss_ratio_display" not in absolute_ids
    assert "selected_loss_ratio_display" not in absolute_ids

    assert "cl_loss_ratio_display" in relative_ids
    assert "bf_loss_ratio_display" in relative_ids
    assert "selected_loss_ratio_display" in relative_ids
    assert "cl_ultimate_display" not in relative_ids
    assert "bf_ultimate_display" not in relative_ids
    assert "ultimate_display" not in relative_ids

    assert "ibnr_display" in absolute_ids
    assert "ibnr_display" in relative_ids


def test_results_table_rows_include_selected_relative_and_na_for_zero_premium() -> None:
    dashboard = _dashboard_stub()
    results = _sample_results_df().copy()
    results.loc["2002", "Premium"] = 0.0

    rows = dashboard._build_results_table_rows(results)

    assert rows[0]["selected_loss_ratio_display"] == "90.00%"
    assert rows[1]["selected_loss_ratio_display"] == "N/A"


def test_results_widths_are_constant_for_absolute_relative_pairs() -> None:
    dashboard = _dashboard_stub()
    rows = dashboard._build_results_table_rows(_sample_results_df())

    style_rules = dashboard._build_results_style_cell_conditional(rows)
    widths = _width_by_column(style_rules)

    assert widths["cl_ultimate_display"] == widths["cl_loss_ratio_display"]
    assert widths["bf_ultimate_display"] == widths["bf_loss_ratio_display"]
    assert widths["ultimate_display"] == widths["selected_loss_ratio_display"]


def test_selection_styles_follow_active_results_view_mode() -> None:
    dashboard = _dashboard_stub()
    selected = {"2002": "bornhuetter_ferguson"}

    absolute_styles = dashboard._build_results_selection_styles(selected, "absolute")
    relative_styles = dashboard._build_results_selection_styles(selected, "relative")

    assert absolute_styles[0]["if"]["column_id"] == "bf_ultimate_display"
    assert relative_styles[0]["if"]["column_id"] == "bf_loss_ratio_display"
