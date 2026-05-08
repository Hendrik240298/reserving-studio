from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.streamlit_data_app import (
    build_data_tab_display,
    build_a2a_color_styles,
    build_emergence_chart,
    build_link_ratio_display,
    build_streamlit_table,
    build_title,
    expand_fit_period_interval,
    get_development_ages,
    selected_cells_to_drops,
    selected_cells_to_tail_actions,
    toggle_tail_attachment,
    toggle_tail_fit_periods,
    toggle_drops,
)


class _FakeFrame:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    def to_frame(self) -> pd.DataFrame:
        return self._dataframe.copy()


class _FakeTriangleSlice:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    def __getitem__(self, key: str) -> _FakeFrame:
        return _FakeFrame(self._frames[key])

    def cum_to_incr(self) -> _FakeTriangleSlice:
        return _FakeTriangleSlice(
            {
                key: dataframe.diff(axis=1).fillna(dataframe)
                for key, dataframe in self._frames.items()
            }
        )


class _FakeTriangle:
    def __init__(self) -> None:
        index = pd.Index([2020, 2021], name="origin")
        columns = pd.Index([12, 24], name="development")
        self._incurred = _FakeTriangleSlice(
            {
                "incurred": pd.DataFrame(
                    [[100.0, 150.0], [50.0, 80.0]],
                    index=index,
                    columns=columns,
                ),
                "Premium_selected": pd.DataFrame(
                    [[1000.0, 1000.0], [900.0, 900.0]],
                    index=index,
                    columns=columns,
                ),
            }
        )
        self._paid = _FakeTriangleSlice(
            {
                "paid": pd.DataFrame(
                    [[70.0, 130.0], [30.0, 60.0]],
                    index=index,
                    columns=columns,
                ),
            }
        )

    def get_triangle(self, metric: str = "incurred") -> _FakeTriangleSlice:
        if metric == "paid":
            return self._paid
        return self._incurred


class _FakeReserving:
    def __init__(self) -> None:
        self._triangle = _FakeTriangle()


def test_ratio_display_uses_cumulative_denominator() -> None:
    reserving = _FakeReserving()

    table, weights, ratio_mode = build_data_tab_display(
        reserving,
        metric="paid",
        triangle_view="incremental",
        divisor="premium",
    )

    assert ratio_mode is True
    assert float(table.loc[2020, 12]) == 0.07
    assert float(table.loc[2020, 24]) == 0.06
    assert float(weights.loc[2020, 24]) == 1000.0


def test_streamlit_table_adds_average_rows_and_readable_labels() -> None:
    reserving = _FakeReserving()
    table, weights, _ratio_mode = build_data_tab_display(
        reserving,
        metric="incurred",
        triangle_view="cumulative",
        divisor="none",
    )

    display = build_streamlit_table(table, weights)

    assert "Simple Avg" in display.index
    assert "Weighted Avg" in display.index
    assert list(display.columns) == ["12", "24"]


def test_title_matches_dash_data_tab_text() -> None:
    assert (
        build_title("outstanding", "incremental", "premium")
        == "Data Triangle - Outstanding / Premium (Incremental)"
    )


def test_factor_table_selection_maps_to_drops() -> None:
    link_ratios = pd.DataFrame(
        [[1.2, 1.1], [1.3, None], [1.15, 1.05], [1.02, 1.01]],
        index=pd.Index([2020, 2021, "LDF", "Tail"]),
        columns=pd.Index([12, 24]),
    )

    table = build_link_ratio_display(link_ratios)
    selected = selected_cells_to_drops(
        table,
        [(0, "24"), {"row": 1, "column": "24"}, (2, "12")],
    )

    assert selected == [["2020", 24]]
    assert toggle_drops([["2020", 12]], selected) == [["2020", 12], ["2020", 24]]
    assert toggle_drops([["2020", 24]], selected) == []
    assert get_development_ages(link_ratios) == [12, 24]


def test_factor_table_selection_maps_to_tail_controls() -> None:
    table = build_link_ratio_display(
        pd.DataFrame(
            [[1.2, 1.1], [1.15, 1.05], [1.02, 1.01]],
            index=pd.Index([2020, "LDF", "Tail"]),
            columns=pd.Index([12, 24]),
        )
    )

    attachment, fit_periods = selected_cells_to_tail_actions(
        table,
        [(1, "12"), (2, "24")],
    )

    assert attachment == 24
    assert fit_periods == [12]
    assert toggle_tail_attachment(24, attachment) is None
    assert toggle_tail_attachment(None, attachment) == 24
    assert toggle_tail_fit_periods([12, 36], fit_periods) == [36]
    assert toggle_tail_fit_periods([], fit_periods) == [12]
    assert expand_fit_period_interval([12, 24], [12, 18, 24, 36]) == [12, 18, 24]


def test_a2a_color_styles_are_column_relative() -> None:
    table = build_link_ratio_display(
        pd.DataFrame(
            [[1.0, 5.0], [2.0, 3.0], [1.5, 4.0], [1.1, 1.1]],
            index=pd.Index([2020, 2021, "LDF", "Tail"]),
            columns=pd.Index([12, 24]),
        )
    )

    styles = build_a2a_color_styles(table)

    assert styles.loc["2020", "12"] != styles.loc["2021", "12"]
    assert styles.loc["2020", "24"] != styles.loc["2021", "24"]
    assert styles.loc["LDF", "12"] == ""
    assert styles.loc["Tail", "24"] == ""


def test_emergence_chart_uses_actual_origins_and_expected_line() -> None:
    columns = pd.MultiIndex.from_product([["Actual", "Expected"], [12, 24]])
    emergence = pd.DataFrame(
        [[0.5, 0.8, 0.4, 0.7], [0.4, 0.9, 0.4, 0.7]],
        index=pd.Index([2020, 2021]),
        columns=columns,
    )

    chart = build_emergence_chart(emergence)

    assert list(chart.columns) == ["2020", "2021", "Expected"]
    assert float(chart.loc[12, "Expected"]) == 0.4
