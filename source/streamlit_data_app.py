from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import (
    _derive_tail_fit_period,
    _derive_tail_projection_settings,
    _infer_months_per_development_period,
    build_reserving,
    build_sample_triangle,
    load_config,
)
from source.presentation.plot_builders import (
    append_data_triangle_average_rows,
    format_triangle_column_labels,
    format_triangle_row_labels,
)
from source.reserving import Reserving

if TYPE_CHECKING:
    from source.triangle import Triangle


METRICS = {
    "incurred": "Incurred",
    "paid": "Paid",
    "outstanding": "Outstanding",
    "premium": "Premium",
}
DIVISORS = {"none": "None", **METRICS}
PAGES = {
    "data": "Data",
    "chainladder": "Chainladder",
    "ave": "Actual vs Expected",
    "bf": "Bornhuetter-Ferguson",
    "results": "Results",
}
TAIL_CURVES = {
    "weibull": "Weibull",
    "exponential": "Exponential",
    "inverse_power": "Inverse power",
}
A2A_COLORS = ["#f5f8fc", "#e7eff9", "#d7e5f5", "#bdd2ec", "#9bbbe0"]


def get_data_tab_triangle(
    reserving: Reserving,
    metric: str,
    triangle_view: str = "cumulative",
) -> pd.DataFrame:
    triangle = reserving._triangle
    incurred = triangle.get_triangle("incurred")
    paid = triangle.get_triangle("paid")

    if triangle_view == "incremental":
        incurred = incurred.cum_to_incr()
        paid = paid.cum_to_incr()

    incurred_df = incurred["incurred"].to_frame()
    metric = (metric or "incurred").lower()

    if metric == "paid":
        return paid["paid"].to_frame()
    if metric == "premium":
        return incurred["Premium_selected"].to_frame().rename(
            columns={"Premium_selected": "premium"}
        )
    if metric == "outstanding":
        paid_df = paid["paid"].to_frame()
        incurred_df, paid_df = incurred_df.align(paid_df, join="outer")
        return incurred_df - paid_df
    return incurred_df


def build_data_tab_display(
    reserving: Reserving,
    metric: str,
    triangle_view: str,
    divisor: str,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    numerator = get_data_tab_triangle(reserving, metric, triangle_view)
    divisor = (divisor or "none").lower()

    if divisor == "none":
        return numerator, get_data_tab_triangle(reserving, "premium", triangle_view), False

    denominator = get_data_tab_triangle(reserving, divisor, "cumulative")
    numerator, denominator = numerator.align(denominator, join="outer")
    return numerator.div(denominator.where(denominator > 0)), denominator, True


def build_title(metric: str, triangle_view: str, divisor: str) -> str:
    title = f"Data Triangle - {METRICS.get(metric, 'Incurred')}"
    if divisor != "none":
        title = f"{title} / {DIVISORS.get(divisor, 'Unknown')}"
    view_label = "Incremental" if triangle_view == "incremental" else "Cumulative"
    return f"{title} ({view_label})"


def build_streamlit_table(
    triangle_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    table = append_data_triangle_average_rows(triangle_df, weights_df).copy()
    table.index = format_triangle_row_labels(table.index)
    table.columns = format_triangle_column_labels(table.columns)
    return table


def style_table(table: pd.DataFrame, ratio_mode: bool) -> pd.io.formats.style.Styler:
    number_format = "{:.3f}" if ratio_mode else "{:,.0f}"
    return table.style.format(number_format, na_rep="")


def load_sample_reserving() -> Reserving:
    config = load_config()
    triangle = build_sample_triangle()
    return build_reserving(triangle, config=config)


def load_sample_triangle() -> Triangle:
    return build_sample_triangle()


def parse_dev_label(label: object) -> int | None:
    try:
        text = str(label).split("-")[0]
        return int(text)
    except (TypeError, ValueError):
        return None


def normalize_drops(rows: list[list[str | int]] | None) -> list[list[str | int]]:
    normalized = []
    for item in rows or []:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            drop = [str(item[0]), int(item[1])]
        except (TypeError, ValueError):
            continue
        if drop not in normalized:
            normalized.append(drop)
    return sorted(normalized, key=lambda item: (item[0], item[1]))


def build_link_ratio_display(link_ratios: pd.DataFrame) -> pd.DataFrame:
    table = link_ratios.copy()
    table.index = format_triangle_row_labels(table.index)
    table.columns = format_triangle_column_labels(table.columns)
    return table


def build_a2a_factor_table(reserving: Reserving) -> pd.DataFrame:
    raw = reserving._triangle.get_triangle().link_ratio["incurred"].to_frame()
    fitted = reserving.get_triangle_heatmap_data()["link_ratios"]
    special_rows = [row for row in ["LDF", "Tail"] if row in fitted.index]
    return pd.concat([raw, fitted.loc[special_rows]])


def build_link_ratio_table(
    link_ratios: pd.DataFrame,
    drops: list[list[str | int]],
    tail_attachment_age: int | None = None,
    tail_fit_period_selection: list[int] | None = None,
    colorcoded: bool = False,
) -> pd.io.formats.style.Styler:
    table = build_link_ratio_display(link_ratios)
    selected = {(str(origin), int(dev)) for origin, dev in normalize_drops(drops)}
    available_ages = [
        age for age in (parse_dev_label(column) for column in table.columns) if age is not None
    ]
    tail_fit_ages = expand_fit_period_interval(
        tail_fit_period_selection or [],
        available_ages,
    )
    styles = build_a2a_color_styles(table) if colorcoded else pd.DataFrame(
        "",
        index=table.index,
        columns=table.columns,
    )

    for origin, dev in selected:
        for column in table.columns:
            if parse_dev_label(column) == dev and origin in styles.index:
                styles.loc[origin, column] = (
                    "background-color: #ffe8cc; color: #8a4b00; "
                    "font-weight: 700; text-decoration: line-through"
                )

    for column in table.columns:
        dev = parse_dev_label(column)
        if dev is None:
            continue
        if "Tail" in styles.index and tail_attachment_age == dev:
            styles.loc["Tail", column] = "background-color: #d9f99d; font-weight: 700"
        if "LDF" in styles.index and dev in tail_fit_ages:
            styles.loc["LDF", column] = "background-color: #dbeafe; font-weight: 700"

    return table.style.format("{:.3f}", na_rep="").apply(lambda _: styles, axis=None)


def build_a2a_color_styles(table: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    origin_rows = [row for row in table.index if row not in {"LDF", "Tail"}]
    if not origin_rows:
        return styles

    for column in table.columns:
        values = pd.to_numeric(table.loc[origin_rows, column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        col_min = float(valid.min())
        col_max = float(valid.max())
        for row, value in values.items():
            if pd.isna(value):
                continue
            normalized = 0.5 if col_max == col_min else (float(value) - col_min) / (
                col_max - col_min
            )
            color_index = min(
                int(normalized * (len(A2A_COLORS) - 1)),
                len(A2A_COLORS) - 1,
            )
            styles.loc[row, column] = f"background-color: {A2A_COLORS[color_index]}"
    return styles


def selected_cells_to_drops(
    table: pd.DataFrame,
    cells: list[object] | None,
) -> list[list[str | int]]:
    drops = []
    for cell in cells or []:
        row_index, column = parse_selected_cell(cell)
        if row_index is None or column is None:
            continue
        if row_index >= len(table.index):
            continue
        origin = str(table.index[row_index])
        dev = parse_dev_label(column)
        if origin in {"LDF", "Tail"} or dev is None or pd.isna(table.loc[origin, column]):
            continue
        drops.append([origin, dev])
    return normalize_drops(drops)


def selected_cells_to_tail_actions(
    table: pd.DataFrame,
    cells: list[object] | None,
) -> tuple[int | None, list[int]]:
    attachment_age = None
    fit_periods = []
    for cell in cells or []:
        row_index, column = parse_selected_cell(cell)
        if row_index is None or column is None or row_index >= len(table.index):
            continue
        row_label = str(table.index[row_index])
        dev = parse_dev_label(column)
        if dev is None or pd.isna(table.loc[row_label, column]):
            continue
        if row_label == "Tail":
            attachment_age = dev
        elif row_label == "LDF" and dev not in fit_periods:
            fit_periods.append(dev)
    return attachment_age, fit_periods


def toggle_tail_attachment(current: int | None, selected: int | None) -> int | None:
    if selected is None:
        return current
    return None if current == selected else selected


def toggle_tail_fit_periods(current: list[int], selected: list[int]) -> list[int]:
    values = []
    for item in current or []:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value not in values:
            values.append(value)
    for value in selected:
        if value in values:
            values.remove(value)
        else:
            values.append(value)
    return sorted(values)


def expand_fit_period_interval(
    selection: list[int],
    available_ages: list[int] | None = None,
) -> list[int]:
    values = sorted({int(value) for value in selection})
    if len(values) <= 1:
        return values
    lower, upper = values[0], values[-1]
    if available_ages is None:
        return list(range(lower, upper + 1))
    return [age for age in available_ages if lower <= age <= upper]


def parse_selected_cell(cell: object) -> tuple[int | None, object | None]:
    if isinstance(cell, dict):
        row = cell.get("row")
        column = cell.get("column")
    elif isinstance(cell, (list, tuple)) and len(cell) == 2:
        row, column = cell
    else:
        return None, None
    try:
        return int(row), column
    except (TypeError, ValueError):
        return None, None


def toggle_drops(
    existing: list[list[str | int]],
    selected: list[list[str | int]],
) -> list[list[str | int]]:
    current = normalize_drops(existing)
    selected = normalize_drops(selected)
    if not selected:
        return current
    if all(drop in current for drop in selected):
        return [drop for drop in current if drop not in selected]
    return normalize_drops(current + [drop for drop in selected if drop not in current])


def build_chainladder_reserving(
    triangle: Triangle,
    *,
    average: str,
    tail_curve: str,
    tail_projection_months: int,
    tail_attachment_age: int | None,
    tail_fit_period_selection: list[int],
    drops: list[list[str | int]],
) -> Reserving:
    config = load_config()
    reserving = Reserving(triangle)
    months_per_dev = _infer_months_per_development_period(
        triangle,
        granularity=config.get_granularity() if config is not None else None,
    )
    extrap_periods, projection_period = _derive_tail_projection_settings(
        tail_projection_months=tail_projection_months,
        months_per_dev=months_per_dev,
    )
    reserving.set_development(
        average=average,
        drop=[(str(origin), int(dev)) for origin, dev in normalize_drops(drops)] or None,
    )
    reserving.set_tail(
        curve=tail_curve,
        attachment_age=tail_attachment_age,
        extrap_periods=extrap_periods,
        projection_period=projection_period,
        fit_period=_derive_tail_fit_period(tail_fit_period_selection),
    )
    reserving.set_bornhuetter_ferguson(apriori=0.6)
    reserving.reserve(final_ultimate="chainladder")
    return reserving


def get_development_ages(link_ratios: pd.DataFrame) -> list[int]:
    ages = []
    for column in link_ratios.columns:
        age = parse_dev_label(column)
        if age is not None and age not in ages:
            ages.append(age)
    return ages


def build_emergence_chart(emergence_pattern: pd.DataFrame) -> pd.DataFrame:
    actual = emergence_pattern["Actual"].T.copy()
    actual.columns = format_triangle_row_labels(actual.columns)
    expected = emergence_pattern["Expected"].iloc[0].rename("Expected")
    return pd.concat([actual, expected], axis=1)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Reserving Studio", layout="wide")

    st.sidebar.title("reserving-studio")
    page = st.sidebar.radio(
        "Workflow",
        options=list(PAGES),
        format_func=PAGES.get,
    )

    if page == "chainladder":
        st.title("Chainladder")
        triangle = st.cache_resource(load_sample_triangle)()

        st.session_state.setdefault("chainladder_drops", [])
        st.session_state.setdefault("chainladder_tail_attachment_age", None)
        st.session_state.setdefault("chainladder_tail_fit_periods", [])
        st.session_state.setdefault("chainladder_factor_table_version", 0)
        col_avg, col_tail, col_tail_len, col_color = st.columns(4)
        average = col_avg.selectbox("Average method", ["volume", "simple"])
        tail_curve = col_tail.selectbox(
            "Tail method",
            list(TAIL_CURVES),
            format_func=TAIL_CURVES.get,
        )
        tail_projection_months = col_tail_len.selectbox(
            "Tail length (months)",
            list(range(0, 241, 12)),
        )
        colorcoded = col_color.toggle("Color-code A2A", value=True)

        reserving = build_chainladder_reserving(
            triangle,
            average=average,
            tail_curve=tail_curve,
            tail_projection_months=tail_projection_months,
            tail_attachment_age=st.session_state.chainladder_tail_attachment_age,
            tail_fit_period_selection=st.session_state.chainladder_tail_fit_periods,
            drops=st.session_state.chainladder_drops,
        )
        link_ratios = build_a2a_factor_table(reserving)

        st.subheader("A2A factor triangle")
        st.caption(
            "Select factor cells to toggle drops. Dropped factors remain visible, "
            "Tail row cells set attachment age; LDF row cells set tail fit periods."
        )
        factor_table = build_link_ratio_display(link_ratios)
        event = st.dataframe(
            build_link_ratio_table(
                link_ratios,
                st.session_state.chainladder_drops,
                st.session_state.chainladder_tail_attachment_age,
                st.session_state.chainladder_tail_fit_periods,
                colorcoded,
            ),
            width="stretch",
            height=520,
            on_select="rerun",
            selection_mode="multi-cell",
            key=f"chainladder_factor_table_{st.session_state.chainladder_factor_table_version}",
        )
        selected_drops = selected_cells_to_drops(
            factor_table,
            event.selection.cells,
        )
        selected_attachment, selected_fit_periods = selected_cells_to_tail_actions(
            factor_table,
            event.selection.cells,
        )
        if selected_drops or selected_attachment is not None or selected_fit_periods:
            st.session_state.chainladder_drops = toggle_drops(
                st.session_state.chainladder_drops,
                selected_drops,
            )
            st.session_state.chainladder_tail_attachment_age = toggle_tail_attachment(
                st.session_state.chainladder_tail_attachment_age,
                selected_attachment,
            )
            st.session_state.chainladder_tail_fit_periods = toggle_tail_fit_periods(
                st.session_state.chainladder_tail_fit_periods,
                selected_fit_periods,
            )
            st.session_state.chainladder_factor_table_version += 1
            st.rerun()

        if st.button("Clear selections"):
            st.session_state.chainladder_drops = []
            st.session_state.chainladder_tail_attachment_age = None
            st.session_state.chainladder_tail_fit_periods = []
            st.session_state.chainladder_factor_table_version += 1
            st.rerun()

        st.subheader("Emergence")
        st.line_chart(build_emergence_chart(reserving.get_emergence_pattern()))
        return

    if page != "data":
        st.title(PAGES[page])
        st.info("This Streamlit prototype currently implements only the Data tab.")
        return

    st.title("Data")
    st.caption("Small Streamlit prototype of the current Dash Data tab.")

    reserving = st.cache_resource(load_sample_reserving)()

    col_metric, col_view, col_divisor = st.columns(3)
    metric = col_metric.selectbox(
        "Triangle metric",
        options=list(METRICS),
        format_func=METRICS.get,
    )
    incremental = col_view.toggle("Incremental view", value=False)
    triangle_view = "incremental" if incremental else "cumulative"
    divisor = col_divisor.selectbox(
        "In relation to",
        options=list(DIVISORS),
        format_func=DIVISORS.get,
    )

    triangle_df, weights_df, ratio_mode = build_data_tab_display(
        reserving,
        metric,
        triangle_view,
        divisor,
    )
    table = build_streamlit_table(triangle_df, weights_df)

    st.subheader(build_title(metric, triangle_view, divisor))
    st.dataframe(style_table(table, ratio_mode), width="stretch", height=720)


if __name__ == "__main__":
    main()
