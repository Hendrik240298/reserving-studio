from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from source.reserving import Reserving


def format_triangle_row_labels(index: pd.Index) -> list[str]:
    labels: list[str] = []
    for value in index:
        if isinstance(value, str):
            labels.append(value)
        elif hasattr(value, "year"):
            labels.append(str(value.year))
        else:
            labels.append(str(value)[:4])
    return labels


def format_triangle_column_labels(columns: pd.Index) -> list[str]:
    labels: list[str] = []
    for value in columns:
        if isinstance(value, tuple) and len(value) > 0:
            labels.append(str(value[-1]))
        else:
            labels.append(str(value))
    return labels


def append_data_triangle_average_rows(
    triangle_df: pd.DataFrame,
    weights_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if triangle_df is None or triangle_df.empty:
        return triangle_df

    df_values = triangle_df.apply(pd.to_numeric, errors="coerce")
    simple_average = df_values.mean(axis=0, skipna=True)

    if weights_df is None:
        premium_weights = df_values.copy() * np.nan
    else:
        premium_weights = weights_df.apply(pd.to_numeric, errors="coerce")
    aligned_values, aligned_weights = df_values.align(
        premium_weights,
        join="left",
    )

    weighted_average_values: dict[object, float] = {}
    for col in aligned_values.columns:
        values_col = pd.to_numeric(aligned_values[col], errors="coerce")
        weights_col = pd.to_numeric(aligned_weights[col], errors="coerce")
        valid = values_col.notna() & weights_col.notna() & (weights_col > 0)
        if valid.any():
            weighted_average_values[col] = float(
                np.average(values_col[valid], weights=weights_col[valid])
            )
        else:
            weighted_average_values[col] = np.nan

    weighted_average = pd.Series(weighted_average_values)

    df_with_averages = df_values.copy()
    df_with_averages.loc["Simple Avg"] = simple_average
    df_with_averages.loc["Weighted Avg"] = weighted_average
    return df_with_averages


def plot_data_triangle_table(
    *,
    triangle_df: pd.DataFrame,
    title: str,
    weights_df: Optional[pd.DataFrame],
    ratio_mode: bool,
    font_family: str,
    figure_font_size: int,
    figure_title_font_size: int,
    table_header_font_size: int,
    table_cell_font_size: int,
    alert_annotation_font_size: int,
    color_text: str,
    color_surface: str,
    color_border: str,
) -> go.Figure:
    if triangle_df is None or triangle_df.empty:
        return go.Figure(
            layout=go.Layout(
                title=f"{title} - No data available",
                annotations=[
                    dict(
                        text="Triangle data not available.",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(
                            color="red",
                            size=alert_annotation_font_size,
                            family=font_family,
                        ),
                    )
                ],
            )
        )

    df_display = append_data_triangle_average_rows(triangle_df, weights_df)
    row_labels = format_triangle_row_labels(df_display.index)
    col_labels = format_triangle_column_labels(df_display.columns)

    def _format_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric_value):
            return ""
        if ratio_mode:
            return f"{numeric_value:.3f}"
        return f"{numeric_value:,.0f}"

    table_values: list[list[str]] = [row_labels]
    for col in df_display.columns:
        table_values.append([_format_cell(v) for v in df_display[col]])

    headers = ["UWY"] + col_labels

    all_column_lengths = [len(str(text)) for text in headers if str(text)]
    for column_values in table_values:
        all_column_lengths.extend(len(str(text)) for text in column_values if str(text))

    data_column_lengths = [len(str(text)) for text in headers[1:] if str(text)]
    for column_values in table_values[1:]:
        data_column_lengths.extend(
            len(str(text)) for text in column_values if str(text)
        )

    max_all_text_length = max(all_column_lengths, default=0)
    max_data_text_length = max(data_column_lengths, default=0)
    row_label_column_width = max(72, min(220, 18 + max_all_text_length * 9))
    standard_data_column_width = max(58, min(180, 12 + max_data_text_length * 7))
    empty_data_column_width = max(16, standard_data_column_width // 4)

    column_widths = [row_label_column_width] * len(headers)
    if headers:
        # Keep the first row-label column sizing behavior unchanged.
        column_widths[0] = row_label_column_width

    for column_index in range(1, len(headers)):
        column_values = table_values[column_index]
        has_any_data = any(bool(str(value)) for value in column_values)
        column_widths[column_index] = (
            standard_data_column_width if has_any_data else empty_data_column_width
        )

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=column_widths,
                header=dict(
                    values=headers,
                    fill_color="#f2f5f9",
                    align="center",
                    line_color=color_border,
                    font=dict(
                        color=color_text,
                        size=table_header_font_size,
                        family=font_family,
                    ),
                    height=28,
                ),
                cells=dict(
                    values=table_values,
                    fill_color=color_surface,
                    align="center",
                    line_color=color_border,
                    font=dict(
                        color=color_text,
                        size=table_cell_font_size,
                        family=font_family,
                    ),
                    height=26,
                ),
            )
        ]
    )

    table_width = sum(column_widths)

    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(color=color_text, size=figure_font_size, family=font_family),
        title_font=dict(
            color=color_text,
            size=figure_title_font_size,
            family=font_family,
        ),
        hoverlabel=dict(
            bgcolor=color_surface,
            bordercolor=color_border,
            font=dict(
                color=color_text,
                size=figure_font_size,
                family=font_family,
            ),
        ),
        margin=dict(l=8, r=8, t=48, b=8),
        width=max(900, table_width + 16),
        height=min(760, 170 + len(df_display.index) * 28),
        autosize=False,
        uirevision="static",
    )
    return fig


def plot_emergence(
    *,
    emergence_pattern,
    title: str,
    font_family: str,
    figure_font_size: int,
    figure_title_font_size: int,
    alert_annotation_font_size: int,
    color_text: str,
    color_surface: str,
    color_border: str,
) -> go.Figure:
    if emergence_pattern is None:
        return go.Figure(
            layout=go.Layout(
                title=f"{title} - No data available",
                annotations=[
                    dict(
                        text="Data not available. Call reserve() first.",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(
                            color="red",
                            size=alert_annotation_font_size,
                            family=font_family,
                        ),
                    )
                ],
            )
        )

    fig = go.Figure()
    actual_data = emergence_pattern["Actual"]
    for idx in actual_data.index:
        year = str(idx)[:4] if hasattr(idx, "year") else str(idx).split("-")[0]
        fig.add_trace(
            go.Scatter(
                x=actual_data.columns,
                y=actual_data.loc[idx],
                mode="lines+markers",
                name=year,
                showlegend=True,
            )
        )

    expected_data = emergence_pattern["Expected"]
    fig.add_trace(
        go.Scatter(
            x=expected_data.columns,
            y=expected_data.iloc[0],
            mode="lines",
            name="Expected",
            line=dict(color="black", width=3, dash="dash"),
            showlegend=True,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Development (Months)",
        yaxis_title="% of Ultimate",
        template="plotly_white",
        font=dict(color=color_text, size=figure_font_size, family=font_family),
        title_font=dict(
            color=color_text,
            size=figure_title_font_size,
            family=font_family,
        ),
        hoverlabel=dict(
            bgcolor=color_surface,
            bordercolor=color_border,
            font=dict(
                color=color_text,
                size=figure_font_size,
                family=font_family,
            ),
        ),
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        height=600,
        autosize=True,
        uirevision="static",
    )

    return fig


def plot_reserving_results_table(
    *,
    results_df,
    title: str,
    font_family: str,
    figure_font_size: int,
    figure_title_font_size: int,
    table_header_font_size: int,
    table_cell_font_size: int,
    alert_annotation_font_size: int,
    color_text: str,
    color_surface: str,
    color_border: str,
) -> go.Figure:
    if results_df is None or len(results_df) == 0:
        return go.Figure(
            layout=go.Layout(
                title=f"{title} - No data available",
                annotations=[
                    dict(
                        text="Reserving results not available. Call reserve() first.",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(
                            color="red",
                            size=alert_annotation_font_size,
                            family=font_family,
                        ),
                    )
                ],
            )
        )

    df_display = results_df.copy()
    uwy_years = []
    for idx in df_display.index:
        if hasattr(idx, "year"):
            uwy_years.append(str(idx.year))
        else:
            uwy_years.append(str(idx)[:4])

    incurred_loss_ratios = []
    for inc, prem in zip(df_display["incurred"], df_display["Premium"]):
        if pd.isna(prem) or prem == 0 or pd.isna(inc):
            incurred_loss_ratios.append("N/A")
        else:
            incurred_loss_ratios.append(f"{(inc / prem):.2%}")

    ibnr_values = df_display["ultimate"] - df_display["incurred"]
    header_values = [
        "UWY",
        "Incurred (€)",
        "Premium (€)",
        "Incurred Loss Ratio",
        "CL Ultimate (€)",
        "CL Loss Ratio",
        "BF Ultimate (€)",
        "BF Loss Ratio",
        "Selected Ultimate (€)",
        "IBNR (€)",
    ]
    cell_values = [
        uwy_years,
        [f"{val:,.0f}" for val in df_display["incurred"]],
        [f"{val:,.0f}" for val in df_display["Premium"]],
        incurred_loss_ratios,
        [f"{val:,.0f}" for val in df_display["cl_ultimate"]],
        [f"{val:.2%}" for val in df_display["cl_loss_ratio"]],
        [f"{val:,.0f}" for val in df_display["bf_ultimate"]],
        [f"{val:.2%}" for val in df_display["bf_loss_ratio"]],
        [f"{val:,.0f}" for val in df_display["ultimate"]],
        [f"{val:,.0f}" for val in ibnr_values],
    ]

    header_colors = [
        "#2c3e50",
        "#34495e",
        "#34495e",
        "#34495e",
        "#3498db",
        "#3498db",
        "#e67e22",
        "#e67e22",
        "#27ae60",
        "#16a085",
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    fill_color=header_colors,
                    align="center",
                    font=dict(
                        color="white",
                        size=table_header_font_size,
                        family=font_family,
                    ),
                    height=35,
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=[
                        [
                            "#f8f9fa" if i % 2 == 0 else "white"
                            for i in range(len(uwy_years))
                        ]
                    ],
                    align=[
                        "center",
                        "right",
                        "right",
                        "center",
                        "right",
                        "center",
                        "right",
                        "center",
                        "right",
                        "right",
                    ],
                    font=dict(
                        color="black",
                        size=table_cell_font_size,
                        family=font_family,
                    ),
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(color=color_text, size=figure_font_size, family=font_family),
        title_font=dict(
            color=color_text,
            size=figure_title_font_size,
            family=font_family,
        ),
        hoverlabel=dict(
            bgcolor=color_surface,
            bordercolor=color_border,
            font=dict(
                color=color_text,
                size=figure_font_size,
                family=font_family,
            ),
        ),
        height=min(650, 200 + len(df_display) * 28),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    return fig


def hash_dataframe(frame: pd.DataFrame) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(frame.shape).encode("utf-8"))
    for value in frame.index:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"|")
    for value in frame.columns:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"|")
    values = np.ascontiguousarray(frame.to_numpy(dtype=float, copy=False))
    hasher.update(values.tobytes())
    return hasher.hexdigest()


def build_heatmap_core_cache_key(
    *,
    segment_key: str,
    triangle_data: pd.DataFrame,
    incurred_data: pd.DataFrame,
    premium_data: pd.DataFrame,
) -> str:
    payload = {
        "segment": segment_key,
        "triangle": hash_dataframe(triangle_data),
        "incurred": hash_dataframe(incurred_data),
        "premium": hash_dataframe(premium_data),
    }
    return json.dumps(payload, sort_keys=True)


def format_millions_array(values: np.ndarray) -> np.ndarray:
    formatted = np.full(values.shape, "", dtype=object)
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return formatted

    abs_values = np.abs(values)
    millions_mask = valid_mask & (abs_values >= 1_000_000)
    thousands_mask = valid_mask & (abs_values >= 1_000) & (~millions_mask)
    units_mask = valid_mask & (~millions_mask) & (~thousands_mask)

    if np.any(millions_mask):
        formatted[millions_mask] = np.char.mod(
            "%.2fm",
            values[millions_mask] / 1_000_000,
        )
    if np.any(thousands_mask):
        formatted[thousands_mask] = np.char.mod(
            "%.2fk",
            values[thousands_mask] / 1_000,
        )
    if np.any(units_mask):
        formatted[units_mask] = np.char.mod("%.0f", values[units_mask])
    return formatted


def build_triangle_customdata(
    *,
    expanded_triangle: pd.DataFrame,
    incurred_data: pd.DataFrame,
    premium_data: pd.DataFrame,
) -> np.ndarray:
    row_keys: list[object] = []
    year_to_incurred_index: dict[int, object] = {}
    for idx in incurred_data.index:
        if hasattr(idx, "year"):
            year_to_incurred_index.setdefault(int(idx.year), idx)

    for tri_idx in expanded_triangle.index:
        if isinstance(tri_idx, str):
            row_keys.append(None)
            continue
        if hasattr(tri_idx, "year"):
            year = int(tri_idx.year)
        else:
            tri_text = str(tri_idx)
            try:
                year = int(tri_text[:4])
            except (TypeError, ValueError):
                row_keys.append(None)
                continue
        row_keys.append(year_to_incurred_index.get(year))

    invalid_column = "__invalid_dev_label__"
    left_columns: list[object] = []
    right_columns: list[object] = []
    valid_dev_columns = np.zeros(len(expanded_triangle.columns), dtype=bool)
    for j, col in enumerate(expanded_triangle.columns):
        try:
            parts = str(col).split("-")
            left_columns.append(int(parts[0]))
            right_columns.append(int(parts[1]))
            valid_dev_columns[j] = True
        except (TypeError, ValueError, IndexError):
            left_columns.append(invalid_column)
            right_columns.append(invalid_column)

    aligned_incurred = incurred_data.reindex(index=row_keys)
    aligned_premium = premium_data.reindex(index=row_keys)

    left_values = aligned_incurred.reindex(columns=left_columns).to_numpy(dtype=float)
    right_values = aligned_incurred.reindex(columns=right_columns).to_numpy(dtype=float)
    premium_values = aligned_premium.reindex(columns=left_columns).to_numpy(dtype=float)

    column_mask = np.broadcast_to(
        valid_dev_columns.reshape(1, -1),
        left_values.shape,
    )
    has_left = (~np.isnan(left_values)) & column_mask
    has_right = (~np.isnan(right_values)) & column_mask
    has_premium = (~np.isnan(premium_values)) & column_mask

    left_display = format_millions_array(left_values)
    right_display = format_millions_array(right_values)
    premium_display = format_millions_array(premium_values)

    incurred_display = np.full(left_values.shape, "", dtype=object)
    both_mask = has_left & has_right
    left_only_mask = has_left & (~has_right)
    if np.any(both_mask):
        merged = np.char.add(
            np.char.add(left_display.astype(str), " --> "),
            right_display.astype(str),
        )
        incurred_display[both_mask] = merged[both_mask]
    if np.any(left_only_mask):
        incurred_display[left_only_mask] = left_display[left_only_mask]

    premium_strings = np.full(premium_values.shape, "", dtype=object)
    if np.any(has_premium):
        premium_strings[has_premium] = premium_display[has_premium]

    return np.stack([incurred_display, premium_strings], axis=-1)


def build_heatmap_core(
    *,
    triangle_data: pd.DataFrame,
    incurred_data: pd.DataFrame,
    premium_data: pd.DataFrame,
    reserving: Reserving,
) -> dict:
    raw_link_ratios = (
        reserving._triangle.get_triangle().link_ratio["incurred"].to_frame()
    )
    raw_link_ratios = raw_link_ratios.reindex(
        index=triangle_data.index,
        columns=triangle_data.columns,
    )
    dropped_mask = triangle_data.isna() & raw_link_ratios.notna()
    for summary_row in ("LDF", "Tail"):
        if summary_row in dropped_mask.index:
            dropped_mask.loc[summary_row, :] = False
    expanded_triangle = triangle_data.fillna(raw_link_ratios)

    link_ratio_row_positions = list(range(max(len(expanded_triangle.index) - 2, 0)))
    ldf_tail_rows = expanded_triangle.index[-2:]

    normalized_data = expanded_triangle.copy()
    for col_idx in range(len(expanded_triangle.columns)):
        col_series = expanded_triangle.iloc[link_ratio_row_positions, col_idx]
        col_data = col_series.dropna()
        if len(col_data) == 0:
            continue
        col_min = col_data.min()
        col_max = col_data.max()
        if col_max > col_min:
            normalized_values = (col_series - col_min) / (col_max - col_min)
            normalized_data.iloc[link_ratio_row_positions, col_idx] = (
                normalized_values.to_numpy()
            )
        else:
            normalized_data.iloc[link_ratio_row_positions, col_idx] = 0.5

    ldf_tail_data = expanded_triangle.loc[ldf_tail_rows]
    ldf_tail_min = ldf_tail_data.min().min()
    ldf_tail_max = ldf_tail_data.max().max()

    if ldf_tail_max > ldf_tail_min and ldf_tail_min > 0:
        log_min = np.log(ldf_tail_min)
        log_max = np.log(ldf_tail_max)
        denom = log_max - log_min
        for row in ldf_tail_rows:
            row_values = expanded_triangle.loc[row]
            row_mask = row_values.notna()
            if row_mask.any():
                normalized_data.loc[row, row_mask] = (
                    np.log(row_values[row_mask]) - log_min
                ) / denom
    else:
        for row in ldf_tail_rows:
            row_values = expanded_triangle.loc[row]
            row_mask = row_values.notna()
            if row_mask.any():
                normalized_data.loc[row, row_mask] = 0.5

    text_values = expanded_triangle.round(3).astype(str).to_numpy(dtype=object)
    text_values = np.where(text_values == "nan", "", text_values)

    z_data = normalized_data.to_numpy(dtype=float, copy=True)
    expanded_values = expanded_triangle.to_numpy(dtype=float)
    z_data = np.where(np.isnan(expanded_values), np.nan, z_data)

    customdata = build_triangle_customdata(
        expanded_triangle=expanded_triangle,
        incurred_data=incurred_data,
        premium_data=premium_data,
    )

    return {
        "expanded_triangle": expanded_triangle,
        "dropped_mask": dropped_mask,
        "z_data": z_data,
        "text_values": text_values,
        "customdata": customdata,
    }


def plot_triangle_heatmap_clean(
    *,
    triangle_data,
    incurred_data,
    premium_data,
    reserving: Reserving,
    title: str,
    tail_attachment_age: Optional[int],
    tail_fit_period_selection: Optional[list[int]],
    parse_dev_label: Callable[[object], Optional[int]],
    derive_tail_fit_period: Callable[
        [Optional[list[int]]], Optional[tuple[int, Optional[int]]]
    ],
    font_family: str,
    figure_font_size: int,
    figure_title_font_size: int,
    heatmap_text_font_size: int,
    color_text: str,
    color_surface: str,
    color_border: str,
    core_payload: Optional[dict] = None,
) -> tuple[go.Figure, dict]:
    render_started = time.perf_counter()
    payload = core_payload
    if payload is None:
        payload = build_heatmap_core(
            triangle_data=triangle_data,
            incurred_data=incurred_data,
            premium_data=premium_data,
            reserving=reserving,
        )

    expanded_triangle = payload["expanded_triangle"]
    dropped_mask = payload["dropped_mask"]
    z_data = payload["z_data"]
    text_values = payload["text_values"]
    customdata = payload["customdata"]

    x_labels = [str(value) for value in expanded_triangle.columns]
    y_labels = [
        str(idx)[:4] if idx not in ["LDF", "Tail"] else str(idx)
        for idx in expanded_triangle.index
    ]

    n_cols = len(triangle_data.columns)
    n_rows = len(triangle_data.index)
    table_width = 130 + (62 * n_cols)
    table_height = min(760, 170 + (n_rows * 28))
    header_bg_value = -0.2

    z_with_headers = np.full(
        (len(y_labels) + 1, len(x_labels) + 1),
        header_bg_value,
        dtype=float,
    )
    z_with_headers[1:, 1:] = z_data

    text_with_headers = np.full(
        (len(y_labels) + 1, len(x_labels) + 1),
        "",
        dtype=object,
    )
    text_with_headers[0, 0] = "<b>UWY</b>"
    text_with_headers[0, 1:] = [f"<b>{label}</b>" for label in x_labels]
    text_with_headers[1:, 0] = [f"<b>{label}</b>" for label in y_labels]
    text_with_headers[1:, 1:] = text_values

    custom_with_headers = np.empty(
        (len(y_labels) + 1, len(x_labels) + 1, 2),
        dtype=object,
    )
    custom_with_headers[:] = ""
    custom_with_headers[1:, 1:] = customdata

    fig = go.Figure(
        data=go.Heatmap(
            z=z_with_headers,
            x=["UWY"] + x_labels,
            y=["Dev"] + y_labels,
            text=text_with_headers,
            texttemplate="%{text}",
            customdata=custom_with_headers,
            hovertemplate="UWY: %{y}<br>Dev Period: %{x}<br>Link Ratio: %{text}<br>Incurred: %{customdata[0]}<br>Premium: %{customdata[1]}<extra></extra>",
            showscale=False,
            zmin=header_bg_value,
            zmax=1,
            hoverongaps=False,
            colorscale=[
                [0.0, "#f2f5f9"],
                [0.1666, "#f2f5f9"],
                [0.1667, "#f5f8fc"],
                [0.375, "#e7eff9"],
                [0.5833, "#d7e5f5"],
                [0.7916, "#bdd2ec"],
                [1.0, "#9bbbe0"],
            ],
            textfont={
                "size": heatmap_text_font_size,
                "family": font_family,
                "color": color_text,
            },
            xgap=1,
            ygap=1,
        )
    )

    shape_defs: list[dict[str, object]] = []

    if tail_attachment_age is not None:
        try:
            tail_row_pos = expanded_triangle.index.get_loc("Tail") + 1
        except KeyError:
            tail_row_pos = None

        if tail_row_pos is not None:
            selected_col = None
            for col in expanded_triangle.columns:
                col_age = parse_dev_label(col)
                if col_age == tail_attachment_age:
                    selected_col = col
                    break

            if selected_col is not None:
                col_pos = expanded_triangle.columns.get_loc(selected_col) + 1
                shape_defs.append(
                    {
                        "type": "rect",
                        "x0": col_pos - 0.5,
                        "y0": tail_row_pos - 0.5,
                        "x1": col_pos + 0.5,
                        "y1": tail_row_pos + 0.5,
                        "line": {"color": "black", "width": 2},
                        "fillcolor": "rgba(0,0,0,0)",
                        "xref": "x",
                        "yref": "y",
                    }
                )

    fit_period = derive_tail_fit_period(tail_fit_period_selection)
    if fit_period is not None:
        try:
            ldf_row_pos = expanded_triangle.index.get_loc("LDF") + 1
        except KeyError:
            ldf_row_pos = None

        if ldf_row_pos is not None:
            lower_value, upper_value = fit_period
            lower_col = None
            upper_col = None
            for col in expanded_triangle.columns:
                col_age = parse_dev_label(col)
                if col_age == lower_value:
                    lower_col = col
                if upper_value is not None and col_age == upper_value:
                    upper_col = col

            if lower_col is not None:
                lower_pos = expanded_triangle.columns.get_loc(lower_col) + 1
                if upper_value is None or upper_col is None:
                    shape_defs.append(
                        {
                            "type": "rect",
                            "x0": lower_pos - 0.5,
                            "y0": ldf_row_pos - 0.5,
                            "x1": lower_pos + 0.5,
                            "y1": ldf_row_pos + 0.5,
                            "line": {"color": "black", "width": 2},
                            "fillcolor": "rgba(0,0,0,0)",
                            "xref": "x",
                            "yref": "y",
                        }
                    )
                else:
                    upper_pos = expanded_triangle.columns.get_loc(upper_col) + 1
                    start_pos = min(lower_pos, upper_pos)
                    end_pos = max(lower_pos, upper_pos)
                    shape_defs.append(
                        {
                            "type": "rect",
                            "x0": start_pos - 0.5,
                            "y0": ldf_row_pos - 0.5,
                            "x1": end_pos + 0.5,
                            "y1": ldf_row_pos + 0.5,
                            "line": {"color": "black", "width": 2},
                            "fillcolor": "rgba(0,0,0,0)",
                            "xref": "x",
                            "yref": "y",
                        }
                    )
                    shape_defs.append(
                        {
                            "type": "rect",
                            "x0": lower_pos - 0.5,
                            "y0": ldf_row_pos - 0.5,
                            "x1": lower_pos + 0.5,
                            "y1": ldf_row_pos + 0.5,
                            "line": {"color": "black", "width": 3},
                            "fillcolor": "rgba(0,0,0,0)",
                            "xref": "x",
                            "yref": "y",
                        }
                    )
                    shape_defs.append(
                        {
                            "type": "rect",
                            "x0": upper_pos - 0.5,
                            "y0": ldf_row_pos - 0.5,
                            "x1": upper_pos + 0.5,
                            "y1": ldf_row_pos + 0.5,
                            "line": {"color": "black", "width": 3},
                            "fillcolor": "rgba(0,0,0,0)",
                            "xref": "x",
                            "yref": "y",
                        }
                    )

    fig.update_layout(
        paper_bgcolor=color_surface,
        plot_bgcolor=color_surface,
        font={
            "family": font_family,
            "color": color_text,
            "size": figure_font_size,
        },
        title_font={
            "family": font_family,
            "color": color_text,
            "size": figure_title_font_size,
        },
        hoverlabel={
            "bgcolor": color_surface,
            "bordercolor": color_border,
            "font": {
                "family": font_family,
                "color": color_text,
                "size": figure_font_size,
            },
        },
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        width=max(900, table_width + 16),
        height=table_height,
        autosize=False,
        xaxis_title=None,
        yaxis_title=None,
        shapes=shape_defs,
        uirevision="static",
    )
    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        side="top",
        range=[-0.5, min(143, n_cols) + 0.5],
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        autorange="reversed",
    )

    _ = (time.perf_counter() - render_started) * 1000
    return fig, payload
