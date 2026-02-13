from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.presentation.plot_builders import (
    build_heatmap_core,
    build_heatmap_core_cache_key,
    build_triangle_customdata,
    format_millions_array,
    plot_data_triangle_table,
    plot_emergence,
    plot_reserving_results_table,
    plot_triangle_heatmap_clean,
)


STYLE = {
    "font_family": "Manrope",
    "figure_font_size": 12,
    "figure_title_font_size": 16,
    "table_header_font_size": 13,
    "table_cell_font_size": 12,
    "heatmap_text_font_size": 10,
    "alert_annotation_font_size": 14,
    "color_text": "#1f2a37",
    "color_surface": "#ffffff",
    "color_border": "#e3e7ee",
}


class _FakeLinkRatioAccessor:
    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    def to_frame(self) -> pd.DataFrame:
        return self._data.copy()


class _FakeTriangleResult:
    def __init__(self, link_ratio_df: pd.DataFrame) -> None:
        self.link_ratio = {"incurred": _FakeLinkRatioAccessor(link_ratio_df)}


class _FakeTriangleService:
    def __init__(self, link_ratio_df: pd.DataFrame) -> None:
        self._link_ratio_df = link_ratio_df

    def get_triangle(self) -> _FakeTriangleResult:
        return _FakeTriangleResult(self._link_ratio_df)


@dataclass
class _FakeReserving:
    _triangle: _FakeTriangleService


def _dev_cols() -> list[str]:
    return ["3-6", "6-9", "9-12"]


def _triangle_df() -> pd.DataFrame:
    index = pd.Index(["2001", "2002", "LDF", "Tail"])
    values = [
        [1.10, np.nan, 1.05],
        [1.20, 1.10, np.nan],
        [1.08, 1.03, 1.01],
        [1.02, 1.01, 1.00],
    ]
    return pd.DataFrame(values, index=index, columns=_dev_cols())


def _raw_link_ratio_df() -> pd.DataFrame:
    index = pd.Index(["2001", "2002", "LDF", "Tail"])
    values = [
        [1.10, 1.15, 1.05],
        [1.20, 1.10, np.nan],
        [1.08, 1.03, 1.01],
        [1.02, 1.01, 1.00],
    ]
    return pd.DataFrame(values, index=index, columns=_dev_cols())


def _incurred_df() -> pd.DataFrame:
    index = pd.Index(["2001", "2002", "LDF", "Tail"])
    columns = [3, 6, 9, 12]
    values = [
        [500_000, 900_000, 1_200_000, 1_300_000],
        [550_000, 980_000, 1_280_000, 1_390_000],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ]
    return pd.DataFrame(values, index=index, columns=columns)


def _premium_df() -> pd.DataFrame:
    index = pd.Index(["2001", "2002", "LDF", "Tail"])
    columns = [3, 6, 9, 12]
    values = [
        [250_000, 250_000, 250_000, 250_000],
        [260_000, 260_000, 260_000, 260_000],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ]
    return pd.DataFrame(values, index=index, columns=columns)


def test_build_heatmap_core_cache_key_is_stable_for_same_input() -> None:
    key_one = build_heatmap_core_cache_key(
        segment_key="quarterly",
        triangle_data=_triangle_df(),
        incurred_data=_incurred_df(),
        premium_data=_premium_df(),
    )
    key_two = build_heatmap_core_cache_key(
        segment_key="quarterly",
        triangle_data=_triangle_df(),
        incurred_data=_incurred_df(),
        premium_data=_premium_df(),
    )
    assert key_one == key_two


def test_build_heatmap_core_cache_key_changes_when_triangle_changes() -> None:
    triangle_one = _triangle_df()
    triangle_two = _triangle_df().copy()
    triangle_two.iloc[0, 0] = 1.11
    key_one = build_heatmap_core_cache_key(
        segment_key="quarterly",
        triangle_data=triangle_one,
        incurred_data=_incurred_df(),
        premium_data=_premium_df(),
    )
    key_two = build_heatmap_core_cache_key(
        segment_key="quarterly",
        triangle_data=triangle_two,
        incurred_data=_incurred_df(),
        premium_data=_premium_df(),
    )
    assert key_one != key_two


def test_format_millions_array_formats_scales_and_nan() -> None:
    raw = np.array([[10.0, 1_500.0, 2_500_000.0, np.nan]])
    rendered = format_millions_array(raw)
    assert rendered[0, 0] == "10"
    assert rendered[0, 1].endswith("k")
    assert rendered[0, 2].endswith("m")
    assert rendered[0, 3] == ""


def test_build_triangle_customdata_shape_matches_triangle() -> None:
    triangle = _triangle_df()
    customdata = build_triangle_customdata(
        expanded_triangle=triangle,
        incurred_data=_incurred_df(),
        premium_data=_premium_df(),
    )
    assert customdata.shape == (len(triangle.index), len(triangle.columns), 2)


def test_build_heatmap_core_returns_required_keys_and_drop_mask() -> None:
    fake_reserving = _FakeReserving(
        _triangle=_FakeTriangleService(_raw_link_ratio_df())
    )
    payload = build_heatmap_core(
        triangle_data=_triangle_df(),
        incurred_data=_incurred_df(),
        premium_data=_premium_df(),
        reserving=fake_reserving,  # type: ignore[arg-type]
    )
    required_keys = {
        "expanded_triangle",
        "dropped_mask",
        "z_data",
        "text_values",
        "customdata",
    }
    assert required_keys.issubset(payload.keys())
    dropped_mask = payload["dropped_mask"]
    assert bool(dropped_mask.loc["2001", "6-9"])


def test_plot_data_triangle_table_returns_table_trace() -> None:
    figure = plot_data_triangle_table(
        triangle_df=_triangle_df().iloc[:2, :],
        title="Data Triangle",
        weights_df=None,
        ratio_mode=False,
        font_family=STYLE["font_family"],
        figure_font_size=STYLE["figure_font_size"],
        figure_title_font_size=STYLE["figure_title_font_size"],
        table_header_font_size=STYLE["table_header_font_size"],
        table_cell_font_size=STYLE["table_cell_font_size"],
        alert_annotation_font_size=STYLE["alert_annotation_font_size"],
        color_text=STYLE["color_text"],
        color_surface=STYLE["color_surface"],
        color_border=STYLE["color_border"],
    )
    assert len(figure.data) == 1
    assert figure.data[0].type == "table"


def test_plot_emergence_returns_actual_and_expected_traces() -> None:
    emergence = pd.concat(
        {
            "Actual": pd.DataFrame(
                [[0.2, 0.5, 0.9], [0.15, 0.45, 0.85]],
                index=["2001", "2002"],
                columns=[12, 24, 36],
            ),
            "Expected": pd.DataFrame(
                [[0.18, 0.48, 0.88], [0.18, 0.48, 0.88]],
                index=["2001", "2002"],
                columns=[12, 24, 36],
            ),
        },
        axis=1,
    )
    figure = plot_emergence(
        emergence_pattern=emergence,
        title="Emergence",
        font_family=STYLE["font_family"],
        figure_font_size=STYLE["figure_font_size"],
        figure_title_font_size=STYLE["figure_title_font_size"],
        alert_annotation_font_size=STYLE["alert_annotation_font_size"],
        color_text=STYLE["color_text"],
        color_surface=STYLE["color_surface"],
        color_border=STYLE["color_border"],
    )
    assert len(figure.data) == 3


def test_plot_reserving_results_table_returns_table_trace() -> None:
    results = pd.DataFrame(
        {
            "incurred": [100.0, 120.0],
            "Premium": [200.0, 220.0],
            "cl_ultimate": [150.0, 170.0],
            "cl_loss_ratio": [0.75, 0.7727],
            "bf_ultimate": [160.0, 175.0],
            "bf_loss_ratio": [0.80, 0.7954],
            "ultimate": [155.0, 172.0],
        },
        index=["2001", "2002"],
    )
    figure = plot_reserving_results_table(
        results_df=results,
        title="Results",
        font_family=STYLE["font_family"],
        figure_font_size=STYLE["figure_font_size"],
        figure_title_font_size=STYLE["figure_title_font_size"],
        table_header_font_size=STYLE["table_header_font_size"],
        table_cell_font_size=STYLE["table_cell_font_size"],
        alert_annotation_font_size=STYLE["alert_annotation_font_size"],
        color_text=STYLE["color_text"],
        color_surface=STYLE["color_surface"],
        color_border=STYLE["color_border"],
    )
    assert len(figure.data) == 1
    assert figure.data[0].type == "table"


def test_plot_triangle_heatmap_clean_returns_figure_and_payload() -> None:
    fake_reserving = _FakeReserving(
        _triangle=_FakeTriangleService(_raw_link_ratio_df())
    )
    figure, payload = plot_triangle_heatmap_clean(
        triangle_data=_triangle_df(),
        incurred_data=_incurred_df(),
        premium_data=_premium_df(),
        reserving=fake_reserving,  # type: ignore[arg-type]
        title="Triangle - Link Ratios",
        tail_attachment_age=3,
        tail_fit_period_selection=[3, 9],
        parse_dev_label=lambda value: int(str(value).split("-")[0]),
        derive_tail_fit_period=lambda selection: (
            (min(selection), max(selection)) if selection else None
        ),
        font_family=STYLE["font_family"],
        figure_font_size=STYLE["figure_font_size"],
        figure_title_font_size=STYLE["figure_title_font_size"],
        heatmap_text_font_size=STYLE["heatmap_text_font_size"],
        color_text=STYLE["color_text"],
        color_surface=STYLE["color_surface"],
        color_border=STYLE["color_border"],
    )
    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"
    assert "expanded_triangle" in payload
    assert len(figure.layout.shapes) >= 2
