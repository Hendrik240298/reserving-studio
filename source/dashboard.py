import json
import hashlib
import logging
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from source.config_manager import ConfigManager
from source.reserving import Reserving
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    callback_context,
    clientside_callback,
    no_update,
    dash_table,
)


_LIVE_RESULTS_BY_SEGMENT: dict[str, dict] = {}

FONT_FAMILY = '"Manrope", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
FIGURE_FONT_SIZE = 12
FIGURE_TITLE_FONT_SIZE = 16
TABLE_HEADER_FONT_SIZE = 13
TABLE_CELL_FONT_SIZE = 12
HEATMAP_TEXT_FONT_SIZE = 10
ALERT_ANNOTATION_FONT_SIZE = 14
COLOR_BG = "#f7f8fa"
COLOR_SURFACE = "#ffffff"
COLOR_BORDER = "#e3e7ee"
COLOR_TEXT = "#1f2a37"
COLOR_MUTED = "#5b6b7b"
COLOR_ACCENT = "#2b6cb0"
COLOR_ACCENT_SOFT = "#e8f1fb"
SHADOW_SOFT = "0 8px 24px rgba(15, 23, 42, 0.06)"
RADIUS_LG = "14px"
RADIUS_MD = "10px"
SIDEBAR_EXPANDED_WIDTH = "240px"
SIDEBAR_COLLAPSED_WIDTH = "0px"


class Dashboard:
    def __init__(
        self,
        reserving: Reserving,
        config: Optional[ConfigManager] = None,
    ):
        """
        Initialize Dashboard with reserving results and configuration.

        Args:
            reserving: Reserving object for the triangle
            config: ConfigManager with paths and settings
        """
        self._reserving = reserving
        self._config = config
        assets_folder = Path(__file__).resolve().parent.parent / "assets"
        self.app = Dash(
            __name__,
            assets_folder=str(assets_folder),
            suppress_callback_exceptions=True,
            external_stylesheets=[
                "https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap",
            ],
        )
        self._default_average = "volume"
        self._default_tail_curve = "weibull"
        self._default_drop_text = ""
        self._default_drop_store: List[List[str | int]] = []
        self._default_tail_attachment_age: Optional[int] = None
        self._default_tail_fit_period_selection: List[int] = []
        self._default_bf_apriori = 0.6
        self._default_bf_apriori_rows: List[dict[str, object]] = []
        self._payload_cache: dict[str, dict] = {}
        self._triangle_figure_cache: dict[str, dict] = {}
        self._emergence_figure_cache: dict[str, dict] = {}
        self._results_table_figure_cache: dict[str, dict] = {}
        self._heatmap_core_cache: dict[str, dict] = {}
        self._cache_max_entries = 32
        self._recalc_request_seq = 0

        self._load_session_defaults()

        # Extract data from domain objects
        self._extract_data()

        self._register_callbacks()
        logging.info("Dashboard initialized successfully")

    def _load_session_defaults(self) -> None:
        if self._config is None:
            self._default_bf_apriori_rows = self._build_bf_apriori_rows()
            return
        session = self._config.load_session()
        self._default_average = session.get("average", self._default_average)
        self._default_tail_curve = session.get(
            "tail_curve",
            self._default_tail_curve,
        )
        self._default_drop_store = self._normalize_drop_store(session.get("drops"))
        tail_attachment_age = session.get("tail_attachment_age")
        if tail_attachment_age is not None:
            try:
                self._default_tail_attachment_age = int(tail_attachment_age)
            except (TypeError, ValueError):
                self._default_tail_attachment_age = None
        self._default_tail_fit_period_selection = self._normalize_tail_fit_selection(
            session.get("tail_fit_period")
        )
        self._default_bf_apriori_rows = self._build_bf_apriori_rows(
            session.get("bf_apriori_by_uwy")
        )

    def _get_segment_key(self) -> str:
        if self._config is None:
            return "default"
        return self._config.get_segment()

    def _normalize_drop_store(self, raw: object) -> List[List[str | int]]:
        normalized: List[List[str | int]] = []
        if not isinstance(raw, list):
            return normalized
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            origin, dev = item[0], item[1]
            try:
                normalized.append([str(origin), int(dev)])
            except (TypeError, ValueError):
                continue
        return normalized

    def _parse_sync_payload(self, raw_payload: object) -> Optional[dict[str, object]]:
        if not raw_payload:
            return None
        if isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            try:
                payload = json.loads(str(raw_payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _extract_data(self):
        """Extract data from domain objects and store as instance variables."""
        try:
            self.emergence_pattern = self._reserving.get_emergence_pattern()
            logging.info(f"Emergence pattern loaded: {self.emergence_pattern.shape}")

            triangle_data = self._reserving.get_triangle_heatmap_data()
            self.triangle = triangle_data["link_ratios"]
            self.incurred = triangle_data["incurred"]
            self.premium = triangle_data["premium"]
            logging.info(f"Triangle data loaded: {self.triangle.shape}")

            self.results = self._reserving.get_results()
            if self.results is not None:
                logging.info(f"Reserving results loaded: {len(self.results)} UWYs")
        except Exception as e:
            logging.error(f"Error extracting data: {e}", exc_info=True)
            self.emergence_pattern = None
            self.triangle = None
            self.incurred = None
            self.premium = None
            self.results = None
            raise  # Re-raise to see the actual error

    def _format_triangle_row_labels(self, index: pd.Index) -> list[str]:
        labels: list[str] = []
        for idx in index:
            if hasattr(idx, "year"):
                labels.append(str(idx.year))
            else:
                labels.append(str(idx))
        return labels

    def _format_triangle_column_labels(self, columns: pd.Index) -> list[str]:
        labels: list[str] = []
        for col in columns:
            if isinstance(col, (int, np.integer)):
                labels.append(str(int(col)))
            else:
                labels.append(str(col))
        return labels

    def _get_data_tab_triangle(
        self,
        metric: str,
        triangle_view: str = "cumulative",
    ) -> pd.DataFrame:
        triangle_obj = self._reserving._triangle
        normalized_view = (triangle_view or "cumulative").lower()

        incurred_triangle = triangle_obj.get_triangle("incurred")
        paid_triangle = triangle_obj.get_triangle("paid")
        if normalized_view == "incremental":
            incurred_triangle = incurred_triangle.cum_to_incr()
            paid_triangle = paid_triangle.cum_to_incr()

        incurred_df = incurred_triangle["incurred"].to_frame()

        metric = (metric or "incurred").lower()
        if metric == "incurred":
            return incurred_df

        if metric == "paid":
            return paid_triangle["paid"].to_frame()

        if metric == "premium":
            return (
                incurred_triangle["Premium_selected"]
                .to_frame()
                .rename(columns={"Premium_selected": "premium"})
            )

        if metric == "outstanding":
            paid_df = paid_triangle["paid"].to_frame()
            incurred_aligned, paid_aligned = incurred_df.align(paid_df, join="outer")
            outstanding = incurred_aligned - paid_aligned
            return outstanding

        return incurred_df

    def _build_data_tab_display(
        self,
        metric: str,
        triangle_view: str,
        divisor: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
        metric_value = (metric or "incurred").lower()
        divisor_value = (divisor or "none").lower()

        numerator_df = self._get_data_tab_triangle(metric_value, triangle_view)
        if divisor_value == "none":
            weights_df = self._get_data_tab_triangle("premium", triangle_view)
            return numerator_df, weights_df, False

        denominator_df = self._get_data_tab_triangle(divisor_value, "cumulative")
        num_aligned, den_aligned = numerator_df.align(denominator_df, join="outer")
        safe_denominator = den_aligned.where(den_aligned > 0)
        divided_df = num_aligned.div(safe_denominator)
        weights_df = den_aligned
        return divided_df, weights_df, True

    def _plot_data_triangle_table(
        self,
        triangle_df: pd.DataFrame,
        title: str,
        weights_df: Optional[pd.DataFrame] = None,
        ratio_mode: bool = False,
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
                                size=ALERT_ANNOTATION_FONT_SIZE,
                                family=FONT_FAMILY,
                            ),
                        )
                    ],
                )
            )

        df_display = self._append_data_triangle_average_rows(triangle_df, weights_df)
        row_labels = self._format_triangle_row_labels(df_display.index)
        col_labels = self._format_triangle_column_labels(df_display.columns)

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

        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=[130] + [62] * len(col_labels),
                    header=dict(
                        values=headers,
                        fill_color="#f2f5f9",
                        align="center",
                        line_color=COLOR_BORDER,
                        font=dict(
                            color=COLOR_TEXT,
                            size=TABLE_HEADER_FONT_SIZE,
                            family=FONT_FAMILY,
                        ),
                        height=28,
                    ),
                    cells=dict(
                        values=table_values,
                        fill_color=COLOR_SURFACE,
                        align="center",
                        line_color=COLOR_BORDER,
                        font=dict(
                            color=COLOR_TEXT,
                            size=TABLE_CELL_FONT_SIZE,
                            family=FONT_FAMILY,
                        ),
                        height=26,
                    ),
                )
            ]
        )

        table_width = 130 + (62 * len(col_labels))

        fig.update_layout(
            title=title,
            template="plotly_white",
            font=dict(color=COLOR_TEXT, size=FIGURE_FONT_SIZE, family=FONT_FAMILY),
            title_font=dict(
                color=COLOR_TEXT,
                size=FIGURE_TITLE_FONT_SIZE,
                family=FONT_FAMILY,
            ),
            hoverlabel=dict(
                bgcolor=COLOR_SURFACE,
                bordercolor=COLOR_BORDER,
                font=dict(
                    color=COLOR_TEXT,
                    size=FIGURE_FONT_SIZE,
                    family=FONT_FAMILY,
                ),
            ),
            margin=dict(l=8, r=8, t=48, b=8),
            width=max(900, table_width + 16),
            height=min(760, 170 + len(df_display.index) * 28),
            autosize=False,
            uirevision="static",
        )
        return fig

    def _append_data_triangle_average_rows(
        self,
        triangle_df: pd.DataFrame,
        weights_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if triangle_df is None or triangle_df.empty:
            return triangle_df

        df_values = triangle_df.apply(pd.to_numeric, errors="coerce")
        simple_average = df_values.mean(axis=0, skipna=True)

        if weights_df is None:
            premium_weights = self._reserving._triangle.get_triangle("incurred")[
                "Premium_selected"
            ].to_frame()
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

    def _parse_drop_text(self, text: str) -> List[Tuple[str, int]]:
        if not text:
            return []
        drops: List[Tuple[str, int]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            origin, dev = parts
            if not origin:
                continue
            try:
                dev_value = int(dev)
            except ValueError:
                continue
            drops.append((str(origin), dev_value))
        return drops

    def _parse_dev_label(self, dev_label: object) -> Optional[int]:
        if dev_label is None:
            return None
        try:
            dev_str = str(dev_label)
            if "-" in dev_str:
                return int(dev_str.split("-")[0])
            return int(dev_str)
        except (TypeError, ValueError):
            return None

    def _normalize_tail_fit_selection(self, raw: object) -> List[int]:
        if raw is None:
            return []
        if isinstance(raw, (int, float, str)):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        normalized: List[int] = []
        for item in raw:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in normalized:
                normalized.append(value)
        return normalized

    def _toggle_tail_fit_selection(self, existing: List[int], dev: int) -> List[int]:
        normalized = []
        for item in existing or []:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in normalized:
                normalized.append(value)
        if dev in normalized:
            normalized = [value for value in normalized if value != dev]
        else:
            normalized.append(dev)
        return normalized

    def _get_uwy_labels(self) -> List[str]:
        triangle = self._reserving._triangle.get_triangle("incurred")["incurred"]
        labels: List[str] = []
        for origin in triangle.origin:
            if hasattr(origin, "year"):
                label = str(origin.year)
            else:
                origin_text = str(origin)
                if len(origin_text) >= 4 and origin_text[:4].isdigit():
                    label = origin_text[:4]
                else:
                    label = origin_text
            if label not in labels:
                labels.append(label)
        return labels

    def _build_bf_apriori_rows(self, raw: object = None) -> List[dict[str, object]]:
        uwy_labels = self._get_uwy_labels()
        mapping: dict[str, float] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if pd.isna(numeric) or numeric < 0:
                    continue
                mapping[str(key)] = numeric

        if isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict):
                    continue
                uwy = str(row.get("uwy", "")).strip()
                if not uwy:
                    continue
                try:
                    numeric = float(row.get("apriori"))
                except (TypeError, ValueError):
                    continue
                if pd.isna(numeric) or numeric < 0:
                    continue
                mapping[uwy] = numeric

        rows: List[dict[str, object]] = []
        for uwy in uwy_labels:
            factor = mapping.get(uwy, self._default_bf_apriori)
            rows.append({"uwy": uwy, "apriori": float(factor)})
        return rows

    def _bf_rows_to_mapping(
        self,
        rows: Optional[List[dict[str, object]]],
    ) -> dict[str, float]:
        normalized_rows = self._build_bf_apriori_rows(rows)
        mapping: dict[str, float] = {}
        for row in normalized_rows:
            uwy = str(row.get("uwy", "")).strip()
            if not uwy:
                continue
            try:
                value = float(row.get("apriori"))
            except (TypeError, ValueError):
                value = self._default_bf_apriori
            if pd.isna(value) or value < 0:
                value = self._default_bf_apriori
            mapping[uwy] = value
        return mapping

    def _derive_tail_fit_period(
        self, selection: Optional[List[int]]
    ) -> Optional[Tuple[int, Optional[int]]]:
        if not selection:
            return None
        normalized = self._normalize_tail_fit_selection(selection)
        if not normalized:
            return None
        sorted_values = sorted(set(normalized))
        if len(sorted_values) == 1:
            return (sorted_values[0], None)
        return (sorted_values[0], sorted_values[-1])

    def _toggle_drop(
        self,
        existing: List[List[str | int]],
        origin: str,
        dev: int,
    ) -> List[List[str | int]]:
        normalized = []
        for item in existing or []:
            if not isinstance(item, list) or len(item) != 2:
                continue
            try:
                normalized.append((str(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue

        entry = (str(origin), int(dev))
        if entry in normalized:
            normalized = [item for item in normalized if item != entry]
        else:
            normalized.append(entry)

        return [[item[0], item[1]] for item in normalized]

    def _drops_to_tuples(
        self, drops: Optional[List[List[str | int]]]
    ) -> Optional[List[Tuple[str, int]]]:
        if not drops:
            return None
        parsed: List[Tuple[str, int]] = []
        for item in drops:
            if not isinstance(item, list) or len(item) != 2:
                continue
            origin, dev = item[0], item[1]
            try:
                parsed.append((str(origin), int(dev)))
            except (TypeError, ValueError):
                continue
        return parsed or None

    def _apply_recalculation(
        self,
        average: str,
        drops: Optional[List[Tuple[str, int]]],
        tail_attachment_age: Optional[int],
        tail_curve: str,
        fit_period: Optional[Tuple[int, Optional[int]]],
        bf_apriori_by_uwy: Optional[dict[str, float]],
    ) -> None:
        self._reserving.set_development(
            average=average,
            drop=drops,
        )
        self._reserving.set_tail(
            curve=tail_curve,
            projection_period=0,
            attachment_age=tail_attachment_age,
            fit_period=fit_period,
        )
        if bf_apriori_by_uwy:
            self._reserving.set_bornhuetter_ferguson(apriori=bf_apriori_by_uwy)
        else:
            self._reserving.set_bornhuetter_ferguson(apriori=self._default_bf_apriori)
        self._reserving.reserve(final_ultimate="chainladder")
        self._extract_data()

    def _build_results_payload(
        self,
        drop_store: Optional[List[List[str | int]]],
        average: Optional[str],
        tail_attachment_age: Optional[int],
        tail_curve: Optional[str],
        tail_fit_period_selection: Optional[List[int]],
        bf_apriori_by_uwy: Optional[dict[str, float]],
    ) -> dict:
        timestamp = datetime.utcnow().isoformat() + "Z"
        display = "None"
        if drop_store:
            display = ", ".join([f"{item[0]}:{item[1]}" for item in drop_store])
        tail_display = "None"
        if tail_attachment_age is not None:
            tail_display = str(tail_attachment_age)
        fit_period = self._derive_tail_fit_period(tail_fit_period_selection)
        if fit_period is None:
            fit_period_display = "lower=None, upper=None"
        else:
            fit_period_display = f"lower={fit_period[0]}, upper={fit_period[1]}"

        visual_cache_key = self._build_visual_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_curve,
            tail_fit_period_selection,
        )
        results_table_cache_key = self._build_results_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
        )

        triangle_figure = self._get_or_build_cached_figure(
            cache=self._triangle_figure_cache,
            cache_key=visual_cache_key,
            label="Triangle figure",
            builder=lambda: self._plot_triangle_heatmap_clean(
                self.triangle,
                "Triangle - Link Ratios",
                tail_attachment_age,
                tail_fit_period_selection,
            ).to_dict(),
        )
        emergence_figure = self._get_or_build_cached_figure(
            cache=self._emergence_figure_cache,
            cache_key=visual_cache_key,
            label="Emergence figure",
            builder=lambda: self._plot_emergence(
                self.emergence_pattern,
                "Emergence Pattern",
            ).to_dict(),
        )
        results_figure = self._get_or_build_cached_figure(
            cache=self._results_table_figure_cache,
            cache_key=results_table_cache_key,
            label="Results table figure",
            builder=lambda: self._plot_reserving_results_table(
                self.results,
                "Reserving Results",
            ).to_dict(),
        )

        return {
            "triangle_figure": triangle_figure,
            "emergence_figure": emergence_figure,
            "results_figure": results_figure,
            "drops_display": display,
            "average": average,
            "tail_curve": tail_curve,
            "drop_store": drop_store or [],
            "tail_attachment_age": tail_attachment_age,
            "tail_attachment_display": tail_display,
            "tail_fit_period_selection": tail_fit_period_selection or [],
            "tail_fit_period_display": fit_period_display,
            "last_updated": timestamp,
        }

    def _build_visual_cache_key(
        self,
        drop_store: Optional[List[List[str | int]]],
        average: Optional[str],
        tail_attachment_age: Optional[int],
        tail_curve: Optional[str],
        tail_fit_period_selection: Optional[List[int]],
    ) -> str:
        return self._build_results_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy=None,
        )

    def _cache_get(self, cache: dict[str, dict], key: str) -> Optional[dict]:
        cached_value = cache.get(key)
        if cached_value is None:
            return None
        cache.pop(key, None)
        cache[key] = cached_value
        return deepcopy(cached_value)

    def _cache_set(self, cache: dict[str, dict], key: str, value: dict) -> None:
        cache.pop(key, None)
        cache[key] = deepcopy(value)
        while len(cache) > self._cache_max_entries:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)

    def _get_or_build_cached_figure(
        self,
        *,
        cache: dict[str, dict],
        cache_key: str,
        label: str,
        builder: Callable[[], dict],
    ) -> dict:
        started = time.perf_counter()
        cached_figure = self._cache_get(cache, cache_key)
        if cached_figure is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logging.info("%s reused in %.0f ms", label, elapsed_ms)
            return cached_figure

        figure_dict = builder()
        elapsed_ms = (time.perf_counter() - started) * 1000
        logging.info("%s built in %.0f ms", label, elapsed_ms)
        self._cache_set(cache, cache_key, figure_dict)
        return figure_dict

    def _hash_dataframe(self, frame: pd.DataFrame) -> str:
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

    def _build_heatmap_core_cache_key(
        self,
        triangle_data: pd.DataFrame,
        incurred_data: pd.DataFrame,
        premium_data: pd.DataFrame,
    ) -> str:
        payload = {
            "segment": self._get_segment_key(),
            "triangle": self._hash_dataframe(triangle_data),
            "incurred": self._hash_dataframe(incurred_data),
            "premium": self._hash_dataframe(premium_data),
        }
        return json.dumps(payload, sort_keys=True)

    def _format_millions_array(self, values: np.ndarray) -> np.ndarray:
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

    def _build_triangle_customdata(
        self,
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

        left_values = aligned_incurred.reindex(columns=left_columns).to_numpy(
            dtype=float
        )
        right_values = aligned_incurred.reindex(columns=right_columns).to_numpy(
            dtype=float
        )
        premium_values = aligned_premium.reindex(columns=left_columns).to_numpy(
            dtype=float
        )

        column_mask = np.broadcast_to(
            valid_dev_columns.reshape(1, -1),
            left_values.shape,
        )
        has_left = (~np.isnan(left_values)) & column_mask
        has_right = (~np.isnan(right_values)) & column_mask
        has_premium = (~np.isnan(premium_values)) & column_mask

        left_display = self._format_millions_array(left_values)
        right_display = self._format_millions_array(right_values)
        premium_display = self._format_millions_array(premium_values)

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

    def _build_heatmap_core(
        self,
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

        customdata = self._build_triangle_customdata(
            expanded_triangle,
            incurred_data,
            premium_data,
        )

        return {
            "expanded_triangle": expanded_triangle,
            "dropped_mask": dropped_mask,
            "z_data": z_data,
            "text_values": text_values,
            "customdata": customdata,
        }

    def _build_results_cache_key(
        self,
        drop_store: Optional[List[List[str | int]]],
        average: Optional[str],
        tail_attachment_age: Optional[int],
        tail_curve: Optional[str],
        tail_fit_period_selection: Optional[List[int]],
        bf_apriori_by_uwy: Optional[dict[str, float]],
    ) -> str:
        normalized_drops: list[list[str | int]] = []
        for item in drop_store or []:
            if not isinstance(item, list) or len(item) != 2:
                continue
            try:
                normalized_drops.append([str(item[0]), int(item[1])])
            except (TypeError, ValueError):
                continue
        normalized_drops.sort(key=lambda item: (str(item[0]), int(item[1])))

        normalized_fit_period = self._normalize_tail_fit_selection(
            tail_fit_period_selection
        )
        normalized_fit_period.sort()

        normalized_bf: dict[str, float] = {}
        for key, value in sorted((bf_apriori_by_uwy or {}).items()):
            try:
                normalized_bf[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        payload = {
            "segment": self._get_segment_key(),
            "average": average or self._default_average,
            "tail_curve": tail_curve or self._default_tail_curve,
            "tail_attachment_age": tail_attachment_age,
            "tail_fit_period_selection": normalized_fit_period,
            "drops": normalized_drops,
            "bf_apriori_by_uwy": normalized_bf,
        }
        return json.dumps(payload, sort_keys=True)

    def _build_params_state(
        self,
        *,
        drop_store: Optional[List[List[str | int]]],
        average: Optional[str],
        tail_attachment_age: Optional[int],
        tail_curve: Optional[str],
        tail_fit_period_selection: Optional[List[int]],
        bf_apriori_by_uwy: Optional[dict[str, float]],
        request_id: int,
        source: str,
        force_recalc: bool,
        sync_version: Optional[int] = None,
    ) -> dict:
        normalized_drop_store = self._normalize_drop_store(drop_store)
        normalized_tail_fit = self._normalize_tail_fit_selection(
            tail_fit_period_selection
        )

        parsed_tail = None
        if tail_attachment_age is not None:
            try:
                parsed_tail = int(tail_attachment_age)
            except (TypeError, ValueError):
                parsed_tail = None

        normalized_bf = {
            key: float(value)
            for key, value in self._bf_rows_to_mapping(
                self._build_bf_apriori_rows(bf_apriori_by_uwy)
            ).items()
        }

        return {
            "request_id": int(request_id),
            "source": source,
            "force_recalc": bool(force_recalc),
            "drop_store": normalized_drop_store,
            "tail_attachment_age": parsed_tail,
            "tail_fit_period_selection": normalized_tail_fit,
            "average": average or self._default_average,
            "tail_curve": tail_curve or self._default_tail_curve,
            "bf_apriori_by_uwy": normalized_bf,
            "sync_version": sync_version,
        }

    def _load_session_params_state(self, request_id: int, force_recalc: bool) -> dict:
        if self._config is None:
            return self._build_params_state(
                drop_store=self._default_drop_store,
                average=self._default_average,
                tail_attachment_age=self._default_tail_attachment_age,
                tail_curve=self._default_tail_curve,
                tail_fit_period_selection=self._default_tail_fit_period_selection,
                bf_apriori_by_uwy=self._bf_rows_to_mapping(
                    self._default_bf_apriori_rows
                ),
                request_id=request_id,
                source="load",
                force_recalc=force_recalc,
                sync_version=None,
            )

        session = self._config.load_session()
        return self._build_params_state(
            drop_store=session.get("drops"),
            average=session.get("average", self._default_average),
            tail_attachment_age=session.get("tail_attachment_age"),
            tail_curve=session.get("tail_curve", self._default_tail_curve),
            tail_fit_period_selection=session.get("tail_fit_period"),
            bf_apriori_by_uwy=session.get("bf_apriori_by_uwy"),
            request_id=request_id,
            source="load",
            force_recalc=force_recalc,
            sync_version=self._config.get_sync_version(),
        )

    def _get_or_build_results_payload(
        self,
        drop_store: Optional[List[List[str | int]]],
        average: Optional[str],
        tail_attachment_age: Optional[int],
        tail_curve: Optional[str],
        tail_fit_period_selection: Optional[List[int]],
        bf_apriori_by_uwy: Optional[dict[str, float]],
        force_recalc: bool = False,
    ) -> dict:
        cache_key = self._build_results_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy,
        )
        cached_payload = (
            None if force_recalc else self._cache_get(self._payload_cache, cache_key)
        )
        if cached_payload is not None:
            logging.info("Using cached reserving payload for current parameters")
            return cached_payload

        fit_period = self._derive_tail_fit_period(tail_fit_period_selection)
        started = time.perf_counter()
        self._apply_recalculation(
            average or self._default_average,
            self._drops_to_tuples(drop_store),
            tail_attachment_age,
            tail_curve or self._default_tail_curve,
            fit_period,
            bf_apriori_by_uwy,
        )
        recalc_elapsed_ms = (time.perf_counter() - started) * 1000
        logging.info("Recalculation completed in %.0f ms", recalc_elapsed_ms)

        payload_started = time.perf_counter()
        payload = self._build_results_payload(
            drop_store=drop_store,
            average=average,
            tail_attachment_age=tail_attachment_age,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
        )
        payload_elapsed_ms = (time.perf_counter() - payload_started) * 1000
        logging.info("Payload build completed in %.0f ms", payload_elapsed_ms)
        payload["cache_key"] = cache_key
        self._cache_set(self._payload_cache, cache_key, payload)
        return payload

    def _register_callbacks(self) -> None:
        clientside_callback(
            """
            function(userKey) {
                if (!userKey) {
                    return false;
                }
                return Boolean(window.ReservingTabSync);
            }
            """,
            Output("sync-ready", "data"),
            Input("sync-user-key", "data"),
        )

        clientside_callback(
            """
            function(userKey, tabId) {
                if (!window.ReservingTabSync) {
                    return tabId || "";
                }
                return window.ReservingTabSync.configure(userKey || "default", tabId || null);
            }
            """,
            Output("sync-tab-id", "data"),
            Input("sync-user-key", "data"),
            State("sync-tab-id", "data"),
        )

        clientside_callback(
            """
            function(message, userKey, tabId) {
                if (!message || !window.ReservingTabSync) {
                    return window.dash_clientside.no_update;
                }
                window.ReservingTabSync.configure(userKey || "default", tabId || null);
                window.ReservingTabSync.publish(message);
                return window.dash_clientside.no_update;
            }
            """,
            Output("sync-publish-signal", "children"),
            Input("sync-publish-message", "data"),
            State("sync-user-key", "data"),
            State("sync-tab-id", "data"),
            prevent_initial_call=True,
        )

        clientside_callback(
            """
            function(paramsState, basePayload, currentOverlaySignature) {
                if (!basePayload || typeof basePayload !== "object") {
                    return [window.dash_clientside.no_update, window.dash_clientside.no_update];
                }

                var baseFigure = basePayload.figure;
                var figureVersion = String(basePayload.figure_version || "0");
                if (!baseFigure || !baseFigure.data || !Array.isArray(baseFigure.data)) {
                    return [window.dash_clientside.no_update, window.dash_clientside.no_update];
                }

                function normalizeDropStore(items) {
                    if (!Array.isArray(items)) {
                        return [];
                    }
                    var normalized = [];
                    for (var i = 0; i < items.length; i += 1) {
                        var item = items[i];
                        if (!Array.isArray(item) || item.length !== 2) {
                            continue;
                        }
                        var origin = String(item[0]);
                        var dev = parseInt(item[1], 10);
                        if (!origin || Number.isNaN(dev)) {
                            continue;
                        }
                        normalized.push([origin, dev]);
                    }
                    normalized.sort(function(a, b) {
                        if (a[0] < b[0]) {
                            return -1;
                        }
                        if (a[0] > b[0]) {
                            return 1;
                        }
                        return a[1] - b[1];
                    });
                    var deduped = [];
                    for (var j = 0; j < normalized.length; j += 1) {
                        if (j === 0 || normalized[j][0] !== normalized[j - 1][0] || normalized[j][1] !== normalized[j - 1][1]) {
                            deduped.push(normalized[j]);
                        }
                    }
                    return deduped;
                }

                var normalizedDropStore = normalizeDropStore(
                    paramsState && paramsState.drop_store
                );
                var nextSignature = JSON.stringify({
                    figure_version: figureVersion,
                    drop_store: normalizedDropStore,
                });
                if (currentOverlaySignature === nextSignature) {
                    return [window.dash_clientside.no_update, window.dash_clientside.no_update];
                }

                var cloned = JSON.parse(JSON.stringify(baseFigure));
                if (!cloned.layout) {
                    cloned.layout = {};
                }

                var heatmap = null;
                for (var i = 0; i < cloned.data.length; i += 1) {
                    if (cloned.data[i] && cloned.data[i].type === "heatmap") {
                        heatmap = cloned.data[i];
                        break;
                    }
                }
                if (!heatmap || !Array.isArray(heatmap.x) || !Array.isArray(heatmap.y)) {
                    return window.dash_clientside.no_update;
                }

                var xValues = heatmap.x.map(function (value) { return String(value); });
                var yValues = heatmap.y.map(function (value) { return String(value); });

                function parseDevStart(label) {
                    if (typeof label !== "string" || label === "UWY") {
                        return null;
                    }
                    var split = label.split("-");
                    var value = parseInt(split[0], 10);
                    return Number.isNaN(value) ? null : value;
                }

                function buildDropLines(rowIndex, colIndex) {
                    return [
                        {
                            type: "line",
                            x0: colIndex - 0.5,
                            y0: rowIndex - 0.5,
                            x1: colIndex + 0.5,
                            y1: rowIndex + 0.5,
                            line: { color: "black", width: 2 },
                            xref: "x",
                            yref: "y"
                        },
                        {
                            type: "line",
                            x0: colIndex - 0.5,
                            y0: rowIndex + 0.5,
                            x1: colIndex + 0.5,
                            y1: rowIndex - 0.5,
                            line: { color: "black", width: 2 },
                            xref: "x",
                            yref: "y"
                        }
                    ];
                }

                var existingShapes = Array.isArray(cloned.layout.shapes)
                    ? cloned.layout.shapes
                    : [];
                var preservedShapes = existingShapes.filter(function (shape) {
                    return !shape || shape.type !== "line";
                });
                var dropShapes = [];

                normalizedDropStore.forEach(function (item) {
                    if (!Array.isArray(item) || item.length !== 2) {
                        return;
                    }
                    var origin = String(item[0]);
                    var dev = parseInt(item[1], 10);
                    if (!origin || Number.isNaN(dev)) {
                        return;
                    }

                    var rowIndex = yValues.indexOf(origin);
                    if (rowIndex < 0) {
                        return;
                    }

                    var colIndex = -1;
                    for (var col = 0; col < xValues.length; col += 1) {
                        var start = parseDevStart(xValues[col]);
                        if (start === dev) {
                            colIndex = col;
                            break;
                        }
                    }
                    if (colIndex < 0) {
                        return;
                    }

                    dropShapes = dropShapes.concat(buildDropLines(rowIndex, colIndex));
                });

                cloned.layout.shapes = preservedShapes.concat(dropShapes);
                return [cloned, nextSignature];
            }
            """,
            Output("triangle-heatmap", "figure"),
            Output("triangle-overlay-signature", "data"),
            Input("params-store", "data"),
            Input("triangle-base-figure", "data"),
            State("triangle-overlay-signature", "data"),
        )

        @self.app.callback(
            Output("active-tab", "data"),
            Input("nav-data", "n_clicks"),
            Input("nav-chainladder", "n_clicks"),
            Input("nav-bf", "n_clicks"),
            Input("nav-results", "n_clicks"),
            State("active-tab", "data"),
        )
        def _set_active_tab(
            data_clicks,
            chainladder_clicks,
            bf_clicks,
            results_clicks,
            current_tab,
        ):
            ctx = callback_context
            if not ctx.triggered:
                return current_tab
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            mapping = {
                "nav-data": "data",
                "nav-chainladder": "chainladder",
                "nav-bf": "bornhuetter_ferguson",
                "nav-results": "results",
            }
            return mapping.get(trigger, current_tab)

        @self.app.callback(
            Output("sidebar-collapsed", "data"),
            Input("sidebar-toggle", "n_clicks"),
            State("sidebar-collapsed", "data"),
        )
        def _toggle_sidebar(collapse_clicks, collapsed):
            if not collapse_clicks:
                return collapsed
            return not bool(collapsed)

        @self.app.callback(
            Output("sidebar", "style"),
            Output("sidebar-title", "style"),
            Output("nav-stack", "style"),
            Output("nav-data", "style"),
            Output("nav-chainladder", "style"),
            Output("nav-bf", "style"),
            Output("nav-results", "style"),
            Output("nav-data", "children"),
            Output("nav-chainladder", "children"),
            Output("nav-bf", "children"),
            Output("nav-results", "children"),
            Output("sidebar-toggle", "children"),
            Output("sidebar-toggle", "style"),
            Input("active-tab", "data"),
            Input("sidebar-collapsed", "data"),
        )
        def _style_sidebar(active_tab, collapsed):
            is_collapsed = bool(collapsed)
            sidebar_style = {
                "width": SIDEBAR_COLLAPSED_WIDTH
                if is_collapsed
                else SIDEBAR_EXPANDED_WIDTH,
                "transition": "width 0.2s ease",
                "background": "#f2f5f9",
                "padding": "0" if is_collapsed else "20px 16px",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "borderRight": "none" if is_collapsed else f"1px solid {COLOR_BORDER}",
                "position": "relative",
                "minHeight": "100vh",
                "height": "100vh",
                "alignSelf": "flex-start",
                "fontFamily": FONT_FAMILY,
                "overflow": "visible",
            }
            title_style = {
                "display": "none" if is_collapsed else "block",
                "textAlign": "center",
                "marginBottom": "18px",
                "color": COLOR_TEXT,
                "letterSpacing": "0.2px",
                "fontSize": "15px",
                "fontWeight": 600,
            }
            nav_stack_style = {
                "display": "none" if is_collapsed else "flex",
                "flexDirection": "column",
                "gap": "8px",
            }
            base_button_style = {
                "width": "100%",
                "textAlign": "left",
                "padding": "10px 12px",
                "border": "1px solid transparent",
                "borderRadius": RADIUS_MD,
                "background": "transparent",
                "cursor": "pointer",
                "fontSize": "14px",
                "marginBottom": "8px",
                "color": COLOR_TEXT,
                "fontFamily": FONT_FAMILY,
                "whiteSpace": "nowrap",
            }
            active_style = {
                **base_button_style,
                "background": COLOR_SURFACE,
                "border": f"1px solid {COLOR_BORDER}",
                "fontWeight": 600,
                "boxShadow": "0 2px 8px rgba(15, 23, 42, 0.06)",
            }
            inactive_style = {
                **base_button_style,
                "color": COLOR_MUTED,
            }
            collapsed_button_style = {
                "display": "none",
            }
            labels_full = {
                "data": "Data",
                "chainladder": "Chainladder",
                "bornhuetter_ferguson": "Bornhuetter-Ferguson",
                "results": "Results",
            }
            labels = labels_full if not is_collapsed else {}
            toggle_label = ">" if is_collapsed else "<"
            toggle_style = {
                "width": "20px",
                "height": "20px",
                "borderRadius": "50%",
                "border": f"1px solid {COLOR_BORDER}",
                "background": COLOR_SURFACE,
                "cursor": "pointer",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "position": "absolute",
                "right": "-10px",
                "top": "50%",
                "transform": "translateY(-50%)",
                "boxShadow": "0 6px 16px rgba(15, 23, 42, 0.12)",
                "padding": "0",
                "color": COLOR_MUTED,
            }
            return (
                sidebar_style,
                title_style,
                nav_stack_style,
                collapsed_button_style
                if is_collapsed
                else (active_style if active_tab == "data" else inactive_style),
                collapsed_button_style
                if is_collapsed
                else (active_style if active_tab == "chainladder" else inactive_style),
                collapsed_button_style
                if is_collapsed
                else (
                    active_style
                    if active_tab == "bornhuetter_ferguson"
                    else inactive_style
                ),
                collapsed_button_style
                if is_collapsed
                else (active_style if active_tab == "results" else inactive_style),
                labels.get("data", ""),
                labels.get("chainladder", ""),
                labels.get("bornhuetter_ferguson", ""),
                labels.get("results", ""),
                toggle_label,
                toggle_style,
            )

        @self.app.callback(
            Output("tab-data", "style"),
            Output("tab-chainladder", "style"),
            Output("tab-bf", "style"),
            Output("tab-results", "style"),
            Output("results-table", "figure"),
            Input("active-tab", "data"),
            Input("results-store", "data"),
        )
        def _toggle_tab_visibility(active_tab, results_payload):
            data_style = {
                "display": "block",
                "background": COLOR_SURFACE,
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": RADIUS_LG,
                "padding": "16px",
                "boxShadow": SHADOW_SOFT,
                "minHeight": "70vh",
                "boxSizing": "border-box",
                "width": "100%",
                "maxWidth": "100%",
                "overflowX": "hidden",
            }
            data_hidden = {
                **data_style,
                "display": "none",
            }
            chainladder_style = {
                "display": "block",
                "minHeight": "70vh",
                "boxSizing": "border-box",
                "width": "100%",
                "maxWidth": "100%",
                "overflowX": "hidden",
            }
            chainladder_hidden = {
                **chainladder_style,
                "display": "none",
            }
            bf_style = {
                "display": "block",
                "background": COLOR_SURFACE,
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": RADIUS_LG,
                "padding": "16px",
                "boxShadow": SHADOW_SOFT,
                "minHeight": "70vh",
                "boxSizing": "border-box",
                "width": "100%",
                "maxWidth": "100%",
                "overflowX": "hidden",
            }
            bf_hidden = {
                **bf_style,
                "display": "none",
            }
            results_style = {
                "display": "block",
                "width": "100%",
                "background": COLOR_SURFACE,
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": RADIUS_LG,
                "padding": "12px",
                "boxShadow": SHADOW_SOFT,
                "minHeight": "70vh",
                "boxSizing": "border-box",
                "maxWidth": "100%",
                "overflowX": "hidden",
            }
            results_hidden = {
                **results_style,
                "display": "none",
            }

            # Plotly tables can render without header text when first drawn inside a
            # hidden container (display: none). Only (re)draw when the Results tab
            # is visible, and bump datarevision to force a redraw.
            results_figure = no_update
            if active_tab == "results":
                if not isinstance(results_payload, dict):
                    figure_dict: dict = self._plot_reserving_results_table(
                        self.results,
                        "Reserving Results",
                    ).to_dict()
                else:
                    figure_dict = deepcopy(results_payload.get("results_figure") or {})

                layout = dict(figure_dict.get("layout") or {})
                layout.setdefault("autosize", True)
                layout["datarevision"] = int(time.time() * 1000)
                figure_dict["layout"] = layout
                results_figure = figure_dict
            return (
                data_style if active_tab == "data" else data_hidden,
                chainladder_style
                if active_tab == "chainladder"
                else chainladder_hidden,
                bf_style if active_tab == "bornhuetter_ferguson" else bf_hidden,
                results_style if active_tab == "results" else results_hidden,
                results_figure,
            )

        @self.app.callback(
            Output("params-store", "data"),
            Input("triangle-heatmap", "clickData"),
            Input("average-method", "value"),
            Input("tail-method", "value"),
            Input("bf-apriori-table", "data"),
            Input("sync-inbox", "value"),
            Input("page-location", "pathname"),
            State("params-store", "data"),
            State("results-store", "data"),
        )
        def _reduce_params(
            click,
            average,
            tail_curve,
            bf_apriori_rows,
            sync_inbox,
            _pathname,
            current_params,
            current_results,
        ):
            ctx = callback_context
            trigger = "page-location"
            if ctx.triggered:
                trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            current_request_id = 0
            if isinstance(current_params, dict):
                try:
                    current_request_id = int(current_params.get("request_id", 0))
                except (TypeError, ValueError):
                    current_request_id = 0

            if trigger == "page-location" or not isinstance(current_params, dict):
                next_params = self._load_session_params_state(
                    request_id=current_request_id + 1,
                    force_recalc=True,
                )
                logging.info(
                    "[params-reducer] load request=%s force_recalc=%s",
                    next_params.get("request_id"),
                    next_params.get("force_recalc"),
                )
                return next_params

            working_params = self._build_params_state(
                drop_store=current_params.get("drop_store"),
                average=current_params.get("average"),
                tail_attachment_age=current_params.get("tail_attachment_age"),
                tail_curve=current_params.get("tail_curve"),
                tail_fit_period_selection=current_params.get(
                    "tail_fit_period_selection"
                ),
                bf_apriori_by_uwy=current_params.get("bf_apriori_by_uwy"),
                request_id=current_request_id,
                source="local",
                force_recalc=False,
                sync_version=None,
            )
            changed = False

            if trigger == "triangle-heatmap":
                if not click or "points" not in click or not click["points"]:
                    return no_update
                point = click["points"][0]
                origin = point.get("y")
                dev_label = point.get("x")
                dev = self._parse_dev_label(dev_label)
                if dev is None:
                    return no_update
                if origin == "Tail":
                    next_tail = (
                        None if working_params["tail_attachment_age"] == dev else dev
                    )
                    if working_params["tail_attachment_age"] != next_tail:
                        working_params["tail_attachment_age"] = next_tail
                        changed = True
                elif origin == "LDF":
                    updated_fit = self._toggle_tail_fit_selection(
                        working_params["tail_fit_period_selection"],
                        dev,
                    )
                    if updated_fit != working_params["tail_fit_period_selection"]:
                        working_params["tail_fit_period_selection"] = updated_fit
                        changed = True
                elif dev_label is not None:
                    updated_drops = self._toggle_drop(
                        working_params["drop_store"],
                        str(origin),
                        dev,
                    )
                    if updated_drops != working_params["drop_store"]:
                        working_params["drop_store"] = updated_drops
                        changed = True

            elif trigger == "average-method":
                next_average = average or self._default_average
                if next_average != working_params["average"]:
                    working_params["average"] = next_average
                    changed = True

            elif trigger == "tail-method":
                next_tail_curve = tail_curve or self._default_tail_curve
                if next_tail_curve != working_params["tail_curve"]:
                    working_params["tail_curve"] = next_tail_curve
                    changed = True

            elif trigger == "bf-apriori-table":
                next_bf = self._bf_rows_to_mapping(bf_apriori_rows)
                if next_bf != working_params["bf_apriori_by_uwy"]:
                    working_params["bf_apriori_by_uwy"] = next_bf
                    changed = True

            elif trigger == "sync-inbox":
                message = self._parse_sync_payload(sync_inbox)
                if not message:
                    return no_update
                if message.get("type") != "session_changed":
                    return no_update
                if message.get("user_key") != self._get_segment_key():
                    return no_update
                incoming_raw = message.get("sync_version", 0)
                try:
                    incoming_version = int(incoming_raw)
                except (TypeError, ValueError):
                    return no_update

                current_sync_version = 0
                if isinstance(current_results, dict):
                    try:
                        current_sync_version = int(
                            current_results.get("sync_version", 0)
                        )
                    except (TypeError, ValueError):
                        current_sync_version = 0
                if incoming_version <= current_sync_version:
                    return no_update
                if self._config is None:
                    return no_update

                session = self._config.load_session()
                synced_params = self._build_params_state(
                    drop_store=session.get("drops"),
                    average=session.get("average", self._default_average),
                    tail_attachment_age=session.get("tail_attachment_age"),
                    tail_curve=session.get("tail_curve", self._default_tail_curve),
                    tail_fit_period_selection=session.get("tail_fit_period"),
                    bf_apriori_by_uwy=session.get("bf_apriori_by_uwy"),
                    request_id=current_request_id + 1,
                    source="sync",
                    force_recalc=False,
                    sync_version=incoming_version,
                )
                logging.info(
                    "[params-reducer] sync request=%s sync_version=%s",
                    synced_params.get("request_id"),
                    incoming_version,
                )
                return synced_params

            if not changed:
                return no_update

            working_params["request_id"] = current_request_id + 1
            working_params["source"] = "local"
            working_params["force_recalc"] = False
            working_params["sync_version"] = None
            logging.info(
                "[params-reducer] local request=%s trigger=%s",
                working_params.get("request_id"),
                trigger,
            )
            return working_params

        @self.app.callback(
            Output("triangle-heatmap", "clickData"),
            Input("params-store", "data"),
            prevent_initial_call=True,
        )
        def _clear_clicks(_params):
            return None

        @self.app.callback(
            Output("results-store", "data"),
            Output("sync-publish-message", "data"),
            Input("params-store", "data"),
            State("results-store", "data"),
            State("sync-ready", "data"),
        )
        def _update_results(params, current_payload, sync_ready):
            if not params or not isinstance(params, dict):
                return no_update, no_update

            callback_started = time.perf_counter()
            request_id = params.get("request_id", "n/a")
            drop_store = params.get("drop_store") or []
            average = params.get("average") or self._default_average
            tail_attachment_age = params.get("tail_attachment_age")
            tail_curve = params.get("tail_curve") or self._default_tail_curve
            tail_fit_period_selection = params.get("tail_fit_period_selection") or []
            bf_apriori_by_uwy = params.get("bf_apriori_by_uwy") or {}
            force_recalc = bool(params.get("force_recalc"))

            cache_key = self._build_results_cache_key(
                drop_store,
                average,
                tail_attachment_age,
                tail_curve,
                tail_fit_period_selection,
                bf_apriori_by_uwy,
            )

            if (
                not force_recalc
                and isinstance(current_payload, dict)
                and current_payload.get("cache_key") == cache_key
            ):
                logging.info(
                    "[recalc] skip unchanged request=%s cache_key=%s",
                    request_id,
                    cache_key,
                )
                total_elapsed_ms = (time.perf_counter() - callback_started) * 1000
                logging.info("Callback total completed in %.0f ms", total_elapsed_ms)
                return current_payload, no_update

            logging.info(
                "[recalc] start request=%s source=%s force=%s",
                request_id,
                params.get("source"),
                force_recalc,
            )
            try:
                results_payload = self._get_or_build_results_payload(
                    drop_store=drop_store,
                    average=average,
                    tail_attachment_age=tail_attachment_age,
                    tail_curve=tail_curve,
                    tail_fit_period_selection=tail_fit_period_selection,
                    bf_apriori_by_uwy=bf_apriori_by_uwy,
                    force_recalc=force_recalc,
                )
            except Exception as exc:
                logging.error(f"Failed to recalculate reserving: {exc}", exc_info=True)
                return no_update, no_update

            try:
                results_payload["figure_version"] = int(request_id)
            except (TypeError, ValueError):
                results_payload["figure_version"] = 0

            source = params.get("source")
            if source == "sync":
                incoming_sync = params.get("sync_version")
                try:
                    sync_version = int(incoming_sync)
                except (TypeError, ValueError):
                    sync_version = 0
                results_payload["sync_version"] = sync_version
                segment_key = self._get_segment_key()
                _LIVE_RESULTS_BY_SEGMENT[segment_key] = results_payload
                logging.info(
                    "[recalc] finish request=%s sync_version=%s",
                    request_id,
                    sync_version,
                )
                total_elapsed_ms = (time.perf_counter() - callback_started) * 1000
                logging.info("Callback total completed in %.0f ms", total_elapsed_ms)
                return results_payload, no_update

            sync_version = 0
            if self._config is not None:
                sync_version = self._config.save_session_with_version(
                    {
                        "average": average,
                        "tail_curve": tail_curve,
                        "drops": drop_store,
                        "tail_attachment_age": tail_attachment_age,
                        "tail_fit_period": tail_fit_period_selection,
                        "bf_apriori_by_uwy": bf_apriori_by_uwy,
                    }
                )
            elif isinstance(current_payload, dict):
                try:
                    sync_version = int(current_payload.get("sync_version", 0)) + 1
                except (TypeError, ValueError):
                    sync_version = 1
            else:
                sync_version = 1

            results_payload["sync_version"] = sync_version
            segment_key = self._get_segment_key()
            _LIVE_RESULTS_BY_SEGMENT[segment_key] = results_payload

            previous_sync_version = 0
            if isinstance(current_payload, dict):
                try:
                    previous_sync_version = int(current_payload.get("sync_version", 0))
                except (TypeError, ValueError):
                    previous_sync_version = 0

            if not bool(sync_ready):
                logging.warning(
                    "Tab sync bridge unavailable; update applied only in current tab"
                )
            if sync_version <= previous_sync_version:
                logging.info(
                    "Session unchanged; skipping sync publish for segment '%s'",
                    segment_key,
                )
                logging.info(
                    "[recalc] finish request=%s sync_version=%s (no publish)",
                    request_id,
                    sync_version,
                )
                total_elapsed_ms = (time.perf_counter() - callback_started) * 1000
                logging.info("Callback total completed in %.0f ms", total_elapsed_ms)
                return results_payload, no_update

            publish_message = {
                "sync_version": sync_version,
                "updated_at": results_payload.get("last_updated"),
            }
            logging.info(
                "Publishing sync update for segment '%s' at version %s",
                segment_key,
                sync_version,
            )
            logging.info(
                "[recalc] finish request=%s sync_version=%s",
                request_id,
                sync_version,
            )
            total_elapsed_ms = (time.perf_counter() - callback_started) * 1000
            logging.info("Callback total completed in %.0f ms", total_elapsed_ms)
            return results_payload, publish_message

        @self.app.callback(
            Output("average-method", "value"),
            Output("tail-method", "value"),
            Output("bf-apriori-table", "data"),
            Input("params-store", "data"),
        )
        def _hydrate_controls(params):
            if not isinstance(params, dict):
                return (
                    self._default_average,
                    self._default_tail_curve,
                    self._build_bf_apriori_rows(self._default_bf_apriori_rows),
                )
            bf_rows = self._build_bf_apriori_rows(params.get("bf_apriori_by_uwy"))
            return (
                params.get("average", self._default_average),
                params.get("tail_curve", self._default_tail_curve),
                bf_rows,
            )

        @self.app.callback(
            Output("triangle-base-figure", "data"),
            Output("emergence-plot", "figure"),
            Input("results-store", "data"),
        )
        def _hydrate_tabs(results_payload):
            if not results_payload:
                return (
                    {
                        "figure": self._plot_triangle_heatmap_clean(
                            self.triangle,
                            "Triangle - Link Ratios",
                            self._default_tail_attachment_age,
                            self._default_tail_fit_period_selection,
                        ).to_dict(),
                        "figure_version": 0,
                    },
                    self._plot_emergence(
                        self.emergence_pattern,
                        "Emergence Pattern",
                    ),
                )

            return (
                {
                    "figure": results_payload.get("triangle_figure"),
                    "figure_version": results_payload.get("figure_version", 0),
                },
                results_payload.get("emergence_figure"),
            )

        @self.app.callback(
            Output("data-view-store", "data"),
            Input("data-view-toggle", "n_clicks"),
            State("data-view-store", "data"),
            prevent_initial_call=True,
        )
        def _toggle_data_view(_n_clicks, current_mode):
            if current_mode == "incremental":
                return "cumulative"
            return "incremental"

        @self.app.callback(
            Output("data-view-toggle-track", "style"),
            Output("data-view-toggle-knob", "style"),
            Output("data-view-mode-label", "children"),
            Input("data-view-store", "data"),
        )
        def _hydrate_data_view_toggle(mode):
            is_incremental = mode == "incremental"
            track_style = {
                "width": "42px",
                "height": "24px",
                "borderRadius": "999px",
                "backgroundColor": COLOR_ACCENT if is_incremental else "#cbd5e1",
                "position": "relative",
                "transition": "background-color 0.2s ease",
                "display": "inline-block",
            }
            knob_style = {
                "width": "18px",
                "height": "18px",
                "borderRadius": "50%",
                "backgroundColor": "#ffffff",
                "position": "absolute",
                "top": "3px",
                "left": "21px" if is_incremental else "3px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.22)",
                "transition": "left 0.2s ease",
            }
            mode_label = "Incremental" if is_incremental else "Cumulative"
            return track_style, knob_style, mode_label

        @self.app.callback(
            Output("data-triangle-view", "figure"),
            Input("data-metric-selector", "value"),
            Input("data-divisor-selector", "value"),
            Input("data-view-store", "data"),
            Input("results-store", "data"),
        )
        def _update_data_triangle(metric, divisor, triangle_view, _results_payload):
            metric_value = metric or "incurred"
            divisor_value = divisor or "none"
            view_value = triangle_view or "cumulative"
            triangle_df, weights_df, ratio_mode = self._build_data_tab_display(
                metric_value,
                view_value,
                divisor_value,
            )
            title_map = {
                "incurred": "Data Triangle - Incurred",
                "paid": "Data Triangle - Paid",
                "outstanding": "Data Triangle - Outstanding (Incurred - Paid)",
                "premium": "Data Triangle - Premium",
            }
            divisor_map = {
                "incurred": "Incurred",
                "paid": "Paid",
                "outstanding": "Outstanding",
                "premium": "Premium",
                "none": "None",
            }
            view_label = "Incremental" if view_value == "incremental" else "Cumulative"
            base_title = title_map.get(metric_value, "Data Triangle - Incurred")
            if divisor_value != "none":
                base_title = (
                    f"{base_title} / {divisor_map.get(divisor_value, 'Unknown')}"
                )
            return self._plot_data_triangle_table(
                triangle_df,
                f"{base_title} ({view_label})",
                weights_df=weights_df,
                ratio_mode=ratio_mode,
            )

    def _plot_emergence(self, emergence_pattern, title):
        """
        Plot emergence pattern with UWY lines and expected line.

        Args:
            emergence_pattern: DataFrame with multi-level columns ('Actual', 'Expected')
            title: Title for the plot

        Returns:
            Plotly Figure object
        """
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
                                size=ALERT_ANNOTATION_FONT_SIZE,
                                family=FONT_FAMILY,
                            ),
                        )
                    ],
                )
            )

        fig = go.Figure()

        # Plot each UWY line from 'Actual' data
        actual_data = emergence_pattern["Actual"]
        for idx in actual_data.index:
            # Extract year from index (handles both datetime and string formats)
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

        # Plot expected line (same for all UWYs, so just take first row)
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
            font=dict(color=COLOR_TEXT, size=FIGURE_FONT_SIZE, family=FONT_FAMILY),
            title_font=dict(
                color=COLOR_TEXT,
                size=FIGURE_TITLE_FONT_SIZE,
                family=FONT_FAMILY,
            ),
            hoverlabel=dict(
                bgcolor=COLOR_SURFACE,
                bordercolor=COLOR_BORDER,
                font=dict(
                    color=COLOR_TEXT,
                    size=FIGURE_FONT_SIZE,
                    family=FONT_FAMILY,
                ),
            ),
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
            height=600,
            autosize=True,
            uirevision="static",
        )

        return fig

    def _plot_reserving_results_table(self, results_df, title):
        """
        Plot reserving results as an interactive table.

        Args:
            results_df: DataFrame with columns [cl_ultimate, cl_loss_ratio, bf_ultimate, bf_loss_ratio, incurred, Premium, ultimate]
            title: Title for the table

        Returns:
            Plotly Figure object with table
        """
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
                                size=ALERT_ANNOTATION_FONT_SIZE,
                                family=FONT_FAMILY,
                            ),
                        )
                    ],
                )
            )

        # Prepare data: extract year from index and format numbers
        df_display = results_df.copy()

        # Extract year from index (handles datetime or string formats)
        uwy_years = []
        for idx in df_display.index:
            if hasattr(idx, "year"):
                uwy_years.append(str(idx.year))
            else:
                # Handle string format like '2020-01-01'
                uwy_years.append(str(idx)[:4])

        # Compute incurred loss ratio
        incurred_loss_ratios = []
        for inc, prem in zip(df_display["incurred"], df_display["Premium"]):
            if pd.isna(prem) or prem == 0 or pd.isna(inc):
                incurred_loss_ratios.append("N/A")
            else:
                incurred_loss_ratios.append(f"{(inc / prem):.2%}")

        # Compute IBNR = selected ultimate - incurred
        ibnr_values = df_display["ultimate"] - df_display["incurred"]

        # Create enhanced table data with formatted values showing both CL and BF methods
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

        # Color coding: neutral for basic data, blue for CL, orange for BF, green for selected
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
                            size=TABLE_HEADER_FONT_SIZE,
                            family=FONT_FAMILY,
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
                            size=TABLE_CELL_FONT_SIZE,
                            family=FONT_FAMILY,
                        ),
                        height=28,
                    ),
                )
            ]
        )

        fig.update_layout(
            title=title,
            template="plotly_white",
            font=dict(color=COLOR_TEXT, size=FIGURE_FONT_SIZE, family=FONT_FAMILY),
            title_font=dict(
                color=COLOR_TEXT,
                size=FIGURE_TITLE_FONT_SIZE,
                family=FONT_FAMILY,
            ),
            hoverlabel=dict(
                bgcolor=COLOR_SURFACE,
                bordercolor=COLOR_BORDER,
                font=dict(
                    color=COLOR_TEXT,
                    size=FIGURE_FONT_SIZE,
                    family=FONT_FAMILY,
                ),
            ),
            height=min(
                650, 200 + len(df_display) * 28
            ),  # Dynamic height based on rows, increased for wider table
            margin=dict(l=20, r=20, t=80, b=20),
        )

        return fig

    def _create_triangle_heatmap(
        self,
        triangle_data,
        incurred_data,
        premium_data,
        title,
        reserving: Reserving,
        tail_attachment_age: Optional[int],
        tail_fit_period_selection: Optional[List[int]],
    ):
        """
        Plot triangle heatmap with link ratios, LDF, and Tail rows.
        Uses column-wise normalization for better contrast.
        Hover tooltip shows link ratios, cumulative incurred, and premium.

        Args:
            triangle_data: DataFrame with link ratios (rows = UWYs, columns = dev periods)
            incurred_data: DataFrame with cumulative incurred values
            premium_data: DataFrame with premium values
            title: Title for the plot
            reserving: Reserving object to access raw triangle data

        Returns:
            Plotly Figure object
        """
        figure_started = time.perf_counter()
        if triangle_data is None:
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
                                size=ALERT_ANNOTATION_FONT_SIZE,
                                family=FONT_FAMILY,
                            ),
                        )
                    ],
                )
            )

        core_started = time.perf_counter()
        core_cache_key = self._build_heatmap_core_cache_key(
            triangle_data,
            incurred_data,
            premium_data,
        )
        core_payload = self._cache_get(self._heatmap_core_cache, core_cache_key)
        if core_payload is None:
            core_payload = self._build_heatmap_core(
                triangle_data,
                incurred_data,
                premium_data,
                reserving,
            )
            self._cache_set(self._heatmap_core_cache, core_cache_key, core_payload)
            core_elapsed_ms = (time.perf_counter() - core_started) * 1000
            logging.info("Triangle heatmap core built in %.0f ms", core_elapsed_ms)
        else:
            core_elapsed_ms = (time.perf_counter() - core_started) * 1000
            logging.info("Triangle heatmap core reused in %.0f ms", core_elapsed_ms)

        expanded_triangle = core_payload["expanded_triangle"]
        z_data = core_payload["z_data"]
        text_values = core_payload["text_values"]
        customdata = core_payload["customdata"]

        # Calculate figure dimensions based on cell size
        n_cols = len(expanded_triangle.columns)
        n_rows = len(expanded_triangle.index)
        cell_width = 42  # pixels per cell (scaled to match 70% browser zoom appearance)
        cell_height = (
            28  # pixels per cell (scaled to match 70% browser zoom appearance)
        )

        fig_width = n_cols * cell_width
        fig_height = n_rows * cell_height + 120  # Extra space for title and axes

        # Create heatmap with normalized colors but original values as text
        fig = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=expanded_triangle.columns,
                y=[
                    str(idx)[:4] if idx not in ["LDF", "Tail"] else idx
                    for idx in expanded_triangle.index
                ],
                text=text_values,
                texttemplate="%{text}",
                textfont={
                    "size": HEATMAP_TEXT_FONT_SIZE,
                    "family": FONT_FAMILY,
                    "color": COLOR_TEXT,
                },
                colorscale="RdBu_r",
                showscale=False,
                zmin=0,
                zmax=1,
                customdata=customdata,
                hovertemplate="UWY: %{y}<br>Dev Period: %{x}<br>Link Ratio: %{text}<br>Incurred: %{customdata[0]}<br>Premium: %{customdata[1]}<extra></extra>",
            )
        )

        # Initially show approximately 143 columns (100/0.7 to match 70% zoom viewing area)
        initial_cols = min(143, n_cols)

        fig.update_layout(
            title=title,
            xaxis_title="Development Period",
            yaxis_title="Underwriting Year",
            template="plotly_white",
            font=dict(color=COLOR_TEXT, size=FIGURE_FONT_SIZE, family=FONT_FAMILY),
            title_font=dict(
                color=COLOR_TEXT,
                size=FIGURE_TITLE_FONT_SIZE,
                family=FONT_FAMILY,
            ),
            hoverlabel=dict(
                bgcolor=COLOR_SURFACE,
                bordercolor=COLOR_BORDER,
                font=dict(
                    color=COLOR_TEXT,
                    size=FIGURE_FONT_SIZE,
                    family=FONT_FAMILY,
                ),
            ),
            width=fig_width,
            height=fig_height,
            yaxis=dict(autorange="reversed"),  # Reverse y-axis so oldest year at top
            xaxis=dict(
                range=[-0.5, initial_cols - 0.5]
            ),  # Show first 100 columns initially
            margin=dict(l=80, r=20, t=80, b=60),  # Reduce margins
            uirevision="static",
        )

        # Add black crosses through dropped cells
        for row_label, col_label in dropped_mask.stack()[dropped_mask.stack()].index:
            row_pos = expanded_triangle.index.get_loc(row_label)
            col_pos = expanded_triangle.columns.get_loc(col_label)
            # Diagonal line 1
            fig.add_shape(
                type="line",
                x0=col_pos - 0.5,
                y0=row_pos - 0.5,
                x1=col_pos + 0.5,
                y1=row_pos + 0.5,
                line=dict(color="black", width=2),
                xref="x",
                yref="y",
            )
            # Diagonal line 2
            fig.add_shape(
                type="line",
                x0=col_pos - 0.5,
                y0=row_pos + 0.5,
                x1=col_pos + 0.5,
                y1=row_pos - 0.5,
                line=dict(color="black", width=2),
                xref="x",
                yref="y",
            )

        if tail_attachment_age is not None:
            try:
                tail_row_pos = expanded_triangle.index.get_loc("Tail")
            except KeyError:
                tail_row_pos = None

            if tail_row_pos is not None:
                selected_col = None
                for col in expanded_triangle.columns:
                    col_age = self._parse_dev_label(col)
                    if col_age == tail_attachment_age:
                        selected_col = col
                        break

                if selected_col is not None:
                    col_pos = expanded_triangle.columns.get_loc(selected_col)
                    fig.add_shape(
                        type="rect",
                        x0=col_pos - 0.5,
                        y0=tail_row_pos - 0.5,
                        x1=col_pos + 0.5,
                        y1=tail_row_pos + 0.5,
                        line=dict(color="black", width=2),
                        fillcolor="rgba(0,0,0,0)",
                        xref="x",
                        yref="y",
                    )

        fit_period = self._derive_tail_fit_period(tail_fit_period_selection)
        if fit_period is not None:
            try:
                ldf_row_pos = expanded_triangle.index.get_loc("LDF")
            except KeyError:
                ldf_row_pos = None

            if ldf_row_pos is not None:
                lower_value, upper_value = fit_period
                lower_col = None
                upper_col = None
                for col in expanded_triangle.columns:
                    col_age = self._parse_dev_label(col)
                    if col_age == lower_value:
                        lower_col = col
                    if upper_value is not None and col_age == upper_value:
                        upper_col = col

                if lower_col is not None:
                    lower_pos = expanded_triangle.columns.get_loc(lower_col)
                    if upper_value is None or upper_col is None:
                        fig.add_shape(
                            type="rect",
                            x0=lower_pos - 0.5,
                            y0=ldf_row_pos - 0.5,
                            x1=lower_pos + 0.5,
                            y1=ldf_row_pos + 0.5,
                            line=dict(color="black", width=2),
                            fillcolor="rgba(0,0,0,0)",
                            xref="x",
                            yref="y",
                        )
                    else:
                        upper_pos = expanded_triangle.columns.get_loc(upper_col)
                        start_pos = min(lower_pos, upper_pos)
                        end_pos = max(lower_pos, upper_pos)
                        fig.add_shape(
                            type="rect",
                            x0=start_pos - 0.5,
                            y0=ldf_row_pos - 0.5,
                            x1=end_pos + 0.5,
                            y1=ldf_row_pos + 0.5,
                            line=dict(color="black", width=2),
                            fillcolor="rgba(0,0,0,0)",
                            xref="x",
                            yref="y",
                        )
                        fig.add_shape(
                            type="rect",
                            x0=lower_pos - 0.5,
                            y0=ldf_row_pos - 0.5,
                            x1=lower_pos + 0.5,
                            y1=ldf_row_pos + 0.5,
                            line=dict(color="black", width=3),
                            fillcolor="rgba(0,0,0,0)",
                            xref="x",
                            yref="y",
                        )
                        fig.add_shape(
                            type="rect",
                            x0=upper_pos - 0.5,
                            y0=ldf_row_pos - 0.5,
                            x1=upper_pos + 0.5,
                            y1=ldf_row_pos + 0.5,
                            line=dict(color="black", width=3),
                            fillcolor="rgba(0,0,0,0)",
                            xref="x",
                            yref="y",
                        )

        total_elapsed_ms = (time.perf_counter() - figure_started) * 1000
        logging.info("Triangle heatmap total built in %.0f ms", total_elapsed_ms)
        return fig

    def _plot_triangle_heatmap(
        self,
        triangle_data,
        title,
        tail_attachment_age: Optional[int],
        tail_fit_period_selection: Optional[List[int]],
    ):
        """
        Plot triangle heatmap with incurred/premium hover values.
        """
        return self._create_triangle_heatmap(
            triangle_data,
            self.incurred,
            self.premium,
            title,
            self._reserving,
            tail_attachment_age,
            tail_fit_period_selection,
        )

    def _plot_triangle_heatmap_clean(
        self,
        triangle_data,
        title,
        tail_attachment_age: Optional[int],
        tail_fit_period_selection: Optional[List[int]],
    ):
        """
        Plot chainladder heatmap with cleaner, table-like styling.
        """
        render_started = time.perf_counter()
        core_cache_key = self._build_heatmap_core_cache_key(
            triangle_data,
            self.incurred,
            self.premium,
        )
        core_payload = self._cache_get(self._heatmap_core_cache, core_cache_key)
        if core_payload is None:
            core_payload = self._build_heatmap_core(
                triangle_data,
                self.incurred,
                self.premium,
                self._reserving,
            )
            self._cache_set(self._heatmap_core_cache, core_cache_key, core_payload)
            core_elapsed_ms = (time.perf_counter() - render_started) * 1000
            logging.info(
                "Triangle heatmap core built for clean view in %.0f ms",
                core_elapsed_ms,
            )
        else:
            core_elapsed_ms = (time.perf_counter() - render_started) * 1000
            logging.info(
                "Triangle heatmap core reused for clean view in %.0f ms",
                core_elapsed_ms,
            )

        expanded_triangle = core_payload["expanded_triangle"]
        dropped_mask = core_payload["dropped_mask"]
        z_data = core_payload["z_data"]
        text_values = core_payload["text_values"]
        customdata = core_payload["customdata"]

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
                    "size": HEATMAP_TEXT_FONT_SIZE,
                    "family": FONT_FAMILY,
                    "color": COLOR_TEXT,
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
                    col_age = self._parse_dev_label(col)
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

        fit_period = self._derive_tail_fit_period(tail_fit_period_selection)
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
                    col_age = self._parse_dev_label(col)
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
            paper_bgcolor=COLOR_SURFACE,
            plot_bgcolor=COLOR_SURFACE,
            font={
                "family": FONT_FAMILY,
                "color": COLOR_TEXT,
                "size": FIGURE_FONT_SIZE,
            },
            title_font={
                "family": FONT_FAMILY,
                "color": COLOR_TEXT,
                "size": FIGURE_TITLE_FONT_SIZE,
            },
            hoverlabel={
                "bgcolor": COLOR_SURFACE,
                "bordercolor": COLOR_BORDER,
                "font": {
                    "family": FONT_FAMILY,
                    "color": COLOR_TEXT,
                    "size": FIGURE_FONT_SIZE,
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

        total_elapsed_ms = (time.perf_counter() - render_started) * 1000
        logging.info("Triangle heatmap clean built in %.0f ms", total_elapsed_ms)
        return fig

    def _create_layout(self):
        """
        Create the Dash layout for the dashboard with tabbed interface.
        Tab 1: Triangle heatmap
        Tab 2: Emergence pattern

        Returns:
            Dash html.Div component
        """
        self._load_session_defaults()
        results_payload = self._build_results_payload(
            drop_store=self._default_drop_store,
            average=self._default_average,
            tail_attachment_age=self._default_tail_attachment_age,
            tail_curve=self._default_tail_curve,
            tail_fit_period_selection=self._default_tail_fit_period_selection,
            bf_apriori_by_uwy=self._bf_rows_to_mapping(self._default_bf_apriori_rows),
        )
        results_payload["figure_version"] = 0
        initial_sync_version = 0
        if self._config is not None:
            initial_sync_version = self._config.get_sync_version()
        results_payload["sync_version"] = initial_sync_version
        segment_key = self._get_segment_key()
        existing_payload = _LIVE_RESULTS_BY_SEGMENT.get(segment_key)
        if existing_payload:
            results_payload = existing_payload
        _LIVE_RESULTS_BY_SEGMENT[segment_key] = results_payload
        initial_data_triangle, initial_data_weights, initial_ratio_mode = (
            self._build_data_tab_display(
                "incurred",
                "cumulative",
                "none",
            )
        )
        initial_bf_apriori_rows = self._build_bf_apriori_rows(
            self._default_bf_apriori_rows
        )
        return html.Div(
            [
                dcc.Location(id="page-location", refresh=False),
                dcc.Store(id="params-store", data=None),
                dcc.Store(id="results-store", data=results_payload),
                dcc.Store(
                    id="triangle-base-figure",
                    data={
                        "figure": results_payload.get("triangle_figure"),
                        "figure_version": results_payload.get("figure_version", 0),
                    },
                ),
                dcc.Store(id="triangle-overlay-signature", data=None),
                dcc.Store(id="data-view-store", data="cumulative"),
                dcc.Store(id="active-tab", data="data"),
                dcc.Store(id="sidebar-collapsed", data=False),
                dcc.Store(id="sync-user-key", data=segment_key),
                dcc.Store(id="sync-tab-id", data=""),
                dcc.Store(id="sync-ready", data=False),
                dcc.Store(id="sync-publish-message", data=None),
                dcc.Input(
                    id="sync-inbox",
                    type="text",
                    value="",
                    style={"display": "none"},
                ),
                html.Div(id="sync-publish-signal", style={"display": "none"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Reserving",
                                    id="sidebar-title",
                                    style={
                                        "fontWeight": 600,
                                        "fontSize": "15px",
                                        "marginBottom": "18px",
                                        "textAlign": "center",
                                        "color": COLOR_TEXT,
                                        "letterSpacing": "0.2px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Button(id="nav-data", n_clicks=0),
                                        html.Button(id="nav-chainladder", n_clicks=0),
                                        html.Button(id="nav-bf", n_clicks=0),
                                        html.Button(id="nav-results", n_clicks=0),
                                    ],
                                    id="nav-stack",
                                ),
                                html.Button(
                                    id="sidebar-toggle",
                                    n_clicks=0,
                                ),
                            ],
                            id="sidebar",
                        ),
                        html.Div(
                            [
                                html.H1(
                                    "Reserving Dashboard",
                                    style={
                                        "textAlign": "left",
                                        "marginBottom": "16px",
                                        "fontFamily": FONT_FAMILY,
                                        "color": COLOR_TEXT,
                                        "fontSize": "24px",
                                        "fontWeight": 600,
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Data",
                                                    style={
                                                        "marginTop": "0",
                                                        "fontFamily": FONT_FAMILY,
                                                        "color": COLOR_TEXT,
                                                        "fontWeight": 600,
                                                        "fontSize": "18px",
                                                    },
                                                ),
                                                html.P(
                                                    "Select a data view and inspect the triangle in chainladder-style tabular form.",
                                                    style={
                                                        "marginTop": "6px",
                                                        "color": COLOR_MUTED,
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Triangle metric",
                                                                    style={
                                                                        "fontWeight": 600,
                                                                        "marginBottom": "6px",
                                                                        "display": "block",
                                                                    },
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="data-metric-selector",
                                                                    options=[
                                                                        {
                                                                            "label": "Incurred",
                                                                            "value": "incurred",
                                                                        },
                                                                        {
                                                                            "label": "Paid",
                                                                            "value": "paid",
                                                                        },
                                                                        {
                                                                            "label": "Outstanding",
                                                                            "value": "outstanding",
                                                                        },
                                                                        {
                                                                            "label": "Premium",
                                                                            "value": "premium",
                                                                        },
                                                                    ],
                                                                    value="incurred",
                                                                    clearable=False,
                                                                    style={
                                                                        "width": "260px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Triangle view",
                                                                    style={
                                                                        "fontWeight": 600,
                                                                        "marginBottom": "6px",
                                                                        "display": "block",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Button(
                                                                            html.Div(
                                                                                id="data-view-toggle-track",
                                                                                children=[
                                                                                    html.Div(
                                                                                        id="data-view-toggle-knob"
                                                                                    )
                                                                                ],
                                                                                style={
                                                                                    "width": "42px",
                                                                                    "height": "24px",
                                                                                    "borderRadius": "999px",
                                                                                    "backgroundColor": "#cbd5e1",
                                                                                    "position": "relative",
                                                                                    "display": "inline-block",
                                                                                },
                                                                            ),
                                                                            id="data-view-toggle",
                                                                            n_clicks=0,
                                                                            style={
                                                                                "border": "none",
                                                                                "background": "transparent",
                                                                                "padding": "0",
                                                                                "cursor": "pointer",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            "Cumulative",
                                                                            id="data-view-mode-label",
                                                                            style={
                                                                                "color": COLOR_TEXT,
                                                                                "fontSize": "14px",
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "display": "flex",
                                                                        "alignItems": "center",
                                                                        "gap": "10px",
                                                                        "marginTop": "4px",
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "In relation to",
                                                                    style={
                                                                        "fontWeight": 600,
                                                                        "marginBottom": "6px",
                                                                        "display": "block",
                                                                    },
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="data-divisor-selector",
                                                                    options=[
                                                                        {
                                                                            "label": "None",
                                                                            "value": "none",
                                                                        },
                                                                        {
                                                                            "label": "Incurred",
                                                                            "value": "incurred",
                                                                        },
                                                                        {
                                                                            "label": "Paid",
                                                                            "value": "paid",
                                                                        },
                                                                        {
                                                                            "label": "Outstanding",
                                                                            "value": "outstanding",
                                                                        },
                                                                        {
                                                                            "label": "Premium",
                                                                            "value": "premium",
                                                                        },
                                                                    ],
                                                                    value="none",
                                                                    clearable=False,
                                                                    style={
                                                                        "width": "200px"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    style={
                                                        "marginTop": "14px",
                                                        "display": "flex",
                                                        "gap": "24px",
                                                        "alignItems": "flex-end",
                                                        "flexWrap": "wrap",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        dcc.Graph(
                                                            id="data-triangle-view",
                                                            figure=self._plot_data_triangle_table(
                                                                initial_data_triangle,
                                                                "Data Triangle - Incurred (Cumulative)",
                                                                weights_df=initial_data_weights,
                                                                ratio_mode=initial_ratio_mode,
                                                            ),
                                                            config={
                                                                "displayModeBar": False,
                                                                "responsive": False,
                                                            },
                                                            style={"width": "100%"},
                                                        )
                                                    ],
                                                    style={
                                                        "marginTop": "16px",
                                                        "background": COLOR_SURFACE,
                                                        "border": f"1px solid {COLOR_BORDER}",
                                                        "borderRadius": RADIUS_LG,
                                                        "padding": "8px",
                                                        "boxShadow": SHADOW_SOFT,
                                                        "overflowX": "auto",
                                                    },
                                                ),
                                            ],
                                            id="tab-data",
                                            style={
                                                "display": "block",
                                                "background": COLOR_SURFACE,
                                                "border": f"1px solid {COLOR_BORDER}",
                                                "borderRadius": RADIUS_LG,
                                                "padding": "16px",
                                                "boxShadow": SHADOW_SOFT,
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Average method"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="average-method",
                                                                    options=[
                                                                        {
                                                                            "label": "Volume",
                                                                            "value": "volume",
                                                                        },
                                                                        {
                                                                            "label": "Simple",
                                                                            "value": "simple",
                                                                        },
                                                                    ],
                                                                    value=self._default_average,
                                                                    clearable=False,
                                                                    style={
                                                                        "width": "100%"
                                                                    },
                                                                ),
                                                            ],
                                                            style={"minWidth": "200px"},
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Tail method"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="tail-method",
                                                                    options=[
                                                                        {
                                                                            "label": "Weibull",
                                                                            "value": "weibull",
                                                                        },
                                                                        {
                                                                            "label": "Exponential",
                                                                            "value": "exponential",
                                                                        },
                                                                        {
                                                                            "label": "Inverse power",
                                                                            "value": "inverse_power",
                                                                        },
                                                                    ],
                                                                    value=self._default_tail_curve,
                                                                    clearable=False,
                                                                    style={
                                                                        "width": "100%"
                                                                    },
                                                                ),
                                                            ],
                                                            style={"minWidth": "200px"},
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "grid",
                                                        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                                                        "gap": "16px",
                                                        "marginBottom": "16px",
                                                        "background": COLOR_SURFACE,
                                                        "border": f"1px solid {COLOR_BORDER}",
                                                        "borderRadius": RADIUS_LG,
                                                        "padding": "16px",
                                                        "boxShadow": SHADOW_SOFT,
                                                        "width": "100%",
                                                        "maxWidth": "100%",
                                                        "boxSizing": "border-box",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                dcc.Graph(
                                                                    id="triangle-heatmap",
                                                                    figure=results_payload.get(
                                                                        "triangle_figure"
                                                                    ),
                                                                    config={
                                                                        "scrollZoom": False,
                                                                        "doubleClick": False,
                                                                        "displayModeBar": False,
                                                                        "responsive": False,
                                                                    },
                                                                    style={
                                                                        "width": "100%",
                                                                        "overflowX": "scroll",
                                                                        "overflowY": "hidden",
                                                                    },
                                                                )
                                                            ],
                                                            style={
                                                                "width": "100%",
                                                                "overflowX": "auto",
                                                                "background": COLOR_SURFACE,
                                                                "border": f"1px solid {COLOR_BORDER}",
                                                                "borderRadius": RADIUS_LG,
                                                                "padding": "8px",
                                                                "boxShadow": SHADOW_SOFT,
                                                                "maxWidth": "100%",
                                                                "boxSizing": "border-box",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Graph(
                                                                    id="emergence-plot",
                                                                    figure=results_payload.get(
                                                                        "emergence_figure"
                                                                    ),
                                                                    config={
                                                                        "responsive": False
                                                                    },
                                                                    style={
                                                                        "width": "100%"
                                                                    },
                                                                )
                                                            ],
                                                            style={
                                                                "width": "100%",
                                                                "background": COLOR_SURFACE,
                                                                "border": f"1px solid {COLOR_BORDER}",
                                                                "borderRadius": RADIUS_LG,
                                                                "padding": "8px",
                                                                "boxShadow": SHADOW_SOFT,
                                                                "marginTop": "16px",
                                                                "maxWidth": "100%",
                                                                "boxSizing": "border-box",
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ],
                                            id="tab-chainladder",
                                            style={"display": "none"},
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Bornhuetter-Ferguson",
                                                    style={
                                                        "marginTop": "0",
                                                        "fontFamily": FONT_FAMILY,
                                                        "color": COLOR_TEXT,
                                                        "fontWeight": 600,
                                                        "fontSize": "18px",
                                                    },
                                                ),
                                                html.P(
                                                    "Set expected loss ratios by underwriting year. These values are applied through the BF exposure input.",
                                                    style={
                                                        "marginTop": "6px",
                                                        "color": COLOR_MUTED,
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        dash_table.DataTable(
                                                            id="bf-apriori-table",
                                                            columns=[
                                                                {
                                                                    "name": "Underwriting year",
                                                                    "id": "uwy",
                                                                    "editable": False,
                                                                },
                                                                {
                                                                    "name": "Expected loss ratio",
                                                                    "id": "apriori",
                                                                    "type": "text",
                                                                    "presentation": "input",
                                                                    "editable": True,
                                                                },
                                                            ],
                                                            data=initial_bf_apriori_rows,
                                                            editable=True,
                                                            row_deletable=False,
                                                            style_as_list_view=True,
                                                            css=[
                                                                {
                                                                    "selector": ".dash-spreadsheet td input",
                                                                    "rule": "text-align: left; caret-color: #1f2a37; width: 100%; border: 0; background: transparent;",
                                                                },
                                                                {
                                                                    "selector": ".dash-spreadsheet td input:focus",
                                                                    "rule": "outline: none;",
                                                                },
                                                                {
                                                                    "selector": ".dash-spreadsheet td",
                                                                    "rule": "cursor: text;",
                                                                },
                                                            ],
                                                            style_table={
                                                                "overflowX": "auto",
                                                                "maxWidth": "560px",
                                                            },
                                                            style_header={
                                                                "fontWeight": 600,
                                                                "backgroundColor": "#f2f5f9",
                                                                "border": f"1px solid {COLOR_BORDER}",
                                                                "fontFamily": FONT_FAMILY,
                                                                "fontSize": f"{TABLE_HEADER_FONT_SIZE}px",
                                                            },
                                                            style_cell={
                                                                "padding": "8px",
                                                                "fontFamily": FONT_FAMILY,
                                                                "fontSize": f"{TABLE_CELL_FONT_SIZE}px",
                                                                "border": f"1px solid {COLOR_BORDER}",
                                                                "textAlign": "left",
                                                            },
                                                            style_data={
                                                                "backgroundColor": COLOR_SURFACE,
                                                            },
                                                            style_cell_conditional=[
                                                                {
                                                                    "if": {
                                                                        "column_id": "uwy"
                                                                    },
                                                                    "width": "50%",
                                                                },
                                                                {
                                                                    "if": {
                                                                        "column_id": "apriori"
                                                                    },
                                                                    "width": "50%",
                                                                },
                                                            ],
                                                        )
                                                    ],
                                                    style={
                                                        "marginTop": "14px",
                                                        "background": COLOR_SURFACE,
                                                        "border": f"1px solid {COLOR_BORDER}",
                                                        "borderRadius": RADIUS_LG,
                                                        "padding": "12px",
                                                        "boxShadow": SHADOW_SOFT,
                                                        "maxWidth": "620px",
                                                    },
                                                ),
                                            ],
                                            id="tab-bf",
                                            style={
                                                "display": "none",
                                                "background": COLOR_SURFACE,
                                                "border": f"1px solid {COLOR_BORDER}",
                                                "borderRadius": RADIUS_LG,
                                                "padding": "16px",
                                                "boxShadow": SHADOW_SOFT,
                                            },
                                        ),
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="results-table",
                                                    # Defer rendering until the Results tab is visible.
                                                    # This avoids Plotly table header text occasionally
                                                    # disappearing when a figure is first rendered in a
                                                    # hidden container.
                                                    figure={},
                                                    config={
                                                        "editable": False,
                                                        "displayModeBar": True,
                                                        "responsive": True,
                                                    },
                                                    style={"width": "100%"},
                                                )
                                            ],
                                            id="tab-results",
                                            style={
                                                "width": "100%",
                                                "display": "none",
                                                "background": COLOR_SURFACE,
                                                "border": f"1px solid {COLOR_BORDER}",
                                                "borderRadius": RADIUS_LG,
                                                "padding": "12px",
                                                "boxShadow": SHADOW_SOFT,
                                            },
                                        ),
                                    ],
                                    style={
                                        "paddingRight": "8px",
                                        "fontFamily": FONT_FAMILY,
                                        "color": COLOR_TEXT,
                                        "width": "100%",
                                        "maxWidth": "100%",
                                        "minWidth": "0",
                                    },
                                ),
                            ],
                            style={
                                "flex": "1",
                                "padding": "16px",
                                "background": COLOR_BG,
                                "borderRadius": RADIUS_LG,
                                "boxShadow": SHADOW_SOFT,
                                "minWidth": "0",
                                "maxWidth": "100%",
                                "overflowX": "hidden",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "alignItems": "stretch",
                        "minHeight": "100vh",
                        "width": "100%",
                        "minWidth": "0",
                    },
                ),
            ],
            style={
                "padding": "10px",
                "width": "100%",
                "background": COLOR_BG,
                "overflowX": "hidden",
                "maxWidth": "100vw",
                "boxSizing": "border-box",
                "fontFamily": FONT_FAMILY,
                "color": COLOR_TEXT,
            },
        )

    def show(self, debug=False, port=8050):
        """
        Launch the dashboard server.

        Args:
            debug: Enable debug mode (default: False - to prevent reloading parent script)
            port: Port to run the server on (default: 8050)
        """
        self.app.layout = self._create_layout
        logging.info(f"Starting dashboard on http://127.0.0.1:{port}")
        self.app.run(debug=debug, port=port, use_reloader=False)
