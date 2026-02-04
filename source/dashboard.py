from source.reserving import Reserving
from source.config_manager import ConfigManager
from typing import Optional, List, Tuple
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
from datetime import datetime
import plotly.graph_objs as go
import pandas as pd
import logging
import numpy as np


_LIVE_RESULTS_BY_SEGMENT: dict[str, dict] = {}


def format_to_millions(value):
    """
    Format values with dynamic units.
    Returns empty string for NaN/None values.

    Args:
        value: Numeric value to format

    Returns:
        Formatted string (e.g., '1.23m', '512.00k', '900') or empty string for NaN
    """
    if pd.isna(value) or value is None:
        return ""
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}m"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:.0f}"


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
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self._default_average = "volume"
        self._default_tail_curve = "weibull"
        self._default_drop_text = ""
        self._default_drop_store: List[List[str | int]] = []
        self._default_tail_attachment_age: Optional[int] = None
        self._default_tail_fit_period_selection: List[int] = []

        self._load_session_defaults()

        # Extract data from domain objects
        self._extract_data()

        self._register_callbacks()
        logging.info("Dashboard initialized successfully")

    def _load_session_defaults(self) -> None:
        if self._config is None:
            return
        session = self._config.load_session()
        self._default_average = session.get("average", self._default_average)
        self._default_tail_curve = session.get(
            "tail_curve",
            self._default_tail_curve,
        )
        raw_drops = session.get("drops", []) or []
        normalized: List[List[str | int]] = []
        for item in raw_drops:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append([item[0], item[1]])
        self._default_drop_store = normalized
        tail_attachment_age = session.get("tail_attachment_age")
        if tail_attachment_age is not None:
            try:
                self._default_tail_attachment_age = int(tail_attachment_age)
            except (TypeError, ValueError):
                self._default_tail_attachment_age = None
        self._default_tail_fit_period_selection = self._normalize_tail_fit_selection(
            session.get("tail_fit_period")
        )

    def _get_segment_key(self) -> str:
        if self._config is None:
            return "default"
        return self._config.get_segment()

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
        self._reserving.set_bornhuetter_ferguson(apriori=0.6)
        self._reserving.reserve(final_ultimate="chainladder")
        self._extract_data()

    def _build_results_payload(
        self,
        drop_store: Optional[List[List[str | int]]],
        average: Optional[str],
        tail_attachment_age: Optional[int],
        tail_curve: Optional[str],
        tail_fit_period_selection: Optional[List[int]],
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

        triangle_fig = self._plot_triangle_heatmap(
            self.triangle,
            "Triangle - Link Ratios",
            tail_attachment_age,
            tail_fit_period_selection,
        )
        emergence_fig = self._plot_emergence(
            self.emergence_pattern,
            "Emergence Pattern",
        )
        results_fig = self._plot_reserving_results_table(
            self.results,
            "Reserving Results",
        )

        return {
            "triangle_figure": triangle_fig.to_dict(),
            "emergence_figure": emergence_fig.to_dict(),
            "results_figure": results_fig.to_dict(),
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

    def _register_callbacks(self) -> None:
        @self.app.callback(
            Output("drop-store", "data"),
            Output("tail-attachment-store", "data"),
            Output("tail-fit-period-store", "data"),
            Input("triangle-heatmap", "clickData"),
            State("drop-store", "data"),
            State("tail-attachment-store", "data"),
            State("tail-fit-period-store", "data"),
        )
        def _toggle_drop_click(
            click,
            drop_store,
            tail_attachment_age,
            tail_fit_period_selection,
        ):
            if not click or "points" not in click or not click["points"]:
                return drop_store, tail_attachment_age, tail_fit_period_selection

            point = click["points"][0]
            origin = point.get("y")
            dev_label = point.get("x")
            dev = self._parse_dev_label(dev_label)
            if dev is None:
                return drop_store, tail_attachment_age, tail_fit_period_selection

            if origin == "Tail":
                if tail_attachment_age == dev:
                    return drop_store, None, tail_fit_period_selection
                return drop_store, dev, tail_fit_period_selection

            if origin == "LDF":
                updated = self._toggle_tail_fit_selection(
                    tail_fit_period_selection or [],
                    dev,
                )
                return drop_store, tail_attachment_age, updated

            if origin == "LDF" or dev_label is None:
                return drop_store, tail_attachment_age, tail_fit_period_selection

            updated = self._toggle_drop(drop_store or [], str(origin), dev)
            return updated, tail_attachment_age, tail_fit_period_selection

        @self.app.callback(
            Output("triangle-heatmap", "clickData"),
            Input("drop-store", "data"),
            Input("tail-attachment-store", "data"),
            Input("tail-fit-period-store", "data"),
            prevent_initial_call=True,
        )
        def _clear_clicks(
            _drop_store, _tail_attachment_age, _tail_fit_period_selection
        ):
            return None

        @self.app.callback(
            Output("results-store", "data"),
            Input("drop-store", "data"),
            Input("tail-attachment-store", "data"),
            Input("tail-fit-period-store", "data"),
            Input("average-method", "value"),
            Input("tail-method", "value"),
            Input("sync-interval", "n_intervals"),
            State("results-store", "data"),
        )
        def _update_results(
            drop_store,
            tail_attachment_age,
            tail_fit_period_selection,
            average,
            tail_curve,
            _n_intervals,
            current_payload,
        ):
            ctx = callback_context
            if not ctx.triggered:
                return current_payload

            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger == "sync-interval":
                segment_key = self._get_segment_key()
                latest_payload = _LIVE_RESULTS_BY_SEGMENT.get(segment_key)
                if not latest_payload:
                    return no_update
                if not current_payload:
                    return latest_payload
                current_updated = current_payload.get("last_updated")
                latest_updated = latest_payload.get("last_updated")
                if latest_updated and latest_updated != current_updated:
                    return latest_payload
                return no_update

            parsed_tail = None
            drops = self._drops_to_tuples(drop_store)
            average = average or self._default_average
            tail_curve = tail_curve or self._default_tail_curve
            parsed_tail = None
            if tail_attachment_age is not None:
                try:
                    parsed_tail = int(tail_attachment_age)
                except (TypeError, ValueError):
                    parsed_tail = None
            fit_period = self._derive_tail_fit_period(tail_fit_period_selection)
            try:
                self._apply_recalculation(
                    average,
                    drops,
                    parsed_tail,
                    tail_curve,
                    fit_period,
                )
            except Exception as exc:
                logging.error(f"Failed to recalculate reserving: {exc}", exc_info=True)

            if self._config is not None:
                self._config.save_session(
                    {
                        "average": average,
                        "tail_curve": tail_curve,
                        "drops": drop_store or [],
                        "tail_attachment_age": parsed_tail,
                        "tail_fit_period": tail_fit_period_selection or [],
                    }
                )

            results_payload = self._build_results_payload(
                drop_store=drop_store,
                average=average,
                tail_attachment_age=parsed_tail,
                tail_curve=tail_curve,
                tail_fit_period_selection=tail_fit_period_selection,
            )

            segment_key = self._get_segment_key()
            _LIVE_RESULTS_BY_SEGMENT[segment_key] = results_payload
            return results_payload

        @self.app.callback(
            Output("triangle-heatmap", "figure"),
            Output("emergence-plot", "figure"),
            Output("results-table", "figure"),
            Output("drops-display", "children"),
            Output("tail-attachment-display", "children"),
            Output("tail-fit-period-display", "children"),
            Output("average-method", "value"),
            Output("tail-method", "value"),
            Input("results-store", "data"),
            Input("analysis-tabs", "value"),
        )
        def _hydrate_tabs(results_payload, _active_tab):
            if not results_payload:
                return (
                    self._plot_triangle_heatmap(
                        self.triangle,
                        "Triangle - Link Ratios",
                        self._default_tail_attachment_age,
                        self._default_tail_fit_period_selection,
                    ),
                    self._plot_emergence(
                        self.emergence_pattern,
                        "Emergence Pattern",
                    ),
                    self._plot_reserving_results_table(
                        self.results,
                        "Reserving Results",
                    ),
                    "None",
                    "None",
                    "lower=None, upper=None",
                    self._default_average,
                    self._default_tail_curve,
                )

            return (
                results_payload.get("triangle_figure"),
                results_payload.get("emergence_figure"),
                results_payload.get("results_figure"),
                results_payload.get("drops_display", "None"),
                results_payload.get("tail_attachment_display", "None"),
                results_payload.get(
                    "tail_fit_period_display", "lower=None, upper=None"
                ),
                results_payload.get("average", self._default_average),
                results_payload.get("tail_curve", self._default_tail_curve),
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
                            font=dict(color="red", size=14),
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
            legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
            height=600,
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
                            font=dict(color="red", size=14),
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
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_values,
                        fill_color=header_colors,
                        align="center",
                        font=dict(color="white", size=11, family="Arial"),
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
                        ],
                        font=dict(color="black", size=10, family="Arial"),
                        height=28,
                    ),
                )
            ]
        )

        fig.update_layout(
            title=title,
            template="plotly_white",
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
                            font=dict(color="red", size=14),
                        )
                    ],
                )
            )

        # Get raw triangle data to include dropped link ratios
        raw_link_ratios = (
            reserving._triangle.get_triangle().link_ratio["incurred"].to_frame()
        )
        # Align raw with processed triangle
        raw_link_ratios = raw_link_ratios.reindex(
            index=triangle_data.index, columns=triangle_data.columns
        )
        dropped_mask = triangle_data.isna() & raw_link_ratios.notna()
        dropped_mask.loc[["LDF", "Tail"]] = False
        expanded_triangle = triangle_data.fillna(raw_link_ratios)

        # Separate link ratios from LDF/Tail rows
        link_ratio_rows = expanded_triangle.index[:-2]  # All except last 2
        ldf_tail_rows = expanded_triangle.index[-2:]  # Last 2: LDF and Tail

        # Column-wise normalization for link ratios
        normalized_data = expanded_triangle.copy()
        for col in expanded_triangle.columns:
            col_data = expanded_triangle.loc[link_ratio_rows, col].dropna()
            if len(col_data) > 0:
                col_min = col_data.min()
                col_max = col_data.max()
                if col_max > col_min:
                    normalized_data.loc[link_ratio_rows, col] = (
                        expanded_triangle.loc[link_ratio_rows, col] - col_min
                    ) / (col_max - col_min)
                else:
                    normalized_data.loc[link_ratio_rows, col] = 0.5

        # Global logarithmic normalization for LDF and Tail rows
        ldf_tail_data = expanded_triangle.loc[ldf_tail_rows]
        ldf_tail_min = ldf_tail_data.min().min()
        ldf_tail_max = ldf_tail_data.max().max()

        if ldf_tail_max > ldf_tail_min and ldf_tail_min > 0:
            log_min = np.log(ldf_tail_min)
            log_max = np.log(ldf_tail_max)
            for col in expanded_triangle.columns:
                for row in ldf_tail_rows:
                    if not pd.isna(expanded_triangle.loc[row, col]):
                        normalized_data.loc[row, col] = (
                            np.log(expanded_triangle.loc[row, col]) - log_min
                        ) / (log_max - log_min)
        else:
            # If all LDF/Tail values are the same, set to mid-range
            for col in expanded_triangle.columns:
                for row in ldf_tail_rows:
                    if not pd.isna(expanded_triangle.loc[row, col]):
                        normalized_data.loc[row, col] = 0.5

        # Set dropped link ratios to special z-value for unique color
        # normalized_data[dropped_mask] = -1  # Removed for boundary approach

        # Format text to hide NaN values
        text_values = expanded_triangle.round(3).astype(str).values
        # Replace 'nan' strings with empty strings
        text_values = np.where(text_values == "nan", "", text_values)

        # Create a mask for NaN values to prevent them from being colored
        z_data = normalized_data.values.copy()
        z_data = np.where(np.isnan(expanded_triangle.values), np.nan, z_data)

        # Prepare customdata for hover: align incurred and premium with link ratio columns
        # Link ratio columns are like '3-6', '6-9' (strings), incurred columns are integers like 3, 6, 9
        # Show both left and right end: for '6-9', show incurred at 6 and at 9
        customdata_incurred = np.empty(
            (len(triangle_data.index), len(triangle_data.columns)), dtype=object
        )
        customdata_premium = np.empty(
            (len(triangle_data.index), len(triangle_data.columns)), dtype=object
        )

        # Convert triangle index to match incurred index (convert Period to year)
        triangle_idx_to_incurred_idx = {}
        for tri_idx in expanded_triangle.index:
            if isinstance(tri_idx, str):
                # LDF and Tail rows - no match in incurred data
                triangle_idx_to_incurred_idx[tri_idx] = None
            else:
                # Convert Period to year and find matching Timestamp in incurred data
                year = (
                    tri_idx.year if hasattr(tri_idx, "year") else int(str(tri_idx)[:4])
                )
                for inc_idx in incurred_data.index:
                    if inc_idx.year == year:
                        triangle_idx_to_incurred_idx[tri_idx] = inc_idx
                        break

        # Fill customdata for all rows
        matches_found = 0
        for i, row_idx in enumerate(expanded_triangle.index):
            # Get corresponding incurred index
            incurred_idx = triangle_idx_to_incurred_idx.get(row_idx)

            for j, col in enumerate(expanded_triangle.columns):
                # Extract left and right values from link ratio column (e.g., '6-9' -> 6, 9)
                try:
                    parts = col.split("-")
                    left_period = int(parts[0])
                    right_period = int(parts[1])
                except (ValueError, AttributeError, IndexError):
                    # If not in expected format, skip
                    customdata_incurred[i, j] = ""
                    customdata_premium[i, j] = ""
                    continue

                # Try to get incurred values for both left and right periods
                if incurred_idx is not None and incurred_idx in incurred_data.index:
                    left_val = None
                    right_val = None

                    if left_period in incurred_data.columns:
                        left_val = incurred_data.loc[incurred_idx, left_period]
                        matches_found += 1

                    if right_period in incurred_data.columns:
                        right_val = incurred_data.loc[incurred_idx, right_period]

                    # Format as "left_value --> right_value"
                    if not bool(pd.isna(left_val)) and not bool(pd.isna(right_val)):
                        customdata_incurred[i, j] = (
                            f"{format_to_millions(left_val)} --> {format_to_millions(right_val)}"
                        )
                    elif not bool(pd.isna(left_val)):
                        customdata_incurred[i, j] = format_to_millions(left_val)
                    else:
                        customdata_incurred[i, j] = ""
                else:
                    customdata_incurred[i, j] = ""

                # Try to get premium value (use left period, premium is stable)
                if (
                    incurred_idx is not None
                    and incurred_idx in premium_data.index
                    and left_period in premium_data.columns
                ):
                    prem_val = premium_data.loc[incurred_idx, left_period]
                    customdata_premium[i, j] = format_to_millions(prem_val)
                else:
                    customdata_premium[i, j] = ""

        import logging

        logging.info(
            f"Matches found for incurred data: {matches_found} out of {len(expanded_triangle.index) * len(expanded_triangle.columns)}"
        )

        # Stack customdata arrays
        customdata = np.stack([customdata_incurred, customdata_premium], axis=-1)

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
                textfont={"size": 10},  # Reduced font size to match 70% zoom
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
            width=fig_width,
            height=fig_height,
            yaxis=dict(autorange="reversed"),  # Reverse y-axis so oldest year at top
            xaxis=dict(
                range=[-0.5, initial_cols - 0.5]
            ),  # Show first 100 columns initially
            margin=dict(l=80, r=20, t=80, b=60),  # Reduce margins
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
        )
        segment_key = self._get_segment_key()
        existing_payload = _LIVE_RESULTS_BY_SEGMENT.get(segment_key)
        if existing_payload:
            results_payload = existing_payload
        if self._default_tail_fit_period_selection:
            fit_period = self._derive_tail_fit_period(
                self._default_tail_fit_period_selection
            )
            if fit_period is None:
                fit_period_display = "lower=None, upper=None"
            else:
                fit_period_display = f"lower={fit_period[0]}, upper={fit_period[1]}"
            results_payload["tail_fit_period_selection"] = (
                self._default_tail_fit_period_selection
            )
            results_payload["tail_fit_period_display"] = fit_period_display
            results_payload["triangle_figure"] = self._plot_triangle_heatmap(
                self.triangle,
                "Triangle - Link Ratios",
                self._default_tail_attachment_age,
                self._default_tail_fit_period_selection,
            ).to_dict()
        _LIVE_RESULTS_BY_SEGMENT[segment_key] = results_payload
        return html.Div(
            [
                html.H1(
                    "Reserving Dashboard",
                    style={"textAlign": "center", "marginBottom": "20px"},
                ),
                dcc.Store(id="drop-store", data=self._default_drop_store),
                dcc.Store(
                    id="tail-attachment-store",
                    data=self._default_tail_attachment_age,
                ),
                dcc.Store(
                    id="tail-fit-period-store",
                    data=self._default_tail_fit_period_selection,
                ),
                dcc.Store(id="results-store", data=results_payload),
                dcc.Interval(id="sync-interval", interval=1000, n_intervals=0),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Average method"),
                                dcc.Dropdown(
                                    id="average-method",
                                    options=[
                                        {"label": "Volume", "value": "volume"},
                                        {"label": "Simple", "value": "simple"},
                                    ],
                                    value=self._default_average,
                                    clearable=False,
                                ),
                            ],
                            style={"minWidth": "200px"},
                        ),
                        html.Div(
                            [
                                html.Label("Tail method"),
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
                    },
                ),
                html.Div(
                    [
                        html.Label("Active drops (click heatmap cells to toggle)"),
                        html.Div(id="drops-display", style={"marginTop": "6px"}),
                        html.Div(
                            [
                                html.Label("Tail attachment age"),
                                html.Div(
                                    id="tail-attachment-display",
                                    style={"marginTop": "6px"},
                                ),
                            ],
                            style={"marginTop": "12px"},
                        ),
                        html.Div(
                            [
                                html.Label("Tail fit period"),
                                html.Div(
                                    id="tail-fit-period-display",
                                    style={"marginTop": "6px"},
                                ),
                            ],
                            style={"marginTop": "12px"},
                        ),
                    ],
                    style={"marginBottom": "24px"},
                ),
                dcc.Tabs(
                    id="analysis-tabs",
                    value="triangles",
                    children=[
                        # Tab 1: Triangle Heatmaps
                        dcc.Tab(
                            label="Triangle Analysis",
                            value="triangles",
                            children=[
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
                                                    },
                                                    style={
                                                        "overflowX": "scroll",
                                                        "overflowY": "hidden",
                                                    },
                                                )
                                            ],
                                            style={
                                                "width": "100%",
                                                "overflowX": "auto",
                                            },
                                        )
                                    ],
                                    style={"width": "100%", "padding": "10px"},
                                )
                            ],
                        ),
                        # Tab 2: Emergence Patterns
                        dcc.Tab(
                            label="Emergence Patterns",
                            value="emergence",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="emergence-plot",
                                                    figure=results_payload.get(
                                                        "emergence_figure"
                                                    ),
                                                )
                                            ],
                                            style={"width": "100%"},
                                        )
                                    ],
                                    style={"width": "100%", "padding": "10px"},
                                )
                            ],
                        ),
                        # Tab 3: Reserving Results
                        dcc.Tab(
                            label="Reserving Results",
                            value="reserving_results",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="results-table",
                                                    figure=results_payload.get(
                                                        "results_figure"
                                                    ),
                                                    config={
                                                        "editable": False,
                                                        "displayModeBar": True,
                                                    },
                                                )
                                            ],
                                            style={"width": "100%"},
                                        )
                                    ],
                                    style={"width": "100%", "padding": "10px"},
                                )
                            ],
                        ),
                    ],
                ),
            ],
            style={
                "padding": "10px",
                "width": "100%",
                "background": "#fafafc",
                "overflowX": "hidden",
                "maxWidth": "100vw",
                "boxSizing": "border-box",
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
