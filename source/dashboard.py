import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
from source.config_manager import ConfigManager
from source.presentation import (
    build_heatmap_core,
    build_heatmap_core_cache_key,
    plot_data_triangle_table,
    plot_emergence,
    plot_reserving_results_table,
    plot_triangle_heatmap_clean,
)
from source.reserving import Reserving
from source.services import (
    CacheService,
    ParamsService,
    ReservingService,
    SessionSyncService,
)
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
        self._default_selected_ultimate_by_uwy: dict[str, str] = {}
        self._payload_cache: dict[str, dict] = {}
        self._triangle_figure_cache: dict[str, dict] = {}
        self._emergence_figure_cache: dict[str, dict] = {}
        self._results_table_figure_cache: dict[str, dict] = {}
        self._heatmap_core_cache: dict[str, dict] = {}
        self._cache_max_entries = 32
        self._recalc_request_seq = 0
        self._cache_service = CacheService(max_entries=self._cache_max_entries)
        self._params_service = ParamsService(
            default_average=self._default_average,
            default_tail_curve=self._default_tail_curve,
            default_bf_apriori=self._default_bf_apriori,
            get_uwy_labels=self._get_uwy_labels,
            load_session=self._config.load_session
            if self._config is not None
            else None,
            get_sync_version=(
                self._config.get_sync_version if self._config is not None else None
            ),
        )

        self._load_session_defaults()

        # Extract data from domain objects
        self._extract_data()

        self._reserving_service = ReservingService(
            reserving=self._reserving,
            params_service=self._params_service,
            cache_service=self._cache_service,
            default_average=self._default_average,
            default_tail_curve=self._default_tail_curve,
            default_bf_apriori=self._default_bf_apriori,
            segment_key_provider=self._get_segment_key,
            extract_data=self._extract_data,
            get_triangle=lambda: self.triangle,
            get_emergence=lambda: self.emergence_pattern,
            get_results=lambda: self.results,
            build_triangle_figure=lambda triangle_data,
            title,
            attachment,
            fit_selection: self._build_triangle_figure_dict(
                triangle_data,
                title,
                attachment,
                fit_selection,
            ),
            build_emergence_figure=lambda emergence_data, title: plot_emergence(
                emergence_pattern=emergence_data,
                title=title,
                font_family=FONT_FAMILY,
                figure_font_size=FIGURE_FONT_SIZE,
                figure_title_font_size=FIGURE_TITLE_FONT_SIZE,
                alert_annotation_font_size=ALERT_ANNOTATION_FONT_SIZE,
                color_text=COLOR_TEXT,
                color_surface=COLOR_SURFACE,
                color_border=COLOR_BORDER,
            ).to_dict(),
            build_results_table_figure=lambda results_df,
            title: plot_reserving_results_table(
                results_df=results_df,
                title=title,
                font_family=FONT_FAMILY,
                figure_font_size=FIGURE_FONT_SIZE,
                figure_title_font_size=FIGURE_TITLE_FONT_SIZE,
                table_header_font_size=TABLE_HEADER_FONT_SIZE,
                table_cell_font_size=TABLE_CELL_FONT_SIZE,
                alert_annotation_font_size=ALERT_ANNOTATION_FONT_SIZE,
                color_text=COLOR_TEXT,
                color_surface=COLOR_SURFACE,
                color_border=COLOR_BORDER,
            ).to_dict(),
            payload_cache=self._payload_cache,
            triangle_cache=self._triangle_figure_cache,
            emergence_cache=self._emergence_figure_cache,
            results_table_cache=self._results_table_figure_cache,
        )
        self._session_sync_service = SessionSyncService(
            config=self._config,
            segment_key_provider=self._get_segment_key,
            live_results_store=_LIVE_RESULTS_BY_SEGMENT,
        )

        self._register_callbacks()
        logging.info("Dashboard initialized successfully")

    def _build_triangle_figure_dict(
        self,
        triangle_data: Optional[pd.DataFrame],
        title: str,
        tail_attachment_age: Optional[int],
        tail_fit_period_selection: Optional[List[int]],
    ) -> dict:
        if triangle_data is None:
            triangle_data = self.triangle
        core_cache_key = build_heatmap_core_cache_key(
            segment_key=self._get_segment_key(),
            triangle_data=triangle_data,
            incurred_data=self.incurred,
            premium_data=self.premium,
        )
        core_payload = self._cache_service.get(self._heatmap_core_cache, core_cache_key)
        if core_payload is None:
            core_payload = build_heatmap_core(
                triangle_data=triangle_data,
                incurred_data=self.incurred,
                premium_data=self.premium,
                reserving=self._reserving,
            )
            self._cache_service.set(
                self._heatmap_core_cache, core_cache_key, core_payload
            )

        figure, _ = plot_triangle_heatmap_clean(
            triangle_data=triangle_data,
            incurred_data=self.incurred,
            premium_data=self.premium,
            reserving=self._reserving,
            title=title,
            tail_attachment_age=tail_attachment_age,
            tail_fit_period_selection=tail_fit_period_selection,
            parse_dev_label=self._params_service.parse_dev_label,
            derive_tail_fit_period=self._params_service.derive_tail_fit_period,
            core_payload=core_payload,
            font_family=FONT_FAMILY,
            figure_font_size=FIGURE_FONT_SIZE,
            figure_title_font_size=FIGURE_TITLE_FONT_SIZE,
            heatmap_text_font_size=HEATMAP_TEXT_FONT_SIZE,
            color_text=COLOR_TEXT,
            color_surface=COLOR_SURFACE,
            color_border=COLOR_BORDER,
        )
        return figure.to_dict()

    def _load_session_defaults(self) -> None:
        if self._config is None:
            self._default_bf_apriori_rows = self._params_service.build_bf_apriori_rows()
            self._default_selected_ultimate_by_uwy = (
                self._params_service.build_selected_ultimate_by_uwy()
            )
            return
        session = self._config.load_session()
        self._default_average = session.get("average", self._default_average)
        self._default_tail_curve = session.get(
            "tail_curve",
            self._default_tail_curve,
        )
        self._default_drop_store = self._params_service.normalize_drop_store(
            session.get("drops")
        )
        tail_attachment_age = session.get("tail_attachment_age")
        if tail_attachment_age is not None:
            try:
                self._default_tail_attachment_age = int(tail_attachment_age)
            except (TypeError, ValueError):
                self._default_tail_attachment_age = None
        self._default_tail_fit_period_selection = (
            self._params_service.normalize_tail_fit_selection(
                session.get("tail_fit_period")
            )
        )
        self._default_bf_apriori_rows = self._params_service.build_bf_apriori_rows(
            session.get("bf_apriori_by_uwy")
        )
        self._default_selected_ultimate_by_uwy = (
            self._params_service.build_selected_ultimate_by_uwy(
                session.get("selected_ultimate_by_uwy")
            )
        )

    def _build_results_table_rows(
        self, results_df: Optional[pd.DataFrame]
    ) -> List[dict]:
        if results_df is None or len(results_df) == 0:
            return []

        rows: List[dict] = []
        for idx, row in results_df.iterrows():
            if hasattr(idx, "year"):
                uwy = str(idx.year)
            else:
                uwy_text = str(idx)
                uwy = uwy_text[:4] if len(uwy_text) >= 4 else uwy_text

            incurred = float(row.get("incurred", 0.0))
            premium = float(row.get("Premium", 0.0))
            cl_ultimate = float(row.get("cl_ultimate", 0.0))
            bf_ultimate = float(row.get("bf_ultimate", 0.0))
            selected_ultimate = float(row.get("ultimate", 0.0))
            ibnr = selected_ultimate - incurred

            if premium > 0:
                incurred_lr_display = f"{(incurred / premium):.2%}"
            else:
                incurred_lr_display = "N/A"

            cl_lr_display = "N/A" if premium <= 0 else f"{(cl_ultimate / premium):.2%}"
            bf_lr_display = "N/A" if premium <= 0 else f"{(bf_ultimate / premium):.2%}"

            rows.append(
                {
                    "uwy": uwy,
                    "incurred_display": f"{incurred:,.0f}",
                    "premium_display": f"{premium:,.0f}",
                    "incurred_loss_ratio_display": incurred_lr_display,
                    "cl_ultimate_display": f"{cl_ultimate:,.2f}",
                    "cl_loss_ratio_display": cl_lr_display,
                    "bf_ultimate_display": f"{bf_ultimate:,.2f}",
                    "bf_loss_ratio_display": bf_lr_display,
                    "ultimate_display": f"{selected_ultimate:,.2f}",
                    "ibnr_display": f"{ibnr:,.2f}",
                }
            )
        return rows

    def _build_results_selection_styles(
        self,
        selected_ultimate_by_uwy: dict[str, str],
    ) -> List[dict]:
        styles: List[dict] = []
        for uwy, method in selected_ultimate_by_uwy.items():
            if method == "chainladder":
                left_column = "cl_ultimate_display"
                right_column = "cl_loss_ratio_display"
            elif method == "bornhuetter_ferguson":
                left_column = "bf_ultimate_display"
                right_column = "bf_loss_ratio_display"
            else:
                continue

            safe_uwy = str(uwy).replace("'", "\\'")
            styles.append(
                {
                    "if": {
                        "filter_query": "{uwy} = '" + safe_uwy + "'",
                        "column_id": left_column,
                    },
                    "borderTop": "0px",
                    "borderRight": f"1px solid {COLOR_SURFACE}",
                    "borderBottom": "0px",
                    "borderLeft": "2px solid black",
                    "boxShadow": "inset 0 2px 0 0 black, inset 0 -2px 0 0 black",
                }
            )
            styles.append(
                {
                    "if": {
                        "filter_query": "{uwy} = '" + safe_uwy + "'",
                        "column_id": right_column,
                    },
                    "borderTop": "0px",
                    "borderRight": "2px solid black",
                    "borderBottom": "0px",
                    "borderLeft": f"1px solid {COLOR_SURFACE}",
                    "boxShadow": "inset 0 2px 0 0 black, inset 0 -2px 0 0 black",
                }
            )
        return styles

    def _get_segment_key(self) -> str:
        if self._config is None:
            return "default"
        return self._config.get_segment()

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
            Input("active-tab", "data"),
        )
        def _toggle_tab_visibility(active_tab):
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

            return (
                data_style if active_tab == "data" else data_hidden,
                chainladder_style
                if active_tab == "chainladder"
                else chainladder_hidden,
                bf_style if active_tab == "bornhuetter_ferguson" else bf_hidden,
                results_style if active_tab == "results" else results_hidden,
            )

        @self.app.callback(
            Output("params-store", "data"),
            Input("triangle-heatmap", "clickData"),
            Input("average-method", "value"),
            Input("tail-method", "value"),
            Input("bf-apriori-table", "data"),
            Input("results-table", "active_cell"),
            Input("sync-inbox", "value"),
            Input("page-location", "pathname"),
            State("params-store", "data"),
            State("results-store", "data"),
            State("results-table", "data"),
        )
        def _reduce_params(
            click,
            average,
            tail_curve,
            bf_apriori_rows,
            results_active_cell,
            sync_inbox,
            _pathname,
            current_params,
            current_results,
            results_rows,
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
                next_params = self._params_service.load_session_params_state(
                    request_id=current_request_id + 1,
                    force_recalc=True,
                    default_drop_store=self._default_drop_store,
                    default_average=self._default_average,
                    default_tail_attachment_age=self._default_tail_attachment_age,
                    default_tail_curve=self._default_tail_curve,
                    default_tail_fit_period_selection=self._default_tail_fit_period_selection,
                    default_bf_apriori_rows=self._default_bf_apriori_rows,
                )
                logging.info(
                    "[params-reducer] load request=%s force_recalc=%s",
                    next_params.get("request_id"),
                    next_params.get("force_recalc"),
                )
                return next_params

            working_params = self._params_service.build_params_state(
                drop_store=current_params.get("drop_store"),
                average=current_params.get("average"),
                tail_attachment_age=current_params.get("tail_attachment_age"),
                tail_curve=current_params.get("tail_curve"),
                tail_fit_period_selection=current_params.get(
                    "tail_fit_period_selection"
                ),
                bf_apriori_by_uwy=current_params.get("bf_apriori_by_uwy"),
                selected_ultimate_by_uwy=current_params.get("selected_ultimate_by_uwy"),
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
                dev = self._params_service.parse_dev_label(dev_label)
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
                    updated_fit = self._params_service.toggle_tail_fit_selection(
                        working_params["tail_fit_period_selection"],
                        dev,
                    )
                    if updated_fit != working_params["tail_fit_period_selection"]:
                        working_params["tail_fit_period_selection"] = updated_fit
                        changed = True
                elif dev_label is not None:
                    updated_drops = self._params_service.toggle_drop(
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
                next_bf = self._params_service.bf_rows_to_mapping(bf_apriori_rows)
                if next_bf != working_params["bf_apriori_by_uwy"]:
                    working_params["bf_apriori_by_uwy"] = next_bf
                    changed = True

            elif trigger == "results-table":
                if not isinstance(results_active_cell, dict):
                    return no_update
                column_id = results_active_cell.get("column_id")
                row_index = results_active_cell.get("row")
                if not isinstance(row_index, int):
                    return no_update
                if not isinstance(results_rows, list) or row_index >= len(results_rows):
                    return no_update
                row_payload = results_rows[row_index]
                if not isinstance(row_payload, dict):
                    return no_update
                uwy = str(row_payload.get("uwy", "")).strip()
                if not uwy:
                    return no_update

                selected_method = None
                if column_id in {"cl_ultimate_display", "cl_loss_ratio_display"}:
                    selected_method = "chainladder"
                elif column_id in {"bf_ultimate_display", "bf_loss_ratio_display"}:
                    selected_method = "bornhuetter_ferguson"
                if selected_method is None:
                    return no_update

                next_selected = self._params_service.set_selected_ultimate_method(
                    working_params.get("selected_ultimate_by_uwy"),
                    uwy,
                    selected_method,
                )
                if next_selected != working_params["selected_ultimate_by_uwy"]:
                    working_params["selected_ultimate_by_uwy"] = next_selected
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
                synced_params = self._params_service.build_params_state(
                    drop_store=session.get("drops"),
                    average=session.get("average", self._default_average),
                    tail_attachment_age=session.get("tail_attachment_age"),
                    tail_curve=session.get("tail_curve", self._default_tail_curve),
                    tail_fit_period_selection=session.get("tail_fit_period"),
                    bf_apriori_by_uwy=session.get("bf_apriori_by_uwy"),
                    selected_ultimate_by_uwy=session.get("selected_ultimate_by_uwy"),
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
            selected_ultimate_by_uwy = (
                self._params_service.build_selected_ultimate_by_uwy(
                    params.get("selected_ultimate_by_uwy")
                )
            )
            force_recalc = bool(params.get("force_recalc"))

            cache_key = self._cache_service.build_results_cache_key(
                segment=self._get_segment_key(),
                default_average=self._default_average,
                default_tail_curve=self._default_tail_curve,
                drop_store=drop_store,
                average=average,
                tail_attachment_age=tail_attachment_age,
                tail_curve=tail_curve,
                tail_fit_period_selection=tail_fit_period_selection,
                bf_apriori_by_uwy=bf_apriori_by_uwy,
                selected_ultimate_by_uwy=selected_ultimate_by_uwy,
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
                results_payload = self._reserving_service.get_or_build_results_payload(
                    drop_store=drop_store,
                    average=average,
                    tail_attachment_age=tail_attachment_age,
                    tail_curve=tail_curve,
                    tail_fit_period_selection=tail_fit_period_selection,
                    bf_apriori_by_uwy=bf_apriori_by_uwy,
                    selected_ultimate_by_uwy=selected_ultimate_by_uwy,
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
                results_payload, _ = (
                    self._session_sync_service.apply_sync_source_payload(
                        results_payload=results_payload,
                        params=params,
                    )
                )
                sync_version = int(results_payload.get("sync_version", 0))
                logging.info(
                    "[recalc] finish request=%s sync_version=%s",
                    request_id,
                    sync_version,
                )
                total_elapsed_ms = (time.perf_counter() - callback_started) * 1000
                logging.info("Callback total completed in %.0f ms", total_elapsed_ms)
                return results_payload, no_update

            if not bool(sync_ready):
                logging.warning(
                    "Tab sync bridge unavailable; update applied only in current tab"
                )

            results_payload, publish_message = (
                self._session_sync_service.apply_local_source_payload(
                    results_payload=results_payload,
                    params=params,
                    current_payload=current_payload
                    if isinstance(current_payload, dict)
                    else None,
                    sync_ready=bool(sync_ready),
                )
            )
            sync_version = int(results_payload.get("sync_version", 0))
            segment_key = self._get_segment_key()

            if publish_message is None:
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
                    self._params_service.build_bf_apriori_rows(
                        self._default_bf_apriori_rows
                    ),
                )
            bf_rows = self._params_service.build_bf_apriori_rows(
                params.get("bf_apriori_by_uwy")
            )
            return (
                params.get("average", self._default_average),
                params.get("tail_curve", self._default_tail_curve),
                bf_rows,
            )

        @self.app.callback(
            Output("results-table", "data"),
            Output("results-table", "style_data_conditional"),
            Input("results-store", "data"),
        )
        def _hydrate_results_table(results_payload):
            rows = []
            if isinstance(results_payload, dict):
                payload_rows = results_payload.get("results_table_rows")
                if isinstance(payload_rows, list):
                    rows = payload_rows
            if not rows:
                rows = self._build_results_table_rows(self.results)
            base_styles = [
                {
                    "if": {"state": "active"},
                    "backgroundColor": "#ffffff",
                    "color": COLOR_TEXT,
                    "boxShadow": "none",
                    "outline": "none",
                },
                {
                    "if": {"state": "selected"},
                    "backgroundColor": "#ffffff",
                    "color": COLOR_TEXT,
                    "boxShadow": "none",
                    "outline": "none",
                },
            ]
            selected_mapping = self._default_selected_ultimate_by_uwy
            if isinstance(results_payload, dict):
                selected_mapping = self._params_service.build_selected_ultimate_by_uwy(
                    results_payload.get("selected_ultimate_by_uwy")
                )
            selection_styles = self._build_results_selection_styles(selected_mapping)
            return rows, base_styles + selection_styles

        @self.app.callback(
            Output("results-table", "active_cell"),
            Input("params-store", "data"),
            prevent_initial_call=True,
        )
        def _clear_results_active_cell(_params):
            return None

        @self.app.callback(
            Output("triangle-base-figure", "data"),
            Output("emergence-plot", "figure"),
            Input("results-store", "data"),
        )
        def _hydrate_tabs(results_payload):
            if not results_payload:
                return (
                    {
                        "figure": self._build_triangle_figure_dict(
                            self.triangle,
                            "Triangle - Link Ratios",
                            self._default_tail_attachment_age,
                            self._default_tail_fit_period_selection,
                        ),
                        "figure_version": 0,
                    },
                    plot_emergence(
                        emergence_pattern=self.emergence_pattern,
                        title="Emergence Pattern",
                        font_family=FONT_FAMILY,
                        figure_font_size=FIGURE_FONT_SIZE,
                        figure_title_font_size=FIGURE_TITLE_FONT_SIZE,
                        alert_annotation_font_size=ALERT_ANNOTATION_FONT_SIZE,
                        color_text=COLOR_TEXT,
                        color_surface=COLOR_SURFACE,
                        color_border=COLOR_BORDER,
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
            return plot_data_triangle_table(
                triangle_df=triangle_df,
                title=f"{base_title} ({view_label})",
                weights_df=weights_df,
                ratio_mode=ratio_mode,
                font_family=FONT_FAMILY,
                figure_font_size=FIGURE_FONT_SIZE,
                figure_title_font_size=FIGURE_TITLE_FONT_SIZE,
                table_header_font_size=TABLE_HEADER_FONT_SIZE,
                table_cell_font_size=TABLE_CELL_FONT_SIZE,
                alert_annotation_font_size=ALERT_ANNOTATION_FONT_SIZE,
                color_text=COLOR_TEXT,
                color_surface=COLOR_SURFACE,
                color_border=COLOR_BORDER,
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
        results_payload = self._reserving_service.build_results_payload(
            drop_store=self._default_drop_store,
            average=self._default_average,
            tail_attachment_age=self._default_tail_attachment_age,
            tail_curve=self._default_tail_curve,
            tail_fit_period_selection=self._default_tail_fit_period_selection,
            bf_apriori_by_uwy=self._params_service.bf_rows_to_mapping(
                self._default_bf_apriori_rows
            ),
            selected_ultimate_by_uwy=self._default_selected_ultimate_by_uwy,
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
        initial_bf_apriori_rows = self._params_service.build_bf_apriori_rows(
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
                                                            figure=plot_data_triangle_table(
                                                                triangle_df=initial_data_triangle,
                                                                title="Data Triangle - Incurred (Cumulative)",
                                                                weights_df=initial_data_weights,
                                                                ratio_mode=initial_ratio_mode,
                                                                font_family=FONT_FAMILY,
                                                                figure_font_size=FIGURE_FONT_SIZE,
                                                                figure_title_font_size=FIGURE_TITLE_FONT_SIZE,
                                                                table_header_font_size=TABLE_HEADER_FONT_SIZE,
                                                                table_cell_font_size=TABLE_CELL_FONT_SIZE,
                                                                alert_annotation_font_size=ALERT_ANNOTATION_FONT_SIZE,
                                                                color_text=COLOR_TEXT,
                                                                color_surface=COLOR_SURFACE,
                                                                color_border=COLOR_BORDER,
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
                                                dash_table.DataTable(
                                                    id="results-table",
                                                    columns=[
                                                        {
                                                            "name": "UWY",
                                                            "id": "uwy",
                                                        },
                                                        {
                                                            "name": "Incurred (EUR)",
                                                            "id": "incurred_display",
                                                        },
                                                        {
                                                            "name": "Premium (EUR)",
                                                            "id": "premium_display",
                                                        },
                                                        {
                                                            "name": "Incurred Loss Ratio",
                                                            "id": "incurred_loss_ratio_display",
                                                        },
                                                        {
                                                            "name": "CL Ultimate (EUR)",
                                                            "id": "cl_ultimate_display",
                                                        },
                                                        {
                                                            "name": "CL Loss Ratio",
                                                            "id": "cl_loss_ratio_display",
                                                        },
                                                        {
                                                            "name": "BF Ultimate (EUR)",
                                                            "id": "bf_ultimate_display",
                                                        },
                                                        {
                                                            "name": "BF Loss Ratio",
                                                            "id": "bf_loss_ratio_display",
                                                        },
                                                        {
                                                            "name": "Selected Ultimate (EUR)",
                                                            "id": "ultimate_display",
                                                        },
                                                        {
                                                            "name": "IBNR (EUR)",
                                                            "id": "ibnr_display",
                                                        },
                                                    ],
                                                    data=results_payload.get(
                                                        "results_table_rows", []
                                                    ),
                                                    sort_action="native",
                                                    css=[
                                                        {
                                                            "selector": ".dash-spreadsheet-container table",
                                                            "rule": "border-collapse: collapse; border-spacing: 0;",
                                                        },
                                                        {
                                                            "selector": ".dash-spreadsheet td.dash-cell.cell--selected, .dash-spreadsheet td.dash-cell.focused",
                                                            "rule": "background-color: #ffffff !important; color: inherit !important; box-shadow: none !important; outline: none !important;",
                                                        },
                                                        {
                                                            "selector": ".dash-spreadsheet td.dash-cell:focus, .dash-spreadsheet td.dash-cell.cell--selected.focused",
                                                            "rule": "box-shadow: none !important; outline: none !important;",
                                                        },
                                                    ],
                                                    style_as_list_view=False,
                                                    style_table={"overflowX": "auto"},
                                                    style_header={
                                                        "fontWeight": 600,
                                                        "backgroundColor": "#f2f5f9",
                                                        "border": f"1px solid {COLOR_BORDER}",
                                                        "fontFamily": FONT_FAMILY,
                                                        "fontSize": f"{TABLE_HEADER_FONT_SIZE}px",
                                                        "textAlign": "center",
                                                    },
                                                    style_cell={
                                                        "padding": "8px",
                                                        "fontFamily": FONT_FAMILY,
                                                        "fontSize": f"{TABLE_CELL_FONT_SIZE}px",
                                                        "border": f"1px solid {COLOR_BORDER}",
                                                        "textAlign": "right",
                                                        "minWidth": "110px",
                                                        "whiteSpace": "nowrap",
                                                    },
                                                    style_data={
                                                        "backgroundColor": COLOR_SURFACE,
                                                    },
                                                    style_cell_conditional=[
                                                        {
                                                            "if": {"column_id": "uwy"},
                                                            "textAlign": "center",
                                                            "minWidth": "70px",
                                                        },
                                                        {
                                                            "if": {
                                                                "column_id": "incurred_loss_ratio_display"
                                                            },
                                                            "textAlign": "center",
                                                        },
                                                        {
                                                            "if": {
                                                                "column_id": "cl_loss_ratio_display"
                                                            },
                                                            "textAlign": "center",
                                                        },
                                                        {
                                                            "if": {
                                                                "column_id": "bf_loss_ratio_display"
                                                            },
                                                            "textAlign": "center",
                                                        },
                                                    ],
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
