from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update

from ai.chat_service import AIChatService
from source.config_manager import ConfigManager
from source.reserving import Reserving


FONT_FAMILY = '"Manrope", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
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
TURTUARY_AVATAR_SRC = "/assets/turtuary.png"
PRESET_PROMPTS = [
    {
        "id": "ai-prompt-quarter-close",
        "label": "Quarter-Close Review",
        "description": "Compare against the prior proxy, surface blockers, rank changes, and prepare sign-off.",
        "prompt": "Run a quarter-close review pack. Compare the current valuation against the prior proxy, flag any recommendation blockers, rank the strongest assumption changes, and tell me what should go to sign-off.",
    },
    {
        "id": "ai-prompt-drop-review",
        "label": "Drop Review",
        "description": "Rank the strongest drop candidates and separate selection-worthy changes from sensitivities.",
        "prompt": "Review whether any ratios should be dropped. Rank the strongest drop candidates, explain reserve impact and fragility, and tell me which candidate is selection-worthy versus sensitivity-only.",
    },
    {
        "id": "ai-prompt-tail-review",
        "label": "Tail Review",
        "description": "Rank tail options, explain attachment continuity, sub-1.0 factors, and stability risk.",
        "prompt": "Review the tail assumptions. Rank the best tail candidates, explain attachment continuity, sub-1.0 late factors, and stability risk, and recommend the strongest selection versus sensitivity.",
    },
    {
        "id": "ai-prompt-reserve-recommendation",
        "label": "Reserve Recommendation",
        "description": "Run the full reserving analysis and recommend the overall booking approach for this segment.",
        "prompt": "Give me your overall reserving recommendation for this segment. Test the strongest overall parameter set and recommend the final booking approach, including any development exclusions, the selected late-development curve settings, and the final reserving method selection by underwriting year where needed.",
    },
    {
        "id": "ai-prompt-anomaly-triage",
        "label": "Anomaly Triage",
        "description": "Classify the main issues, explain reserve relevance, and say whether recommendations should pause.",
        "prompt": "Run anomaly triage before any assumption changes. Classify the main issues, explain reserve relevance, say whether recommendations should pause, and tell me the next diagnostic to run.",
    },
    {
        "id": "ai-prompt-movement-review",
        "label": "Movement Review",
        "description": "Separate observed evidence from inference and identify whether the move is data or selection driven.",
        "prompt": "Explain the biggest reserve movement this quarter. Separate observed evidence from inference, identify the main drivers by UWY or assumption, and tell me whether the movement points to a data issue or a selection issue.",
    },
]


class AIDashboard:
    def __init__(
        self,
        reserving: Reserving,
        *,
        config: ConfigManager | None = None,
        chat_service: AIChatService | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._config = config
        self._chat_service = chat_service
        self._chat_id = chat_id
        assets_folder = Path(__file__).resolve().parent.parent / "assets"
        self.app = Dash(
            __name__,
            assets_folder=str(assets_folder),
            suppress_callback_exceptions=True,
            external_stylesheets=[
                "https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap",
            ],
        )
        self._register_callbacks()

    def _register_callbacks(self) -> None:
        preset_inputs = [Input(spec["id"], "n_clicks") for spec in PRESET_PROMPTS]

        @self.app.callback(
            Output("ai-sidebar-open", "data"),
            Output("ai-sidebar", "style"),
            Output("ai-sidebar-toggle", "children"),
            Input("ai-sidebar-toggle", "n_clicks"),
            State("ai-sidebar-open", "data"),
            prevent_initial_call=True,
        )
        def _toggle_sidebar(_n_clicks, is_open):
            open_state = not bool(is_open)
            return (
                open_state,
                self._sidebar_style(open_state),
                "Hide Sidebar" if open_state else "Show Sidebar",
            )

        @self.app.callback(
            Output("ai-chat-history-store", "data"),
            Output("ai-chat-tool-events-store", "data"),
            Output("ai-chat-scenario-ledger-store", "data"),
            Output("ai-chat-analysis-basis-store", "data"),
            Output("ai-chat-transcript", "children"),
            Output("ai-analysis-trace", "data"),
            Output("ai-chat-evidence-trace", "data"),
            Output("ai-scenario-ledger", "data"),
            Output("ai-analysis-basis", "data"),
            Input("ai-refresh-review", "n_clicks"),
        )
        def _refresh_review(_n_clicks):
            history = self._initial_chat_history()
            tool_events = self._initial_tool_events()
            scenario_ledger = self._initial_scenario_ledger()
            analysis_basis = self._initial_analysis_basis()
            return (
                history,
                tool_events,
                scenario_ledger,
                analysis_basis,
                self._render_chat_messages(history),
                self._tool_event_rows(tool_events),
                self._chat_evidence_rows(tool_events),
                self._scenario_ledger_rows(scenario_ledger),
                self._analysis_basis_rows(analysis_basis),
            )

        @self.app.callback(
            Output("ai-chat-history-store", "data", allow_duplicate=True),
            Output("ai-chat-tool-events-store", "data", allow_duplicate=True),
            Output("ai-chat-scenario-ledger-store", "data", allow_duplicate=True),
            Output("ai-chat-analysis-basis-store", "data", allow_duplicate=True),
            Output("ai-chat-transcript", "children", allow_duplicate=True),
            Output("ai-chat-input", "value"),
            Output("ai-chat-status", "children"),
            Output("ai-analysis-trace", "data", allow_duplicate=True),
            Output("ai-chat-evidence-trace", "data", allow_duplicate=True),
            Output("ai-scenario-ledger", "data", allow_duplicate=True),
            Output("ai-analysis-basis", "data", allow_duplicate=True),
            Output("ai-chat-poll", "disabled", allow_duplicate=True),
            Input("ai-chat-send", "n_clicks"),
            *preset_inputs,
            State("ai-chat-input", "value"),
            State("ai-chat-history-store", "data"),
            prevent_initial_call=True,
        )
        def _chat(_n_clicks, *_args):
            prompt = _args[-2] if len(_args) >= 2 else ""
            history = _args[-1] if _args else []
            triggered_id = ctx.triggered_id
            prompt_text = self._prompt_text_for_trigger(
                triggered_id=triggered_id,
                typed_prompt=prompt,
            )
            if not prompt_text:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            if self._chat_service is None or not self._chat_id:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "AI chat backend is not configured.",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    True,
                )
            rows = history if isinstance(history, list) else []
            normalized = [dict(item) for item in rows if isinstance(item, dict)]
            try:
                response = self._chat_service.send_message(self._chat_id, prompt_text)
            except Exception as error:
                normalized.append({"role": "user", "content": prompt_text})
                normalized.append(
                    {
                        "role": "assistant",
                        "content": f"AI assistant error: {error}",
                    }
                )
                return (
                    normalized,
                    self._initial_tool_events(),
                    self._initial_scenario_ledger(),
                    self._initial_analysis_basis(),
                    self._render_chat_messages(normalized),
                    "",
                    "AI response failed.",
                    self._tool_event_rows(self._initial_tool_events()),
                    self._chat_evidence_rows(self._initial_tool_events()),
                    self._scenario_ledger_rows(self._initial_scenario_ledger()),
                    self._analysis_basis_rows(self._initial_analysis_basis()),
                    True,
                )
            messages = response.get("messages")
            if not isinstance(messages, list):
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "",
                    "AI response did not contain a transcript.",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    True,
                )
            tool_events = response.get("tool_events")
            scenario_ledger = response.get("scenario_ledger")
            analysis_basis = response.get("analysis_basis")
            normalized_tool_events = (
                [dict(item) for item in tool_events if isinstance(item, dict)]
                if isinstance(tool_events, list)
                else []
            )
            normalized_scenario_ledger = (
                [dict(item) for item in scenario_ledger if isinstance(item, dict)]
                if isinstance(scenario_ledger, list)
                else []
            )
            normalized_analysis_basis = (
                dict(analysis_basis) if isinstance(analysis_basis, dict) else {}
            )
            status = (
                "AI fallback summary used." if response.get("fallback_used") else ""
            )
            return (
                messages,
                normalized_tool_events,
                normalized_scenario_ledger,
                normalized_analysis_basis,
                self._render_chat_messages(messages),
                "",
                status,
                self._tool_event_rows(normalized_tool_events),
                self._chat_evidence_rows(normalized_tool_events),
                self._scenario_ledger_rows(normalized_scenario_ledger),
                self._analysis_basis_rows(normalized_analysis_basis),
                not bool(response.get("streaming")),
            )

        @self.app.callback(
            Output("ai-chat-history-store", "data", allow_duplicate=True),
            Output("ai-chat-tool-events-store", "data", allow_duplicate=True),
            Output("ai-chat-scenario-ledger-store", "data", allow_duplicate=True),
            Output("ai-chat-analysis-basis-store", "data", allow_duplicate=True),
            Output("ai-chat-transcript", "children", allow_duplicate=True),
            Output("ai-chat-status", "children", allow_duplicate=True),
            Output("ai-analysis-trace", "data", allow_duplicate=True),
            Output("ai-chat-evidence-trace", "data", allow_duplicate=True),
            Output("ai-scenario-ledger", "data", allow_duplicate=True),
            Output("ai-analysis-basis", "data", allow_duplicate=True),
            Output("ai-chat-poll", "disabled", allow_duplicate=True),
            Input("ai-chat-poll", "n_intervals"),
            prevent_initial_call=True,
        )
        def _poll_chat(_n_intervals):
            if self._chat_service is None or not self._chat_id:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    True,
                )
            response = self._chat_service.build_chat_response(self._chat_id)
            messages = response.get("messages")
            tool_events = response.get("tool_events")
            scenario_ledger = response.get("scenario_ledger")
            analysis_basis = response.get("analysis_basis")
            if not isinstance(messages, list):
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    True,
                )
            normalized_tool_events = (
                [dict(item) for item in tool_events if isinstance(item, dict)]
                if isinstance(tool_events, list)
                else []
            )
            normalized_scenario_ledger = (
                [dict(item) for item in scenario_ledger if isinstance(item, dict)]
                if isinstance(scenario_ledger, list)
                else []
            )
            normalized_analysis_basis = (
                dict(analysis_basis) if isinstance(analysis_basis, dict) else {}
            )
            status = (
                "AI fallback summary used." if response.get("fallback_used") else ""
            )
            return (
                messages,
                normalized_tool_events,
                normalized_scenario_ledger,
                normalized_analysis_basis,
                self._render_chat_messages(messages),
                status,
                self._tool_event_rows(normalized_tool_events),
                self._chat_evidence_rows(normalized_tool_events),
                self._scenario_ledger_rows(normalized_scenario_ledger),
                self._analysis_basis_rows(normalized_analysis_basis),
                not bool(response.get("streaming")),
            )

    def _create_layout(self):
        history = self._initial_chat_history()
        return html.Div(
            [
                dcc.Store(id="ai-sidebar-open", data=False),
                dcc.Store(id="ai-chat-history-store", data=history),
                dcc.Store(
                    id="ai-chat-tool-events-store", data=self._initial_tool_events()
                ),
                dcc.Store(
                    id="ai-chat-scenario-ledger-store",
                    data=self._initial_scenario_ledger(),
                ),
                dcc.Store(
                    id="ai-chat-analysis-basis-store",
                    data=self._initial_analysis_basis(),
                ),
                dcc.Interval(
                    id="ai-chat-poll", interval=1000, n_intervals=0, disabled=True
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Reserving Studio AI",
                                    style={"fontSize": "28px", "fontWeight": 700},
                                ),
                                html.Div(
                                    "Chat-first actuarial workspace with evidence and traceability attached to the conversation.",
                                    style={"color": COLOR_MUTED, "marginTop": "6px"},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Show Sidebar",
                                    id="ai-sidebar-toggle",
                                    n_clicks=0,
                                    style=self._secondary_button_style(),
                                ),
                                html.Button(
                                    "Refresh Chat",
                                    id="ai-refresh-review",
                                    n_clicks=0,
                                    style=self._primary_button_style(),
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "10px",
                                "flexWrap": "wrap",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "16px",
                        "flexWrap": "wrap",
                        "marginBottom": "14px",
                        "flex": "0 0 auto",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                self._panel(
                                    "Actuarial Chat",
                                    [
                                        html.Div(
                                            "Ask about movements, assumptions, anomalies, scenarios, or evidence and Turtuary will inspect the current reserving data and use the available toolset to help you.",
                                            style={
                                                "color": COLOR_MUTED,
                                                "fontSize": "13px",
                                                "marginBottom": "10px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                self._intro_chat_message(),
                                                html.Div(
                                                    self._render_chat_messages(history),
                                                    id="ai-chat-transcript",
                                                    style={
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "gap": "22px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "flex": "1 1 auto",
                                                "minHeight": "0",
                                                "overflowY": "auto",
                                                "padding": "8px 4px 18px 4px",
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "gap": "22px",
                                                "background": "transparent",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                dcc.Textarea(
                                                    id="ai-chat-input",
                                                    placeholder="Ask the reserving assistant a question...",
                                                    style={
                                                        "flex": "1 1 auto",
                                                        "minHeight": "72px",
                                                        "maxHeight": "160px",
                                                        "border": "none",
                                                        "outline": "none",
                                                        "padding": "14px 2px 0 2px",
                                                        "fontFamily": FONT_FAMILY,
                                                        "fontSize": "16px",
                                                        "resize": "none",
                                                        "background": "transparent",
                                                        "color": COLOR_TEXT,
                                                    },
                                                ),
                                                html.Button(
                                                    ">",
                                                    id="ai-chat-send",
                                                    n_clicks=0,
                                                    style={
                                                        "width": "46px",
                                                        "height": "46px",
                                                        "borderRadius": "23px",
                                                        "border": "none",
                                                        "background": COLOR_ACCENT,
                                                        "color": "#ffffff",
                                                        "fontSize": "28px",
                                                        "fontWeight": 700,
                                                        "cursor": "pointer",
                                                        "flex": "0 0 auto",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "flex-end",
                                                "gap": "12px",
                                                "padding": "12px 14px",
                                                "border": "1px solid #d9dee8",
                                                "borderRadius": "28px",
                                                "background": "#ffffff",
                                                "boxShadow": SHADOW_SOFT,
                                            },
                                        ),
                                        html.Div(
                                            "",
                                            id="ai-chat-status",
                                            style={
                                                "color": COLOR_MUTED,
                                                "fontSize": "13px",
                                                "padding": "0 6px",
                                            },
                                        ),
                                    ],
                                    extra_style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "height": "100%",
                                        "minHeight": "0",
                                    },
                                ),
                            ],
                            style={
                                "minWidth": "0",
                                "flex": "1 1 auto",
                                "height": "100%",
                                "overflow": "hidden",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Evidence And Traceability",
                                            style={
                                                "fontSize": "18px",
                                                "fontWeight": 700,
                                            },
                                        ),
                                        html.Div(
                                            "Open the sections below when you want the evidence cited in chat or the tool and scenario history behind the conversation.",
                                            style={
                                                "color": COLOR_MUTED,
                                                "fontSize": "13px",
                                                "marginTop": "8px",
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "14px"},
                                ),
                                html.Details(
                                    [
                                        html.Summary(
                                            "Chat Evidence References",
                                            style={
                                                "cursor": "pointer",
                                                "fontWeight": 600,
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Evidence IDs and plain-English explanations cited by the assistant during chat.",
                                                    style={
                                                        "color": COLOR_MUTED,
                                                        "fontSize": "13px",
                                                        "margin": "12px 0",
                                                    },
                                                ),
                                                dash_table.DataTable(
                                                    id="ai-chat-evidence-trace",
                                                    columns=[
                                                        {
                                                            "name": "Evidence ID",
                                                            "id": "evidence_id",
                                                        },
                                                        {"name": "Code", "id": "code"},
                                                        {
                                                            "name": "Metric",
                                                            "id": "metric_id",
                                                        },
                                                        {
                                                            "name": "Value",
                                                            "id": "value",
                                                        },
                                                        {
                                                            "name": "Meaning",
                                                            "id": "plain_explanation",
                                                        },
                                                    ],
                                                    data=self._chat_evidence_rows(
                                                        self._initial_tool_events()
                                                    ),
                                                    style_table={"overflowX": "auto"},
                                                    style_cell=self._table_cell_style(),
                                                    style_header=self._table_header_style(),
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={
                                        "background": COLOR_SURFACE,
                                        "border": f"1px solid {COLOR_BORDER}",
                                        "borderRadius": RADIUS_LG,
                                        "padding": "14px",
                                        "boxShadow": SHADOW_SOFT,
                                        "overflowX": "auto",
                                    },
                                ),
                                html.Details(
                                    [
                                        html.Summary(
                                            "Traceability",
                                            style={
                                                "cursor": "pointer",
                                                "fontWeight": 600,
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Tool calls and scenario tests remain available for audit, but stay out of the way during normal chat use.",
                                                    style={
                                                        "color": COLOR_MUTED,
                                                        "fontSize": "13px",
                                                        "margin": "12px 0",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        self._panel(
                                                            "AI Analysis Trace",
                                                            [
                                                                dash_table.DataTable(
                                                                    id="ai-analysis-trace",
                                                                    columns=[
                                                                        {
                                                                            "name": "Tool",
                                                                            "id": "tool",
                                                                        },
                                                                        {
                                                                            "name": "Focus",
                                                                            "id": "focus",
                                                                        },
                                                                        {
                                                                            "name": "Summary",
                                                                            "id": "summary",
                                                                        },
                                                                    ],
                                                                    data=self._tool_event_rows(
                                                                        self._initial_tool_events()
                                                                    ),
                                                                    style_table={
                                                                        "overflowX": "auto"
                                                                    },
                                                                    style_cell=self._table_cell_style(),
                                                                    style_header=self._table_header_style(),
                                                                )
                                                            ],
                                                        ),
                                                        self._panel(
                                                            "Analysis Basis",
                                                            [
                                                                dash_table.DataTable(
                                                                    id="ai-analysis-basis",
                                                                    columns=[
                                                                        {
                                                                            "name": "Field",
                                                                            "id": "field",
                                                                        },
                                                                        {
                                                                            "name": "Value",
                                                                            "id": "value",
                                                                        },
                                                                    ],
                                                                    data=self._analysis_basis_rows(
                                                                        self._initial_analysis_basis()
                                                                    ),
                                                                    style_table={
                                                                        "overflowX": "auto"
                                                                    },
                                                                    style_cell=self._table_cell_style(),
                                                                    style_header=self._table_header_style(),
                                                                )
                                                            ],
                                                        ),
                                                        self._panel(
                                                            "Scenario Ledger",
                                                            [
                                                                dash_table.DataTable(
                                                                    id="ai-scenario-ledger",
                                                                    columns=[
                                                                        {
                                                                            "name": "Scenario",
                                                                            "id": "scenario_id",
                                                                        },
                                                                        {
                                                                            "name": "Score",
                                                                            "id": "score",
                                                                        },
                                                                        {
                                                                            "name": "Tier",
                                                                            "id": "tier",
                                                                        },
                                                                        {
                                                                            "name": "Transform",
                                                                            "id": "transform",
                                                                        },
                                                                        {
                                                                            "name": "Summary",
                                                                            "id": "summary",
                                                                        },
                                                                    ],
                                                                    data=self._scenario_ledger_rows(
                                                                        self._initial_scenario_ledger()
                                                                    ),
                                                                    style_table={
                                                                        "overflowX": "auto"
                                                                    },
                                                                    style_cell=self._table_cell_style(),
                                                                    style_header=self._table_header_style(),
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "grid",
                                                        "gap": "14px",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={
                                        "background": COLOR_SURFACE,
                                        "border": f"1px solid {COLOR_BORDER}",
                                        "borderRadius": RADIUS_LG,
                                        "padding": "14px",
                                        "boxShadow": SHADOW_SOFT,
                                        "overflowX": "auto",
                                    },
                                ),
                            ],
                            id="ai-sidebar",
                            style=self._sidebar_style(False),
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "14px",
                        "alignItems": "stretch",
                        "flex": "1 1 auto",
                        "minHeight": "0",
                        "overflow": "hidden",
                    },
                ),
            ],
            style={
                "padding": "14px 16px 12px 16px",
                "background": COLOR_BG,
                "height": "calc(100dvh - 6px)",
                "fontFamily": FONT_FAMILY,
                "color": COLOR_TEXT,
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
                "boxSizing": "border-box",
            },
        )

    def _initial_chat_history(self) -> list[dict[str, Any]]:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if session is not None:
                return [dict(item) for item in session.messages]
        return []

    def _initial_tool_events(self) -> list[dict[str, Any]]:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if session is not None:
                return [dict(item) for item in session.tool_events]
        return []

    def _initial_scenario_ledger(self) -> list[dict[str, Any]]:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if session is not None:
                return [dict(item) for item in session.scenario_ledger]
        return []

    def _initial_analysis_basis(self) -> dict[str, Any]:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if session is not None:
                basis = (
                    session.working_memory.get("analysis_basis")
                    if isinstance(session.working_memory.get("analysis_basis"), dict)
                    else {}
                )
                return dict(basis)
        return {}

    @staticmethod
    def _preset_prompt_specs() -> list[dict[str, str]]:
        return [dict(item) for item in PRESET_PROMPTS]

    @classmethod
    def _prompt_text_for_trigger(
        cls,
        *,
        triggered_id: object,
        typed_prompt: object,
    ) -> str:
        trigger = str(triggered_id or "").strip()
        if trigger == "ai-chat-send":
            return str(typed_prompt or "").strip()
        for spec in cls._preset_prompt_specs():
            if spec["id"] == trigger:
                return spec["prompt"]
        return str(typed_prompt or "").strip()

    def _render_chat_messages(self, history: list[dict[str, Any]]) -> list:
        items = [item for item in history if isinstance(item, dict)]
        rendered: list = []
        if not items:
            return rendered

        for item in items:
            role = str(item.get("role", "assistant")).strip().lower()
            content = str(item.get("content", "")).strip()
            is_streaming = bool(item.get("streaming", False))
            if not content and not is_streaming:
                continue
            is_user = role == "user"
            bubble_body: list[Any]
            if is_streaming and not content and not is_user:
                bubble_body = [
                    html.Div(
                        [
                            html.Span(className="ai-thinking-dot"),
                            html.Span(className="ai-thinking-dot"),
                            html.Span(className="ai-thinking-dot"),
                        ],
                        className="ai-thinking-indicator",
                    )
                ]
            else:
                bubble_body = [
                    dcc.Markdown(
                        content,
                        style={
                            "lineHeight": "1.65",
                            "margin": "0",
                            "fontSize": "16px",
                        },
                    )
                ]
                if is_streaming and not is_user:
                    bubble_body.append(
                        html.Div(
                            [
                                html.Span(className="ai-thinking-dot"),
                                html.Span(className="ai-thinking-dot"),
                                html.Span(className="ai-thinking-dot"),
                            ],
                            className="ai-thinking-indicator",
                            style={"marginTop": "10px"},
                        )
                    )
            rendered.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=TURTUARY_AVATAR_SRC,
                                    alt="Turtuary avatar",
                                    style={
                                        "width": "52px",
                                        "height": "52px",
                                        "borderRadius": "26px",
                                        "objectFit": "cover",
                                        "flex": "0 0 auto",
                                        "border": f"1px solid {COLOR_BORDER}",
                                        "background": "#f1f5f9",
                                        "display": "none" if is_user else "block",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "You" if is_user else "Turtuary",
                                            style={
                                                "fontSize": "12px",
                                                "fontWeight": 700,
                                                "color": COLOR_MUTED,
                                                "marginBottom": "6px",
                                            },
                                        ),
                                        *bubble_body,
                                    ],
                                    style={"minWidth": "0", "flex": "1 1 auto"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "12px",
                                "alignItems": "flex-start",
                            },
                        ),
                    ],
                    style={
                        "alignSelf": "flex-end" if is_user else "flex-start",
                        "maxWidth": "78%",
                        "padding": "14px 18px",
                        "borderRadius": "22px",
                        "background": "#eef4fb" if is_user else "#ffffff",
                        "color": COLOR_TEXT,
                        "border": f"1px solid {COLOR_BORDER}",
                        "boxShadow": SHADOW_SOFT,
                    },
                )
            )
        return rendered

    @staticmethod
    def _intro_chat_message():
        return html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=TURTUARY_AVATAR_SRC,
                            alt="Turtuary avatar",
                            style={
                                "width": "52px",
                                "height": "52px",
                                "borderRadius": "26px",
                                "objectFit": "cover",
                                "flex": "0 0 auto",
                                "border": f"1px solid {COLOR_BORDER}",
                                "background": "#f1f5f9",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Turtuary",
                                    style={
                                        "fontSize": "12px",
                                        "fontWeight": 700,
                                        "color": COLOR_MUTED,
                                        "marginBottom": "6px",
                                    },
                                ),
                                html.Div(
                                    "Hi, I'm Turtuary. I can inspect your current reserving data, run the available workflows and diagnostics, and help you work through the evidence.",
                                    style={"lineHeight": "1.6"},
                                ),
                                AIDashboard._preset_prompt_grid(),
                            ],
                            style={"minWidth": "0"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "alignItems": "flex-start",
                    },
                ),
            ],
            style={
                "alignSelf": "flex-start",
                "maxWidth": "78%",
                "padding": "14px 18px",
                "background": "#ffffff",
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": "22px",
                "boxShadow": SHADOW_SOFT,
            },
        )

    @staticmethod
    def _preset_prompt_grid():
        return html.Div(
            [
                html.Button(
                    [
                        html.Div(
                            spec["label"],
                            style={
                                "fontSize": "13px",
                                "fontWeight": 700,
                                "marginBottom": "4px",
                                "color": COLOR_TEXT,
                            },
                        ),
                        html.Div(
                            spec["description"],
                            style={
                                "fontSize": "12px",
                                "lineHeight": "1.45",
                                "color": COLOR_MUTED,
                            },
                        ),
                    ],
                    id=spec["id"],
                    n_clicks=0,
                    style={
                        "textAlign": "left",
                        "padding": "12px 12px",
                        "border": f"1px solid {COLOR_BORDER}",
                        "borderRadius": RADIUS_MD,
                        "background": COLOR_ACCENT_SOFT,
                        "cursor": "pointer",
                        "boxShadow": "none",
                    },
                )
                for spec in PRESET_PROMPTS
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                "gap": "10px",
                "marginTop": "14px",
            },
        )

    @staticmethod
    def _tool_event_rows(tool_events: list[dict[str, Any]]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in reversed(tool_events[-12:]):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).replace("tool_", "")
            arguments = (
                item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            )
            result_summary = (
                item.get("result_summary")
                if isinstance(item.get("result_summary"), dict)
                else {}
            )
            focus = (
                arguments.get("scenario_id")
                or arguments.get("uwy")
                or arguments.get("segment")
                or arguments.get("session_id")
                or ""
            )
            summary_bits: list[str] = []
            for key in (
                "finding_count",
                "recommendation_count",
                "scenario_count",
                "result_row_count",
            ):
                value = result_summary.get(key)
                if value is not None:
                    summary_bits.append(f"{key}={value}")
            metrics = result_summary.get("iteration_metrics")
            if isinstance(metrics, dict) and metrics.get("best_scenario_id"):
                summary_bits.append(f"best={metrics.get('best_scenario_id')}")
            gov = result_summary.get("governance")
            if isinstance(gov, dict) and gov.get("tier"):
                summary_bits.append(f"tier={gov.get('tier')}")
            rows.append(
                {
                    "tool": name,
                    "focus": str(focus),
                    "summary": "; ".join(summary_bits) or "detail lookup",
                }
            )
        if not rows:
            return [
                {
                    "tool": "No chat analysis yet",
                    "focus": "",
                    "summary": "Ask the assistant a question to start diagnostics, scenario testing, and evidence lookups.",
                }
            ]
        return rows

    @staticmethod
    def _chat_evidence_rows(tool_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in reversed(tool_events):
            if not isinstance(item, dict):
                continue
            result_summary = (
                item.get("result_summary")
                if isinstance(item.get("result_summary"), dict)
                else {}
            )
            for key in ("top_findings", "top_recommendations"):
                entries = result_summary.get(key)
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    evidence_id = str(entry.get("evidence_id", "")).strip()
                    row_key = (
                        evidence_id or f"{entry.get('code')}|{entry.get('metric_id')}"
                    )
                    if row_key in seen:
                        continue
                    seen.add(row_key)
                    rows.append(
                        {
                            "evidence_id": evidence_id,
                            "code": entry.get("code"),
                            "metric_id": entry.get("metric_id"),
                            "value": entry.get("value"),
                            "plain_explanation": entry.get("plain_explanation")
                            or entry.get("message"),
                        }
                    )
            matches = result_summary.get("matches")
            if not isinstance(matches, list):
                continue
            for match in matches:
                if not isinstance(match, dict):
                    continue
                evidence = (
                    match.get("evidence")
                    if isinstance(match.get("evidence"), dict)
                    else {}
                )
                evidence_id = str(evidence.get("evidence_id", "")).strip()
                row_key = (
                    evidence_id or f"{match.get('code')}|{evidence.get('metric_id')}"
                )
                if row_key in seen:
                    continue
                seen.add(row_key)
                rows.append(
                    {
                        "evidence_id": evidence_id,
                        "code": match.get("code"),
                        "metric_id": evidence.get("metric_id"),
                        "value": evidence.get("value"),
                        "plain_explanation": match.get("plain_explanation")
                        or match.get("message"),
                    }
                )
        if not rows:
            return [
                {
                    "evidence_id": "No chat evidence yet",
                    "code": "",
                    "metric_id": "",
                    "value": "",
                    "plain_explanation": "When the assistant references diagnostics or scenario evidence in chat, the IDs and plain-English explanations will appear here.",
                }
            ]
        return rows[:20]

    @staticmethod
    def _scenario_ledger_rows(
        scenario_ledger: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = [dict(item) for item in scenario_ledger if isinstance(item, dict)]
        if rows:
            return rows
        return [
            {
                "scenario_id": "No chat scenarios tested yet",
                "score": "",
                "tier": "",
                "transform": "",
                "summary": "Scenario searches and bespoke scenario tests launched from the chat will accumulate here.",
            }
        ]

    @staticmethod
    def _analysis_basis_rows(analysis_basis: dict[str, Any]) -> list[dict[str, str]]:
        if not isinstance(analysis_basis, dict) or not analysis_basis:
            return [
                {
                    "field": "Status",
                    "value": "No chat basis locked yet. After a recommendation or exact numeric follow-up, the active analysis basis will appear here.",
                }
            ]
        parameters = (
            analysis_basis.get("parameters")
            if isinstance(analysis_basis.get("parameters"), dict)
            else {}
        )
        tail = (
            parameters.get("tail") if isinstance(parameters.get("tail"), dict) else {}
        )
        rows = [
            {"field": "Basis Type", "value": str(analysis_basis.get("basis_type", ""))},
            {
                "field": "Scenario",
                "value": str(analysis_basis.get("scenario_id") or "baseline"),
            },
            {
                "field": "Matches Active Session",
                "value": "yes"
                if bool(analysis_basis.get("is_active_session"))
                else "no",
            },
            {"field": "Average", "value": str(parameters.get("average", ""))},
            {"field": "Drops", "value": str(parameters.get("drop", []))},
            {"field": "Tail Curve", "value": str(tail.get("curve", ""))},
            {
                "field": "Tail Attachment",
                "value": str(tail.get("attachment_age", "")),
            },
            {"field": "Tail Fit Period", "value": str(tail.get("fit_period", []))},
            {
                "field": "BF Apriori",
                "value": str(parameters.get("bf_apriori", {})),
            },
            {
                "field": "Method Overrides",
                "value": str(parameters.get("selected_ultimate_by_uwy", {})),
            },
        ]
        return rows

    @staticmethod
    def _panel(title: str, children: list, extra_style: dict | None = None):
        style = {
            "background": COLOR_SURFACE,
            "border": f"1px solid {COLOR_BORDER}",
            "borderRadius": RADIUS_LG,
            "padding": "14px",
            "boxShadow": SHADOW_SOFT,
            "overflowX": "auto",
        }
        if isinstance(extra_style, dict):
            style.update(extra_style)
        return html.Div(
            [html.H3(title, style={"marginTop": "0"})] + children, style=style
        )

    @staticmethod
    def _sidebar_style(is_open: bool) -> dict[str, str]:
        return {
            "width": "720px" if is_open else "0px",
            "minWidth": "720px" if is_open else "0px",
            "overflowX": "hidden",
            "overflowY": "auto",
            "transition": "width 0.2s ease, min-width 0.2s ease",
            "flex": "0 0 auto",
            "display": "grid" if is_open else "none",
            "gap": "14px",
            "gridAutoRows": "min-content",
            "alignContent": "start",
            "background": COLOR_SURFACE,
            "border": f"1px solid {COLOR_BORDER}",
            "borderRadius": RADIUS_LG,
            "padding": "14px",
            "boxShadow": SHADOW_SOFT,
            "height": "100%",
            "minHeight": "0",
        }

    @staticmethod
    def _primary_button_style() -> dict[str, str]:
        return {
            "padding": "10px 14px",
            "background": COLOR_ACCENT,
            "color": "#ffffff",
            "border": "none",
            "borderRadius": RADIUS_MD,
            "fontWeight": 600,
            "cursor": "pointer",
        }

    @staticmethod
    def _secondary_button_style() -> dict[str, str]:
        return {
            "padding": "10px 14px",
            "background": COLOR_SURFACE,
            "color": COLOR_TEXT,
            "border": f"1px solid {COLOR_BORDER}",
            "borderRadius": RADIUS_MD,
            "fontWeight": 600,
            "cursor": "pointer",
        }

    @staticmethod
    def _table_cell_style() -> dict[str, str]:
        return {
            "fontFamily": FONT_FAMILY,
            "fontSize": "12px",
            "padding": "8px",
            "border": f"1px solid {COLOR_BORDER}",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
        }

    @staticmethod
    def _table_header_style() -> dict[str, str]:
        return {
            "fontWeight": 600,
            "backgroundColor": COLOR_ACCENT_SOFT,
            "border": f"1px solid {COLOR_BORDER}",
            "textAlign": "left",
        }

    def show(self, debug: bool = False, port: int = 8052) -> None:
        self.app.layout = self._create_layout
        logging.info("Starting AI dashboard on http://127.0.0.1:%s", port)
        self.app.run(debug=debug, port=port, use_reloader=False)


def launch_ai_dashboard(
    reserving: Reserving,
    *,
    config: ConfigManager | None = None,
    chat_service: AIChatService | None = None,
    chat_id: str | None = None,
    debug: bool = False,
    port: int = 8052,
) -> AIDashboard:
    dashboard = AIDashboard(
        reserving,
        config=config,
        chat_service=chat_service,
        chat_id=chat_id,
    )
    dashboard.show(debug=debug, port=port)
    return dashboard
