from __future__ import annotations

import json
import logging
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

from source.ai_review import AIReviewService
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


class AIDashboard:
    def __init__(
        self,
        reserving: Reserving,
        *,
        config: ConfigManager | None = None,
    ) -> None:
        self._config = config
        self._review_service = AIReviewService(reserving, config=config)
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
        @self.app.callback(
            Output("ai-review-store", "data"),
            Output("ai-chat-history-store", "data"),
            Output("ai-chat-transcript", "children"),
            Output("ai-summary-text", "children"),
            Output("ai-governance-card", "children"),
            Output("ai-uncertainty-card", "children"),
            Output("ai-best-scenario-card", "children"),
            Output("ai-scenario-matrix", "data"),
            Output("ai-evidence-trace", "data"),
            Input("ai-refresh-review", "n_clicks"),
        )
        def _refresh_review(_n_clicks):
            review_data = self._review_service.build_review_payload()
            history = [
                {"role": "assistant", "content": review_data.get("ai_commentary", "")}
            ]
            return (
                review_data,
                history,
                AIReviewService.render_chat_transcript(history),
                review_data.get("ai_commentary", ""),
                self._governance_card(review_data),
                self._uncertainty_card(review_data),
                self._best_scenario_card(review_data),
                review_data.get("scenario_matrix", []),
                review_data.get("evidence_trace", []),
            )

        @self.app.callback(
            Output("ai-chat-history-store", "data", allow_duplicate=True),
            Output("ai-chat-transcript", "children", allow_duplicate=True),
            Output("ai-chat-input", "value"),
            Input("ai-chat-send", "n_clicks"),
            State("ai-chat-input", "value"),
            State("ai-chat-history-store", "data"),
            State("ai-review-store", "data"),
            prevent_initial_call=True,
        )
        def _chat(_n_clicks, prompt, history, review_data):
            prompt_text = str(prompt or "").strip()
            if not prompt_text:
                return no_update, no_update, no_update
            rows = history if isinstance(history, list) else []
            normalized = [item for item in rows if isinstance(item, dict)]
            normalized.append({"role": "user", "content": prompt_text})
            answer = AIReviewService.answer_chat_prompt(prompt_text, review_data)
            normalized.append({"role": "assistant", "content": answer})
            return normalized, AIReviewService.render_chat_transcript(normalized), ""

        @self.app.callback(
            Output("ai-review-store", "data", allow_duplicate=True),
            Output("ai-decision-status", "children"),
            Input("ai-save-decision", "n_clicks"),
            State("ai-decision", "value"),
            State("ai-approver", "value"),
            State("ai-rationale", "value"),
            State("ai-review-store", "data"),
            prevent_initial_call=True,
        )
        def _save_decision(_n_clicks, decision, approver, rationale, review_data):
            if not isinstance(review_data, dict):
                return no_update, "Run review first."
            updated = dict(review_data)
            updated["ai_override"] = {
                "decision": str(decision or "pending"),
                "approver": str(approver or "").strip(),
                "rationale": str(rationale or "").strip(),
            }
            self._review_service.persist_review(updated)
            return updated, "Decision saved."

        @self.app.callback(
            Output("ai-decision-download", "data"),
            Input("ai-export-decision", "n_clicks"),
            State("ai-review-store", "data"),
            prevent_initial_call=True,
        )
        def _export_decision(_n_clicks, review_data):
            if not isinstance(review_data, dict):
                return no_update
            packet = AIReviewService.build_decision_packet(review_data)
            segment = str(review_data.get("segment", "segment"))
            return {
                "content": json.dumps(packet, indent=2, default=str),
                "filename": f"{segment}-ai-decision-packet.json",
                "type": "application/json",
            }

    def _create_layout(self):
        review_data = self._review_service.build_review_payload()
        history = [
            {"role": "assistant", "content": review_data.get("ai_commentary", "")}
        ]
        return html.Div(
            [
                dcc.Store(id="ai-review-store", data=review_data),
                dcc.Store(id="ai-chat-history-store", data=history),
                dcc.Download(id="ai-decision-download"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Reserving Studio AI Review",
                                    style={"fontSize": "28px", "fontWeight": 700},
                                ),
                                html.Div(
                                    "Single-page actuarial review workspace for scenario comparison, evidence trace, and sign-off.",
                                    style={"color": COLOR_MUTED, "marginTop": "6px"},
                                ),
                            ]
                        ),
                        html.Button(
                            "Refresh Review",
                            id="ai-refresh-review",
                            n_clicks=0,
                            style=self._primary_button_style(),
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "16px",
                        "flexWrap": "wrap",
                        "marginBottom": "18px",
                    },
                ),
                html.Div(
                    [
                        self._card(
                            "Governance",
                            self._governance_card(review_data),
                            "ai-governance-card",
                        ),
                        self._card(
                            "Uncertainty",
                            self._uncertainty_card(review_data),
                            "ai-uncertainty-card",
                        ),
                        self._card(
                            "Best Scenario",
                            self._best_scenario_card(review_data),
                            "ai-best-scenario-card",
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
                        "gap": "14px",
                        "marginBottom": "18px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                self._panel(
                                    "Chat",
                                    [
                                        dcc.Textarea(
                                            id="ai-chat-input",
                                            placeholder="Ask about governance, uncertainty, scenarios, or evidence...",
                                            style={
                                                "width": "100%",
                                                "minHeight": "74px",
                                                "border": f"1px solid {COLOR_BORDER}",
                                                "borderRadius": RADIUS_MD,
                                                "padding": "10px",
                                                "fontFamily": FONT_FAMILY,
                                            },
                                        ),
                                        html.Button(
                                            "Send",
                                            id="ai-chat-send",
                                            n_clicks=0,
                                            style={
                                                **self._primary_button_style(),
                                                "marginTop": "10px",
                                            },
                                        ),
                                        dcc.Markdown(
                                            AIReviewService.render_chat_transcript(
                                                history
                                            ),
                                            id="ai-chat-transcript",
                                            style={
                                                "marginTop": "12px",
                                                "minHeight": "220px",
                                                "padding": "12px",
                                                "border": f"1px solid {COLOR_BORDER}",
                                                "borderRadius": RADIUS_MD,
                                                "background": "#fbfcfe",
                                            },
                                        ),
                                    ],
                                ),
                                self._panel(
                                    "Commentary",
                                    [
                                        html.Div(
                                            review_data.get("ai_commentary", ""),
                                            id="ai-summary-text",
                                            style={"lineHeight": "1.6"},
                                        )
                                    ],
                                ),
                            ],
                            style={"display": "grid", "gap": "14px"},
                        ),
                        html.Div(
                            [
                                self._panel(
                                    "Scenario Matrix",
                                    [
                                        dash_table.DataTable(
                                            id="ai-scenario-matrix",
                                            columns=[
                                                {
                                                    "name": "Scenario",
                                                    "id": "scenario_id",
                                                },
                                                {"name": "Score", "id": "score"},
                                                {
                                                    "name": "Tier",
                                                    "id": "governance_tier",
                                                },
                                                {
                                                    "name": "Transform",
                                                    "id": "transform",
                                                },
                                                {
                                                    "name": "Evidence",
                                                    "id": "evidence_refs",
                                                },
                                            ],
                                            data=review_data.get("scenario_matrix", []),
                                            style_table={"overflowX": "auto"},
                                            style_cell=self._table_cell_style(),
                                            style_header=self._table_header_style(),
                                        )
                                    ],
                                ),
                                self._panel(
                                    "Evidence Trace",
                                    [
                                        dash_table.DataTable(
                                            id="ai-evidence-trace",
                                            columns=[
                                                {"name": "Kind", "id": "kind"},
                                                {
                                                    "name": "Evidence ID",
                                                    "id": "evidence_id",
                                                },
                                                {"name": "Code", "id": "code"},
                                                {"name": "Severity", "id": "severity"},
                                                {"name": "Metric", "id": "metric"},
                                                {"name": "Value", "id": "value"},
                                            ],
                                            data=review_data.get("evidence_trace", []),
                                            style_table={"overflowX": "auto"},
                                            style_cell=self._table_cell_style(),
                                            style_header=self._table_header_style(),
                                        )
                                    ],
                                ),
                            ],
                            style={"display": "grid", "gap": "14px"},
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(320px, 1.05fr) minmax(360px, 1fr)",
                        "gap": "14px",
                        "alignItems": "start",
                    },
                ),
                self._panel(
                    "Override and Sign-off",
                    [
                        dcc.Dropdown(
                            id="ai-decision",
                            options=[
                                {"label": "Approve", "value": "approve"},
                                {
                                    "label": "Approve with conditions",
                                    "value": "approve_with_conditions",
                                },
                                {"label": "Escalate", "value": "escalate"},
                                {"label": "Reject", "value": "reject"},
                            ],
                            value="approve_with_conditions",
                            clearable=False,
                            style={"maxWidth": "340px", "marginBottom": "10px"},
                        ),
                        dcc.Input(
                            id="ai-approver",
                            type="text",
                            placeholder="Approver name",
                            style={
                                "width": "340px",
                                "padding": "10px",
                                "border": f"1px solid {COLOR_BORDER}",
                                "borderRadius": RADIUS_MD,
                                "marginBottom": "10px",
                            },
                        ),
                        dcc.Textarea(
                            id="ai-rationale",
                            placeholder="Document rationale, conditions, and sign-off notes...",
                            style={
                                "width": "100%",
                                "minHeight": "110px",
                                "padding": "10px",
                                "border": f"1px solid {COLOR_BORDER}",
                                "borderRadius": RADIUS_MD,
                            },
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Save Decision",
                                    id="ai-save-decision",
                                    n_clicks=0,
                                    style=self._primary_button_style(),
                                ),
                                html.Button(
                                    "Export Decision Packet",
                                    id="ai-export-decision",
                                    n_clicks=0,
                                    style=self._secondary_button_style(),
                                ),
                                html.Span(
                                    "",
                                    id="ai-decision-status",
                                    style={"color": COLOR_MUTED, "fontSize": "13px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "10px",
                                "alignItems": "center",
                                "flexWrap": "wrap",
                                "marginTop": "12px",
                            },
                        ),
                    ],
                    extra_style={"marginTop": "18px"},
                ),
            ],
            style={
                "padding": "20px",
                "background": COLOR_BG,
                "minHeight": "100vh",
                "fontFamily": FONT_FAMILY,
                "color": COLOR_TEXT,
            },
        )

    @staticmethod
    def _card(title: str, content: str, element_id: str):
        return html.Div(
            [
                html.Div(
                    title,
                    style={
                        "fontSize": "12px",
                        "color": COLOR_MUTED,
                        "marginBottom": "6px",
                    },
                ),
                html.Div(
                    content,
                    id=element_id,
                    style={"fontSize": "14px", "fontWeight": 600, "lineHeight": "1.5"},
                ),
            ],
            style={
                "background": COLOR_SURFACE,
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": RADIUS_LG,
                "padding": "14px",
                "boxShadow": SHADOW_SOFT,
            },
        )

    @staticmethod
    def _panel(title: str, children: list, extra_style: dict | None = None):
        style = {
            "background": COLOR_SURFACE,
            "border": f"1px solid {COLOR_BORDER}",
            "borderRadius": RADIUS_LG,
            "padding": "14px",
            "boxShadow": SHADOW_SOFT,
        }
        if isinstance(extra_style, dict):
            style.update(extra_style)
        return html.Div(
            [html.H3(title, style={"marginTop": "0"})] + children, style=style
        )

    @staticmethod
    def _governance_card(review_data: dict) -> str:
        return AIReviewService.summarize_governance(
            review_data.get("governance", {}),
            review_data.get("uncertainty", {}),
        )

    @staticmethod
    def _uncertainty_card(review_data: dict) -> str:
        uncertainty = review_data.get("uncertainty", {})
        baseline = (
            uncertainty.get("baseline", {}) if isinstance(uncertainty, dict) else {}
        )
        bootstrap = (
            uncertainty.get("bootstrap", {}) if isinstance(uncertainty, dict) else {}
        )
        return (
            f"Process CV={baseline.get('total_process_cv')}; "
            f"P50={bootstrap.get('p50')}; P90={bootstrap.get('p90')}"
        )

    @staticmethod
    def _best_scenario_card(review_data: dict) -> str:
        scenario_matrix = review_data.get("scenario_matrix", [])
        if not isinstance(scenario_matrix, list) or not scenario_matrix:
            return "No scenarios available."
        top = scenario_matrix[0]
        return f"{top.get('scenario_id')} | score={top.get('score')} | tier={top.get('governance_tier')}"

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

    def show(self, debug: bool = False, port: int = 8050) -> None:
        self.app.layout = self._create_layout
        logging.info("Starting AI dashboard on http://127.0.0.1:%s", port)
        self.app.run(debug=debug, port=port, use_reloader=False)


def launch_ai_dashboard(
    reserving: Reserving,
    *,
    config: ConfigManager | None = None,
    debug: bool = False,
    port: int = 8050,
) -> AIDashboard:
    dashboard = AIDashboard(reserving, config=config)
    dashboard.show(debug=debug, port=port)
    return dashboard
