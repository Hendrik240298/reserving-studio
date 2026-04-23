from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, State, ctx, dcc, html, dash_table, no_update

from ai.chat_service import AIChatService
from source.config_manager import ConfigManager
from source.reserving import Reserving
from source.services.memory_authoring_service import MemoryAuthoringService


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
        self._memory_authoring_service = MemoryAuthoringService()
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
            Output("ai-chat-proposal-basis-store", "data"),
            Output("ai-memory-proposals-store", "data"),
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
            execution_records = self._initial_execution_records()
            scenario_ledger = self._initial_scenario_ledger()
            analysis_basis = self._initial_analysis_basis()
            proposal_basis = self._initial_proposal_basis()
            proposals = self._initial_memory_proposals()
            return (
                history,
                tool_events,
                scenario_ledger,
                analysis_basis,
                proposal_basis,
                proposals,
                self._render_chat_messages(history, proposal_basis),
                self._execution_record_rows(execution_records),
                self._chat_evidence_rows(tool_events),
                self._scenario_ledger_rows(scenario_ledger),
                self._analysis_basis_rows(analysis_basis),
            )

        @self.app.callback(
            Output("ai-chat-history-store", "data", allow_duplicate=True),
            Output("ai-chat-tool-events-store", "data", allow_duplicate=True),
            Output("ai-chat-scenario-ledger-store", "data", allow_duplicate=True),
            Output("ai-chat-analysis-basis-store", "data", allow_duplicate=True),
            Output("ai-chat-proposal-basis-store", "data", allow_duplicate=True),
            Output("ai-memory-proposals-store", "data", allow_duplicate=True),
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
                    no_update,
                    no_update,
                    "AI chat backend is not configured.",
                    no_update,
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
                    self._initial_proposal_basis(),
                    self._initial_memory_proposals(),
                    self._render_chat_messages(
                        normalized,
                        self._initial_proposal_basis(),
                    ),
                    "",
                    "AI response failed.",
                    self._execution_record_rows(self._initial_execution_records()),
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
            execution_records = response.get("execution_records")
            scenario_ledger = response.get("scenario_ledger")
            analysis_basis = response.get("accepted_analysis_basis")
            proposal_basis = response.get("proposal_basis")
            normalized_tool_events = (
                [dict(item) for item in tool_events if isinstance(item, dict)]
                if isinstance(tool_events, list)
                else []
            )
            normalized_execution_records = (
                [dict(item) for item in execution_records if isinstance(item, dict)]
                if isinstance(execution_records, list)
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
            normalized_proposal_basis = (
                dict(proposal_basis) if isinstance(proposal_basis, dict) else {}
            )
            proposals = response.get("memory_update_proposals")
            normalized_proposals = (
                [dict(item) for item in proposals if isinstance(item, dict)]
                if isinstance(proposals, list)
                else []
            )
            status = (
                "AI fallback summary used." if response.get("fallback_used") else ""
            )
            return (
                messages,
                normalized_tool_events,
                normalized_scenario_ledger,
                normalized_analysis_basis,
                normalized_proposal_basis,
                normalized_proposals,
                self._render_chat_messages(messages, normalized_proposal_basis),
                "",
                status,
                self._execution_record_rows(normalized_execution_records),
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
            Output("ai-chat-proposal-basis-store", "data", allow_duplicate=True),
            Output("ai-memory-proposals-store", "data", allow_duplicate=True),
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
                    no_update,
                    no_update,
                    True,
                )
            response = self._chat_service.build_chat_response(self._chat_id)
            messages = response.get("messages")
            tool_events = response.get("tool_events")
            execution_records = response.get("execution_records")
            scenario_ledger = response.get("scenario_ledger")
            analysis_basis = response.get("accepted_analysis_basis")
            proposal_basis = response.get("proposal_basis")
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
                    no_update,
                    no_update,
                    True,
                )
            normalized_tool_events = (
                [dict(item) for item in tool_events if isinstance(item, dict)]
                if isinstance(tool_events, list)
                else []
            )
            normalized_execution_records = (
                [dict(item) for item in execution_records if isinstance(item, dict)]
                if isinstance(execution_records, list)
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
            normalized_proposal_basis = (
                dict(proposal_basis) if isinstance(proposal_basis, dict) else {}
            )
            proposals = response.get("memory_update_proposals")
            normalized_proposals = (
                [dict(item) for item in proposals if isinstance(item, dict)]
                if isinstance(proposals, list)
                else []
            )
            status = (
                "AI fallback summary used." if response.get("fallback_used") else ""
            )
            return (
                messages,
                normalized_tool_events,
                normalized_scenario_ledger,
                normalized_analysis_basis,
                normalized_proposal_basis,
                normalized_proposals,
                self._render_chat_messages(messages, normalized_proposal_basis),
                status,
                self._execution_record_rows(normalized_execution_records),
                self._chat_evidence_rows(normalized_tool_events),
                self._scenario_ledger_rows(normalized_scenario_ledger),
                self._analysis_basis_rows(normalized_analysis_basis),
                not bool(response.get("streaming")),
            )

        @self.app.callback(
            Output("ai-chat-history-store", "data", allow_duplicate=True),
            Output("ai-chat-analysis-basis-store", "data", allow_duplicate=True),
            Output("ai-chat-proposal-basis-store", "data", allow_duplicate=True),
            Output("ai-chat-transcript", "children", allow_duplicate=True),
            Output("ai-analysis-basis", "data", allow_duplicate=True),
            Output("ai-chat-status", "children", allow_duplicate=True),
            Input("ai-chat-proposal-accept", "n_clicks"),
            Input("ai-chat-proposal-reject", "n_clicks"),
            State("ai-chat-proposal-basis-store", "data"),
            prevent_initial_call=True,
        )
        def _handle_chat_proposal(accept_clicks, reject_clicks, proposal_basis):
            action = self._proposal_action_from_trigger(
                triggered_id=ctx.triggered_id,
                accept_clicks=accept_clicks,
                reject_clicks=reject_clicks,
            )
            if not action:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            normalized_proposal_basis = (
                dict(proposal_basis) if isinstance(proposal_basis, dict) else {}
            )
            proposal_id = str(normalized_proposal_basis.get("proposal_id") or "").strip()
            if not proposal_id:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "No pending proposal is available.",
                )
            if self._chat_service is None or not self._chat_id:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "AI chat backend is not configured.",
                )
            try:
                if action == "accept":
                    response = self._chat_service.accept_proposal(
                        self._chat_id,
                        proposal_id,
                    )
                    status = "Proposal accepted. Analysis Basis updated for this chat."
                else:
                    response = self._chat_service.reject_proposal(
                        self._chat_id,
                        proposal_id,
                    )
                    status = "Proposal rejected. Analysis Basis unchanged."
            except Exception as error:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    f"Proposal action failed: {error}",
                )
            messages = response.get("messages")
            accepted_analysis_basis = response.get("accepted_analysis_basis")
            next_proposal_basis = response.get("proposal_basis")
            normalized_analysis_basis = (
                dict(accepted_analysis_basis)
                if isinstance(accepted_analysis_basis, dict)
                else {}
            )
            normalized_next_proposal_basis = (
                dict(next_proposal_basis) if isinstance(next_proposal_basis, dict) else {}
            )
            return (
                [dict(item) for item in messages if isinstance(item, dict)]
                if isinstance(messages, list)
                else no_update,
                normalized_analysis_basis,
                normalized_next_proposal_basis,
                self._render_chat_messages(messages, normalized_next_proposal_basis)
                if isinstance(messages, list)
                else no_update,
                self._analysis_basis_rows(normalized_analysis_basis),
                status,
            )

        @self.app.callback(
            Output("ai-segment-memory-store", "data"),
            Output("ai-memory-segment-overview", "value"),
            Output("ai-memory-known-issues", "value"),
            Output("ai-memory-house-preferences", "value"),
            Output("ai-memory-recent-quarter-notes", "value"),
            Output("ai-memory-open-items", "value"),
            Output("ai-memory-structured-preferences", "children"),
            Output("ai-memory-change-log", "data"),
            Output("ai-memory-status", "children"),
            Input("ai-refresh-review", "n_clicks"),
        )
        def _refresh_memory(_n_clicks):
            return self._memory_payload_outputs(
                self._initial_segment_memory_payload(),
                status="",
            )

        @self.app.callback(
            Output("ai-segment-memory-store", "data", allow_duplicate=True),
            Output("ai-memory-segment-overview", "value", allow_duplicate=True),
            Output("ai-memory-known-issues", "value", allow_duplicate=True),
            Output("ai-memory-house-preferences", "value", allow_duplicate=True),
            Output("ai-memory-recent-quarter-notes", "value", allow_duplicate=True),
            Output("ai-memory-open-items", "value", allow_duplicate=True),
            Output(
                "ai-memory-structured-preferences", "children", allow_duplicate=True
            ),
            Output("ai-memory-change-log", "data", allow_duplicate=True),
            Output("ai-memory-status", "children", allow_duplicate=True),
            Input("ai-memory-save-button", "n_clicks"),
            State("ai-memory-segment-overview", "value"),
            State("ai-memory-known-issues", "value"),
            State("ai-memory-house-preferences", "value"),
            State("ai-memory-recent-quarter-notes", "value"),
            State("ai-memory-open-items", "value"),
            prevent_initial_call=True,
        )
        def _save_memory(
            n_clicks,
            segment_overview,
            known_issues,
            house_preferences,
            recent_quarter_notes,
            open_items,
        ):
            if not n_clicks:
                return self._memory_payload_outputs(no_update, status=no_update)
            if self._config is None:
                return self._memory_payload_outputs(
                    self._initial_segment_memory_payload(),
                    status="Memory persistence is not configured.",
                )
            memory = self._memory_authoring_service.save_manual_update(
                config=self._config,
                segment=self._current_segment(),
                fields={
                    "segment_overview": segment_overview,
                    "known_issues_text": known_issues,
                    "house_preferences_text": house_preferences,
                    "recent_quarter_notes_text": recent_quarter_notes,
                    "open_items_text": open_items,
                },
                editor="ai_dashboard",
            )
            return self._memory_payload_outputs(
                self._memory_authoring_service.build_ui_payload(memory),
                status="Segment memory saved.",
            )

        @self.app.callback(
            Output("ai-memory-proposal-selector", "options"),
            Output("ai-memory-proposal-selector", "value"),
            Output("ai-memory-proposal-field", "children"),
            Output("ai-memory-proposal-rationale", "children"),
            Output("ai-memory-proposal-evidence", "children"),
            Output("ai-memory-proposal-editor", "value"),
            Input("ai-memory-proposals-store", "data"),
            State("ai-memory-proposal-selector", "value"),
        )
        def _sync_memory_proposals(proposals, selected_proposal_id):
            normalized = (
                [dict(item) for item in proposals if isinstance(item, dict)]
                if isinstance(proposals, list)
                else []
            )
            options = self._memory_proposal_options(normalized)
            selected = (
                selected_proposal_id
                if any(
                    str(item.get("value")) == str(selected_proposal_id)
                    for item in options
                )
                else (options[0]["value"] if options else None)
            )
            proposal = self._proposal_by_id(normalized, selected)
            return (
                options,
                selected,
                self._proposal_field_label(proposal),
                self._proposal_rationale_label(proposal),
                self._proposal_evidence_label(proposal),
                self._proposal_editable_value(proposal),
            )

        @self.app.callback(
            Output("ai-memory-proposals-store", "data", allow_duplicate=True),
            Output("ai-segment-memory-store", "data", allow_duplicate=True),
            Output("ai-memory-segment-overview", "value", allow_duplicate=True),
            Output("ai-memory-known-issues", "value", allow_duplicate=True),
            Output("ai-memory-house-preferences", "value", allow_duplicate=True),
            Output("ai-memory-recent-quarter-notes", "value", allow_duplicate=True),
            Output("ai-memory-open-items", "value", allow_duplicate=True),
            Output(
                "ai-memory-structured-preferences", "children", allow_duplicate=True
            ),
            Output("ai-memory-change-log", "data", allow_duplicate=True),
            Output("ai-memory-status", "children", allow_duplicate=True),
            Input("ai-memory-apply-proposal", "n_clicks"),
            Input("ai-memory-reject-proposal", "n_clicks"),
            State("ai-memory-proposals-store", "data"),
            State("ai-memory-proposal-selector", "value"),
            State("ai-memory-proposal-editor", "value"),
            prevent_initial_call=True,
        )
        def _handle_memory_proposal(
            apply_clicks,
            reject_clicks,
            proposals,
            proposal_id,
            edited_value,
        ):
            del apply_clicks, reject_clicks
            normalized = (
                [dict(item) for item in proposals if isinstance(item, dict)]
                if isinstance(proposals, list)
                else []
            )
            proposal = self._proposal_by_id(normalized, proposal_id)
            if not proposal:
                return (
                    normalized,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "Select a memory proposal first.",
                )
            if self._config is None:
                return (
                    normalized,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "Memory persistence is not configured.",
                )
            triggered = str(ctx.triggered_id or "")
            if triggered == "ai-memory-reject-proposal":
                updated_proposals = [
                    item
                    for item in normalized
                    if item.get("proposal_id") != proposal_id
                ]
                self._persist_memory_proposals(updated_proposals)
                return (
                    updated_proposals,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    "Memory proposal rejected.",
                )
            applied_memory, _approved = (
                self._memory_authoring_service.apply_memory_proposal(
                    config=self._config,
                    segment=self._current_segment(),
                    proposal=proposal,
                    approver="ai_dashboard",
                    edited_value=self._coerce_edited_proposal_value(
                        proposal, edited_value
                    ),
                )
            )
            updated_proposals = [
                item for item in normalized if item.get("proposal_id") != proposal_id
            ]
            self._persist_memory_proposals(updated_proposals)
            return (
                updated_proposals,
                *self._memory_payload_outputs(
                    self._memory_authoring_service.build_ui_payload(applied_memory),
                    status="Memory proposal applied.",
                ),
            )

    def _create_layout(self):
        history = self._initial_chat_history()
        initial_memory_payload = self._initial_segment_memory_payload()
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
                dcc.Store(
                    id="ai-chat-proposal-basis-store",
                    data=self._initial_proposal_basis(),
                ),
                dcc.Store(
                    id="ai-segment-memory-store",
                    data=initial_memory_payload,
                ),
                dcc.Store(
                    id="ai-memory-proposals-store",
                    data=self._initial_memory_proposals(),
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
                                                    self._render_chat_messages(
                                                        history,
                                                        self._initial_proposal_basis(),
                                                    ),
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
                                            "Memory, Evidence And Traceability",
                                            style={
                                                "fontSize": "18px",
                                                "fontWeight": 700,
                                            },
                                        ),
                                        html.Div(
                                            "Use the sections below to maintain segment memory, review pending AI memory proposals, and inspect the evidence behind the conversation.",
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
                                            "Segment Memory",
                                            style={
                                                "cursor": "pointer",
                                                "fontWeight": 600,
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "This is the persisted segment memory the AI can use for future continuity and review context.",
                                                    style={
                                                        "color": COLOR_MUTED,
                                                        "fontSize": "13px",
                                                        "margin": "12px 0",
                                                    },
                                                ),
                                                self._memory_field_label(
                                                    "Segment Overview",
                                                    (
                                                        "What it means: a short persistent description of the segment, its reserving context, and any enduring framing the AI should keep in mind.\n\n"
                                                        "Impact: gives the assistant high-level background for future conversations and helps it interpret results in the right segment context."
                                                    ),
                                                ),
                                                dcc.Textarea(
                                                    id="ai-memory-segment-overview",
                                                    value=initial_memory_payload.get(
                                                        "segment_overview", ""
                                                    ),
                                                    style=self._memory_textarea_style(
                                                        height="88px"
                                                    ),
                                                ),
                                                self._memory_field_label(
                                                    "Known Issues",
                                                    (
                                                        "What it means: persistent segment-specific problems, caveats, or distortions that can affect interpretation of the reserving analysis.\n\n"
                                                        "Impact: the assistant reads these as important background constraints and should factor them into diagnostics, caveats, and recommendations."
                                                    ),
                                                    margin_top="12px",
                                                ),
                                                dcc.Textarea(
                                                    id="ai-memory-known-issues",
                                                    value=initial_memory_payload.get(
                                                        "known_issues_text", ""
                                                    ),
                                                    style=self._memory_textarea_style(
                                                        height="110px"
                                                    ),
                                                ),
                                                self._memory_field_label(
                                                    "House Preferences",
                                                    (
                                                        "What it means: stable reserving preferences, judgment style, and governance tendencies the team usually wants followed.\n\n"
                                                        "Impact: these currently influence AI context and recommendation framing as guidance, but they are not all enforced as hard deterministic rules unless explicitly implemented in code."
                                                    ),
                                                    margin_top="12px",
                                                ),
                                                dcc.Textarea(
                                                    id="ai-memory-house-preferences",
                                                    value=initial_memory_payload.get(
                                                        "house_preferences_text", ""
                                                    ),
                                                    style=self._memory_textarea_style(
                                                        height="90px"
                                                    ),
                                                ),
                                                html.Div(
                                                    self._structured_preference_summary(
                                                        initial_memory_payload.get(
                                                            "structured_house_preferences",
                                                            [],
                                                        )
                                                    ),
                                                    id="ai-memory-structured-preferences",
                                                    style={
                                                        "marginTop": "8px",
                                                        "fontSize": "12px",
                                                        "color": COLOR_MUTED,
                                                        "whiteSpace": "pre-wrap",
                                                    },
                                                ),
                                                self._memory_field_label(
                                                    "Recent Quarter Notes",
                                                    (
                                                        "What it means: recent quarter-specific observations or decisions that matter for short-term continuity.\n\n"
                                                        "Impact: the assistant uses these for continuity context, but only the latest four notes are loaded into AI context by default to stay token efficient."
                                                    ),
                                                    margin_top="12px",
                                                ),
                                                dcc.Textarea(
                                                    id="ai-memory-recent-quarter-notes",
                                                    value=initial_memory_payload.get(
                                                        "recent_quarter_notes_text", ""
                                                    ),
                                                    style=self._memory_textarea_style(
                                                        height="110px"
                                                    ),
                                                ),
                                                html.Div(
                                                    "Use one line per note, formatted as '2026Q1 | note'.",
                                                    style={
                                                        "fontSize": "12px",
                                                        "color": COLOR_MUTED,
                                                        "marginTop": "6px",
                                                    },
                                                ),
                                                self._memory_field_label(
                                                    "Open Items",
                                                    (
                                                        "What it means: unresolved questions, follow-ups, or issues that should be carried into future review cycles.\n\n"
                                                        "Impact: the assistant can use these to preserve continuity, highlight pending work, and suggest what still needs investigation."
                                                    ),
                                                    margin_top="12px",
                                                ),
                                                dcc.Textarea(
                                                    id="ai-memory-open-items",
                                                    value=initial_memory_payload.get(
                                                        "open_items_text", ""
                                                    ),
                                                    style=self._memory_textarea_style(
                                                        height="96px"
                                                    ),
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Save Memory",
                                                            id="ai-memory-save-button",
                                                            n_clicks=0,
                                                            style=self._primary_button_style(),
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "gap": "10px",
                                                        "marginTop": "14px",
                                                    },
                                                ),
                                                html.Div(
                                                    "",
                                                    id="ai-memory-status",
                                                    style={
                                                        "fontSize": "12px",
                                                        "color": COLOR_MUTED,
                                                        "marginTop": "10px",
                                                    },
                                                ),
                                                dash_table.DataTable(
                                                    id="ai-memory-change-log",
                                                    columns=[
                                                        {
                                                            "name": "Field",
                                                            "id": "field",
                                                        },
                                                        {
                                                            "name": "Action",
                                                            "id": "action",
                                                        },
                                                        {
                                                            "name": "Summary",
                                                            "id": "summary",
                                                        },
                                                        {
                                                            "name": "When",
                                                            "id": "updated_at",
                                                        },
                                                    ],
                                                    data=initial_memory_payload.get(
                                                        "memory_change_log", []
                                                    ),
                                                    style_table={
                                                        "overflowX": "auto",
                                                        "marginTop": "14px",
                                                    },
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
                                            "Pending Memory Proposals",
                                            style={
                                                "cursor": "pointer",
                                                "fontWeight": 600,
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "The assistant can suggest memory updates, but nothing persists until you explicitly apply it.",
                                                    style={
                                                        "color": COLOR_MUTED,
                                                        "fontSize": "13px",
                                                        "margin": "12px 0",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="ai-memory-proposal-selector",
                                                    options=self._memory_proposal_options(
                                                        self._initial_memory_proposals()
                                                    ),
                                                    value=(
                                                        self._memory_proposal_options(
                                                            self._initial_memory_proposals()
                                                        )[0]["value"]
                                                        if self._memory_proposal_options(
                                                            self._initial_memory_proposals()
                                                        )
                                                        else None
                                                    ),
                                                    placeholder="No pending memory proposals",
                                                    clearable=False,
                                                ),
                                                html.Div(
                                                    "Field: none",
                                                    id="ai-memory-proposal-field",
                                                    style={
                                                        "fontSize": "13px",
                                                        "fontWeight": 600,
                                                        "marginTop": "12px",
                                                    },
                                                ),
                                                html.Div(
                                                    "Rationale: none",
                                                    id="ai-memory-proposal-rationale",
                                                    style={
                                                        "fontSize": "12px",
                                                        "color": COLOR_MUTED,
                                                        "marginTop": "8px",
                                                        "whiteSpace": "pre-wrap",
                                                    },
                                                ),
                                                html.Div(
                                                    "Evidence: none",
                                                    id="ai-memory-proposal-evidence",
                                                    style={
                                                        "fontSize": "12px",
                                                        "color": COLOR_MUTED,
                                                        "marginTop": "6px",
                                                    },
                                                ),
                                                dcc.Textarea(
                                                    id="ai-memory-proposal-editor",
                                                    value="",
                                                    style=self._memory_textarea_style(
                                                        height="96px"
                                                    ),
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Apply Proposal",
                                                            id="ai-memory-apply-proposal",
                                                            n_clicks=0,
                                                            style=self._primary_button_style(),
                                                        ),
                                                        html.Button(
                                                            "Reject Proposal",
                                                            id="ai-memory-reject-proposal",
                                                            n_clicks=0,
                                                            style=self._secondary_button_style(),
                                                        ),
                                                    ],
                                                    style={
                                                        "display": "flex",
                                                        "gap": "10px",
                                                        "marginTop": "14px",
                                                        "flexWrap": "wrap",
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
                                                                            "name": "Status",
                                                                            "id": "status",
                                                                        },
                                                                        {
                                                                            "name": "Requested",
                                                                            "id": "requested",
                                                                        },
                                                                        {
                                                                            "name": "Effective",
                                                                            "id": "effective",
                                                                        },
                                                                        {
                                                                            "name": "Notes",
                                                                            "id": "notes",
                                                                        },
                                                                    ],
                                                                    data=self._execution_record_rows(
                                                                        self._initial_execution_records()
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
                                                                            "name": "Label",
                                                                            "id": "candidate_id",
                                                                        },
                                                                        {
                                                                            "name": "Basis Key",
                                                                            "id": "basis_key",
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

    def _initial_execution_records(self) -> list[dict[str, Any]]:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if session is not None:
                return [dict(item) for item in session.execution_records]
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
                    session.accepted_analysis_basis
                    if isinstance(session.accepted_analysis_basis, dict)
                    else {}
                )
                return dict(basis)
        return {}

    def _initial_proposal_basis(self) -> dict[str, Any]:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if session is not None:
                proposal = (
                    session.proposal_basis
                    if isinstance(session.proposal_basis, dict)
                    else {}
                )
                return dict(proposal)
        return {}

    def _current_segment(self) -> str | None:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if (
                session is not None
                and isinstance(session.segment, str)
                and session.segment.strip()
            ):
                return session.segment.strip()
        if self._config is not None:
            segment = self._config.get_segment()
            if isinstance(segment, str) and segment.strip():
                return segment.strip()
        return None

    def _initial_segment_memory_payload(self) -> dict[str, Any]:
        memory = self._memory_authoring_service.load_for_segment(
            config=self._config,
            segment=self._current_segment(),
        )
        return self._memory_authoring_service.build_ui_payload(memory)

    def _initial_memory_proposals(self) -> list[dict[str, Any]]:
        if self._chat_service is not None and self._chat_id:
            session = self._chat_service.get_chat(self._chat_id)
            if session is not None and isinstance(
                session.working_memory.get("memory_update_proposals"), list
            ):
                return [
                    dict(item)
                    for item in session.working_memory.get(
                        "memory_update_proposals", []
                    )
                    if isinstance(item, dict)
                ]
        return []

    def _persist_memory_proposals(self, proposals: list[dict[str, Any]]) -> None:
        if self._chat_service is None or not self._chat_id:
            return
        self._chat_service.update_working_memory_fields(
            self._chat_id,
            fields={
                "memory_update_proposals": [
                    dict(item) for item in proposals if isinstance(item, dict)
                ]
            },
        )

    @staticmethod
    def _memory_payload_outputs(payload: Any, *, status: Any) -> tuple[Any, ...]:
        if payload is no_update:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                status,
            )
        if not isinstance(payload, dict):
            payload = {}
        return (
            payload,
            payload.get("segment_overview", ""),
            payload.get("known_issues_text", ""),
            payload.get("house_preferences_text", ""),
            payload.get("recent_quarter_notes_text", ""),
            payload.get("open_items_text", ""),
            AIDashboard._structured_preference_summary(
                payload.get("structured_house_preferences", [])
            ),
            payload.get("memory_change_log", []),
            status,
        )

    @staticmethod
    def _structured_preference_summary(preferences: list[dict[str, Any]]) -> str:
        if not isinstance(preferences, list) or not preferences:
            return "No structured house preferences stored."
        rows = []
        for item in preferences:
            if not isinstance(item, dict):
                continue
            pref_type = str(item.get("type", "")).strip()
            value = item.get("value")
            if pref_type:
                rows.append(
                    f"Structured preferences preserved on save: {pref_type}={value}"
                )
        return "\n".join(rows) if rows else "No structured house preferences stored."

    @staticmethod
    def _memory_proposal_options(
        proposals: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for item in proposals:
            if not isinstance(item, dict):
                continue
            proposal_id = str(item.get("proposal_id", "")).strip()
            field = str(item.get("field", "")).strip() or "memory"
            if not proposal_id:
                continue
            options.append({"label": f"{field}: {proposal_id}", "value": proposal_id})
        return options

    @staticmethod
    def _proposal_by_id(
        proposals: list[dict[str, Any]],
        proposal_id: object,
    ) -> dict[str, Any]:
        target = str(proposal_id or "").strip()
        for item in proposals:
            if str(item.get("proposal_id", "")).strip() == target:
                return dict(item)
        return {}

    @staticmethod
    def _proposal_field_label(proposal: dict[str, Any]) -> str:
        if not proposal:
            return "Field: none"
        return f"Field: {proposal.get('field')} ({proposal.get('operation')})"

    @staticmethod
    def _proposal_rationale_label(proposal: dict[str, Any]) -> str:
        if not proposal:
            return "Rationale: none"
        rationale = (
            str(proposal.get("rationale", "")).strip() or "No rationale provided."
        )
        return f"Rationale: {rationale}"

    @staticmethod
    def _proposal_evidence_label(proposal: dict[str, Any]) -> str:
        if not proposal:
            return "Evidence: none"
        evidence_ids = (
            proposal.get("evidence_ids")
            if isinstance(proposal.get("evidence_ids"), list)
            else []
        )
        return (
            "Evidence: "
            + ", ".join(str(item) for item in evidence_ids if str(item).strip())
            if evidence_ids
            else "Evidence: none"
        )

    @staticmethod
    def _proposal_editable_value(proposal: dict[str, Any]) -> str:
        if not proposal:
            return ""
        value = proposal.get("value")
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                return "\n".join(
                    f"{item.get('period', '')} | {item.get('note', '')}".strip()
                    for item in value
                    if isinstance(item, dict)
                )
            return "\n".join(str(item) for item in value if str(item).strip())
        return str(value or "")

    @staticmethod
    def _coerce_edited_proposal_value(
        proposal: dict[str, Any], edited_value: Any
    ) -> Any:
        field = str(proposal.get("field", "")).strip()
        if field in {"known_issues", "open_items"}:
            return [
                line.strip()
                for line in str(edited_value or "").splitlines()
                if line.strip()
            ]
        if field == "recent_quarter_notes":
            return MemoryAuthoringService._parse_recent_quarter_notes(edited_value)
        return str(edited_value or "").strip()

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

    @staticmethod
    def _proposal_action_from_trigger(
        *,
        triggered_id: object,
        accept_clicks: object,
        reject_clicks: object,
    ) -> str:
        trigger = str(triggered_id or "").strip()
        if trigger == "ai-chat-proposal-accept":
            try:
                return "accept" if int(accept_clicks or 0) > 0 else ""
            except (TypeError, ValueError):
                return ""
        if trigger == "ai-chat-proposal-reject":
            try:
                return "reject" if int(reject_clicks or 0) > 0 else ""
            except (TypeError, ValueError):
                return ""
        return ""

    def _render_chat_messages(
        self,
        history: list[dict[str, Any]],
        proposal_basis: dict[str, Any] | None = None,
    ) -> list:
        items = [item for item in history if isinstance(item, dict)]
        rendered: list = []
        if not items:
            return rendered
        current_proposal = dict(proposal_basis) if isinstance(proposal_basis, dict) else {}

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
            message_id = str(item.get("message_id") or "").strip()
            embedded_proposal = (
                dict(item.get("proposal_basis"))
                if isinstance(item.get("proposal_basis"), dict)
                else {}
            )
            active_proposal = embedded_proposal
            if (
                not active_proposal
                and current_proposal
                and message_id
                and message_id
                == str(current_proposal.get("presented_in_message_id") or "").strip()
            ):
                active_proposal = current_proposal
            proposal_component = self._render_message_proposal(
                active_proposal,
                current_proposal=current_proposal,
            )
            message_children = [
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
            ]
            if proposal_component is not None:
                message_children.append(proposal_component)
            rendered.append(
                html.Div(
                    message_children,
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "10px",
                        "alignSelf": "flex-end" if is_user else "flex-start",
                        "maxWidth": "78%",
                    },
                )
            )
        return rendered

    @staticmethod
    def _proposal_parameter_summary(proposal: dict[str, Any]) -> str:
        parameters = (
            proposal.get("parameters")
            if isinstance(proposal.get("parameters"), dict)
            else {}
        )
        tail = parameters.get("tail") if isinstance(parameters.get("tail"), dict) else {}
        drops = parameters.get("drop") if isinstance(parameters.get("drop"), list) else []
        return (
            f"Average: {parameters.get('average', '')} | "
            f"Drops: {len(drops)} | "
            f"Tail: {tail.get('curve', '')} @ {tail.get('attachment_age', 'n/a')}"
        )

    @classmethod
    def _render_message_proposal(
        cls,
        proposal: dict[str, Any],
        *,
        current_proposal: dict[str, Any],
    ):
        if not isinstance(proposal, dict) or not proposal:
            return None
        if str(proposal.get("status") or "").strip() != "pending":
            return None
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        current_id = str(current_proposal.get("proposal_id") or "").strip()
        if not proposal_id or proposal_id != current_id:
            return None
        scenario_label = str(
            proposal.get("scenario_label")
            or proposal.get("candidate_id")
            or proposal.get("scenario_id")
            or "proposed basis"
        )
        caveats = (
            [str(item) for item in proposal.get("caveats", []) if str(item).strip()][:3]
            if isinstance(proposal.get("caveats"), list)
            else []
        )
        children: list[Any] = [
            html.Div(
                "Recommended Change Pending Acceptance",
                style={"fontSize": "12px", "fontWeight": 700, "color": COLOR_MUTED},
            ),
            html.Div(
                scenario_label,
                style={"fontSize": "16px", "fontWeight": 700, "color": COLOR_TEXT},
            ),
            html.Div(
                f"Strength: {proposal.get('recommendation_strength') or 'recommended'}",
                style={"fontSize": "13px", "color": COLOR_MUTED},
            ),
            html.Div(
                cls._proposal_parameter_summary(proposal),
                style={"fontSize": "13px", "color": COLOR_TEXT},
            ),
        ]
        if caveats:
            children.append(
                html.Div(
                    "Caveats: " + "; ".join(caveats),
                    style={"fontSize": "12px", "color": COLOR_MUTED, "lineHeight": "1.5"},
                )
            )
        children.append(
            html.Div(
                [
                    html.Button(
                        "Yes",
                        id="ai-chat-proposal-accept",
                        n_clicks=0,
                        style=cls._primary_button_style(),
                    ),
                    html.Button(
                        "No",
                        id="ai-chat-proposal-reject",
                        n_clicks=0,
                        style=cls._secondary_button_style(),
                    ),
                ],
                style={"display": "flex", "gap": "10px", "marginTop": "4px"},
            )
        )
        return html.Div(
            children,
            style={
                "background": COLOR_ACCENT_SOFT,
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": RADIUS_MD,
                "padding": "12px 14px",
                "boxShadow": SHADOW_SOFT,
            },
        )

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
    def _execution_record_rows(
        execution_records: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in reversed(execution_records[-12:]):
            if not isinstance(item, dict):
                continue
            requested_inputs = (
                item.get("requested_inputs")
                if isinstance(item.get("requested_inputs"), dict)
                else {}
            )
            effective_inputs = (
                item.get("effective_inputs")
                if isinstance(item.get("effective_inputs"), dict)
                else {}
            )
            warnings = (
                [str(entry) for entry in item.get("warnings", []) if str(entry).strip()]
                if isinstance(item.get("warnings"), list)
                else []
            )
            requested_focus = (
                requested_inputs.get("scenario_id")
                or requested_inputs.get("uwy")
                or requested_inputs.get("segment")
                or requested_inputs.get("session_id")
                or ""
            )
            effective_focus = (
                effective_inputs.get("scenario_id")
                or effective_inputs.get("uwy")
                or effective_inputs.get("segment")
                or effective_inputs.get("session_id")
                or ""
            )
            rows.append(
                {
                    "tool": str(item.get("tool_name") or item.get("workflow_name") or ""),
                    "status": str(item.get("execution_status") or ""),
                    "requested": str(requested_focus),
                    "effective": str(effective_focus),
                    "notes": "; ".join(warnings[:3]) or "executed without adjustments",
                }
            )
        if rows:
            return rows
        return [
            {
                "tool": "No execution records yet",
                "status": "",
                "requested": "",
                "effective": "",
                "notes": "Material tool runs will appear here with requested and effective inputs, execution status, and adjustments.",
            }
        ]

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
        rows: list[dict[str, Any]] = []
        for item in scenario_ledger:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["candidate_id"] = str(
                row.get("candidate_id") or row.get("scenario_id") or ""
            ).strip()
            row["basis_key"] = str(
                row.get("basis_key") or row.get("scenario_key") or row.get("scenario_id") or ""
            ).strip()
            rows.append(row)
        if rows:
            return rows
        return [
            {
                "candidate_id": "No chat scenarios tested yet",
                "basis_key": "",
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
                    "field": "Meaning",
                    "value": "No accepted Analysis Basis is set for this chat yet. Recommendations can appear in chat, but the basis shown here changes only after an explicit acceptance action.",
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
        tail_active = tail.get("attachment_age") is not None
        tail_mode = "attached" if tail_active else "reference_fit_only"
        basis_type = str(analysis_basis.get("basis_type") or "").strip().lower()
        basis_key = str(analysis_basis.get("basis_key") or "").strip()
        scenario_id = str(analysis_basis.get("scenario_id") or "").strip()
        scenario_label = str(analysis_basis.get("scenario_label") or "").strip()
        candidate_id = str(analysis_basis.get("candidate_id") or "").strip()
        if not scenario_label and scenario_id:
            scenario_label = candidate_id or scenario_id
        elif basis_type == "baseline":
            scenario_label = "active baseline session"
        elif basis_type == "bespoke":
            scenario_label = "custom parameter basis"
        elif not scenario_label:
            scenario_label = "unnamed scenario basis"
        rows = [
            {
                "field": "Meaning",
                "value": "This is the currently accepted reasoning basis for this chat. Future basis-aware analysis uses this accepted basis until you explicitly switch basis or accept a different proposed basis.",
            },
            {
                "field": "Used For Future Analysis",
                "value": "yes, this accepted basis is what the AI will use for future basis-aware analysis",
            },
            {"field": "Basis Type", "value": str(analysis_basis.get("basis_type", ""))},
            {
                "field": "Scenario Label",
                "value": scenario_label,
            },
            {
                "field": "Basis Key",
                "value": basis_key,
            },
            {
                "field": "Scenario ID",
                "value": scenario_id,
            },
            {
                "field": "Matches Active Session",
                "value": "yes, this accepted basis matches the current active session"
                if bool(analysis_basis.get("is_active_session"))
                else "no, this accepted basis differs from the current active session",
            },
            {
                "field": "Source Tool",
                "value": str(analysis_basis.get("source_tool") or ""),
            },
            {"field": "Average", "value": str(parameters.get("average", ""))},
            {"field": "Drops", "value": str(parameters.get("drop", []))},
            {"field": "Tail Curve", "value": str(tail.get("curve", ""))},
            {
                "field": "Tail Active",
                "value": "yes" if tail_active else "no",
            },
            {"field": "Tail Mode", "value": tail_mode},
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
    def _memory_textarea_style(*, height: str) -> dict[str, str]:
        return {
            "width": "100%",
            "minHeight": height,
            "border": f"1px solid {COLOR_BORDER}",
            "borderRadius": RADIUS_MD,
            "padding": "10px 12px",
            "fontFamily": FONT_FAMILY,
            "fontSize": "13px",
            "resize": "vertical",
            "boxSizing": "border-box",
            "marginTop": "6px",
        }

    @staticmethod
    def _memory_field_label(
        text: str,
        tooltip: str,
        *,
        margin_top: str | None = None,
    ):
        style = {
            "fontWeight": 600,
            "fontSize": "13px",
            "display": "flex",
            "alignItems": "center",
            "gap": "6px",
        }
        if margin_top:
            style["marginTop"] = margin_top
        return html.Div(
            [
                html.Span(text),
                html.Span(
                    "?",
                    title=tooltip,
                    style={
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "width": "16px",
                        "height": "16px",
                        "borderRadius": "999px",
                        "border": f"1px solid {COLOR_BORDER}",
                        "background": COLOR_ACCENT_SOFT,
                        "color": COLOR_MUTED,
                        "fontSize": "11px",
                        "cursor": "help",
                        "lineHeight": "1",
                    },
                ),
            ],
            style=style,
        )

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
