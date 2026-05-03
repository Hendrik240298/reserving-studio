from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

from ai.api_tools import ReservingApiTools
from ai.backend_tools import BackendReservingTools
from ai.basis_manager import BasisManager
from ai.context_loader import load_segment_note
from ai.control_plane_types import (
    basis_key_from_parameters,
    normalize_accepted_analysis_basis,
    normalize_preview_basis,
    normalize_proposal_basis,
)
from ai.deterministic_packet import build_deterministic_packet
from ai.execution_records import execution_status_is_successful, latest_execution_record
from ai.memory_store import SegmentMemoryStore
from ai.narration import (
    NarrationAssembler,
    build_narration_prompt,
    render_narration_fallback,
)
from ai.openrouter_client import OpenRouterClient
from ai.plan_models import ExecutionPlan, PlanStep
from ai.planner import PlaybookPlanner
from ai.proposal_manager import ProposalManager
from ai.recommendation_policy import RecommendationPolicy
from ai.reviewer import ReviewerGate
from ai.tool_payloads import (
    build_analysis_basis,
    build_baseline_analysis_basis,
    build_memory_snapshot,
    build_tool_specs,
    compact_tool_result_for_model,
    render_memory_hint,
)
from ai.tool_contract import normalize_tool_result
from ai.workflow_definitions import (
    select_workflow_definitions,
    select_workflow_name,
)
from source.services.memory_authoring_service import MemoryAuthoringService
from source.services.segment_memory_service import SegmentMemoryService


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "off"}


def _load_prompt_file(filename: str) -> str:
    prompt_path = Path(__file__).resolve().parents[1] / filename
    if not prompt_path.exists():
        return ""
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


SYSTEM_PROMPT = (
    "You are Turtuary, an actuarial reserving diagnostics assistant. "
    "You are calm, helpful, and a little personable without being gimmicky. "
    "You can introduce yourself as Turtuary when it is natural, but keep answers professionally useful for actuaries. "
    "Always prefer tool calls over assumptions. "
    "Use compact summary tools first, then use detail tools only when specific evidence is needed. "
    "For data-view tools, only use these metric names: incurred, paid, outstanding, premium. "
    "If the user says 'claims' without a modifier, interpret that as incurred. "
    "Map common user phrases onto those exact names: 'incurred claims' -> incurred, 'paid claims' -> paid, 'outstanding claims' -> outstanding, 'earned premium' or 'gross written premium' -> premium. "
    "For data-view tools, only use these view names: cumulative or incremental. "
    "For tail curve methods, only use these supported method names: exponential, inverse_power, weibull. Treat 'power' or 'power_curve' as inverse_power. "
    "If the user specifies a notation preference, terminology preference, or unit preference, follow it consistently for the rest of the conversation unless they change it again. "
    "If the user asks about movements 'this quarter' or 'current quarter', interpret that as the latest valuation period / latest diagonal. "
    "For questions about claims movements this quarter, inspect incurred incremental latest-diagonal movement first, not premium first. "
    "When recommending tail settings, proactively check for late selected LDFs below 1.0 and for a sharp drop between the selected LDF just before attachment and the first fitted tail LDF. "
    "Do not wait for the user to point those issues out. If either issue appears, revise the tail recommendation and explain the issue plainly. "
    "Prefer earlier attachment to smooth late around-1.0 fluctuation, but avoid recommendations where the first tail factor creates a material cut versus the previous selected LDF. "
    "When the user asks which drops were used or why a drop was used, load exact scenario or derived-drop detail first. "
    "When an earlier answer recommended a tested scenario, keep exact numeric follow-up answers bound to that same scenario unless the user explicitly switches back to baseline or current session. "
    "When the conversation has a locked Analysis Basis and a tool supports basis fields, pass that basis into the tool call unless the user explicitly switches basis. "
    "Every exact numeric answer must start by stating the basis used. "
    "Only assign a drop reason if the tool output gives explicit support for that exact AY/development pair. Otherwise say the exact driver is not confirmed from current evidence. "
    "Do not relabel a drop as 'below 1.0', 'negative development', 'high outlier', or similar unless that label is directly supported by the tool output for that same drop. "
    "Ground all material statements in tool outputs. "
    "Decide for yourself whether scenario iteration is needed. "
    "Use scenario iteration when the user is asking for recommendations, best alternatives, scenario comparisons, or changes to drops, tail fitting, BF apriori, or final selection. "
    "Do not use scenario iteration for simple observational questions unless it materially helps answer the user. "
    "Do not present a scenario, parameter set, or tail setting as a recommendation unless it has been tested in the tools during the current conversation or is explicitly reported as untested. "
    "Explain statistical jargon like z-scores in plain English when you use it. "
    "If you cite an evidence ID, explain what that evidence refers to and only cite IDs that are present in the tool results you saw. "
    "Include uncertainty interpretation when available. "
    "If evidence is missing, say so explicitly."
)

AI_CONTEXT_PROMPT = _load_prompt_file("AI_CONTEXT.md")
AI_PLAYBOOKS_PROMPT = _load_prompt_file("AI_PLAYBOOKS.md")
AI_EXAMPLES_PROMPT = _load_prompt_file("AI_EXAMPLES.md")
AI_POLICY_PROMPT = _load_prompt_file("AI_POLICY.md")
USE_FULL_REFERENCE_PROMPTS = _env_flag("AI_FULL_REFERENCE_PROMPTS")

COMPACT_CONTEXT_PROMPT = (
    "Reference context. "
    "'Current quarter' and 'this quarter' mean the latest valuation period / latest diagonal. "
    "Development ages are month-based labels, so month 45 means 45 months of development, not 45 years. "
    "If the user says claims without a modifier, treat that as incurred. "
    "For data-view tools use only metric names incurred, paid, outstanding, premium and only view names cumulative or incremental. "
    "Distinguish observed a2a factors from the selected LDF vector. "
    "For monotone selected LDF questions, default to monotone decay: LDF_i >= LDF_{i+1}. "
    "Use the existing Reserving workflow only. Supported tail methods are exponential, inverse_power, and weibull; map power and power_curve to inverse_power. "
    "Analysis Basis carries forward for basis-aware tools unless the user explicitly asks for baseline or current session. "
    "For exact numeric factor, vector, or table questions, load exact assumption detail first and answer only from that payload."
)

COMPACT_PLAYBOOKS_PROMPT = (
    "Playbook reference. "
    "Movement Review: for claims movement this quarter, inspect incremental incurred latest-diagonal movement first, add incurred-on-premium or LDF consistency only as supporting context, and do not lead with recommendations. "
    "Scenario Recommendation: run diagnostics then scenario iteration before recommending changes. "
    "Drop Review, Tail Selection, BF Suitability, Data Anomaly Triage, and Quarter-Close: prefer the composite review tool first and use drilldowns only for follow-up evidence. "
    "Reserve Change Explanation: use reserve-change attribution against the current analysis basis unless baseline is requested. "
    "Data Exploration: start with summary or comparison tools before detailed rows."
)

COMPACT_POLICY_PROMPT = (
    "Policy reference. "
    "Keep recommendations tool-backed and tested, distinguish observation from inference and judgment, and weaken or hold recommendations when evidence is incomplete or governance/pause flags remain unresolved. "
    "Keep memory compact and structured."
)

RECENT_HISTORY_LIMIT = 6


class AssistantService:
    def __init__(self, *, tool_executor: Any) -> None:
        self._client = OpenRouterClient()
        self._tools = tool_executor
        self._planner = PlaybookPlanner()
        self._reviewer = ReviewerGate()
        self._recommendation_policy = RecommendationPolicy()
        self._segment_memory_store = SegmentMemoryStore()
        self._memory_authoring = MemoryAuthoringService()
        self._deterministic_orchestration_enabled = os.environ.get(
            "AI_DETERMINISTIC_ORCHESTRATION", "1"
        ).strip().lower() not in {"0", "false", "off"}
        self._observability_enabled = os.environ.get(
            "AI_OBSERVABILITY", "1"
        ).strip().lower() not in {"0", "false", "off"}

    @classmethod
    def from_api_base_url(cls, *, api_base_url: str) -> "AssistantService":
        return cls(tool_executor=ReservingApiTools(base_url=api_base_url))

    @classmethod
    def from_backend(cls, *, backend: Any) -> "AssistantService":
        return cls(
            tool_executor=BackendReservingTools(
                backend=backend,
                tool_specs=build_tool_specs(),
            )
        )

    def bootstrap_workflow(
        self,
        *,
        segment: str,
        claims_rows: list[dict[str, Any]],
        premium_rows: list[dict[str, Any]],
        granularity: str | None = None,
    ) -> dict[str, Any]:
        return self._tools.create_workflow(
            segment=segment,
            claims_rows=claims_rows,
            premium_rows=premium_rows,
            granularity=granularity,
        )

    def answer(self, *, user_prompt: str, max_steps: int = 14) -> str:
        return self.run_turn(user_prompt=user_prompt, max_steps=max_steps)["content"]

    def run_turn(
        self,
        *,
        user_prompt: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_context: dict[str, Any] | None = None,
        accepted_analysis_basis: dict[str, Any] | None = None,
        proposal_basis: dict[str, Any] | None = None,
        preview_basis: dict[str, Any] | None = None,
        execution_records: list[dict[str, Any]] | None = None,
        basis_transition_history: list[dict[str, Any]] | None = None,
        working_memory: dict[str, Any] | None = None,
        event_callback: Any | None = None,
        max_steps: int = 14,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for prompt_text in self._reference_prompts_for_turn(user_prompt):
            messages.append({"role": "system", "content": prompt_text})
        intent_hint = self._build_intent_hint(user_prompt)
        if intent_hint:
            messages.append({"role": "system", "content": intent_hint})
        playbook_hint = self._build_playbook_hint(user_prompt)
        if playbook_hint:
            messages.append({"role": "system", "content": playbook_hint})
        session_hint = self._build_session_context_hint(session_context)
        if session_hint:
            messages.append({"role": "system", "content": session_hint})
        memory_hint = render_memory_hint(working_memory)
        if memory_hint:
            messages.append({"role": "system", "content": memory_hint})
        recent_history = (conversation_history or [])[-RECENT_HISTORY_LIMIT:]
        memory_authoring = (
            getattr(self, "_memory_authoring", None) or MemoryAuthoringService()
        )
        segment_memory = self._load_segment_memory(session_context)
        segment_memory_hint = memory_authoring.render_context_text(segment_memory)
        if segment_memory_hint:
            messages.append({"role": "system", "content": segment_memory_hint})
        segment_note = self._load_segment_note(session_context)
        if segment_note:
            messages.append({"role": "system", "content": segment_note})
        for item in recent_history:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        tool_outputs: dict[str, dict[str, Any]] = {}
        tool_events: list[dict[str, Any]] = []
        current_accepted_basis = BasisManager.accepted_basis(
            accepted_analysis_basis=accepted_analysis_basis,
            legacy_working_memory=working_memory if isinstance(working_memory, dict) else None,
        )
        current_proposal_basis = normalize_proposal_basis(
            proposal_basis
            if proposal_basis is not None
            else (working_memory or {}).get("proposal_basis")
        )
        current_preview_basis = normalize_preview_basis(
            preview_basis
            if preview_basis is not None
            else (working_memory or {}).get("preview_basis")
        )
        memory_state = build_memory_snapshot(
            session_summary=(working_memory or {}).get("session_summary")
            if isinstance(working_memory, dict)
            else None,
            diagnostics_summary=(working_memory or {}).get("diagnostics_summary")
            if isinstance(working_memory, dict)
            else None,
            iteration_summary=(working_memory or {}).get("iteration_summary")
            if isinstance(working_memory, dict)
            else None,
            results_summary=(working_memory or {}).get("results_summary")
            if isinstance(working_memory, dict)
            else None,
            data_view_summary=(working_memory or {}).get("data_view_summary")
            if isinstance(working_memory, dict)
            else None,
            movement_summary=(working_memory or {}).get("movement_summary")
            if isinstance(working_memory, dict)
            else None,
            reserve_change_summary=(working_memory or {}).get("reserve_change_summary")
            if isinstance(working_memory, dict)
            else None,
            review_summary=(working_memory or {}).get("review_summary")
            if isinstance(working_memory, dict)
            else None,
            existing_scenario_ledger=(working_memory or {}).get("scenario_ledger")
            if isinstance(working_memory, dict)
            else None,
            existing_accepted_analysis_basis=current_accepted_basis,
            existing_scenario_basis_cache=(working_memory or {}).get(
                "scenario_basis_cache"
            )
            if isinstance(working_memory, dict)
            else None,
        )
        memory_state["proposal_basis"] = current_proposal_basis
        memory_state["preview_basis"] = current_preview_basis
        memory_state["execution_records"] = (
            [dict(item) for item in execution_records if isinstance(item, dict)]
            if isinstance(execution_records, list)
            else []
        )
        memory_state["basis_transition_history"] = (
            [dict(item) for item in basis_transition_history if isinstance(item, dict)]
            if isinstance(basis_transition_history, list)
            else []
        )
        pending_proposal_response = self._pending_proposal_acceptance_response(
            user_prompt=user_prompt,
            proposal_basis=current_proposal_basis,
            accepted_analysis_basis=current_accepted_basis,
        )
        if pending_proposal_response:
            self._emit_streamed_content(event_callback, pending_proposal_response)
            return {
                "content": pending_proposal_response,
                "fallback_used": False,
                "session_id": session_context.get("session_id")
                if isinstance(session_context, dict)
                else None,
                "tool_events": tool_events,
                "memory_snapshot": memory_state,
                "deterministic_packet": memory_state.get("deterministic_packet", {}),
                "narration_packet": memory_state.get("narration_packet", {}),
                "memory_update_proposals": memory_state.get(
                    "memory_update_proposals", []
                ),
            }
        tool_specs = self._tools.tool_specs
        known_tool_names = self._tool_names(tool_specs)
        guardrail_state: dict[str, bool] = {
            "portfolio_shift_unconfirmed": False,
            "paid_incurred_conflict": False,
            "low_confidence": False,
            "tail_instability": False,
            "high_process_uncertainty": False,
        }
        workflow_state: dict[str, Any] = {
            "session_id": None,
            "ran_diagnostics": False,
            "ran_iteration": False,
            "exact_data_required": False,
            "exact_data_loaded": False,
            "current_user_prompt": user_prompt,
        }
        if isinstance(working_memory, dict) and isinstance(
            working_memory.get("memory_update_proposals"), list
        ):
            memory_state["memory_update_proposals"] = [
                dict(item)
                for item in working_memory.get("memory_update_proposals", [])
                if isinstance(item, dict)
            ]
        self._prime_context_for_prompt(
            user_prompt=user_prompt,
            session_context=session_context,
            messages=messages,
            tool_outputs=tool_outputs,
            tool_events=tool_events,
            memory_state=memory_state,
            workflow_state=workflow_state,
            event_callback=event_callback,
        )
        memory_state["segment_memory"] = segment_memory
        deterministic_packet = self._run_deterministic_orchestration(
            user_prompt=user_prompt,
            session_context=session_context,
            tool_outputs=tool_outputs,
            tool_events=tool_events,
            memory_state=memory_state,
            guardrail_state=guardrail_state,
            workflow_state=workflow_state,
            event_callback=event_callback,
            segment_memory=segment_memory,
        )
        if deterministic_packet:
            reviewer = getattr(self, "_reviewer", None) or ReviewerGate()
            validated_proposals = reviewer.validate_memory_update_proposals(
                memory_authoring.propose_updates_for_turn(
                    user_prompt=user_prompt,
                    deterministic_packet=deterministic_packet,
                ),
                evidence_packets=deterministic_packet.get("evidence_packets", []),
            )
            deterministic_packet["memory_update_proposals"] = validated_proposals
            memory_state["memory_update_proposals"] = validated_proposals
            proposal = ProposalManager.build_from_deterministic_packet(
                deterministic_packet=deterministic_packet,
                accepted_analysis_basis=memory_state.get("accepted_analysis_basis"),
                basis_cache=memory_state.get("scenario_basis_cache"),
            )
            if proposal:
                memory_state["proposal_basis"] = proposal
            deterministic_packet["accepted_analysis_basis"] = dict(
                memory_state.get("accepted_analysis_basis", {})
            )
            deterministic_packet["proposal_basis"] = dict(
                memory_state.get("proposal_basis", {})
            )
            presentation = (
                deterministic_packet.get("presentation")
                if isinstance(deterministic_packet.get("presentation"), dict)
                else {}
            )
            presentation["basis_confirmation_line"] = BasisManager.label(
                memory_state.get("accepted_analysis_basis")
            )
            if proposal:
                presentation["proposal_confirmation_line"] = (
                    "Recommended change pending acceptance: "
                    f"{proposal.get('scenario_label') or proposal.get('scenario_id') or 'proposed basis'}."
                )
            deterministic_packet["presentation"] = presentation
            narration_packet = NarrationAssembler.build(
                deterministic_packet=deterministic_packet,
                accepted_analysis_basis=memory_state.get("accepted_analysis_basis"),
                proposal_basis=memory_state.get("proposal_basis"),
                execution_records=memory_state.get("execution_records"),
                guardrail_state=guardrail_state,
            )
            deterministic_packet["narration_packet"] = narration_packet
            memory_state["narration_packet"] = narration_packet
            messages.append(
                {
                    "role": "system",
                    "content": self._build_narration_prompt(narration_packet),
                }
            )
            memory_state["deterministic_packet"] = deterministic_packet
            self._persist_segment_memory(
                session_context=session_context,
                segment_memory=segment_memory,
                memory_state=memory_state,
                deterministic_packet=deterministic_packet,
            )

        for _ in range(max_steps):
            available_tool_specs = self._filter_tool_specs_for_turn(
                tool_specs=tool_specs,
                deterministic_packet=memory_state.get("deterministic_packet", {}),
                user_prompt=str(workflow_state.get("current_user_prompt") or ""),
                exact_data_required=bool(workflow_state.get("exact_data_required")),
            )
            available_tool_names = self._tool_names(available_tool_specs)
            if self._observability_enabled:
                tool_schema_chars = 0
                if available_tool_specs:
                    tool_schema_chars = len(
                        json.dumps(available_tool_specs, ensure_ascii=True)
                    )
                prompt_chars = sum(
                    len(str(item.get("content", "")))
                    for item in messages
                    if isinstance(item, dict)
                )
                logger.info(
                    "[OBS] ai.step request_openrouter messages=%s tools=%s prompt_chars=%s tool_schema_chars=%s",
                    len(messages),
                    len(available_tool_specs),
                    prompt_chars,
                    tool_schema_chars,
                )
            try:
                self._emit_event(event_callback, "status", {"message": "Thinking"})
                response = self._client.chat_completion(
                    messages=messages,
                    tools=available_tool_specs,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=1200,
                )
            except RuntimeError as error:
                narration_packet = memory_state.get("narration_packet")
                if isinstance(narration_packet, dict) and narration_packet:
                    fallback = render_narration_fallback(narration_packet)
                    self._emit_streamed_content(event_callback, fallback)
                    return {
                        "content": fallback,
                        "fallback_used": True,
                        "fallback_reason": "provider_error_after_deterministic_packet",
                        "fallback_detail": str(error).strip(),
                        "session_id": workflow_state.get("session_id"),
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
                        "narration_packet": narration_packet,
                        "memory_update_proposals": memory_state.get(
                            "memory_update_proposals", []
                        ),
                    }
                if workflow_state.get("ran_diagnostics"):
                    session_id = workflow_state.get("session_id")
                    fallback = self._build_deterministic_fallback_commentary(
                        session_id=session_id if isinstance(session_id, str) else None,
                        diagnostics=tool_outputs.get("tool_run_diagnostics_summary"),
                        iteration=tool_outputs.get("tool_iterate_diagnostics_summary"),
                        results=tool_outputs.get("tool_get_results_summary"),
                    )
                    if fallback:
                        self._emit_streamed_content(event_callback, fallback)
                        return {
                            "content": fallback,
                            "fallback_used": True,
                            "fallback_reason": "provider_error_after_tool_execution",
                            "fallback_detail": str(error).strip(),
                            "session_id": session_id,
                            "tool_events": tool_events,
                            "memory_snapshot": memory_state,
                            "deterministic_packet": memory_state.get(
                                "deterministic_packet", {}
                            ),
                            "narration_packet": memory_state.get(
                                "narration_packet", {}
                            ),
                            "memory_update_proposals": memory_state.get(
                                "memory_update_proposals", []
                            ),
                        }
                    suffix = (
                        f" session_id={session_id}"
                        if isinstance(session_id, str)
                        else ""
                    )
                    return {
                        "content": (
                            "Model provider is temporarily unavailable after deterministic tool execution. "
                            "Diagnostics and scenario evaluation completed successfully; "
                            "retry to generate narrative commentary." + suffix
                        ),
                        "fallback_used": True,
                        "fallback_reason": "provider_error_after_tool_execution",
                        "fallback_detail": str(error).strip(),
                        "session_id": session_id,
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
                        "narration_packet": memory_state.get("narration_packet", {}),
                        "memory_update_proposals": memory_state.get(
                            "memory_update_proposals", []
                        ),
                    }
                provider_error = str(error).strip()
                if provider_error:
                    provider_error = f" ({provider_error})"
                friendly = (
                    "The AI model provider timed out before it could start tool-backed analysis. "
                    "No reserving diagnostics or scenario tests were run. Please retry."
                )
                if "timed out" not in str(error).lower():
                    friendly = (
                        "The AI model provider is temporarily unavailable before analysis could start. "
                        "No reserving diagnostics or scenario tests were run. Please retry."
                    )
                self._emit_streamed_content(event_callback, friendly)
                return {
                    "content": friendly + provider_error,
                    "fallback_used": True,
                    "fallback_reason": "provider_error_before_analysis",
                    "fallback_detail": str(error).strip(),
                    "session_id": workflow_state.get("session_id"),
                    "tool_events": tool_events,
                    "memory_snapshot": memory_state,
                    "deterministic_packet": memory_state.get(
                        "deterministic_packet", {}
                    ),
                    "narration_packet": memory_state.get("narration_packet", {}),
                    "memory_update_proposals": memory_state.get(
                        "memory_update_proposals", []
                    ),
                }
                raise error
            choice = response["choices"][0]["message"]
            tool_calls = choice.get("tool_calls") or []
            if self._observability_enabled:
                logger.info("[OBS] ai.step tool_calls=%s", len(tool_calls))

            if not tool_calls:
                content = choice.get("content")
                if isinstance(content, str):
                    if not content.strip():
                        fallback_result = self._narration_fallback_result(
                            memory_state=memory_state,
                            workflow_state=workflow_state,
                            tool_events=tool_events,
                            event_callback=event_callback,
                            reason="empty_model_content",
                        )
                        if fallback_result is not None:
                            return fallback_result
                    self._emit_streamed_content(event_callback, content)
                    if self._should_block_for_missing_exact_data(workflow_state):
                        blocked = self._exact_data_guardrail_message()
                        return {
                            "content": blocked,
                            "fallback_used": False,
                            "session_id": workflow_state.get("session_id"),
                            "tool_events": tool_events,
                            "memory_snapshot": memory_state,
                            "deterministic_packet": memory_state.get(
                                "deterministic_packet", {}
                            ),
                            "narration_packet": memory_state.get(
                                "narration_packet", {}
                            ),
                            "memory_update_proposals": memory_state.get(
                                "memory_update_proposals", []
                            ),
                        }
                    return {
                        "content": self._apply_narrative_guardrails(
                            content,
                            guardrail_state,
                            memory_state.get("execution_records", []),
                            memory_state.get("narration_packet", {}),
                        ),
                        "fallback_used": False,
                        "session_id": workflow_state.get("session_id"),
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
                        "narration_packet": memory_state.get("narration_packet", {}),
                        "memory_update_proposals": memory_state.get(
                            "memory_update_proposals", []
                        ),
                    }
                if isinstance(content, list):
                    parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ]
                    merged = "\n".join(part for part in parts if part)
                    if not merged.strip():
                        fallback_result = self._narration_fallback_result(
                            memory_state=memory_state,
                            workflow_state=workflow_state,
                            tool_events=tool_events,
                            event_callback=event_callback,
                            reason="empty_model_content",
                        )
                        if fallback_result is not None:
                            return fallback_result
                    self._emit_streamed_content(event_callback, merged)
                    if self._should_block_for_missing_exact_data(workflow_state):
                        blocked = self._exact_data_guardrail_message()
                        return {
                            "content": blocked,
                            "fallback_used": False,
                            "session_id": workflow_state.get("session_id"),
                            "tool_events": tool_events,
                            "memory_snapshot": memory_state,
                            "deterministic_packet": memory_state.get(
                                "deterministic_packet", {}
                            ),
                            "narration_packet": memory_state.get(
                                "narration_packet", {}
                            ),
                            "memory_update_proposals": memory_state.get(
                                "memory_update_proposals", []
                            ),
                        }
                    return {
                        "content": self._apply_narrative_guardrails(
                            merged,
                            guardrail_state,
                            memory_state.get("execution_records", []),
                            memory_state.get("narration_packet", {}),
                        ),
                        "fallback_used": False,
                        "session_id": workflow_state.get("session_id"),
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
                        "narration_packet": memory_state.get("narration_packet", {}),
                        "memory_update_proposals": memory_state.get(
                            "memory_update_proposals", []
                        ),
                    }
                fallback_result = self._narration_fallback_result(
                    memory_state=memory_state,
                    workflow_state=workflow_state,
                    tool_events=tool_events,
                    event_callback=event_callback,
                    reason="non_text_model_content",
                )
                if fallback_result is not None:
                    return fallback_result
                return {
                    "content": "No response content was produced by the model.",
                    "fallback_used": False,
                    "session_id": workflow_state.get("session_id"),
                    "tool_events": tool_events,
                    "memory_snapshot": memory_state,
                    "deterministic_packet": memory_state.get(
                        "deterministic_packet", {}
                    ),
                    "narration_packet": memory_state.get("narration_packet", {}),
                    "memory_update_proposals": memory_state.get(
                        "memory_update_proposals", []
                    ),
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                function_name = call["function"]["name"]
                raw_args = call["function"].get("arguments", "{}")
                try:
                    args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except json.JSONDecodeError:
                    args = {}

                if self._observability_enabled:
                    logger.info(
                        "[OBS] ai.tool.call name=%s args=%s",
                        function_name,
                        self._short_json(args),
                    )
                if known_tool_names and (
                    function_name not in known_tool_names
                    or function_name not in available_tool_names
                ):
                    logger.warning(
                        "Model requested unavailable tool: %s", function_name
                    )
                    fallback_result = self._narration_fallback_result(
                        memory_state=memory_state,
                        workflow_state=workflow_state,
                        tool_events=tool_events,
                        event_callback=event_callback,
                        reason="unavailable_tool_requested",
                        detail=function_name,
                    )
                    if fallback_result is not None:
                        return fallback_result
                    friendly = (
                        "The assistant requested an unavailable analysis tool before it could complete the answer. "
                        "No unsupported tool was executed; please retry the request."
                    )
                    self._emit_streamed_content(event_callback, friendly)
                    return {
                        "content": friendly,
                        "fallback_used": True,
                        "fallback_reason": "unavailable_tool_requested",
                        "fallback_detail": function_name,
                        "session_id": workflow_state.get("session_id"),
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
                        "narration_packet": memory_state.get(
                            "narration_packet", {}
                        ),
                        "memory_update_proposals": memory_state.get(
                            "memory_update_proposals", []
                        ),
                    }
                self._emit_event(
                    event_callback,
                    "status",
                    {"message": self._tool_status_label(function_name, args)},
                )

                tool_result, memory_state = self._execute_tool_call(
                    function_name=function_name,
                    args=args,
                    tool_outputs=tool_outputs,
                    tool_events=tool_events,
                    memory_state=memory_state,
                    workflow_state=workflow_state,
                    guardrail_state=guardrail_state,
                    event_callback=event_callback,
                )
                if self._observability_enabled:
                    logger.info(
                        "[OBS] ai.tool.result name=%s summary=%s",
                        function_name,
                        self._summarize_tool_result(tool_result),
                    )
                model_tool_result = compact_tool_result_for_model(
                    tool_name=function_name,
                    result=tool_result,
                )
                model_tool_json = json.dumps(model_tool_result, ensure_ascii=True)
                if self._observability_enabled:
                    raw_tool_json = json.dumps(tool_result, ensure_ascii=True)
                    logger.info(
                        "[OBS] ai.tool.replay_payload name=%s raw_chars=%s compact_chars=%s",
                        function_name,
                        len(raw_tool_json),
                        len(model_tool_json),
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": function_name,
                        "content": model_tool_json,
                    }
                )

        return {
            "content": (
                "Tool-call step limit reached before the assistant produced a final answer. "
                "Please retry with a narrower question."
            ),
            "fallback_used": True,
            "fallback_reason": "tool_step_limit_reached",
            "session_id": workflow_state.get("session_id"),
            "tool_events": tool_events,
            "memory_snapshot": memory_state,
            "deterministic_packet": memory_state.get("deterministic_packet", {}),
            "narration_packet": memory_state.get("narration_packet", {}),
            "memory_update_proposals": memory_state.get("memory_update_proposals", []),
        }

    def _run_deterministic_orchestration(
        self,
        *,
        user_prompt: str,
        session_context: dict[str, Any] | None,
        tool_outputs: dict[str, dict[str, Any]],
        tool_events: list[dict[str, Any]],
        memory_state: dict[str, Any],
        guardrail_state: dict[str, bool],
        workflow_state: dict[str, Any],
        event_callback: Any | None,
        segment_memory: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not getattr(self, "_deterministic_orchestration_enabled", True):
            return None
        if bool(workflow_state.get("exact_data_required")):
            return None
        planner = getattr(self, "_planner", None) or PlaybookPlanner()
        prompt_text = str(user_prompt or "").strip().lower()
        planning_basis = self._basis_for_recommendation_turn(
            prompt=prompt_text,
            memory_state=memory_state,
            session_context=session_context,
        )
        combined_drop_context = self._build_prior_drop_combination_context(
            user_prompt=user_prompt,
            session_context=session_context,
            memory_state=memory_state,
        )
        bf_recalculation_context = self._build_bf_recalculation_context(
            user_prompt=user_prompt,
            session_context=session_context,
            memory_state=memory_state,
            planning_basis=planning_basis,
        )
        plan = (
            combined_drop_context.get("plan")
            if isinstance(combined_drop_context.get("plan"), ExecutionPlan)
            else None
        )
        if plan is None and isinstance(
            bf_recalculation_context.get("plan"), ExecutionPlan
        ):
            plan = bf_recalculation_context["plan"]
        if plan is None:
            plan = planner.plan(
                user_prompt=user_prompt,
                session_context=session_context,
                segment_memory=segment_memory,
                analysis_basis=planning_basis,
            )
        if plan is None:
            return None
        if self._observability_enabled:
            logger.info(
                "[OBS] deterministic.playbook.selected playbook=%s session_id=%s segment=%s steps=%s min_evidence=%s",
                plan.playbook,
                plan.session_id,
                plan.segment,
                len(plan.steps),
                plan.minimum_evidence_count,
            )

        self._emit_event(
            event_callback,
            "status",
            {"message": f"Running deterministic playbook: {plan.playbook}"},
        )
        envelopes: list[dict[str, Any]] = []
        for step in plan.steps:
            step_args = step.args
            if (
                bf_recalculation_context
                and step.evidence_key == "bf_recalculation"
            ):
                step_args = self._build_bf_recalculation_args(
                    context=bf_recalculation_context,
                    memory_state=memory_state,
                )
            if self._observability_enabled:
                logger.info(
                    "[OBS] deterministic.step.execute playbook=%s tool=%s evidence_key=%s",
                    plan.playbook,
                    step.tool_name,
                    step.evidence_key,
                )
            tool_result, updated_memory = self._execute_tool_call(
                function_name=step.tool_name,
                args=step_args,
                tool_outputs=tool_outputs,
                tool_events=tool_events,
                memory_state=memory_state,
                workflow_state=workflow_state,
                guardrail_state=guardrail_state,
                event_callback=event_callback,
            )
            if (
                combined_drop_context
                and step.evidence_key == "combined_drop_recalculation"
            ):
                tool_result = self._enrich_combined_drop_recalculation_result(
                    tool_result=tool_result,
                    context=combined_drop_context,
                )
                tool_outputs[step.tool_name] = tool_result
                if tool_events:
                    tool_events[-1]["result_summary"] = tool_result
                updated_memory = self._update_memory_state(
                    updated_memory,
                    function_name=step.tool_name,
                    tool_result=tool_result,
                )
            if bf_recalculation_context and step.evidence_key == "bf_recalculation":
                tool_result = self._enrich_bf_recalculation_result(
                    tool_result=tool_result,
                    context=bf_recalculation_context,
                    args=step_args,
                )
                tool_outputs[step.tool_name] = tool_result
                if tool_events:
                    tool_events[-1]["result_summary"] = tool_result
                updated_memory = self._update_memory_state(
                    updated_memory,
                    function_name=step.tool_name,
                    tool_result=tool_result,
                )
            memory_state.clear()
            memory_state.update(updated_memory)
            envelopes.append(
                normalize_tool_result(
                    tool_name=step.tool_name,
                    args=step_args,
                    result=tool_result,
                    segment=plan.segment,
                    evidence_key=step.evidence_key,
                )
            )

        reviewer = getattr(self, "_reviewer", None) or ReviewerGate()
        review_outcome = reviewer.review(plan=plan, evidence_packets=envelopes)
        if self._observability_enabled:
            logger.info(
                "[OBS] deterministic.review.completed playbook=%s status=%s collected=%s missing=%s issues=%s caveats=%s",
                plan.playbook,
                review_outcome.status,
                len(review_outcome.collected_evidence),
                len(review_outcome.missing_evidence),
                len(review_outcome.issues),
                len(review_outcome.caveats),
            )
        policy = getattr(self, "_recommendation_policy", None) or RecommendationPolicy()
        recommendation = policy.decide(
            review=review_outcome,
            evidence_packets=envelopes,
        )
        if self._observability_enabled:
            logger.info(
                "[OBS] deterministic.recommendation.completed playbook=%s status=%s basis_key=%s alternatives=%s",
                plan.playbook,
                recommendation.status,
                recommendation.recommended_basis_key,
                len(recommendation.alternative_basis_keys),
            )
        memory_authoring = (
            getattr(self, "_memory_authoring", None) or MemoryAuthoringService()
        )
        memory_update_proposals = reviewer.validate_memory_update_proposals(
            memory_authoring.propose_updates_for_turn(
                user_prompt=user_prompt,
                deterministic_packet={},
            ),
            evidence_packets=envelopes,
        )
        return build_deterministic_packet(
            plan=plan.to_dict(),
            review=review_outcome.to_dict(),
            recommendation=recommendation.to_dict(),
            evidence_packets=envelopes,
            memory_update_proposals=memory_update_proposals,
        )

    @staticmethod
    def _build_prior_drop_combination_context(
        *,
        user_prompt: str,
        session_context: dict[str, Any] | None,
        memory_state: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = str(user_prompt or "").strip().lower()
        if not AssistantService._prompt_requests_prior_drop_combination(prompt):
            return {}
        review_summary = (
            memory_state.get("review_summary")
            if isinstance(memory_state.get("review_summary"), dict)
            else {}
        )
        if str(review_summary.get("review_type") or "").strip() != "drop_review":
            return {}
        candidates = review_summary.get("top_candidates")
        if not isinstance(candidates, list) or not candidates:
            return {}
        candidate_count = AssistantService._requested_drop_candidate_count(
            prompt=prompt,
            available_count=len(candidates),
        )
        selected_candidates = [
            dict(item)
            for item in candidates[:candidate_count]
            if isinstance(item, dict)
        ]
        if not selected_candidates:
            return {}

        base = (
            review_summary.get("analysis_basis")
            if isinstance(review_summary.get("analysis_basis"), dict)
            else {}
        )
        base_parameters = (
            dict(base.get("parameters"))
            if isinstance(base.get("parameters"), dict)
            else {}
        )
        parameters = AssistantService._combine_drop_candidate_parameters(
            base_parameters=base_parameters,
            candidates=selected_candidates,
        )
        if not parameters.get("drop"):
            return {}
        session_id = ""
        segment = None
        if isinstance(session_context, dict):
            session_id = str(session_context.get("session_id") or "").strip()
            segment = session_context.get("segment")
        if not session_id:
            session_id = str(review_summary.get("session_id") or "").strip()
        if not session_id:
            return {}

        candidate_id = f"combined_top_{len(selected_candidates)}_drop_candidates"
        args = {
            "session_id": session_id,
            **parameters,
        }
        plan = ExecutionPlan(
            playbook="prior_drop_combination",
            workflow_name="prior_drop_combination",
            goal="Recalculate the requested combination of prior drop-review candidates.",
            segment=str(segment).strip() if isinstance(segment, str) and segment.strip() else None,
            session_id=session_id,
            intent_class="scenario_recommendation",
            required_capabilities=["recalculate"],
            steps=[
                PlanStep(
                    tool_name="tool_recalculate",
                    args=args,
                    evidence_key="combined_drop_recalculation",
                    basis_aware=True,
                )
            ],
            required_evidence=["combined_drop_recalculation"],
            minimum_evidence_count=1,
            stopping_rule="Stop after the combined drop candidate basis is recalculated.",
            basis_behavior=["use_accepted_basis", "proposal_possible"],
            answer_contract="recommendation_with_proposal",
            requires_continuity=False,
        )
        return {
            "plan": plan,
            "candidate_id": candidate_id,
            "scenario_id": candidate_id,
            "selected_candidates": selected_candidates,
            "parameters": parameters,
            "session_id": session_id,
        }

    @staticmethod
    def _prompt_requests_prior_drop_combination(prompt: str) -> bool:
        if "drop" not in prompt and "drops" not in prompt:
            return False
        reference_terms = {
            "all",
            "these",
            "those",
            "your",
            "recommended",
            "recommendations",
            "top",
            "first",
            "include",
            "use",
            "want",
        }
        if not any(term in prompt for term in reference_terms):
            return False
        count_terms = {
            "two",
            "three",
            "four",
            "five",
            "2",
            "3",
            "4",
            "5",
        }
        return "all" in prompt or "these" in prompt or "those" in prompt or any(
            term in prompt for term in count_terms
        )

    @staticmethod
    def _requested_drop_candidate_count(*, prompt: str, available_count: int) -> int:
        count_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }
        for word, value in count_words.items():
            if word in prompt:
                return max(1, min(value, available_count))
        number_match = re.search(r"\b([1-9])\b", prompt)
        if number_match is not None:
            return max(1, min(int(number_match.group(1)), available_count))
        if "all" in prompt:
            return available_count
        return min(3, available_count)

    @staticmethod
    def _combine_drop_candidate_parameters(
        *,
        base_parameters: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        parameters = {
            "average": base_parameters.get("average", "volume"),
            "drop": [
                list(item)
                for item in base_parameters.get("drop", [])
                if isinstance(item, list | tuple) and len(item) >= 2
            ],
            "drop_valuation": base_parameters.get("drop_valuation", []),
            "tail": base_parameters.get(
                "tail",
                {
                    "curve": "weibull",
                    "attachment_age": None,
                    "projection_period": 0,
                    "fit_period": [],
                },
            ),
            "bf_apriori": base_parameters.get("bf_apriori", {}),
            "final_ultimate": base_parameters.get("final_ultimate", "chainladder"),
            "selected_ultimate_by_uwy": base_parameters.get(
                "selected_ultimate_by_uwy", {}
            ),
        }
        seen = {tuple(item[:2]) for item in parameters["drop"]}
        for candidate in candidates:
            candidate_parameters = (
                candidate.get("parameters")
                if isinstance(candidate.get("parameters"), dict)
                else {}
            )
            drops = candidate_parameters.get("drop")
            if not isinstance(drops, list):
                continue
            for drop in drops:
                if not isinstance(drop, list | tuple) or len(drop) < 2:
                    continue
                normalized = [str(drop[0]), int(drop[1])]
                identity = tuple(normalized)
                if identity in seen:
                    continue
                seen.add(identity)
                parameters["drop"].append(normalized)
        return parameters

    @staticmethod
    def _enrich_combined_drop_recalculation_result(
        *,
        tool_result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(tool_result)
        parameters = (
            dict(context.get("parameters"))
            if isinstance(context.get("parameters"), dict)
            else {}
        )
        candidate_id = str(context.get("candidate_id") or "combined_drop_candidates")
        scenario_id = str(context.get("scenario_id") or candidate_id)
        basis = build_analysis_basis(
            session_id=str(context.get("session_id") or enriched.get("session_id") or ""),
            basis_type="review_candidate",
            parameters=parameters,
            scenario_id=scenario_id,
            candidate_id=candidate_id,
            source_tool="tool_recalculate",
            source_review_type="combined_drop_recalculation",
            is_active_session=False,
        )
        enriched["analysis_basis"] = basis
        enriched["review_type"] = "combined_drop_recalculation"
        enriched["selected_drops"] = parameters.get("drop", [])
        enriched["top_candidates"] = [
            dict(item)
            for item in context.get("selected_candidates", [])
            if isinstance(item, dict)
        ]
        enriched["recommendation"] = {
            "recommendation_class": "reasonable_alternative",
            "candidate_id": candidate_id,
            "basis_key": basis.get("basis_key"),
            "scenario_id": scenario_id,
            "summary": (
                f"Use the top {len(enriched['top_candidates'])} drop-review candidates together."
            ),
            "recommended_changes": [
                {
                    "candidate_id": candidate_id,
                    "basis_key": basis.get("basis_key"),
                    "scenario_id": scenario_id,
                    "parameters": parameters,
                }
            ],
        }
        enriched.setdefault("continuity_notes", [])
        enriched.setdefault("policy_trace", {})
        return enriched

    @staticmethod
    def _build_bf_recalculation_context(
        *,
        user_prompt: str,
        session_context: dict[str, Any] | None,
        memory_state: dict[str, Any],
        planning_basis: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt = str(user_prompt or "").strip().lower()
        if not AssistantService._prompt_requests_bf_recalculation(prompt):
            return {}
        base_basis = AssistantService._basis_for_bf_recalculation(
            prompt=prompt,
            memory_state=memory_state,
            planning_basis=planning_basis,
        )
        base_parameters = (
            dict(base_basis.get("parameters"))
            if isinstance(base_basis.get("parameters"), dict)
            else {}
        )
        if not base_parameters:
            return {}
        session_id = ""
        segment = None
        if isinstance(session_context, dict):
            session_id = str(session_context.get("session_id") or "").strip()
            segment = session_context.get("segment")
        if not session_id:
            session_id = str(base_basis.get("session_id") or "").strip()
        if not session_id:
            return {}
        results_args = {"session_id": session_id}
        basis_args = AssistantService._assumption_detail_basis_args(base_basis)
        results_args.update(basis_args)
        plan = ExecutionPlan(
            playbook="bf_recalculation",
            workflow_name="bf_recalculation",
            goal="Apply explicit BF settings incrementally to the current analysis basis.",
            segment=str(segment).strip() if isinstance(segment, str) and segment.strip() else None,
            session_id=session_id,
            intent_class="scenario_recommendation",
            required_capabilities=["results_summary", "recalculate"],
            steps=[
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args=results_args,
                    evidence_key="bf_base_results",
                    basis_aware=True,
                ),
                PlanStep(
                    tool_name="tool_recalculate",
                    args={"session_id": session_id},
                    evidence_key="bf_recalculation",
                    basis_aware=True,
                ),
            ],
            required_evidence=["bf_base_results", "bf_recalculation"],
            minimum_evidence_count=2,
            stopping_rule="Stop after applying the explicit BF overrides to the selected basis.",
            basis_behavior=["use_accepted_basis", "proposal_possible"],
            answer_contract="recommendation_with_proposal",
            requires_continuity=False,
        )
        return {
            "plan": plan,
            "prompt": prompt,
            "base_basis": base_basis,
            "base_parameters": base_parameters,
            "session_id": session_id,
            "apriori": AssistantService._parse_bf_apriori(prompt),
            "newest_count": AssistantService._parse_newest_ay_count(prompt),
        }

    @staticmethod
    def _prompt_requests_bf_recalculation(prompt: str) -> bool:
        if "bf" not in prompt and "bornhuetter" not in prompt:
            return False
        explicit_terms = {
            "apriori",
            "a priori",
            "lr",
            "loss ratio",
            "newest",
            "newer",
            "chainladder else",
            "chainladder for",
            "apply",
            "test",
        }
        return any(term in prompt for term in explicit_terms)

    @staticmethod
    def _basis_for_bf_recalculation(
        *,
        prompt: str,
        memory_state: dict[str, Any],
        planning_basis: dict[str, Any] | None,
    ) -> dict[str, Any]:
        correction_terms = {
            "why have you changed",
            "changed the drops",
            "changed the tail",
            "setting we had",
            "settings we had",
            "with the setting we had",
        }
        if any(term in prompt for term in correction_terms):
            previous_basis = AssistantService._previous_accepted_basis(memory_state)
            if previous_basis:
                return previous_basis
        return dict(planning_basis) if isinstance(planning_basis, dict) else {}

    @staticmethod
    def _previous_accepted_basis(memory_state: dict[str, Any]) -> dict[str, Any]:
        transitions = (
            memory_state.get("basis_transition_history")
            if isinstance(memory_state.get("basis_transition_history"), list)
            else []
        )
        if not transitions:
            return {}
        last_transition = transitions[-1] if isinstance(transitions[-1], dict) else {}
        previous_key = str(last_transition.get("from_basis_key") or "").strip()
        if not previous_key:
            return {}
        current_basis = (
            memory_state.get("accepted_analysis_basis")
            if isinstance(memory_state.get("accepted_analysis_basis"), dict)
            else {}
        )
        basis_cache = (
            memory_state.get("scenario_basis_cache")
            if isinstance(memory_state.get("scenario_basis_cache"), dict)
            else {}
        )
        return BasisManager.lookup_basis_by_key(
            basis_key=previous_key,
            current_basis=current_basis,
            basis_cache=basis_cache,
        )

    @staticmethod
    def _parse_bf_apriori(prompt: str) -> float:
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", prompt)
        if percent_match is not None:
            return float(percent_match.group(1)) / 100.0
        decimal_match = re.search(r"(?:apriori|a priori|lr|loss ratio)\D*(0\.\d+|1\.0|1)", prompt)
        if decimal_match is not None:
            return float(decimal_match.group(1))
        return 0.6

    @staticmethod
    def _parse_newest_ay_count(prompt: str) -> int:
        count_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }
        for word, value in count_words.items():
            if f"{word} newest" in prompt or f"{word} newer" in prompt:
                return value
        number_match = re.search(r"\b([1-9])\s+(?:newest|newer|latest)", prompt)
        if number_match is not None:
            return int(number_match.group(1))
        return 3

    @staticmethod
    def _build_bf_recalculation_args(
        *,
        context: dict[str, Any],
        memory_state: dict[str, Any],
    ) -> dict[str, Any]:
        parameters = dict(context.get("base_parameters", {}))
        years = AssistantService._newest_uwys_from_results_summary(
            memory_state.get("results_summary"),
            count=int(context.get("newest_count") or 3),
        )
        apriori = float(context.get("apriori") or 0.6)
        bf_apriori = dict(parameters.get("bf_apriori", {}))
        selected_ultimate_by_uwy = dict(parameters.get("selected_ultimate_by_uwy", {}))
        for year in years:
            bf_apriori[str(year)] = apriori
            selected_ultimate_by_uwy[str(year)] = "bornhuetter_ferguson"
        parameters["bf_apriori"] = bf_apriori
        parameters["selected_ultimate_by_uwy"] = selected_ultimate_by_uwy
        parameters["final_ultimate"] = parameters.get("final_ultimate", "chainladder")
        return {
            "session_id": str(context.get("session_id") or ""),
            **parameters,
        }

    @staticmethod
    def _newest_uwys_from_results_summary(
        results_summary: object,
        *,
        count: int,
    ) -> list[str]:
        if not isinstance(results_summary, dict):
            return []
        rows = []
        latest_rows = results_summary.get("latest_rows")
        if isinstance(latest_rows, list) and latest_rows:
            rows.extend(item for item in latest_rows if isinstance(item, dict))
        else:
            top_rows = results_summary.get("top_rows")
            if isinstance(top_rows, list):
                rows.extend(item for item in top_rows if isinstance(item, dict))
        years = sorted(
            {str(item.get("uwy") or "").strip() for item in rows if str(item.get("uwy") or "").strip()},
            key=lambda value: int(value) if value.isdigit() else value,
        )
        return years[-max(1, count):]

    @staticmethod
    def _enrich_bf_recalculation_result(
        *,
        tool_result: dict[str, Any],
        context: dict[str, Any],
        args: dict[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(tool_result)
        parameters = {
            key: value
            for key, value in dict(args).items()
            if key != "session_id"
        }
        basis = build_analysis_basis(
            session_id=str(context.get("session_id") or enriched.get("session_id") or ""),
            basis_type="bespoke",
            parameters=parameters,
            scenario_id="bf_incremental_recalculation",
            candidate_id="bf_incremental_recalculation",
            source_tool="tool_recalculate",
            source_review_type="bf_recalculation",
            is_active_session=False,
        )
        enriched["analysis_basis"] = basis
        enriched["review_type"] = "bf_recalculation"
        return enriched

    def _filter_tool_specs_for_turn(
        self,
        *,
        tool_specs: list[dict[str, Any]],
        deterministic_packet: dict[str, Any] | None,
        user_prompt: str,
        exact_data_required: bool,
    ) -> list[dict[str, Any]]:
        filtered = list(tool_specs)
        prompt_filtered = self._filter_tool_specs_for_prompt(
            tool_specs=filtered,
            user_prompt=user_prompt,
            exact_data_required=exact_data_required,
        )
        if prompt_filtered:
            filtered = prompt_filtered
        if not isinstance(deterministic_packet, dict) or not deterministic_packet:
            return filtered
        plan = (
            deterministic_packet.get("plan", {})
            if isinstance(deterministic_packet.get("plan"), dict)
            else {}
        )
        if str(plan.get("answer_contract") or "").strip() == "recommendation_with_proposal":
            proposal = (
                deterministic_packet.get("proposal_basis")
                if isinstance(deterministic_packet.get("proposal_basis"), dict)
                else {}
            )
            if str(proposal.get("status") or "").strip() == "pending":
                return []
        review = (
            deterministic_packet.get("review", {})
            if isinstance(deterministic_packet.get("review"), dict)
            else {}
        )
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        if not steps:
            return filtered
        if str(review.get("status", "")).strip().lower() == "hard_fail":
            return filtered

        suppressed_names = {
            str(step.get("tool_name", "")).strip()
            for step in steps
            if isinstance(step, dict) and str(step.get("tool_name", "")).strip()
        }
        if not suppressed_names:
            return filtered

        deterministic_filtered: list[dict[str, Any]] = []
        for spec in filtered:
            if not isinstance(spec, dict):
                continue
            function = spec.get("function")
            if not isinstance(function, dict):
                deterministic_filtered.append(spec)
                continue
            name = str(function.get("name", "")).strip()
            if name in suppressed_names:
                continue
            deterministic_filtered.append(spec)
        if self._observability_enabled and len(deterministic_filtered) != len(filtered):
            logger.info(
                "[OBS] deterministic.tools.suppressed count=%s names=%s",
                len(filtered) - len(deterministic_filtered),
                sorted(suppressed_names),
            )
        return deterministic_filtered

    def _filter_tool_specs_for_prompt(
        self,
        *,
        tool_specs: list[dict[str, Any]],
        user_prompt: str,
        exact_data_required: bool,
    ) -> list[dict[str, Any]]:
        prompt = str(user_prompt or "").strip().lower()
        if not prompt:
            return tool_specs
        allowed_names = self._prompt_tool_whitelist(
            prompt=prompt,
            exact_data_required=exact_data_required,
        )
        if not allowed_names:
            return tool_specs
        filtered = []
        for spec in tool_specs:
            function = spec.get("function") if isinstance(spec, dict) else None
            if not isinstance(function, dict):
                filtered.append(spec)
                continue
            name = str(function.get("name", "")).strip()
            if name in allowed_names:
                filtered.append(spec)
        if self._observability_enabled and len(filtered) != len(tool_specs):
            logger.info(
                "[OBS] prompt.tools.filtered playbook=%s count=%s names=%s",
                self._select_playbook(prompt) or "unclassified",
                len(tool_specs) - len(filtered),
                sorted(allowed_names),
            )
        return filtered or tool_specs

    @staticmethod
    def _prompt_tool_whitelist(
        *,
        prompt: str,
        exact_data_required: bool,
    ) -> set[str]:
        if exact_data_required or AssistantService._is_exact_numeric_question(prompt):
            return {
                "tool_get_assumption_context_detail",
                "tool_get_result_for_uwy",
                "tool_get_scenario_detail",
                "tool_get_last_derived_drop_detail",
            }

        definitions = select_workflow_definitions(prompt)
        if not definitions:
            return set()
        allowed: set[str] = set()
        for definition in definitions:
            allowed.update(definition.tool_whitelist)
        return allowed

    @staticmethod
    def _tool_names(tool_specs: list[dict[str, Any]]) -> set[str]:
        names: set[str] = set()
        for spec in tool_specs:
            function = spec.get("function") if isinstance(spec, dict) else None
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name:
                names.add(name)
        return names

    def _narration_fallback_result(
        self,
        *,
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
        tool_events: list[dict[str, Any]],
        event_callback: Any | None,
        reason: str = "deterministic_narration_fallback",
        detail: str | None = None,
    ) -> dict[str, Any] | None:
        narration_packet = memory_state.get("narration_packet")
        if not isinstance(narration_packet, dict) or not narration_packet:
            return None
        fallback = render_narration_fallback(narration_packet)
        self._emit_streamed_content(event_callback, fallback)
        return {
            "content": fallback,
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_detail": detail or "",
            "session_id": workflow_state.get("session_id"),
            "tool_events": tool_events,
            "memory_snapshot": memory_state,
            "deterministic_packet": memory_state.get("deterministic_packet", {}),
            "narration_packet": narration_packet,
            "memory_update_proposals": memory_state.get("memory_update_proposals", []),
        }

    @staticmethod
    def _pending_proposal_acceptance_response(
        *,
        user_prompt: str,
        proposal_basis: dict[str, Any] | None,
        accepted_analysis_basis: dict[str, Any] | None,
    ) -> str:
        proposal = normalize_proposal_basis(proposal_basis)
        if not proposal or str(proposal.get("status") or "").strip() != "pending":
            return ""
        prompt = str(user_prompt or "").strip().lower()
        action_terms = {
            "accept",
            "apply",
            "use it",
            "use the strongest",
            "add it",
            "add this",
            "attach it",
            "make it",
            "set it",
            "use this",
            "baseline",
            "analysis basis",
        }
        if not any(term in prompt for term in action_terms):
            return ""
        label = str(
            proposal.get("scenario_label")
            or proposal.get("candidate_id")
            or proposal.get("scenario_id")
            or "the pending proposal"
        ).strip()
        current_label = BasisManager.label(accepted_analysis_basis)
        return (
            f"{current_label}\n\n"
            "### Pending Proposal\n"
            f"`{label}` is ready to be added to the Analysis Basis, but I cannot accept proposals from a chat text reply. "
            "Use the proposal card's accept button to update the Analysis Basis.\n\n"
            "### Current Status\n"
            "No new tool run was started and the Analysis Basis is unchanged until that explicit proposal acceptance happens."
        )

    def _execute_tool_call(
        self,
        *,
        function_name: str,
        args: dict[str, Any],
        tool_outputs: dict[str, dict[str, Any]],
        tool_events: list[dict[str, Any]],
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
        guardrail_state: dict[str, bool],
        event_callback: Any | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        args = self._apply_default_analysis_basis_args(
            function_name=function_name,
            args=args,
            memory_state=memory_state,
            workflow_state=workflow_state,
        )
        for event in tool_events:
            if not isinstance(event, dict):
                continue
            if event.get("name") == function_name and event.get("arguments") == args:
                cached = event.get("result_summary")
                if isinstance(cached, dict):
                    return cached, memory_state

        tool_result = self._tools.call_tool(function_name, args)
        execution_record = None
        if isinstance(tool_result, dict):
            execution_record = tool_result.pop("execution_record", None)
        tool_outputs[function_name] = tool_result
        next_memory = self._update_memory_state(
            memory_state,
            function_name=function_name,
            tool_result=tool_result,
        )
        if isinstance(execution_record, dict) and execution_record:
            existing_records = (
                list(next_memory.get("execution_records"))
                if isinstance(next_memory.get("execution_records"), list)
                else []
            )
            existing_records.append(dict(execution_record))
            next_memory["execution_records"] = existing_records
        tool_event = {
            "name": function_name,
            "arguments": args,
            "result_summary": tool_result,
        }
        tool_events.append(tool_event)
        self._emit_event(event_callback, "tool_event", tool_event)
        self._update_workflow_state(
            workflow_state=workflow_state,
            function_name=function_name,
            args=args,
            tool_result=tool_result,
        )
        self._update_guardrail_state(guardrail_state, tool_result)
        return tool_result, next_memory

    @staticmethod
    def _apply_default_analysis_basis_args(
        *,
        function_name: str,
        args: dict[str, Any],
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
    ) -> dict[str, Any]:
        supported = {
            "tool_recalculate",
            "tool_evaluate_tail_fit",
            "tool_run_diagnostics_summary",
            "tool_iterate_diagnostics_summary",
            "tool_run_drop_review",
            "tool_run_tail_review",
            "tool_run_bf_suitability_review",
            "tool_run_anomaly_triage",
            "tool_run_quarter_close_review",
            "tool_get_results_summary",
            "tool_get_data_view_summary",
            "tool_get_data_view",
            "tool_get_assumption_context_detail",
            "tool_run_ldf_consistency_diagnostics",
            "tool_project_late_emergence_benchmark",
            "tool_explain_reserve_change",
            "tool_run_highest_a2a_drop_scenario",
            "tool_rank_link_ratios",
            "tool_run_derived_drop_scenario",
        }
        if function_name not in supported:
            return args
        if function_name in {"tool_recalculate", "tool_evaluate_tail_fit"}:
            recalculate_args = AssistantService._compile_flat_recalculate_args(
                args=args,
                memory_state=memory_state,
                workflow_state=workflow_state,
            )
            if recalculate_args:
                return recalculate_args
        if function_name == "tool_explain_reserve_change":
            compare_args = AssistantService._compile_compare_current_basis_to_baseline_args(
                args=args,
                memory_state=memory_state,
                workflow_state=workflow_state,
            )
            if compare_args:
                return compare_args
        baseline_args = AssistantService._compile_explicit_baseline_basis_args(
            args=args,
            memory_state=memory_state,
            workflow_state=workflow_state,
        )
        if baseline_args:
            return baseline_args
        parameter_field = (
            "basis_parameters"
            if function_name == "tool_explain_reserve_change"
            else "parameters"
        )
        has_basis_type = args.get("basis_type") not in (None, "", {})
        has_basis_key = args.get("basis_key") not in (None, "", {})
        has_scenario_id = args.get("scenario_id") not in (None, "", {})
        has_parameters = args.get(parameter_field) not in (None, "", {})
        if has_parameters and isinstance(args.get(parameter_field), dict):
            computed_basis_key = basis_key_from_parameters(args.get(parameter_field))
            supplied_basis_key = str(args.get("basis_key") or "").strip()
            if computed_basis_key and supplied_basis_key and supplied_basis_key != computed_basis_key:
                corrected = dict(args)
                corrected["basis_key"] = computed_basis_key
                corrected.pop("scenario_id", None)
                return corrected
        if has_basis_type and has_basis_key and has_parameters:
            return args
        has_any_basis_arg = has_basis_type or has_basis_key or has_scenario_id or has_parameters

        basis = AssistantService._resolve_tool_call_basis(
            args=args,
            memory_state=memory_state,
            workflow_state=workflow_state,
            has_any_basis_arg=has_any_basis_arg,
        )
        if not basis:
            return args

        enriched = dict(args)
        basis_type = basis.get("basis_type")
        basis_key = basis.get("basis_key")
        parameters = basis.get("parameters")
        if not has_basis_type and isinstance(basis_type, str) and basis_type.strip():
            enriched["basis_type"] = basis_type.strip()
        if not has_basis_key and isinstance(basis_key, str) and basis_key.strip():
            enriched["basis_key"] = basis_key.strip()
        if not has_parameters and isinstance(parameters, dict) and parameters:
            if function_name == "tool_explain_reserve_change":
                enriched["basis_parameters"] = parameters
            else:
                enriched["parameters"] = parameters
        return enriched

    @staticmethod
    def _compile_explicit_baseline_basis_args(
        *,
        args: dict[str, Any],
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = str(workflow_state.get("current_user_prompt") or "").strip().lower()
        if not AssistantService._prompt_requests_baseline_basis(prompt):
            return {}
        basis_type = str(args.get("basis_type") or "").strip().lower()
        scenario_id = str(args.get("scenario_id") or "").strip().lower()
        if basis_type != "baseline" and scenario_id != "baseline":
            return {}
        baseline_basis = BasisManager.baseline_basis_from_memory(
            memory_state=memory_state,
            session_context=None,
        )
        baseline_parameters = (
            dict(baseline_basis.get("parameters"))
            if isinstance(baseline_basis.get("parameters"), dict)
            else {}
        )
        compiled = dict(args)
        compiled["basis_type"] = "baseline"
        compiled.pop("scenario_id", None)
        baseline_key = str(baseline_basis.get("basis_key") or "").strip()
        if baseline_key:
            compiled["basis_key"] = baseline_key
        else:
            compiled.pop("basis_key", None)
        if baseline_parameters:
            compiled["parameters"] = baseline_parameters
        else:
            compiled.pop("parameters", None)
        return compiled

    @staticmethod
    def _compile_flat_recalculate_args(
        *,
        args: dict[str, Any],
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
    ) -> dict[str, Any]:
        explicit_fields = {
            "average",
            "drop",
            "drop_valuation",
            "tail",
            "bf_apriori",
            "final_ultimate",
            "selected_ultimate_by_uwy",
        }
        has_explicit = any(field in args and args.get(field) is not None for field in explicit_fields)
        has_selector = any(
            args.get(field) not in (None, "", {})
            for field in ("basis_type", "basis_key", "scenario_id")
        )
        if not has_explicit and not has_selector:
            return {}

        basis = AssistantService._resolve_tool_call_basis(
            args=args,
            memory_state=memory_state,
            workflow_state=workflow_state,
            has_any_basis_arg=has_selector,
        )
        basis_parameters = (
            dict(basis.get("parameters"))
            if isinstance(basis, dict) and isinstance(basis.get("parameters"), dict)
            else {}
        )
        compiled = {
            key: value
            for key, value in args.items()
            if key not in {"basis_type", "basis_key", "scenario_id", "parameters"}
        }
        if basis_parameters:
            for field in explicit_fields:
                if field not in compiled or compiled.get(field) is None:
                    if field in basis_parameters:
                        compiled[field] = basis_parameters[field]
        return compiled

    @staticmethod
    def _compile_compare_current_basis_to_baseline_args(
        *,
        args: dict[str, Any],
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = str(workflow_state.get("current_user_prompt") or "").strip().lower()
        if not AssistantService._prompt_requests_baseline_basis(prompt):
            return {}
        current_terms = (
            "this basis",
            "current basis",
            "accepted basis",
            "analysis basis",
            "scenario before",
            "before all the modifications",
            "before the modifications",
            "before modifications",
        )
        if not any(term in prompt for term in current_terms):
            return {}
        accepted_basis = (
            memory_state.get("accepted_analysis_basis")
            if isinstance(memory_state.get("accepted_analysis_basis"), dict)
            else {}
        )
        accepted_params = (
            accepted_basis.get("parameters")
            if isinstance(accepted_basis.get("parameters"), dict)
            else {}
        )
        if not accepted_params:
            return {}
        baseline_basis = BasisManager.baseline_basis_from_memory(
            memory_state=memory_state,
            session_context=None,
        )
        baseline_params = (
            baseline_basis.get("parameters")
            if isinstance(baseline_basis.get("parameters"), dict)
            else {}
        )
        if not baseline_params:
            return {}
        compiled = dict(args)
        compiled.update(dict(accepted_params))
        compiled["basis_type"] = baseline_basis.get("basis_type") or "baseline"
        baseline_key = str(baseline_basis.get("basis_key") or "").strip()
        if baseline_key:
            compiled["basis_key"] = baseline_key
        compiled["basis_parameters"] = dict(baseline_params)
        compiled.pop("scenario_id", None)
        compiled.pop("parameters", None)
        return compiled

    @staticmethod
    def _resolve_tool_call_basis(
        *,
        args: dict[str, Any],
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
        has_any_basis_arg: bool,
    ) -> dict[str, Any]:
        prompt = str(workflow_state.get("current_user_prompt") or "").strip().lower()
        if AssistantService._prompt_requests_baseline_basis(prompt):
            return BasisManager.baseline_basis_from_memory(
                memory_state=memory_state,
                session_context=None,
            )

        basis_key = str(args.get("basis_key") or "").strip()
        scenario_id = str(args.get("scenario_id") or "").strip()
        current_basis = (
            memory_state.get("accepted_analysis_basis")
            if isinstance(memory_state.get("accepted_analysis_basis"), dict)
            else {}
        )
        basis_cache = (
            memory_state.get("scenario_basis_cache")
            if isinstance(memory_state.get("scenario_basis_cache"), dict)
            else {}
        )
        if basis_key:
            resolved_basis = BasisManager.lookup_basis_by_key(
                basis_key=basis_key,
                current_basis=current_basis,
                basis_cache=basis_cache,
            )
            if resolved_basis:
                return resolved_basis

        if scenario_id:
            if scenario_id == "baseline":
                return BasisManager.baseline_basis_from_memory(
                    memory_state=memory_state,
                    session_context=None,
                )
            return {}

        accepted_analysis_basis = current_basis
        if accepted_analysis_basis:
            return dict(accepted_analysis_basis)
        if has_any_basis_arg:
            return {}
        return BasisManager.baseline_basis_from_memory(
            memory_state=memory_state,
            session_context=None,
        )

    @staticmethod
    def _build_narration_prompt(narration_packet: dict[str, Any]) -> str:
        return build_narration_prompt(narration_packet)

    @staticmethod
    def _build_deterministic_packet_prompt(packet: dict[str, Any]) -> str:
        compact_packet = AssistantService._compact_deterministic_packet(packet)
        return (
            "Deterministic control packet already prepared for this turn. "
            "Use it as the primary evidence frame for the answer. Do not contradict its review status or recommendation status. "
            "Do not say the Analysis Basis changed unless explicit acceptance occurred; distinguish the current accepted basis from any proposed change.\n"
            + json.dumps(compact_packet, ensure_ascii=True)
        )

    @staticmethod
    def _compact_deterministic_packet(packet: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(packet, dict):
            return {}
        plan = packet.get("plan") if isinstance(packet.get("plan"), dict) else {}
        review = packet.get("review") if isinstance(packet.get("review"), dict) else {}
        recommendation = (
            packet.get("recommendation")
            if isinstance(packet.get("recommendation"), dict)
            else {}
        )
        composite_review = (
            packet.get("composite_review")
            if isinstance(packet.get("composite_review"), dict)
            else {}
        )
        composite_summary = (
            composite_review.get("summary")
            if isinstance(composite_review.get("summary"), dict)
            else {}
        )
        return {
            "plan": {
                "playbook": plan.get("playbook"),
                "workflow_name": plan.get("workflow_name"),
                "segment": plan.get("segment"),
                "intent_class": plan.get("intent_class"),
                "minimum_evidence_count": plan.get("minimum_evidence_count"),
                "basis_behavior": plan.get("basis_behavior", []),
                "answer_contract": plan.get("answer_contract"),
                "steps": [
                    {
                        "tool_name": item.get("tool_name"),
                        "evidence_key": item.get("evidence_key"),
                    }
                    for item in (plan.get("steps") or [])[:6]
                    if isinstance(item, dict)
                ],
            },
            "review": {
                "status": review.get("status"),
                "issues": review.get("issues", [])[:5],
                "caveats": review.get("caveats", [])[:5],
                "missing_evidence": review.get("missing_evidence", [])[:5],
            },
            "recommendation": {
                "status": recommendation.get("status"),
                "recommendation_class": recommendation.get("recommendation_class"),
                "recommended_basis_key": recommendation.get("recommended_basis_key"),
                "recommended_scenario_id": recommendation.get(
                    "recommended_scenario_id"
                ),
                "recommended_basis_id": recommendation.get("recommended_basis_id"),
                "alternative_basis_keys": recommendation.get(
                    "alternative_basis_keys", []
                )[:3],
                "alternative_scenario_ids": recommendation.get(
                    "alternative_scenario_ids", []
                )[:3],
            },
            "presentation": packet.get("presentation", {}),
            "composite_review": {
                "evidence_key": composite_review.get("evidence_key"),
                "governance": composite_review.get("governance", {}),
                "recommendation": composite_summary.get("recommendation", {}),
                "top_candidates": composite_summary.get("top_candidates", [])[:3],
                "top_ranked": composite_summary.get("top_ranked", [])[:3],
            },
            "continuity_notes": packet.get("continuity_notes", [])[:3],
            "score_breakdown": packet.get("score_breakdown", {}),
            "policy_trace": packet.get("policy_trace", {}),
            "recommended_changes": packet.get("recommended_changes", [])[:3],
            "accepted_analysis_basis": packet.get("accepted_analysis_basis", {}),
            "proposal_basis": packet.get("proposal_basis", {}),
        }

    @staticmethod
    def _build_segment_memory_hint(segment_memory: dict[str, Any]) -> str:
        if not isinstance(segment_memory, dict) or not segment_memory:
            return ""
        parts: list[str] = []
        last_selection = segment_memory.get("last_selection")
        if isinstance(last_selection, dict):
            tail = (
                last_selection.get("tail")
                if isinstance(last_selection.get("tail"), dict)
                else {}
            )
            parts.append(
                "Segment memory: "
                f"valuation_date={last_selection.get('valuation_date')}, "
                f"tail={tail.get('estimator') or tail.get('curve')}, "
                f"attachment_age={tail.get('attachment_age')}"
            )
        known_issues = segment_memory.get("known_issues")
        if isinstance(known_issues, list) and known_issues:
            parts.append(
                "Known issues: " + ", ".join(str(item) for item in known_issues[:4])
            )
        dispositions = segment_memory.get("scenario_dispositions")
        if isinstance(dispositions, list) and dispositions:
            rejected = [
                item
                for item in dispositions
                if isinstance(item, dict)
                and str(item.get("decision", "")).strip().lower() == "rejected"
            ]
        else:
            rejected = []
        if rejected:
            parts.append(
                "Previously rejected scenarios: "
                + "; ".join(
                    str(item.get("scenario_signature") or item.get("scenario_id"))
                    for item in rejected[:3]
                )
            )
        preferences = segment_memory.get("house_preferences")
        if isinstance(preferences, list) and preferences:
            parts.append(
                "House preferences: " + ", ".join(str(item) for item in preferences[:3])
            )
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _load_segment_note(session_context: dict[str, Any] | None) -> str:
        if not isinstance(session_context, dict):
            return ""
        return load_segment_note(session_context.get("segment"))

    def _load_segment_memory(
        self,
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(session_context, dict):
            return {}
        segment = session_context.get("segment")
        store = getattr(self, "_segment_memory_store", None)
        if store is None:
            return {}
        try:
            return store.load(segment=segment)
        except Exception:
            return {}

    def _persist_segment_memory(
        self,
        *,
        session_context: dict[str, Any] | None,
        segment_memory: dict[str, Any],
        memory_state: dict[str, Any],
        deterministic_packet: dict[str, Any],
    ) -> None:
        if not isinstance(session_context, dict):
            return
        segment = session_context.get("segment")
        if not isinstance(segment, str) or not segment.strip():
            return
        store = getattr(self, "_segment_memory_store", None)
        if store is None:
            return
        next_memory = dict(segment_memory)
        next_memory["segment_id"] = segment.strip()
        next_memory["schema_version"] = SegmentMemoryService.SCHEMA_VERSION
        session_summary = (
            memory_state.get("session_summary")
            if isinstance(memory_state.get("session_summary"), dict)
            else {}
        )
        params = (
            session_summary.get("params")
            if isinstance(session_summary.get("params"), dict)
            else {}
        )
        last_selection = dict(next_memory.get("last_selection", {}))
        if params:
            last_selection.update(
                {
                    "average": params.get("average"),
                    "tail": {
                        "curve": params.get("tail_curve"),
                        "attachment_age": params.get("tail_attachment_age"),
                    },
                }
            )
        next_memory["last_selection"] = last_selection
        if isinstance(memory_state.get("scenario_ledger"), list):
            next_memory["scenario_ledger"] = memory_state.get("scenario_ledger")
        recommendation = deterministic_packet.get("recommendation")
        if isinstance(recommendation, dict):
            next_memory["last_recommendation"] = recommendation
        review = deterministic_packet.get("review")
        if isinstance(review, dict):
            next_memory["last_review"] = review
        valuation_context = (
            session_summary.get("valuation_context")
            if isinstance(session_summary.get("valuation_context"), dict)
            else {}
        )
        memory_service = SegmentMemoryService()
        current_snapshot = (
            valuation_context.get("current")
            if isinstance(valuation_context.get("current"), dict)
            else None
        )
        prior_proxy_snapshot = (
            valuation_context.get("prior_proxy")
            if isinstance(valuation_context.get("prior_proxy"), dict)
            else None
        )
        next_memory = memory_service.append_valuation_snapshot(
            memory=next_memory,
            snapshot=current_snapshot,
        )
        next_memory = memory_service.append_valuation_snapshot(
            memory=next_memory,
            snapshot=prior_proxy_snapshot,
        )
        try:
            store.save(segment=segment.strip(), memory=next_memory)
        except Exception:
            return

    @staticmethod
    def _build_session_context_hint(session_context: dict[str, Any] | None) -> str:
        if not isinstance(session_context, dict):
            return ""
        session_id = session_context.get("session_id")
        segment = session_context.get("segment")
        details: list[str] = []
        if isinstance(segment, str) and segment.strip():
            details.append(f"segment={segment.strip()}")
        if isinstance(session_id, str) and session_id.strip():
            details.append(f"session_id={session_id.strip()}")
        if not details:
            return ""
        return (
            "Current reserving workspace context: "
            + ", ".join(details)
            + ". Reuse this context in tool calls unless the user asks to start a different session."
        )

    def _reference_prompts_for_turn(self, user_prompt: str) -> list[str]:
        if USE_FULL_REFERENCE_PROMPTS:
            prompts = []
            for prompt_text in (
                AI_CONTEXT_PROMPT,
                AI_PLAYBOOKS_PROMPT,
                AI_EXAMPLES_PROMPT,
                AI_POLICY_PROMPT,
            ):
                if prompt_text:
                    prompts.append(prompt_text)
            return prompts

        prompt = str(user_prompt or "").strip().lower()
        prompts = [COMPACT_CONTEXT_PROMPT, COMPACT_PLAYBOOKS_PROMPT]
        definitions = select_workflow_definitions(prompt)
        if self._is_recommendation_question(prompt) or (
            definitions and any(definition.policy_prompt_relevant for definition in definitions)
        ):
            prompts.append(COMPACT_POLICY_PROMPT)
        return prompts

    def _prime_context_for_prompt(
        self,
        *,
        user_prompt: str,
        session_context: dict[str, Any] | None,
        messages: list[dict[str, Any]],
        tool_outputs: dict[str, dict[str, Any]],
        tool_events: list[dict[str, Any]],
        memory_state: dict[str, Any],
        workflow_state: dict[str, Any],
        event_callback: Any | None,
    ) -> None:
        session_id = None
        if isinstance(session_context, dict):
            raw_session_id = session_context.get("session_id")
            if isinstance(raw_session_id, str) and raw_session_id.strip():
                session_id = raw_session_id.strip()
        if not session_id:
            return
        prompt = str(user_prompt or "").strip().lower()
        if self._is_exact_numeric_question(prompt):
            workflow_state["exact_data_required"] = True
            args = self._build_exact_detail_args(session_id=session_id, prompt=prompt)
            basis = self._resolve_exact_question_basis(
                prompt=prompt,
                memory_state=memory_state,
                session_context=session_context,
            )
            if isinstance(basis, dict) and basis:
                args.update(self._assumption_detail_basis_args(basis))
            basis_label = self._analysis_basis_label(basis)
            self._emit_event(
                event_callback,
                "status",
                {
                    "message": self._tool_status_label(
                        "tool_get_assumption_context_detail",
                        args,
                    )
                },
            )
            tool_result = self._tools.call_tool(
                "tool_get_assumption_context_detail", args
            )
            self._append_prefetch_execution_record(memory_state, tool_result)
            compact_tool_result = compact_tool_result_for_model(
                tool_name="tool_get_assumption_context_detail",
                result=tool_result,
            )
            tool_outputs["prefetch:assumption_detail"] = tool_result
            memory_state.update(
                self._update_memory_state(
                    dict(memory_state),
                    function_name="tool_get_assumption_context_detail",
                    tool_result=tool_result,
                )
            )
            tool_event = {
                "name": "tool_get_assumption_context_detail",
                "arguments": args,
                "result_summary": tool_result,
                "prefetch": True,
            }
            tool_events.append(tool_event)
            self._emit_event(event_callback, "tool_event", tool_event)
            self._update_workflow_state(
                workflow_state=workflow_state,
                function_name="tool_get_assumption_context_detail",
                args=args,
                tool_result=tool_result,
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Exact numeric follow-up detected. {basis_label} Begin the answer with 'Basis used: ...'. "
                        "Use this exact-detail payload as the primary evidence source. "
                        "Do not switch back to baseline unless the user explicitly asks for baseline or current session. "
                        "Do not invent tables, vectors, or quoted values outside this payload. If a requested value is missing here, say it is not verified from current tool outputs."
                        + (
                            " The payload marks the tail as inactive/reference-only, so do not describe it as an applied tail override or as affecting ultimates."
                            if tool_result.get("tail_active") is False
                            else ""
                        )
                        + "\n"
                        + json.dumps(compact_tool_result, ensure_ascii=True)
                    ),
                }
            )
        if not self._is_movement_question(prompt):
            return
        if "claims" not in prompt and "incurred" not in prompt:
            return

        preloads = [
            (
                "tool_get_data_view_summary",
                {
                    "session_id": session_id,
                    "metric": "incurred",
                    "view": "incremental",
                },
                "latest_diagonal_incurred_incremental",
            ),
            (
                "tool_get_data_view_summary",
                {
                    "session_id": session_id,
                    "metric": "incurred",
                    "view": "cumulative",
                    "denominator": "premium",
                },
                "latest_diagonal_incurred_on_premium",
            ),
            (
                "tool_run_ldf_consistency_diagnostics",
                {"session_id": session_id},
                "a2a_ldf_consistency",
            ),
        ]

        evidence_payload: dict[str, Any] = {}
        for function_name, args, evidence_key in preloads:
            self._emit_event(
                event_callback,
                "status",
                {"message": self._tool_status_label(function_name, args)},
            )
            tool_result = self._tools.call_tool(function_name, args)
            self._append_prefetch_execution_record(memory_state, tool_result)
            tool_outputs[f"prefetch:{evidence_key}"] = tool_result
            memory_state.update(
                self._update_memory_state(
                    dict(memory_state),
                    function_name=function_name,
                    tool_result=tool_result,
                )
            )
            tool_event = {
                "name": function_name,
                "arguments": args,
                "result_summary": tool_result,
                "prefetch": True,
            }
            tool_events.append(tool_event)
            self._emit_event(event_callback, "tool_event", tool_event)
            self._update_workflow_state(
                workflow_state=workflow_state,
                function_name=function_name,
                args=args,
                tool_result=tool_result,
            )
            evidence_payload[evidence_key] = tool_result

        messages.append(
            {
                "role": "system",
                "content": (
                    "Preloaded evidence for this claims-movement question. "
                    "Base the first answer primarily on this evidence: latest-diagonal incremental incurred movement, "
                    "incurred-on-premium context, and age-to-age/LDF consistency. "
                    "Do not lead with premium or generic recommendations unless this evidence supports it.\n"
                    + json.dumps(evidence_payload, ensure_ascii=True)
                ),
            }
        )

    @staticmethod
    def _build_intent_hint(user_prompt: str) -> str:
        prompt = str(user_prompt or "").strip().lower()
        if not prompt:
            return ""
        if AssistantService._is_movement_question(prompt):
            return (
                "This user is asking an observational movement question, not asking for a scenario recommendation yet. "
                "Prefer movement/data-view tools first. Answer the movement question directly from current data and diagnostics. "
                "If they say claims without a modifier, treat that as incurred. "
                "If they ask about this quarter/current quarter, focus on the latest diagonal or in-quarter incremental movement first. "
                "Do not lead with premium unless they explicitly asked about premium or it is clearly secondary supporting context. "
                "Do not run scenario-search unless the user asks for recommendations, drops, method changes, or scenario comparisons."
            )
        if AssistantService._is_recommendation_question(prompt):
            return (
                "This user is asking for recommendations or alternative scenarios. "
                "Use scenario iteration before final recommendations unless the answer is already directly established by stronger evidence."
            )
        return ""

    @staticmethod
    def _build_playbook_hint(user_prompt: str) -> str:
        prompt = str(user_prompt or "").strip().lower()
        if not prompt:
            return ""
        definitions = select_workflow_definitions(prompt)
        if len(definitions) > 1:
            return (
                "Selected playbooks: "
                + "; ".join(definition.workflow_name for definition in definitions)
                + ". Run each matched deterministic workflow and summarize the separate evidence. "
                "Do not collapse multiple proposal-capable workflows into one implicit accepted-basis change."
            )
        return definitions[0].prompt_hint if definitions else ""

    @staticmethod
    def _is_recommendation_question(prompt: str) -> bool:
        recommendation_keywords = {
            "recommend",
            "scenario",
            "drop",
            "tail",
            "bf",
            "bornhuetter",
            "change",
            "adjust",
            "what should",
            "which should",
            "best",
            "optimi",
            "recal",
            "compare",
            "trade-off",
            "tradeoff",
        }
        return any(keyword in prompt for keyword in recommendation_keywords)

    @staticmethod
    def _select_playbook(prompt: str) -> str:
        return select_workflow_name(prompt)

    @staticmethod
    def _is_movement_question(prompt: str) -> bool:
        movement_keywords = {
            "movement",
            "movements",
            "unexpected",
            "unusual",
            "this quarter",
            "current quarter",
            "latest diagonal",
            "in quarter",
            "what happened",
        }
        claims_keywords = {"claims", "incurred", "paid", "outstanding", "premium"}
        return any(keyword in prompt for keyword in movement_keywords) and any(
            keyword in prompt for keyword in claims_keywords
        )

    @staticmethod
    def _is_exact_numeric_question(prompt: str) -> bool:
        if AssistantService._is_recommendation_question(prompt):
            review_terms = {
                "review",
                "rank",
                "recommend",
                "selection-worthy",
                "sensitivity",
                "candidate",
                "candidates",
            }
            exact_override_terms = {
                "exact",
                "show exact",
                "list exact",
                "exact values",
                "exact settings",
                "exact parameters",
                "table of",
                "vector",
            }
            exact_detail_request_terms = {
                "what are",
                "what is",
                "show me",
                "list",
                "table",
                "vector",
            }
            exact_detail_subject_terms = {
                "ldf",
                "ldfs",
                "fitted tail",
                "fitted ldf",
                "values",
                "factors",
            }
            has_explicit_exact_override = any(
                term in prompt for term in exact_override_terms
            ) or (
                any(term in prompt for term in exact_detail_request_terms)
                and any(term in prompt for term in exact_detail_subject_terms)
            )
            if any(term in prompt for term in review_terms) and not has_explicit_exact_override:
                return False
            exact_terms = {
                "exact",
                "which exact",
                "which setting",
                "which settings",
                "what setting",
                "what settings",
                "configuration",
                "parameters",
                "vector",
                "values",
                "factors",
                "ldf",
                "ldfs",
                "a2a",
                "age-to-age",
            }
            if not any(term in prompt for term in exact_terms):
                return False
        request_terms = {
            "show me",
            "compare",
            "comparison",
            "side by side",
            "what is",
            "what are",
            "which exact",
            "which setting",
            "which settings",
            "what setting",
            "what settings",
            "why",
            "list",
            "table",
            "vector",
            "values",
            "factors",
            "rows",
            "configuration",
            "parameters",
        }
        subject_terms = {
            "tail",
            "curve",
            "weibull",
            "inverse power",
            "inverse_power",
            "exponential",
            "setting",
            "settings",
            "configuration",
            "parameters",
            "ldf",
            "ldfs",
            "a2a",
            "age-to-age",
            "link ratio",
            "link ratios",
            "non-fitted",
            "non fitted",
            "original",
            "observed",
            "apriori",
            "selected method",
            "selected methods",
            "fitted tail",
            "fitted ldf",
            "uwy",
        }
        return any(term in prompt for term in request_terms) and any(
            term in prompt for term in subject_terms
        )

    @staticmethod
    def _build_exact_detail_args(*, session_id: str, prompt: str) -> dict[str, Any]:
        args: dict[str, Any] = {"session_id": session_id}
        age_range = re.search(
            r"(?:from|between|ages?|months?)\s+(\d+)\s*(?:-|to|up to|through|until|and)\s*(\d+)",
            prompt,
        )
        if age_range is not None:
            left = int(age_range.group(1))
            right = int(age_range.group(2))
            args["start_age"] = min(left, right)
            args["end_age"] = max(left, right)
        development_period = re.search(r"period\s+(\d+)\s*[-/]\s*(\d+)", prompt)
        if development_period is not None:
            args["development_period"] = int(development_period.group(1))
        return args

    @staticmethod
    def _assumption_detail_basis_args(basis: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {}
        basis_type = basis.get("basis_type")
        basis_key = basis.get("basis_key")
        parameters = basis.get("parameters")
        if isinstance(basis_type, str) and basis_type.strip():
            args["basis_type"] = basis_type.strip()
        if isinstance(basis_key, str) and basis_key.strip():
            args["basis_key"] = basis_key.strip()
        if isinstance(parameters, dict) and parameters:
            args["parameters"] = parameters
        return args

    @staticmethod
    def _analysis_basis_label(basis: dict[str, Any] | None) -> str:
        return BasisManager.label(basis)

    @staticmethod
    def _resolve_exact_question_basis(
        *,
        prompt: str,
        memory_state: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        baseline_basis = BasisManager.baseline_basis_from_memory(
            memory_state=memory_state,
            session_context=session_context,
        )
        if AssistantService._prompt_requests_baseline_basis(prompt):
            return baseline_basis
        current_basis = (
            memory_state.get("accepted_analysis_basis")
            if isinstance(memory_state.get("accepted_analysis_basis"), dict)
            else {}
        )
        if current_basis and BasisManager.basis_is_mentioned_in_prompt(
            prompt=prompt,
            basis=current_basis,
        ):
            return dict(current_basis)
        basis_cache = (
            memory_state.get("scenario_basis_cache")
            if isinstance(memory_state.get("scenario_basis_cache"), dict)
            else {}
        )
        matched_scenario = BasisManager.scenario_id_mentioned_in_prompt(
            prompt=prompt,
            basis_cache=basis_cache,
        )
        if matched_scenario:
            resolved_cached = BasisManager.lookup_basis_by_requested_id(
                requested_id=matched_scenario,
                current_basis=current_basis,
                basis_cache=basis_cache,
            )
            if resolved_cached:
                return dict(resolved_cached)
        if current_basis:
            return dict(current_basis)
        return baseline_basis

    @staticmethod
    def _baseline_basis_from_memory(
        *,
        memory_state: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return BasisManager.baseline_basis_from_memory(
            memory_state=memory_state,
            session_context=session_context,
        )

    @staticmethod
    def _prompt_requests_baseline_basis(prompt: str) -> bool:
        return any(
            phrase in prompt
            for phrase in (
                "baseline",
                "base line",
                "current session",
                "current baseline",
                "active session",
                "beginning",
                "start of chat",
                "start of the chat",
                "before all the modifications",
                "before the modifications",
                "before modifications",
                "before we changed",
                "original scenario",
                "original basis",
            )
        )

    @staticmethod
    def _scenario_id_mentioned_in_prompt(
        *,
        prompt: str,
        basis_cache: dict[str, Any],
    ) -> str | None:
        return BasisManager.scenario_id_mentioned_in_prompt(
            prompt=prompt,
            basis_cache=basis_cache,
        )

    @staticmethod
    def _basis_identifier_aliases(basis: dict[str, Any] | None) -> list[str]:
        return BasisManager.identifier_aliases(basis)

    @staticmethod
    def _basis_is_mentioned_in_prompt(*, prompt: str, basis: dict[str, Any]) -> bool:
        return BasisManager.basis_is_mentioned_in_prompt(prompt=prompt, basis=basis)

    @staticmethod
    def _lookup_basis_by_requested_id(
        *,
        requested_id: str,
        current_basis: dict[str, Any],
        basis_cache: dict[str, Any],
    ) -> dict[str, Any]:
        return BasisManager.lookup_basis_by_requested_id(
            requested_id=requested_id,
            current_basis=current_basis,
            basis_cache=basis_cache,
        )

    @staticmethod
    def _basis_for_recommendation_turn(
        *,
        prompt: str,
        memory_state: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if AssistantService._prompt_requests_baseline_basis(prompt):
            return BasisManager.baseline_basis_from_memory(
                memory_state=memory_state,
                session_context=session_context,
            )
        accepted_analysis_basis = (
            memory_state.get("accepted_analysis_basis")
            if isinstance(memory_state.get("accepted_analysis_basis"), dict)
            else {}
        )
        if accepted_analysis_basis:
            return dict(accepted_analysis_basis)
        return None

    @staticmethod
    def _should_block_for_missing_exact_data(workflow_state: dict[str, Any]) -> bool:
        return bool(workflow_state.get("exact_data_required")) and not bool(
            workflow_state.get("exact_data_loaded")
        )

    @staticmethod
    def _exact_data_guardrail_message() -> str:
        return (
            "Guardrail: exact numeric answer blocked because no exact-data tool result was loaded for this question. "
            "Load exact assumption/detail evidence first, then answer from that payload only."
        )

    @staticmethod
    def _emit_event(
        event_callback: Any | None, event_type: str, payload: dict[str, Any]
    ) -> None:
        if callable(event_callback):
            event_callback(event_type, payload)

    @staticmethod
    def _emit_streamed_content(event_callback: Any | None, content: str) -> None:
        if not callable(event_callback):
            return
        for chunk in AssistantService._chunk_text(content):
            event_callback("content_chunk", {"text": chunk})
            time.sleep(0.02)
        event_callback("status", {"message": "Done"})

    @staticmethod
    def _chunk_text(content: str, words_per_chunk: int = 10) -> list[str]:
        words = content.split()
        if not words:
            return [content] if content else []
        chunks: list[str] = []
        for index in range(0, len(words), words_per_chunk):
            part = " ".join(words[index : index + words_per_chunk])
            if index + words_per_chunk < len(words):
                part += " "
            chunks.append(part)
        return chunks

    @staticmethod
    def _append_prefetch_execution_record(
        memory_state: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> None:
        if not isinstance(tool_result, dict):
            return
        execution_record = tool_result.get("execution_record")
        if not isinstance(execution_record, dict) or not execution_record:
            return
        records = (
            list(memory_state.get("execution_records"))
            if isinstance(memory_state.get("execution_records"), list)
            else []
        )
        records.append(dict(execution_record))
        memory_state["execution_records"] = records

    @staticmethod
    def _tool_status_label(function_name: str, args: dict[str, Any]) -> str:
        labels = {
            "tool_get_session_summary": "Loading session summary",
            "tool_evaluate_tail_fit": "Evaluating tail fit",
            "tool_get_data_view_summary": "Loading data summary",
            "tool_get_data_view": "Loading detailed data view",
            "tool_get_assumption_context_detail": "Loading exact assumption detail",
            "tool_compare_data_views": "Comparing data views",
            "tool_run_diagnostics": "Running diagnostics",
            "tool_run_diagnostics_summary": "Running diagnostics",
            "tool_run_drop_review": "Running drop review",
            "tool_run_tail_review": "Running tail review",
            "tool_run_bf_suitability_review": "Running BF suitability review",
            "tool_run_anomaly_triage": "Running anomaly triage",
            "tool_run_quarter_close_review": "Running quarter-close review",
            "tool_get_quarter_close_pack": "Building quarter-close pack",
            "tool_iterate_diagnostics": "Testing scenarios",
            "tool_run_movement_diagnostics": "Running movement diagnostics",
            "tool_run_ldf_consistency_diagnostics": "Checking LDF consistency",
            "tool_project_late_emergence_benchmark": "Projecting late emergence",
            "tool_iterate_diagnostics_summary": "Testing scenarios",
            "tool_rank_link_ratios": "Ranking link ratios",
            "tool_run_derived_drop_scenario": "Running derived drop scenario",
            "tool_run_highest_a2a_drop_scenario": "Running highest a2a drop scenario",
            "tool_get_last_derived_drop_detail": "Loading exact derived drop detail",
            "tool_get_results_summary": "Loading results summary",
            "tool_get_finding_detail": "Inspecting diagnostic evidence",
            "tool_get_scenario_detail": "Inspecting scenario detail",
            "tool_get_result_for_uwy": "Inspecting underwriting year detail",
            "tool_recalculate": "Running bespoke recalculation",
            "tool_explain_reserve_change": "Explaining reserve change",
        }
        label = labels.get(function_name, function_name.replace("tool_", ""))
        scenario_id = args.get("scenario_id") if isinstance(args, dict) else None
        uwy = args.get("uwy") if isinstance(args, dict) else None
        if scenario_id:
            return f"{label}: {scenario_id}"
        if uwy:
            return f"{label}: UWY {uwy}"
        return label

    @staticmethod
    def _update_memory_state(
        memory_state: dict[str, Any],
        *,
        function_name: str,
        tool_result: dict[str, Any],
    ) -> dict[str, Any]:
        current = dict(memory_state)
        if function_name == "tool_get_session_summary":
            current = build_memory_snapshot(
                session_summary=dict(tool_result),
                diagnostics_summary=current.get("diagnostics_summary"),
                iteration_summary=current.get("iteration_summary"),
                results_summary=current.get("results_summary"),
                data_view_summary=current.get("data_view_summary"),
                movement_summary=current.get("movement_summary"),
                reserve_change_summary=current.get("reserve_change_summary"),
                review_summary=current.get("review_summary"),
                existing_scenario_ledger=current.get("scenario_ledger"),
                existing_accepted_analysis_basis=current.get("accepted_analysis_basis"),
                existing_scenario_basis_cache=current.get("scenario_basis_cache"),
            )
        elif function_name == "tool_get_assumption_context_detail":
            current["assumption_detail"] = dict(tool_result)
        elif function_name in {"tool_get_data_view_summary", "tool_get_data_view"}:
            current["data_view_summary"] = dict(tool_result)
        elif function_name in {"tool_run_diagnostics", "tool_run_diagnostics_summary"}:
            current["diagnostics_summary"] = dict(tool_result)
        elif function_name in {
            "tool_run_movement_diagnostics",
            "tool_run_ldf_consistency_diagnostics",
            "tool_project_late_emergence_benchmark",
        }:
            current["movement_summary"] = dict(tool_result)
        elif function_name in {
            "tool_iterate_diagnostics",
            "tool_iterate_diagnostics_summary",
        }:
            current = build_memory_snapshot(
                session_summary=current.get("session_summary"),
                diagnostics_summary=current.get("diagnostics_summary"),
                iteration_summary=dict(tool_result),
                results_summary=current.get("results_summary"),
                data_view_summary=current.get("data_view_summary"),
                movement_summary=current.get("movement_summary"),
                reserve_change_summary=current.get("reserve_change_summary"),
                review_summary=current.get("review_summary"),
                existing_scenario_ledger=current.get("scenario_ledger"),
                existing_accepted_analysis_basis=current.get("accepted_analysis_basis"),
                existing_scenario_basis_cache=current.get("scenario_basis_cache"),
            )
        elif function_name in {"tool_get_results_summary", "tool_recalculate"}:
            current["results_summary"] = dict(tool_result)
            if function_name == "tool_recalculate":
                basis = tool_result.get("analysis_basis")
                if (
                    isinstance(basis, dict)
                    and basis
                    and bool(basis.get("is_active_session"))
                ):
                    current["session_summary"] = (
                        AssistantService._session_summary_with_basis(
                            current.get("session_summary"),
                            basis,
                        )
                    )
                    current = AssistantService._sync_baseline_basis_cache_from_session_summary(
                        current
                    )
                    current["preview_basis"] = {}
                elif isinstance(basis, dict) and basis:
                    current["preview_basis"] = dict(basis)
                    basis_key = str(basis.get("basis_key") or "").strip()
                    if basis_key:
                        cache = (
                            dict(current.get("scenario_basis_cache"))
                            if isinstance(current.get("scenario_basis_cache"), dict)
                            else {}
                        )
                        cache[basis_key] = dict(basis)
                        current["scenario_basis_cache"] = cache
        elif function_name == "tool_explain_reserve_change":
            current["reserve_change_summary"] = dict(tool_result)
        elif function_name in {
            "tool_run_drop_review",
            "tool_run_tail_review",
            "tool_run_bf_suitability_review",
            "tool_run_anomaly_triage",
            "tool_run_quarter_close_review",
            "tool_get_quarter_close_pack",
        }:
            current = build_memory_snapshot(
                session_summary=current.get("session_summary"),
                diagnostics_summary=current.get("diagnostics_summary"),
                iteration_summary=current.get("iteration_summary"),
                results_summary=current.get("results_summary"),
                data_view_summary=current.get("data_view_summary"),
                movement_summary=current.get("movement_summary"),
                reserve_change_summary=current.get("reserve_change_summary"),
                review_summary=dict(tool_result),
                existing_scenario_ledger=current.get("scenario_ledger"),
                existing_accepted_analysis_basis=current.get("accepted_analysis_basis"),
                existing_scenario_basis_cache=current.get("scenario_basis_cache"),
            )
        elif function_name in {
            "tool_run_highest_a2a_drop_scenario",
            "tool_run_derived_drop_scenario",
        }:
            current["reserve_change_summary"] = dict(tool_result)
            scenario = (
                tool_result.get("scenario")
                if isinstance(tool_result.get("scenario"), dict)
                else tool_result
            )
            parameters = (
                scenario.get("parameters")
                if isinstance(scenario.get("parameters"), dict)
                else {}
            )
            if parameters:
                basis = build_analysis_basis(
                    session_id=tool_result.get("session_id"),
                    basis_type="scenario",
                    parameters=parameters,
                    scenario_id=scenario.get("scenario_id"),
                    source_tool=function_name,
                    is_active_session=False,
                )
                basis_key = str(basis.get("basis_key") or "").strip()
                if basis_key:
                    cache = (
                        dict(current.get("scenario_basis_cache"))
                        if isinstance(current.get("scenario_basis_cache"), dict)
                        else {}
                    )
                    cache[basis_key] = basis
                    current["scenario_basis_cache"] = cache
        elif function_name == "tool_rank_link_ratios":
            current["data_view_summary"] = dict(tool_result)
        for key in (
            "accepted_analysis_basis",
            "proposal_basis",
            "preview_basis",
            "execution_records",
            "basis_transition_history",
            "memory_update_proposals",
            "narration_packet",
        ):
            if key in memory_state and key not in current:
                value = memory_state.get(key)
                if isinstance(value, dict):
                    current[key] = dict(value)
                elif isinstance(value, list):
                    current[key] = [
                        dict(item) if isinstance(item, dict) else item for item in value
                    ]
                else:
                    current[key] = value
        return current

    @staticmethod
    def _session_summary_with_basis(
        session_summary: object,
        analysis_basis: dict[str, Any],
    ) -> dict[str, Any]:
        summary = dict(session_summary) if isinstance(session_summary, dict) else {}
        params = (
            dict(summary.get("params"))
            if isinstance(summary.get("params"), dict)
            else {}
        )
        basis_params = (
            analysis_basis.get("parameters")
            if isinstance(analysis_basis.get("parameters"), dict)
            else {}
        )
        tail = (
            basis_params.get("tail")
            if isinstance(basis_params.get("tail"), dict)
            else {}
        )
        if basis_params:
            params.update(
                {
                    "average": basis_params.get("average", params.get("average")),
                    "drop_store": basis_params.get(
                        "drop", params.get("drop_store", [])
                    ),
                    "tail_curve": tail.get("curve", params.get("tail_curve")),
                    "tail_attachment_age": tail.get(
                        "attachment_age", params.get("tail_attachment_age")
                    ),
                    "tail_projection_months": tail.get(
                        "projection_period", params.get("tail_projection_months", 0)
                    ),
                    "tail_fit_period_selection": tail.get(
                        "fit_period", params.get("tail_fit_period_selection", [])
                    ),
                    "bf_apriori_by_uwy": basis_params.get(
                        "bf_apriori", params.get("bf_apriori_by_uwy", {})
                    ),
                    "selected_ultimate_by_uwy": basis_params.get(
                        "selected_ultimate_by_uwy",
                        params.get("selected_ultimate_by_uwy", {}),
                    ),
                    "drop_count": len(basis_params.get("drop", [])),
                    "bf_apriori_year_count": len(basis_params.get("bf_apriori", {})),
                    "selected_ultimate_overrides": len(
                        basis_params.get("selected_ultimate_by_uwy", {})
                    ),
                }
            )
        summary["params"] = params
        return summary

    @staticmethod
    def _sync_baseline_basis_cache_from_session_summary(
        memory_state: dict[str, Any],
    ) -> dict[str, Any]:
        current = dict(memory_state)
        session_summary = (
            current.get("session_summary")
            if isinstance(current.get("session_summary"), dict)
            else {}
        )
        baseline_basis = build_baseline_analysis_basis(session_summary)
        if not baseline_basis:
            return current
        basis_cache = (
            dict(current.get("scenario_basis_cache"))
            if isinstance(current.get("scenario_basis_cache"), dict)
            else {}
        )
        baseline_key = str(baseline_basis.get("basis_key") or "").strip()
        if baseline_key:
            basis_cache[baseline_key] = baseline_basis
        current["scenario_basis_cache"] = basis_cache
        return current

    @staticmethod
    def _update_workflow_state(
        *,
        workflow_state: dict[str, Any],
        function_name: str,
        args: dict[str, Any],
        tool_result: dict[str, Any],
    ) -> None:
        session_from_args = args.get("session_id") if isinstance(args, dict) else None
        session_from_result = tool_result.get("session_id")
        session_id = session_from_args or session_from_result
        if isinstance(session_id, str) and session_id:
            workflow_state["session_id"] = session_id

        if function_name in {"tool_run_diagnostics", "tool_run_diagnostics_summary"}:
            workflow_state["ran_diagnostics"] = True
        if function_name in {
            "tool_iterate_diagnostics",
            "tool_iterate_diagnostics_summary",
        }:
            workflow_state["ran_iteration"] = True
        if function_name in {
            "tool_get_assumption_context_detail",
            "tool_get_data_view",
            "tool_get_result_for_uwy",
            "tool_get_last_derived_drop_detail",
            "tool_evaluate_tail_fit",
            "tool_rank_link_ratios",
            "tool_run_bf_suitability_review",
        }:
            workflow_state["exact_data_loaded"] = True

    @staticmethod
    def _update_guardrail_state(
        state: dict[str, bool],
        tool_result: dict[str, Any],
    ) -> None:
        findings = tool_result.get("findings")
        if isinstance(findings, list):
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                code = str(finding.get("code", ""))
                if code.startswith(
                    "PORTFOLIO_SHIFT_SIGNAL_UNCONFIRMED"
                ) or code.startswith("PORTFOLIO_SHIFT_CONFLICT"):
                    state["portfolio_shift_unconfirmed"] = True
                if "PAID_INCURRED_COHERENCE" in code:
                    state["paid_incurred_conflict"] = True

        metrics = tool_result.get("metrics")
        if isinstance(metrics, dict):
            confidence = metrics.get("assessment_confidence")
            try:
                if confidence is not None and float(confidence) < 0.5:
                    state["low_confidence"] = True
            except (TypeError, ValueError):
                pass
            tier = str(metrics.get("governance_tier", "")).lower()
            if tier in {"amber", "red"}:
                state["low_confidence"] = True
            uncertainty = metrics.get("uncertainty")
            AssistantService._update_uncertainty_state(state, uncertainty)

        top_uncertainty = tool_result.get("uncertainty")
        AssistantService._update_uncertainty_state(state, top_uncertainty)

        scenarios = tool_result.get("scenarios") or tool_result.get("top_scenarios")
        if isinstance(scenarios, list):
            for scenario in scenarios:
                if not isinstance(scenario, dict):
                    continue
                findings_items = scenario.get("findings")
                if isinstance(findings_items, list):
                    for finding in findings_items:
                        if not isinstance(finding, dict):
                            continue
                        code = str(finding.get("code", ""))
                        if code.startswith(
                            "PORTFOLIO_SHIFT_SIGNAL_UNCONFIRMED"
                        ) or code.startswith("PORTFOLIO_SHIFT_CONFLICT"):
                            state["portfolio_shift_unconfirmed"] = True
                        if "PAID_INCURRED_COHERENCE" in code:
                            state["paid_incurred_conflict"] = True

                scenario_uncertainty = scenario.get("uncertainty")
                AssistantService._update_uncertainty_state(state, scenario_uncertainty)

    @staticmethod
    def _update_uncertainty_state(
        state: dict[str, bool],
        uncertainty_payload: object,
    ) -> None:
        if not isinstance(uncertainty_payload, dict):
            return
        cv_raw = uncertainty_payload.get("total_process_cv")
        if cv_raw is None:
            cv_raw = uncertainty_payload.get("process_cv")
        try:
            if cv_raw is not None and float(cv_raw) >= 0.35:
                state["high_process_uncertainty"] = True
        except (TypeError, ValueError):
            pass
        if bool(uncertainty_payload.get("instability_flag")) or bool(
            uncertainty_payload.get("tail_instability")
        ):
            state["tail_instability"] = True

        baseline = uncertainty_payload.get("baseline")
        if isinstance(baseline, dict):
            cv_raw = baseline.get("total_process_cv")
            try:
                if cv_raw is not None and float(cv_raw) >= 0.35:
                    state["high_process_uncertainty"] = True
            except (TypeError, ValueError):
                pass

        tail_model = uncertainty_payload.get("tail_model")
        if isinstance(tail_model, dict):
            if bool(tail_model.get("instability_flag")):
                state["tail_instability"] = True

    @staticmethod
    def _apply_narrative_guardrails(
        content: str,
        state: dict[str, bool],
        execution_records: list[dict[str, Any]] | None = None,
        narration_packet: dict[str, Any] | None = None,
    ) -> str:
        guarded = AssistantService._strip_control_blocks(content).strip()
        if not guarded:
            return guarded

        guarded = AssistantService._apply_execution_narrative_guardrails(
            guarded,
            execution_records,
        )
        guarded = AssistantService._apply_proposal_narrative_guardrails(
            guarded,
            narration_packet,
        )

        coherence_claim = re.search(
            r"paid\s+and\s+incurred\s+(are|is)\s+(consistent|aligned)",
            guarded,
            flags=re.IGNORECASE,
        )
        if state.get("paid_incurred_conflict") and coherence_claim is not None:
            return (
                "Guardrail: narrative blocked because it claims paid/incurred consistency while "
                "coherence diagnostics indicate conflict. Re-run with evidence-constrained wording."
            )

        if state.get("portfolio_shift_unconfirmed"):
            guarded = re.sub(
                r"\bportfolio shift\b",
                "possible portfolio shift signal",
                guarded,
                flags=re.IGNORECASE,
            )
            causal_pattern = re.search(
                r"(because|due to|driven by|caused by)",
                guarded,
                flags=re.IGNORECASE,
            )
            if causal_pattern is not None:
                guarded = (
                    "Guardrail notice: shift evidence is mixed; causal attribution is not confirmed. "
                    + guarded
                )

        if state.get("low_confidence") and "uncertain" not in guarded.lower():
            guarded += "\n\nUncertainty note: diagnostic confidence is reduced; treat recommendations as provisional and review before sign-off."

        if state.get("tail_instability") and "tail instability" not in guarded.lower():
            guarded += "\n\nTail uncertainty note: tail scenario dispersion indicates instability; avoid over-confident tail curve selection and document rationale for chosen tail assumptions."

        if (
            state.get("high_process_uncertainty")
            and "process variability" not in guarded.lower()
        ):
            guarded += "\n\nProcess variability note: aggregate reserve variability is elevated (high process CV); communicate a range-based view using bootstrap quantiles rather than point estimates only."

        if "z-score" in guarded.lower() and "how far" not in guarded.lower():
            guarded += "\n\nPlain-language note: a z-score measures how far a result is from the typical range; bigger absolute values mean the year looks more unusual."

        if "evidence:" in guarded.lower() and "evidence id" not in guarded.lower():
            guarded += "\n\nReference note: each evidence ID points to a specific diagnostic record shown in the analysis trace or evidence panel, including the metric tested and the observed value."

        return guarded

    @staticmethod
    def _apply_proposal_narrative_guardrails(
        content: str,
        narration_packet: dict[str, Any] | None,
    ) -> str:
        packet = narration_packet if isinstance(narration_packet, dict) else {}
        proposal = packet.get("proposal") if isinstance(packet.get("proposal"), dict) else {}
        if bool(proposal.get("exists")):
            return content
        lowered = content.lower()
        pending_claim = "pending" in lowered and "proposal" in lowered
        accept_request = "would you like to accept" in lowered or "accept this" in lowered
        if not pending_claim and not accept_request:
            return content
        basis = packet.get("basis") if isinstance(packet.get("basis"), dict) else {}
        accepted = (
            basis.get("accepted_basis")
            if isinstance(basis.get("accepted_basis"), dict)
            else {}
        )
        label = str(
            accepted.get("scenario_label")
            or accepted.get("candidate_id")
            or accepted.get("scenario_id")
            or "the current Analysis Basis"
        ).strip()
        key = str(accepted.get("basis_key") or "").strip()
        suffix = f" (basis key: `{key[:8]}`)" if key else ""
        correction = (
            "Proposal status: no pending basis proposal is attached to this answer. "
            f"The current accepted Analysis Basis is {label}{suffix}."
        )
        cleaned = re.sub(
            r"\n{0,2}\*{0,2}Proposal Status:?\*{0,2}.*?(?=\n#{1,3}\s|\Z)",
            "\n\n" + correction,
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if correction.lower() not in cleaned.lower():
            cleaned = cleaned.rstrip() + "\n\n" + correction
        return cleaned

    @staticmethod
    def _apply_execution_narrative_guardrails(
        content: str,
        execution_records: list[dict[str, Any]] | None,
    ) -> str:
        latest_record = latest_execution_record(execution_records)
        if not latest_record:
            return content
        status = str(latest_record.get("execution_status") or "").strip()
        if execution_status_is_successful(status):
            return content
        success_style_pattern = re.search(
            r"\b(applied|uses|used|ran successfully|executed successfully|completed successfully|succeeded)\b",
            content,
            flags=re.IGNORECASE,
        )
        if status == "rejected":
            return (
                "Execution note: the latest tool request was rejected, so no success-style scenario change was executed. "
                "Describe the rejection and any blocking input issue explicitly.\n\n"
                + content
            )
        if success_style_pattern is not None:
            return (
                "Execution note: the latest tool run did not execute exactly as requested. "
                "Base the explanation on the effective executed inputs and explicit adjustments, not the original request wording.\n\n"
                + content
            )
        warnings = latest_record.get("warnings")
        if isinstance(warnings, list) and warnings:
            return (
                "Execution note: the latest tool run required input adjustments. "
                + "; ".join(str(item) for item in warnings if str(item).strip())
                + ".\n\n"
                + content
            )
        return content

    @staticmethod
    def _strip_control_blocks(content: str) -> str:
        cleaned = re.sub(
            r"<system-reminder>.*?</system-reminder>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned

    @staticmethod
    def _short_json(payload: dict[str, Any], max_len: int = 280) -> str:
        raw = json.dumps(payload, default=str, ensure_ascii=True)
        if len(raw) <= max_len:
            return raw
        return raw[: max_len - 3] + "..."

    @staticmethod
    def _summarize_tool_result(result: dict[str, Any]) -> str:
        return json.dumps(result, ensure_ascii=True)

    @staticmethod
    def _build_deterministic_fallback_commentary(
        *,
        session_id: str | None,
        diagnostics: dict[str, Any] | None,
        iteration: dict[str, Any] | None,
        results: dict[str, Any] | None,
    ) -> str:
        sections: list[str] = [
            "Model provider is temporarily unavailable, so this is a deterministic summary from completed tool outputs.",
        ]

        if session_id:
            sections.append(f"session_id={session_id}")

        if isinstance(diagnostics, dict):
            findings = diagnostics.get("top_findings")
            recommendations = diagnostics.get("top_recommendations")
            governance = diagnostics.get("governance")
            metrics = diagnostics.get("metrics")

            finding_count = int(diagnostics.get("finding_count") or 0)
            recommendation_count = int(diagnostics.get("recommendation_count") or 0)
            tier = "unknown"
            requires_review = None
            if isinstance(governance, dict):
                tier = str(governance.get("tier", "unknown")).upper()
                requires_review = governance.get("requires_human_review")

            diagnostic_line = f"Diagnostics: {finding_count} findings, {recommendation_count} recommendations, governance tier {tier}"
            if requires_review is not None:
                diagnostic_line += (
                    ", human review required"
                    if bool(requires_review)
                    else ", no mandatory human review flag"
                )
            sections.append(diagnostic_line + ".")

            if isinstance(findings, list) and findings:
                top_findings = []
                for item in findings[:5]:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("code", "unknown"))
                    severity = str(item.get("severity", "unknown"))
                    message = str(item.get("message", "")).strip()
                    top_findings.append(f"- [{severity}] {code}: {message}")
                if top_findings:
                    sections.append("Top findings:\n" + "\n".join(top_findings))

            if isinstance(recommendations, list) and recommendations:
                top_recommendations = []
                for item in recommendations[:5]:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("code", "unknown"))
                    priority = str(item.get("priority", "unknown"))
                    message = str(item.get("message", "")).strip()
                    top_recommendations.append(f"- [{priority}] {code}: {message}")
                if top_recommendations:
                    sections.append(
                        "Top recommendations:\n" + "\n".join(top_recommendations)
                    )

            uncertainty = diagnostics.get("uncertainty")
            if isinstance(uncertainty, dict):
                notes: list[str] = []
                if uncertainty.get("process_cv") is not None:
                    notes.append(f"process_cv={uncertainty.get('process_cv')}")
                if uncertainty.get("bootstrap_p50") is not None:
                    notes.append(f"bootstrap_p50={uncertainty.get('bootstrap_p50')}")
                if uncertainty.get("bootstrap_p90") is not None:
                    notes.append(f"bootstrap_p90={uncertainty.get('bootstrap_p90')}")
                if uncertainty.get("tail_instability") is not None:
                    notes.append(
                        f"tail_instability={uncertainty.get('tail_instability')}"
                    )
                if notes:
                    sections.append("Uncertainty: " + "; ".join(notes) + ".")

        if isinstance(iteration, dict):
            scenarios = iteration.get("top_scenarios")
            iteration_metrics = iteration.get("iteration_metrics")
            scenario_count = int(iteration.get("scenario_count") or 0)
            best_scenario_id = None
            if isinstance(iteration_metrics, dict):
                best_scenario_id = iteration_metrics.get("best_scenario_id")
            sections.append(
                f"Scenario search: evaluated {scenario_count} scenarios; best scenario={best_scenario_id or 'none'}."
            )
            if isinstance(scenarios, list) and scenarios:
                best_item = None
                if best_scenario_id is not None:
                    for item in scenarios:
                        if (
                            isinstance(item, dict)
                            and item.get("scenario_id") == best_scenario_id
                        ):
                            best_item = item
                            break
                if best_item is None and isinstance(scenarios[0], dict):
                    best_item = scenarios[0]
                if isinstance(best_item, dict):
                    summary = str(best_item.get("summary", "")).strip()
                    score = best_item.get("score")
                    if summary or score is not None:
                        best_line = "Best scenario detail:"
                        if score is not None:
                            best_line += f" score={score}."
                        if summary:
                            best_line += f" {summary}"
                        sections.append(best_line)

        if isinstance(results, dict):
            row_count = results.get("result_row_count")
            if row_count is not None:
                sections.append(
                    f"Results snapshot: {row_count} underwriting years in the latest results table."
                )

        sections.append(
            "Retry later if you want a model-written narrative, but the diagnostics and scenario search above completed successfully."
        )
        return "\n\n".join(section for section in sections if section.strip())
