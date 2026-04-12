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
from ai.context_loader import load_segment_note
from ai.deterministic_packet import build_deterministic_packet
from ai.memory_store import SegmentMemoryStore
from ai.openrouter_client import OpenRouterClient
from ai.planner import PlaybookPlanner
from ai.recommendation_policy import RecommendationPolicy
from ai.reviewer import ReviewerGate
from ai.tool_payloads import (
    build_baseline_analysis_basis,
    build_memory_snapshot,
    build_tool_specs,
    render_memory_hint,
)
from ai.tool_contract import normalize_tool_result
from source.services.memory_authoring_service import MemoryAuthoringService
from source.services.segment_memory_service import SegmentMemoryService


logger = logging.getLogger(__name__)


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
        working_memory: dict[str, Any] | None = None,
        event_callback: Any | None = None,
        max_steps: int = 14,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if AI_CONTEXT_PROMPT:
            messages.append({"role": "system", "content": AI_CONTEXT_PROMPT})
        if AI_PLAYBOOKS_PROMPT:
            messages.append({"role": "system", "content": AI_PLAYBOOKS_PROMPT})
        if AI_EXAMPLES_PROMPT:
            messages.append({"role": "system", "content": AI_EXAMPLES_PROMPT})
        if AI_POLICY_PROMPT:
            messages.append({"role": "system", "content": AI_POLICY_PROMPT})
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
            existing_analysis_basis=(working_memory or {}).get("analysis_basis")
            if isinstance(working_memory, dict)
            else None,
            existing_scenario_basis_cache=(working_memory or {}).get(
                "scenario_basis_cache"
            )
            if isinstance(working_memory, dict)
            else None,
        )
        tool_specs = self._tools.tool_specs
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
            analysis_basis = self._bind_analysis_basis_to_packet(
                memory_state=memory_state,
                deterministic_packet=deterministic_packet,
                session_context=session_context,
            )
            if analysis_basis:
                memory_state["analysis_basis"] = analysis_basis
            messages.append(
                {
                    "role": "system",
                    "content": self._build_deterministic_packet_prompt(
                        deterministic_packet
                    ),
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
            )
            if self._observability_enabled:
                logger.info(
                    "[OBS] ai.step request_openrouter messages=%s tools=%s",
                    len(messages),
                    len(available_tool_specs),
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
                            "session_id": session_id,
                            "tool_events": tool_events,
                            "memory_snapshot": memory_state,
                            "deterministic_packet": memory_state.get(
                                "deterministic_packet", {}
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
                        "session_id": session_id,
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
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
                    "session_id": workflow_state.get("session_id"),
                    "tool_events": tool_events,
                    "memory_snapshot": memory_state,
                    "deterministic_packet": memory_state.get(
                        "deterministic_packet", {}
                    ),
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
                            "memory_update_proposals": memory_state.get(
                                "memory_update_proposals", []
                            ),
                        }
                    return {
                        "content": self._apply_narrative_guardrails(
                            content,
                            guardrail_state,
                        ),
                        "fallback_used": False,
                        "session_id": workflow_state.get("session_id"),
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
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
                            "memory_update_proposals": memory_state.get(
                                "memory_update_proposals", []
                            ),
                        }
                    return {
                        "content": self._apply_narrative_guardrails(
                            merged,
                            guardrail_state,
                        ),
                        "fallback_used": False,
                        "session_id": workflow_state.get("session_id"),
                        "tool_events": tool_events,
                        "memory_snapshot": memory_state,
                        "deterministic_packet": memory_state.get(
                            "deterministic_packet", {}
                        ),
                        "memory_update_proposals": memory_state.get(
                            "memory_update_proposals", []
                        ),
                    }
                return {
                    "content": "No response content was produced by the model.",
                    "fallback_used": False,
                    "session_id": workflow_state.get("session_id"),
                    "tool_events": tool_events,
                    "memory_snapshot": memory_state,
                    "deterministic_packet": memory_state.get(
                        "deterministic_packet", {}
                    ),
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": function_name,
                        "content": json.dumps(tool_result, ensure_ascii=True),
                    }
                )

        return {
            "content": (
                "Tool-call step limit reached before the assistant produced a final answer. "
                "Please retry with a narrower question."
            ),
            "fallback_used": False,
            "session_id": workflow_state.get("session_id"),
            "tool_events": tool_events,
            "memory_snapshot": memory_state,
            "deterministic_packet": memory_state.get("deterministic_packet", {}),
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
        planner = getattr(self, "_planner", None) or PlaybookPlanner()
        prompt_text = str(user_prompt or "").strip().lower()
        planning_basis = self._basis_for_recommendation_turn(
            prompt=prompt_text,
            memory_state=memory_state,
            session_context=session_context,
        )
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
            if self._observability_enabled:
                logger.info(
                    "[OBS] deterministic.step.execute playbook=%s tool=%s evidence_key=%s",
                    plan.playbook,
                    step.tool_name,
                    step.evidence_key,
                )
            tool_result, updated_memory = self._execute_tool_call(
                function_name=step.tool_name,
                args=step.args,
                tool_outputs=tool_outputs,
                tool_events=tool_events,
                memory_state=memory_state,
                workflow_state=workflow_state,
                guardrail_state=guardrail_state,
                event_callback=event_callback,
            )
            memory_state.clear()
            memory_state.update(updated_memory)
            envelopes.append(
                normalize_tool_result(
                    tool_name=step.tool_name,
                    args=step.args,
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
                "[OBS] deterministic.recommendation.completed playbook=%s status=%s scenario_id=%s alternatives=%s",
                plan.playbook,
                recommendation.status,
                recommendation.recommended_scenario_id,
                len(recommendation.alternative_scenario_ids),
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

    def _filter_tool_specs_for_turn(
        self,
        *,
        tool_specs: list[dict[str, Any]],
        deterministic_packet: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(deterministic_packet, dict) or not deterministic_packet:
            return tool_specs
        plan = (
            deterministic_packet.get("plan", {})
            if isinstance(deterministic_packet.get("plan"), dict)
            else {}
        )
        review = (
            deterministic_packet.get("review", {})
            if isinstance(deterministic_packet.get("review"), dict)
            else {}
        )
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        if not steps:
            return tool_specs
        if str(review.get("status", "")).strip().lower() == "hard_fail":
            return tool_specs

        suppressed_names = {
            str(step.get("tool_name", "")).strip()
            for step in steps
            if isinstance(step, dict) and str(step.get("tool_name", "")).strip()
        }
        if not suppressed_names:
            return tool_specs

        filtered: list[dict[str, Any]] = []
        for spec in tool_specs:
            if not isinstance(spec, dict):
                continue
            function = spec.get("function")
            if not isinstance(function, dict):
                filtered.append(spec)
                continue
            name = str(function.get("name", "")).strip()
            if name in suppressed_names:
                continue
            filtered.append(spec)
        if self._observability_enabled and len(filtered) != len(tool_specs):
            logger.info(
                "[OBS] deterministic.tools.suppressed count=%s names=%s",
                len(tool_specs) - len(filtered),
                sorted(suppressed_names),
            )
        return filtered

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
        tool_outputs[function_name] = tool_result
        next_memory = self._update_memory_state(
            memory_state,
            function_name=function_name,
            tool_result=tool_result,
        )
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
        parameter_field = (
            "basis_parameters"
            if function_name == "tool_explain_reserve_change"
            else "parameters"
        )
        has_basis_type = args.get("basis_type") not in (None, "", {})
        has_scenario_id = args.get("scenario_id") not in (None, "", {})
        has_parameters = args.get(parameter_field) not in (None, "", {})
        if has_basis_type and has_scenario_id and has_parameters:
            return args
        has_any_basis_arg = has_basis_type or has_scenario_id or has_parameters

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
        scenario_id = basis.get("scenario_id")
        parameters = basis.get("parameters")
        if not has_basis_type and isinstance(basis_type, str) and basis_type.strip():
            enriched["basis_type"] = basis_type.strip()
        if not has_scenario_id and isinstance(scenario_id, str) and scenario_id.strip():
            enriched["scenario_id"] = scenario_id.strip()
        if not has_parameters and isinstance(parameters, dict) and parameters:
            if function_name == "tool_explain_reserve_change":
                enriched["basis_parameters"] = parameters
            else:
                enriched["parameters"] = parameters
        return enriched

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
            return AssistantService._baseline_basis_from_memory(
                memory_state=memory_state,
                session_context=None,
            )

        scenario_id = str(args.get("scenario_id") or "").strip()
        if scenario_id:
            if scenario_id == "baseline":
                return AssistantService._baseline_basis_from_memory(
                    memory_state=memory_state,
                    session_context=None,
                )
            basis_cache = (
                memory_state.get("scenario_basis_cache")
                if isinstance(memory_state.get("scenario_basis_cache"), dict)
                else {}
            )
            cached = basis_cache.get(scenario_id)
            if isinstance(cached, dict) and cached:
                return dict(cached)

        analysis_basis = (
            memory_state.get("analysis_basis")
            if isinstance(memory_state.get("analysis_basis"), dict)
            else {}
        )
        if analysis_basis:
            return dict(analysis_basis)
        if has_any_basis_arg:
            return {}
        return AssistantService._baseline_basis_from_memory(
            memory_state=memory_state,
            session_context=None,
        )

    @staticmethod
    def _build_deterministic_packet_prompt(packet: dict[str, Any]) -> str:
        return (
            "Deterministic control packet already prepared for this turn. "
            "Use it as the primary evidence frame for the answer. Do not contradict its review status or recommendation status.\n"
            + json.dumps(packet, ensure_ascii=True)
        )

    @staticmethod
    def _bind_analysis_basis_to_packet(
        *,
        memory_state: dict[str, Any],
        deterministic_packet: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        recommendation = (
            deterministic_packet.get("recommendation")
            if isinstance(deterministic_packet.get("recommendation"), dict)
            else {}
        )
        recommended_scenario_id = str(
            recommendation.get("recommended_scenario_id") or ""
        ).strip()
        basis_cache = (
            memory_state.get("scenario_basis_cache")
            if isinstance(memory_state.get("scenario_basis_cache"), dict)
            else {}
        )
        analysis_basis: dict[str, Any] = {}
        if recommended_scenario_id and isinstance(
            basis_cache.get(recommended_scenario_id), dict
        ):
            analysis_basis = dict(basis_cache[recommended_scenario_id])
        elif recommended_scenario_id in {"", "baseline"}:
            analysis_basis = AssistantService._baseline_basis_from_memory(
                memory_state=memory_state,
                session_context=session_context,
            )
        elif isinstance(memory_state.get("analysis_basis"), dict):
            analysis_basis = dict(memory_state.get("analysis_basis", {}))
        if not analysis_basis:
            analysis_basis = AssistantService._baseline_basis_from_memory(
                memory_state=memory_state,
                session_context=session_context,
            )
        deterministic_packet["analysis_basis"] = analysis_basis
        presentation = (
            deterministic_packet.get("presentation")
            if isinstance(deterministic_packet.get("presentation"), dict)
            else {}
        )
        presentation["basis_confirmation_line"] = (
            AssistantService._analysis_basis_label(analysis_basis)
        )
        deterministic_packet["presentation"] = presentation
        return analysis_basis

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
        analysis_basis = memory_state.get("analysis_basis")
        if isinstance(analysis_basis, dict):
            next_memory["last_analysis_basis"] = analysis_basis
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
                        "Do not invent tables, vectors, or quoted values outside this payload. If a requested value is missing here, say it is not verified from current tool outputs.\n"
                        + json.dumps(tool_result, ensure_ascii=True)
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
        playbook = AssistantService._select_playbook(prompt)
        if playbook == "movement_review":
            return (
                "Selected playbook: Movement Review. "
                "Use the Movement Review workflow from AI_PLAYBOOKS.md. "
                "Start with data-view summaries and movement-focused evidence, then answer directly."
            )
        if playbook == "quarter_close_review":
            return (
                "Selected playbook: Quarter-Close Review. "
                "Use the composite deterministic quarter-close review first, then use drilldown tools only for follow-up evidence."
            )
        if playbook == "drop_review":
            return (
                "Selected playbook: Drop Review. "
                "Use the composite drop review first and treat its ranked candidates, continuity notes, and policy trace as the primary evidence base."
            )
        if playbook == "scenario_recommendation":
            return (
                "Selected playbook: Scenario Recommendation. "
                "Use the Scenario Recommendation workflow from AI_PLAYBOOKS.md. "
                "Favor diagnostics plus scenario iteration before recommending changes."
            )
        if playbook == "reserve_change_explanation":
            return (
                "Selected playbook: Reserve Change Explanation. "
                "Use attribution against baseline before broad scenario discussion."
            )
        if playbook == "late_emergence_review":
            return (
                "Selected playbook: Late Emergence Review. "
                "Use historical continuation evidence before broad recommendations."
            )
        if playbook == "method_suitability_review":
            return (
                "Selected playbook: Method Suitability Review. "
                "Use the composite BF suitability review first and treat its UWY-level suitability conclusions as the primary evidence base."
            )
        if playbook == "tail_selection":
            return (
                "Selected playbook: Tail Selection. "
                "Use the composite tail review first before drilldown tail-fit testing. "
                "Proactively comment on sub-1 late selected LDFs, whether the tail smooths them from above, and whether the attachment creates too sharp a cut from the previous selected LDF."
            )
        if playbook == "data_anomaly_triage":
            return (
                "Selected playbook: Data Anomaly Triage. "
                "Lead with the composite anomaly triage result before any parameter recommendation."
            )
        if playbook == "data_exploration":
            return (
                "Selected playbook: Data Exploration. "
                "Use summary data tools first and only request detailed rows if needed."
            )
        return ""

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
        if any(
            keyword in prompt
            for keyword in {
                "quarter close",
                "quarter-close",
                "close pack",
                "close review",
                "quarterly review pack",
            }
        ):
            return "quarter_close_review"
        if AssistantService._is_movement_question(prompt):
            return "movement_review"
        if any(
            keyword in prompt
            for keyword in {
                "why did reserve",
                "why does reserve",
                "explain reserve change",
                "driver of reserve",
                "reserve change",
                "impact on reserve",
            }
        ):
            return "reserve_change_explanation"
        if any(
            keyword in prompt
            for keyword in {
                "data quality",
                "anomaly",
                "triage",
                "missing diagonal",
                "impossible link ratio",
                "calendar year distortion",
                "large loss contamination",
            }
        ):
            return "data_anomaly_triage"
        if any(
            keyword in prompt
            for keyword in {
                "drop review",
                "drop any ratios",
                "which ratios should be dropped",
                "which ratio should be dropped",
                "should be dropped",
                "should we drop",
                "should i drop",
                "drop ratios",
            }
        ):
            return "drop_review"
        if any(
            keyword in prompt
            for keyword in {
                "how much more",
                "still emerge",
                "late emergence",
                "still come",
                "still develop",
            }
        ):
            return "late_emergence_review"
        if any(
            keyword in prompt
            for keyword in {
                "tail",
                "weibull",
                "inverse power",
                "inverse_power",
                "exponential",
                "r2",
                "fit period",
                "tail fit",
            }
        ):
            return "tail_selection"
        if any(
            keyword in prompt
            for keyword in {
                "cl vs bf",
                "chainladder vs bf",
                "bornhuetter",
                "method suitable",
                "bf better",
                "chainladder better",
            }
        ):
            return "method_suitability_review"
        if AssistantService._is_recommendation_question(prompt):
            return "scenario_recommendation"
        if any(
            keyword in prompt
            for keyword in {
                "show me",
                "compare data",
                "triangle",
                "view",
                "ratio",
                "table",
            }
        ):
            return "data_exploration"
        return ""

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
        request_terms = {
            "show me",
            "compare",
            "side by side",
            "what is",
            "what are",
            "list",
            "table",
            "vector",
            "values",
            "factors",
            "rows",
        }
        subject_terms = {
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
        scenario_id = basis.get("scenario_id")
        parameters = basis.get("parameters")
        if isinstance(basis_type, str) and basis_type.strip():
            args["basis_type"] = basis_type.strip()
        if isinstance(scenario_id, str) and scenario_id.strip():
            args["scenario_id"] = scenario_id.strip()
        if isinstance(parameters, dict) and parameters:
            args["parameters"] = parameters
        return args

    @staticmethod
    def _analysis_basis_label(basis: dict[str, Any] | None) -> str:
        if not isinstance(basis, dict) or not basis:
            return "Basis used: current baseline session."
        basis_type = str(basis.get("basis_type") or "").strip().lower()
        scenario_id = str(basis.get("scenario_id") or "").strip()
        if scenario_id and scenario_id != "baseline":
            return f"Basis used: scenario {scenario_id}."
        if basis_type == "bespoke":
            return "Basis used: custom conversation basis."
        return "Basis used: current baseline session."

    @staticmethod
    def _resolve_exact_question_basis(
        *,
        prompt: str,
        memory_state: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        baseline_basis = AssistantService._baseline_basis_from_memory(
            memory_state=memory_state,
            session_context=session_context,
        )
        if AssistantService._prompt_requests_baseline_basis(prompt):
            return baseline_basis
        basis_cache = (
            memory_state.get("scenario_basis_cache")
            if isinstance(memory_state.get("scenario_basis_cache"), dict)
            else {}
        )
        matched_scenario = AssistantService._scenario_id_mentioned_in_prompt(
            prompt=prompt,
            basis_cache=basis_cache,
        )
        if matched_scenario:
            cached = basis_cache.get(matched_scenario)
            if isinstance(cached, dict):
                return dict(cached)
        current_basis = (
            memory_state.get("analysis_basis")
            if isinstance(memory_state.get("analysis_basis"), dict)
            else {}
        )
        if current_basis:
            return dict(current_basis)
        return baseline_basis

    @staticmethod
    def _baseline_basis_from_memory(
        *,
        memory_state: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        session_summary = (
            memory_state.get("session_summary")
            if isinstance(memory_state.get("session_summary"), dict)
            else {}
        )
        baseline_basis = build_baseline_analysis_basis(session_summary)
        if baseline_basis:
            return baseline_basis
        session_id = None
        if isinstance(session_context, dict):
            raw_session_id = session_context.get("session_id")
            if isinstance(raw_session_id, str) and raw_session_id.strip():
                session_id = raw_session_id.strip()
        return {
            "basis_type": "baseline",
            "session_id": session_id or "",
            "scenario_id": "baseline",
            "is_active_session": True,
            "parameters": {},
        }

    @staticmethod
    def _prompt_requests_baseline_basis(prompt: str) -> bool:
        return any(
            phrase in prompt
            for phrase in (
                "baseline",
                "current session",
                "current baseline",
                "active session",
            )
        )

    @staticmethod
    def _scenario_id_mentioned_in_prompt(
        *,
        prompt: str,
        basis_cache: dict[str, Any],
    ) -> str | None:
        for scenario_id in basis_cache:
            candidate = str(scenario_id).strip().lower()
            if candidate and candidate != "baseline" and candidate in prompt:
                return str(scenario_id)
        return None

    @staticmethod
    def _basis_for_recommendation_turn(
        *,
        prompt: str,
        memory_state: dict[str, Any],
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if AssistantService._prompt_requests_baseline_basis(prompt):
            return AssistantService._baseline_basis_from_memory(
                memory_state=memory_state,
                session_context=session_context,
            )
        analysis_basis = (
            memory_state.get("analysis_basis")
            if isinstance(memory_state.get("analysis_basis"), dict)
            else {}
        )
        if analysis_basis:
            return dict(analysis_basis)
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
                existing_analysis_basis=current.get("analysis_basis"),
                existing_scenario_basis_cache=current.get("scenario_basis_cache"),
            )
        elif function_name == "tool_get_assumption_context_detail":
            current["assumption_detail"] = dict(tool_result)
            basis = tool_result.get("analysis_basis")
            if isinstance(basis, dict) and basis:
                current["analysis_basis"] = dict(basis)
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
                existing_analysis_basis=current.get("analysis_basis"),
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
                existing_analysis_basis=current.get("analysis_basis"),
                existing_scenario_basis_cache=current.get("scenario_basis_cache"),
            )
        elif function_name in {
            "tool_run_highest_a2a_drop_scenario",
            "tool_run_derived_drop_scenario",
        }:
            current["reserve_change_summary"] = dict(tool_result)
        elif function_name == "tool_rank_link_ratios":
            current["data_view_summary"] = dict(tool_result)
        basis = tool_result.get("analysis_basis")
        if isinstance(basis, dict) and basis:
            current["analysis_basis"] = dict(basis)
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
        basis_cache["baseline"] = baseline_basis
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
    ) -> str:
        guarded = AssistantService._strip_control_blocks(content).strip()
        if not guarded:
            return guarded

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
