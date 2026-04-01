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
from ai.tool_payloads import build_memory_snapshot, build_tool_specs, render_memory_hint
from ai.tool_contract import normalize_tool_result


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
        segment_memory = self._load_segment_memory(session_context)
        segment_memory_hint = self._build_segment_memory_hint(segment_memory)
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
            existing_scenario_ledger=(working_memory or {}).get("scenario_ledger")
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
        }
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
                    }
                if isinstance(content, list):
                    parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ]
                    merged = "\n".join(part for part in parts if part)
                    self._emit_streamed_content(event_callback, merged)
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
        plan = planner.plan(
            user_prompt=user_prompt,
            session_context=session_context,
            segment_memory=segment_memory,
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
        return build_deterministic_packet(
            plan=plan.to_dict(),
            review=review_outcome.to_dict(),
            recommendation=recommendation.to_dict(),
            evidence_packets=envelopes,
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
    def _build_deterministic_packet_prompt(packet: dict[str, Any]) -> str:
        return (
            "Deterministic control packet already prepared for this turn. "
            "Use it as the primary evidence frame for the answer. Do not contradict its review status or recommendation status.\n"
            + json.dumps(packet, ensure_ascii=True)
        )

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
        rejected = segment_memory.get("rejected_scenarios")
        if isinstance(rejected, list) and rejected:
            parts.append(
                "Previously rejected scenarios: "
                + "; ".join(
                    str(item.get("scenario_hash") or item.get("scenario_id"))
                    for item in rejected[:3]
                    if isinstance(item, dict)
                )
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
                "Use diagnostics, a2a/LDF consistency, and incurred/premium context."
            )
        if playbook == "tail_selection":
            return (
                "Selected playbook: Tail Selection. "
                "Use tested tail-fit evaluation before recommending or comparing tail methods. "
                "Proactively comment on sub-1 late selected LDFs, whether the tail smooths them from above, and whether the attachment creates too sharp a cut from the previous selected LDF."
            )
        if playbook == "data_anomaly_triage":
            return (
                "Selected playbook: Data Anomaly Triage. "
                "Lead with diagnostics, movement evidence, and LDF consistency before any parameter recommendation."
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
            "tool_compare_data_views": "Comparing data views",
            "tool_run_diagnostics": "Running diagnostics",
            "tool_run_diagnostics_summary": "Running diagnostics",
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
            current["session_summary"] = dict(tool_result)
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
                existing_scenario_ledger=current.get("scenario_ledger"),
            )
        elif function_name in {"tool_get_results_summary", "tool_recalculate"}:
            current["results_summary"] = dict(tool_result)
        elif function_name == "tool_explain_reserve_change":
            current["reserve_change_summary"] = dict(tool_result)
        elif function_name in {
            "tool_run_highest_a2a_drop_scenario",
            "tool_run_derived_drop_scenario",
        }:
            current["reserve_change_summary"] = dict(tool_result)
        elif function_name == "tool_rank_link_ratios":
            current["data_view_summary"] = dict(tool_result)
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
