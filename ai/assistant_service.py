from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ai.api_tools import ReservingApiTools
from ai.openrouter_client import OpenRouterClient


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an actuarial reserving diagnostics assistant. "
    "Always prefer tool calls over assumptions. "
    "Ground all material statements in tool outputs. "
    "Include checks for latest diagonal actual-vs-expected emergence and incurred-on-premium development at matched ages. "
    "When available, run iterative scenario analysis using tool_iterate_diagnostics "
    "before final recommendations on drops, tail fitting, and BF apriori. "
    "Include uncertainty interpretation (MSEP/error, bootstrap quantiles, tail instability/model averaging) when available. "
    "If evidence is missing, say so explicitly."
)


class AssistantService:
    def __init__(self, *, api_base_url: str) -> None:
        self._client = OpenRouterClient()
        self._tools = ReservingApiTools(base_url=api_base_url)
        self._observability_enabled = os.environ.get(
            "AI_OBSERVABILITY", "1"
        ).strip().lower() not in {"0", "false", "off"}

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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
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
            "forced_iteration_prompt": False,
        }

        for _ in range(max_steps):
            if self._observability_enabled:
                logger.info(
                    "[OBS] ai.step request_openrouter messages=%s", len(messages)
                )
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    tools=tool_specs,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=1200,
                )
            except RuntimeError as error:
                if workflow_state.get("ran_diagnostics"):
                    session_id = workflow_state.get("session_id")
                    suffix = (
                        f" session_id={session_id}"
                        if isinstance(session_id, str)
                        else ""
                    )
                    return (
                        "Model provider is temporarily unavailable after deterministic tool execution. "
                        "Diagnostics and scenario evaluation completed successfully; "
                        "retry to generate narrative commentary." + suffix
                    )
                raise error
            choice = response["choices"][0]["message"]
            tool_calls = choice.get("tool_calls") or []
            if self._observability_enabled:
                logger.info("[OBS] ai.step tool_calls=%s", len(tool_calls))

            if not tool_calls:
                if (
                    workflow_state.get("ran_diagnostics")
                    and not workflow_state.get("ran_iteration")
                    and workflow_state.get("session_id")
                    and not workflow_state.get("forced_iteration_prompt")
                ):
                    workflow_state["forced_iteration_prompt"] = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Before finalizing recommendations, run tool_iterate_diagnostics "
                                f"for session_id={workflow_state['session_id']} with include_baseline=true. "
                                "Then summarize robust scenario tradeoffs with evidence."
                            ),
                        }
                    )
                    continue

                content = choice.get("content")
                if isinstance(content, str):
                    return self._apply_narrative_guardrails(content, guardrail_state)
                if isinstance(content, list):
                    parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ]
                    merged = "\n".join(part for part in parts if part)
                    return self._apply_narrative_guardrails(merged, guardrail_state)
                return "No response content was produced by the model."

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

                tool_result = self._tools.call_tool(function_name, args)
                self._update_workflow_state(
                    workflow_state=workflow_state,
                    function_name=function_name,
                    args=args,
                    tool_result=tool_result,
                )
                self._update_guardrail_state(guardrail_state, tool_result)
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
                        "content": json.dumps(tool_result),
                    }
                )

        return (
            "Tool-call step limit reached before the assistant produced a final answer. "
            "Please retry with a narrower question."
        )

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

        if function_name == "tool_run_diagnostics":
            workflow_state["ran_diagnostics"] = True
        if function_name == "tool_iterate_diagnostics":
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

        scenarios = tool_result.get("scenarios")
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
        try:
            if cv_raw is not None and float(cv_raw) >= 0.35:
                state["high_process_uncertainty"] = True
        except (TypeError, ValueError):
            pass
        if bool(uncertainty_payload.get("instability_flag")):
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
        keys = sorted(result.keys())
        summary: dict[str, Any] = {"keys": keys[:8]}
        if "findings" in result and isinstance(result.get("findings"), list):
            summary["finding_count"] = len(result["findings"])
        if "recommendations" in result and isinstance(
            result.get("recommendations"), list
        ):
            summary["recommendation_count"] = len(result["recommendations"])
        if "scenarios" in result and isinstance(result.get("scenarios"), list):
            summary["scenario_count"] = len(result["scenarios"])
        if "iteration_metrics" in result and isinstance(
            result.get("iteration_metrics"), dict
        ):
            metrics = result["iteration_metrics"]
            summary["duration_ms"] = metrics.get("duration_ms")
            summary["best_scenario_id"] = metrics.get("best_scenario_id")
        return json.dumps(summary, ensure_ascii=True)
