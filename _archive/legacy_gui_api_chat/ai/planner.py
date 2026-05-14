from __future__ import annotations

from typing import Any

from ai.control_plane_types import normalize_accepted_analysis_basis
from ai.plan_models import ExecutionPlan, PlanStep
from ai.playbook_registry import build_execution_plan
from ai.workflow_definitions import WorkflowDefinition, select_workflow_definitions


class PlaybookPlanner:
    def select_playbook(self, user_prompt: str) -> str:
        definitions = select_workflow_definitions(user_prompt)
        if len(definitions) > 1:
            return "multi_review"
        return definitions[0].workflow_name if definitions else ""

    def plan(
        self,
        *,
        user_prompt: str,
        session_context: dict[str, Any] | None,
        segment_memory: dict[str, Any] | None = None,
        analysis_basis: dict[str, Any] | None = None,
    ) -> ExecutionPlan | None:
        definitions = select_workflow_definitions(user_prompt)
        if not definitions:
            return None
        session_id = None
        segment = None
        if isinstance(session_context, dict):
            raw_session_id = session_context.get("session_id")
            raw_segment = session_context.get("segment")
            if isinstance(raw_session_id, str) and raw_session_id.strip():
                session_id = raw_session_id.strip()
            if isinstance(raw_segment, str) and raw_segment.strip():
                segment = raw_segment.strip()
        if len(definitions) > 1:
            return _build_multi_workflow_plan(
                definitions=definitions,
                session_id=session_id,
                segment=segment,
                segment_memory=segment_memory,
                analysis_basis=analysis_basis,
            )
        playbook = definitions[0].workflow_name
        return build_execution_plan(
            playbook=playbook,
            session_id=session_id,
            segment=segment,
            segment_memory=segment_memory,
            analysis_basis=analysis_basis,
        )


def _build_multi_workflow_plan(
    *,
    definitions: tuple[WorkflowDefinition, ...],
    session_id: str | None,
    segment: str | None,
    segment_memory: dict[str, Any] | None,
    analysis_basis: dict[str, Any] | None,
) -> ExecutionPlan | None:
    if not session_id:
        return None
    steps: list[PlanStep] = []
    seen_steps: set[tuple[str, str]] = set()
    required_capabilities: list[str] = []
    required_evidence: list[str] = []
    basis_behavior: set[str] = {"use_accepted_basis", "proposal_disallowed"}
    requires_continuity = False
    for definition in definitions:
        required_capabilities.extend(definition.required_capabilities)
        required_evidence.extend(definition.required_evidence)
        requires_continuity = requires_continuity or definition.requires_continuity
        for step in definition.steps:
            identity = (step.tool_name, step.evidence_key)
            if identity in seen_steps:
                continue
            seen_steps.add(identity)
            steps.append(
                PlanStep(
                    tool_name=step.tool_name,
                    args=_step_args(
                        session_id=session_id,
                        default_args=step.default_args,
                        analysis_basis=analysis_basis,
                        basis_aware=step.basis_aware,
                    ),
                    evidence_key=step.evidence_key,
                    basis_aware=step.basis_aware,
                )
            )
    workflow_names = [definition.workflow_name for definition in definitions]
    return ExecutionPlan(
        playbook="multi_review",
        workflow_name="multi_review",
        goal="Run the matched deterministic workflows: " + ", ".join(workflow_names) + ".",
        segment=segment,
        session_id=session_id,
        intent_class="multi_intent_review",
        required_capabilities=_dedupe(required_capabilities),
        steps=steps,
        required_evidence=_dedupe(required_evidence),
        minimum_evidence_count=len(_dedupe(required_evidence)),
        stopping_rule="Stop after all matched deterministic workflow evidence is collected.",
        basis_behavior=sorted(basis_behavior),
        answer_contract="review_summary_only",
        requires_continuity=requires_continuity,
    )


def _step_args(
    *,
    session_id: str,
    default_args: dict[str, Any],
    analysis_basis: dict[str, Any] | None,
    basis_aware: bool,
) -> dict[str, Any]:
    merged = {"session_id": session_id, **dict(default_args)}
    if not basis_aware or not isinstance(analysis_basis, dict) or not analysis_basis:
        return merged
    normalized_basis = normalize_accepted_analysis_basis(analysis_basis)
    basis_type = normalized_basis.get("basis_type")
    basis_key = normalized_basis.get("basis_key")
    parameters = normalized_basis.get("parameters")
    if isinstance(basis_type, str) and basis_type.strip():
        merged["basis_type"] = basis_type.strip()
    if isinstance(basis_key, str) and basis_key.strip():
        merged["basis_key"] = basis_key.strip()
    if isinstance(parameters, dict) and parameters:
        merged["parameters"] = dict(parameters)
    return merged


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
