from __future__ import annotations

from typing import Any

from ai.plan_models import ExecutionPlan, PlanStep
from ai.workflow_definitions import get_workflow_definition


def build_execution_plan(
    *,
    playbook: str,
    session_id: str | None,
    segment: str | None,
    segment_memory: dict[str, Any] | None = None,
    analysis_basis: dict[str, Any] | None = None,
) -> ExecutionPlan | None:
    if not session_id:
        return None

    playbook = str(playbook or "").strip().lower()
    definition = get_workflow_definition(playbook)
    if definition is None:
        return None
    steps = [
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
        for step in definition.steps
    ]
    return ExecutionPlan(
        playbook=definition.workflow_name,
        workflow_name=definition.workflow_name,
        goal=definition.goal,
        segment=segment,
        session_id=session_id,
        intent_class=definition.intent_class,
        required_capabilities=list(definition.required_capabilities),
        steps=steps,
        required_evidence=list(definition.required_evidence),
        minimum_evidence_count=definition.minimum_evidence_count,
        stopping_rule=definition.rendered_stopping_rule(segment_memory=segment_memory),
        basis_behavior=list(definition.basis_behavior),
        answer_contract=definition.answer_contract,
        requires_continuity=definition.requires_continuity,
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
    basis_type = analysis_basis.get("basis_type")
    scenario_id = analysis_basis.get("scenario_id")
    parameters = analysis_basis.get("parameters")
    if isinstance(basis_type, str) and basis_type.strip():
        merged["basis_type"] = basis_type.strip()
    if isinstance(scenario_id, str) and scenario_id.strip():
        merged["scenario_id"] = scenario_id.strip()
    if isinstance(parameters, dict) and parameters:
        merged["parameters"] = dict(parameters)
    return merged
