from __future__ import annotations

from typing import Any

from ai.plan_models import ExecutionPlan, PlanStep


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
    if playbook == "quarter_close_review":
        return ExecutionPlan(
            playbook=playbook,
            goal="Run the deterministic quarter-close workflow and produce a review-ready recommendation packet.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_quarter_close_review",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="quarter_close_review",
                ),
            ],
            required_evidence=["quarter_close_review"],
            minimum_evidence_count=1,
            stopping_rule="Stop after the composite quarter-close review packet is collected unless follow-up drilldown is needed.",
        )

    if playbook == "drop_review":
        return ExecutionPlan(
            playbook=playbook,
            goal="Review candidate development drops using the composite deterministic drop-review workflow.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_drop_review",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id, "candidate_limit": 5},
                        analysis_basis,
                    ),
                    evidence_key="drop_review",
                ),
            ],
            required_evidence=["drop_review"],
            minimum_evidence_count=1,
            stopping_rule="Stop after the ranked drop review result is collected unless follow-up evidence is requested.",
        )

    if playbook == "scenario_recommendation":
        return ExecutionPlan(
            playbook=playbook,
            goal="Recommend a tested reserving scenario from deterministic evidence.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_diagnostics_summary",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="diagnostics_summary",
                ),
                PlanStep(
                    tool_name="tool_iterate_diagnostics_summary",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id, "max_scenarios": 12},
                        analysis_basis,
                    ),
                    evidence_key="scenario_comparison",
                ),
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="results_summary",
                ),
            ],
            required_evidence=[
                "diagnostics_summary",
                "scenario_comparison",
                "results_summary",
            ],
            minimum_evidence_count=2,
            stopping_rule="Stop when diagnostics and scenario comparison are complete, or earlier if reviewer hard-fails.",
        )

    if playbook == "movement_review":
        return ExecutionPlan(
            playbook=playbook,
            goal="Explain observed current-period movement using deterministic evidence.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_get_data_view_summary",
                    args={
                        "session_id": session_id,
                        "metric": "incurred",
                        "view": "incremental",
                    },
                    evidence_key="latest_diagonal_incurred_incremental",
                ),
                PlanStep(
                    tool_name="tool_get_data_view_summary",
                    args={
                        "session_id": session_id,
                        "metric": "incurred",
                        "view": "cumulative",
                        "denominator": "premium",
                    },
                    evidence_key="incurred_on_premium",
                ),
                PlanStep(
                    tool_name="tool_run_ldf_consistency_diagnostics",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="ldf_consistency",
                ),
                PlanStep(
                    tool_name="tool_run_movement_diagnostics",
                    args={"session_id": session_id},
                    evidence_key="movement_diagnostics",
                ),
            ],
            required_evidence=[
                "latest_diagonal_incurred_incremental",
                "incurred_on_premium",
                "ldf_consistency",
                "movement_diagnostics",
            ],
            minimum_evidence_count=3,
            stopping_rule="Stop when latest-diagonal movement and consistency evidence are gathered.",
        )

    if playbook == "method_suitability_review":
        return ExecutionPlan(
            playbook=playbook,
            goal="Assess CL versus BF suitability using deterministic diagnostics.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_bf_suitability_review",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="bf_suitability_review",
                ),
            ],
            required_evidence=["bf_suitability_review"],
            minimum_evidence_count=1,
            stopping_rule="Stop after the composite BF suitability review is gathered unless UWY drilldown is needed.",
        )

    if playbook == "reserve_change_explanation":
        return ExecutionPlan(
            playbook=playbook,
            goal="Explain reserve change drivers against the current baseline.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_diagnostics_summary",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="diagnostics_summary",
                ),
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="results_summary",
                ),
            ],
            required_evidence=["diagnostics_summary", "results_summary"],
            minimum_evidence_count=2,
            stopping_rule="Stop after collecting baseline diagnostics and current results snapshot.",
        )

    if playbook == "late_emergence_review":
        return ExecutionPlan(
            playbook=playbook,
            goal="Assess how much development may still emerge from later maturities.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_project_late_emergence_benchmark",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="late_emergence",
                ),
                PlanStep(
                    tool_name="tool_run_ldf_consistency_diagnostics",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="ldf_consistency",
                ),
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="results_summary",
                ),
            ],
            required_evidence=[
                "late_emergence",
                "ldf_consistency",
                "results_summary",
            ],
            minimum_evidence_count=2,
            stopping_rule="Stop after benchmarking residual emergence and current selection context.",
        )

    if playbook == "data_exploration":
        return ExecutionPlan(
            playbook=playbook,
            goal="Answer the data question with summary-level evidence first.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="results_summary",
                ),
            ],
            required_evidence=["results_summary"],
            minimum_evidence_count=1,
            stopping_rule="Stop after one relevant summary unless follow-up evidence is required.",
        )

    if playbook == "data_anomaly_triage":
        return ExecutionPlan(
            playbook=playbook,
            goal="Triage data-quality or anomaly concerns before parameter recommendations.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_anomaly_triage",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id}, analysis_basis
                    ),
                    evidence_key="anomaly_triage",
                ),
            ],
            required_evidence=["anomaly_triage"],
            minimum_evidence_count=1,
            stopping_rule="Stop when anomaly signals are triaged and pause guidance is available.",
        )

    if playbook == "tail_selection":
        memory_tail = (
            segment_memory.get("last_selection", {}).get("tail", {})
            if isinstance(segment_memory, dict)
            else {}
        )
        return ExecutionPlan(
            playbook=playbook,
            goal="Review tail suitability with deterministic diagnostics before narrative guidance.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_tail_review",
                    args=_merge_analysis_basis_args(
                        {"session_id": session_id, "candidate_limit": 12},
                        analysis_basis,
                    ),
                    evidence_key="tail_review",
                ),
            ],
            required_evidence=["tail_review"],
            minimum_evidence_count=1,
            stopping_rule=(
                "Stop after the composite tail review; use segment tail memory as context when available. "
                f"Current stored tail context: {memory_tail}."
            ),
        )

    return None


def _merge_analysis_basis_args(
    args: dict[str, Any],
    analysis_basis: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(args)
    if not isinstance(analysis_basis, dict) or not analysis_basis:
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
