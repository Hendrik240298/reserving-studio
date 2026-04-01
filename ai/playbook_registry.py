from __future__ import annotations

from typing import Any

from ai.plan_models import ExecutionPlan, PlanStep


def build_execution_plan(
    *,
    playbook: str,
    session_id: str | None,
    segment: str | None,
    segment_memory: dict[str, Any] | None = None,
) -> ExecutionPlan | None:
    if not session_id:
        return None

    playbook = str(playbook or "").strip().lower()
    if playbook == "scenario_recommendation":
        return ExecutionPlan(
            playbook=playbook,
            goal="Recommend a tested reserving scenario from deterministic evidence.",
            segment=segment,
            session_id=session_id,
            steps=[
                PlanStep(
                    tool_name="tool_run_diagnostics_summary",
                    args={"session_id": session_id},
                    evidence_key="diagnostics_summary",
                ),
                PlanStep(
                    tool_name="tool_iterate_diagnostics_summary",
                    args={"session_id": session_id, "max_scenarios": 12},
                    evidence_key="scenario_comparison",
                ),
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args={"session_id": session_id},
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
                    args={"session_id": session_id},
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
                    tool_name="tool_run_diagnostics_summary",
                    args={"session_id": session_id},
                    evidence_key="diagnostics_summary",
                ),
                PlanStep(
                    tool_name="tool_run_ldf_consistency_diagnostics",
                    args={"session_id": session_id},
                    evidence_key="ldf_consistency",
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
            ],
            required_evidence=[
                "diagnostics_summary",
                "ldf_consistency",
                "incurred_on_premium",
            ],
            minimum_evidence_count=2,
            stopping_rule="Stop after method-suitability evidence set is gathered.",
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
                    args={"session_id": session_id},
                    evidence_key="diagnostics_summary",
                ),
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args={"session_id": session_id},
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
                    args={"session_id": session_id},
                    evidence_key="late_emergence",
                ),
                PlanStep(
                    tool_name="tool_run_ldf_consistency_diagnostics",
                    args={"session_id": session_id},
                    evidence_key="ldf_consistency",
                ),
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args={"session_id": session_id},
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
                    args={"session_id": session_id},
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
                    tool_name="tool_run_diagnostics_summary",
                    args={"session_id": session_id},
                    evidence_key="diagnostics_summary",
                ),
                PlanStep(
                    tool_name="tool_run_movement_diagnostics",
                    args={"session_id": session_id},
                    evidence_key="movement_diagnostics",
                ),
                PlanStep(
                    tool_name="tool_run_ldf_consistency_diagnostics",
                    args={"session_id": session_id},
                    evidence_key="ldf_consistency",
                ),
            ],
            required_evidence=[
                "diagnostics_summary",
                "movement_diagnostics",
                "ldf_consistency",
            ],
            minimum_evidence_count=2,
            stopping_rule="Stop when anomaly signals are triaged and governance implications are clear.",
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
                    tool_name="tool_run_diagnostics_summary",
                    args={"session_id": session_id},
                    evidence_key="diagnostics_summary",
                ),
                PlanStep(
                    tool_name="tool_get_results_summary",
                    args={"session_id": session_id},
                    evidence_key="results_summary",
                ),
            ],
            required_evidence=["diagnostics_summary", "results_summary"],
            minimum_evidence_count=2,
            stopping_rule=(
                "Stop after diagnostics and current results; use segment tail memory as context when available. "
                f"Current stored tail context: {memory_tail}."
            ),
        )

    return None
