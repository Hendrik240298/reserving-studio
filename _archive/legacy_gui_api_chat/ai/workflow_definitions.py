from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowStepDefinition:
    tool_name: str
    evidence_key: str
    default_args: dict[str, Any] = field(default_factory=dict)
    basis_aware: bool = True


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_name: str
    intent_class: str
    goal: str
    required_capabilities: tuple[str, ...]
    required_evidence: tuple[str, ...]
    minimum_evidence_count: int
    stopping_rule_template: str
    basis_behavior: tuple[str, ...]
    answer_contract: str
    steps: tuple[WorkflowStepDefinition, ...]
    prompt_hint: str
    tool_whitelist: tuple[str, ...]
    selection_keywords: tuple[str, ...] = ()
    selection_requires_any: tuple[str, ...] = ()
    selection_mode: str = "keywords"
    policy_prompt_relevant: bool = False
    requires_continuity: bool = False

    def rendered_stopping_rule(self, *, segment_memory: dict[str, Any] | None) -> str:
        if "{memory_tail}" not in self.stopping_rule_template:
            return self.stopping_rule_template
        memory_tail = (
            segment_memory.get("last_selection", {}).get("tail", {})
            if isinstance(segment_memory, dict)
            else {}
        )
        return self.stopping_rule_template.format(memory_tail=memory_tail)


def _movement_question(prompt: str) -> bool:
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


def _recommendation_question(prompt: str) -> bool:
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


def _reserve_change_question(prompt: str) -> bool:
    baseline_terms = {
        "baseline",
        "base line",
        "beginning",
        "start of chat",
        "start of the chat",
        "before all the modifications",
        "before the modifications",
        "before modifications",
        "before we changed",
        "original scenario",
        "original basis",
    }
    if not any(term in prompt for term in baseline_terms):
        return False
    compare_terms = {
        "comparison",
        "compare this basis",
        "compare the basis",
        "compare current basis",
        "compare accepted basis",
        "compare analysis basis",
        "this basis to baseline",
        "basis to baseline",
        "analysis basis and",
        "analysis basis to",
        "ibnr comparison",
        "reserve comparison",
        "scenario before",
        "before all the modifications",
        "before the modifications",
    }
    return any(term in prompt for term in compare_terms)


def _additional_drop_request(prompt: str) -> bool:
    drop_terms = {"drop", "drops", "dropped"}
    add_terms = {"add", "additional", "append", "more", "another"}
    impact_terms = {"impact", "positive", "highest", "largest", "most", "outlier"}
    return (
        any(term in prompt for term in drop_terms)
        and any(term in prompt for term in add_terms)
        and any(term in prompt for term in impact_terms)
    )


WORKFLOW_DEFINITIONS: tuple[WorkflowDefinition, ...] = (
    WorkflowDefinition(
        workflow_name="quarter_close_review",
        intent_class="quarter_close",
        goal="Run the deterministic quarter-close workflow and produce a review-ready recommendation packet.",
        required_capabilities=("quarter_close_review",),
        required_evidence=("quarter_close_review",),
        minimum_evidence_count=1,
        stopping_rule_template="Stop after the composite quarter-close review packet is collected unless follow-up drilldown is needed.",
        basis_behavior=("use_accepted_basis", "proposal_possible"),
        answer_contract="recommendation_with_proposal",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_quarter_close_review",
                evidence_key="quarter_close_review",
                default_args={},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Quarter-Close Review. Use the composite deterministic quarter-close review first, then use drilldown tools only for follow-up evidence.",
        tool_whitelist=(
            "tool_run_quarter_close_review",
            "tool_get_quarter_close_pack",
            "tool_get_assumption_context_detail",
            "tool_get_results_summary",
            "tool_get_result_for_uwy",
            "tool_get_finding_detail",
            "tool_get_scenario_detail",
        ),
        selection_keywords=(
            "quarter close",
            "quarter-close",
            "close pack",
            "close review",
            "quarterly review pack",
        ),
        policy_prompt_relevant=True,
        requires_continuity=True,
    ),
    WorkflowDefinition(
        workflow_name="data_anomaly_triage",
        intent_class="anomaly_triage",
        goal="Triage data-quality or anomaly concerns before parameter recommendations.",
        required_capabilities=("anomaly_triage",),
        required_evidence=("anomaly_triage",),
        minimum_evidence_count=1,
        stopping_rule_template="Stop when anomaly signals are triaged and pause guidance is available.",
        basis_behavior=("use_accepted_basis", "proposal_disallowed"),
        answer_contract="review_summary_only",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_anomaly_triage",
                evidence_key="anomaly_triage",
                default_args={},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Data Anomaly Triage. Lead with the composite anomaly triage result before any parameter recommendation.",
        tool_whitelist=(
            "tool_run_anomaly_triage",
            "tool_run_movement_diagnostics",
            "tool_run_ldf_consistency_diagnostics",
            "tool_get_data_view_summary",
            "tool_get_data_view",
            "tool_compare_data_views",
            "tool_get_finding_detail",
        ),
        selection_keywords=(
            "data quality",
            "anomaly",
            "triage",
            "missing diagonal",
            "impossible link ratio",
            "calendar year distortion",
            "large loss contamination",
        ),
        policy_prompt_relevant=True,
    ),
    WorkflowDefinition(
        workflow_name="drop_review",
        intent_class="assumption_review",
        goal="Review candidate development drops using the composite deterministic drop-review workflow.",
        required_capabilities=("drop_review",),
        required_evidence=("drop_review",),
        minimum_evidence_count=1,
        stopping_rule_template="Stop after the ranked drop review result is collected unless follow-up evidence is requested.",
        basis_behavior=("use_accepted_basis", "proposal_possible"),
        answer_contract="recommendation_with_proposal",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_drop_review",
                evidence_key="drop_review",
                default_args={"candidate_limit": 5},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Drop Review. Use the composite drop review first and treat its ranked candidates, continuity notes, and policy trace as the primary evidence base.",
        tool_whitelist=(
            "tool_run_drop_review",
            "tool_get_finding_detail",
            "tool_get_scenario_detail",
            "tool_get_last_derived_drop_detail",
            "tool_explain_reserve_change",
        ),
        selection_keywords=(
            "drop review",
            "drop any ratios",
            "which ratios should be dropped",
            "which ratio should be dropped",
            "should be dropped",
            "should we drop",
            "should i drop",
            "drop",
            "drops",
            "drop ratios",
        ),
        policy_prompt_relevant=True,
        requires_continuity=True,
    ),
    WorkflowDefinition(
        workflow_name="derived_drop_expansion",
        intent_class="scenario_recommendation",
        goal="Build and review an expanded drop scenario from deterministic link-ratio rules.",
        required_capabilities=("derived_drop_scenario",),
        required_evidence=("derived_drop_scenario",),
        minimum_evidence_count=1,
        stopping_rule_template="Stop after the derived drop scenario is built from the current accepted basis.",
        basis_behavior=("use_accepted_basis", "proposal_possible"),
        answer_contract="recommendation_with_proposal",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_derived_drop_scenario",
                evidence_key="derived_drop_scenario",
                default_args={
                    "source": "link_ratios",
                    "selection_mode": "max",
                    "scope": "global",
                    "limit": 3,
                    "include_existing_drops": True,
                },
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Derived Drop Expansion. Build the proposed drop list from the derived-drop tool result; do not substitute scenario-iteration labels or unrelated drop lists.",
        tool_whitelist=(
            "tool_run_derived_drop_scenario",
            "tool_get_last_derived_drop_detail",
            "tool_explain_reserve_change",
            "tool_get_assumption_context_detail",
        ),
        selection_mode="additional_drop_request",
        policy_prompt_relevant=True,
    ),
    WorkflowDefinition(
        workflow_name="movement_review",
        intent_class="observational_review",
        goal="Explain observed current-period movement using deterministic evidence.",
        required_capabilities=("data_view_summary", "movement_diagnostics", "ldf_consistency"),
        required_evidence=(
            "latest_diagonal_incurred_incremental",
            "incurred_on_premium",
            "ldf_consistency",
            "movement_diagnostics",
        ),
        minimum_evidence_count=3,
        stopping_rule_template="Stop when latest-diagonal movement and consistency evidence are gathered.",
        basis_behavior=("use_active_session_only", "proposal_disallowed"),
        answer_contract="observational_explanation",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_get_data_view_summary",
                evidence_key="latest_diagonal_incurred_incremental",
                default_args={"metric": "incurred", "view": "incremental"},
                basis_aware=False,
            ),
            WorkflowStepDefinition(
                tool_name="tool_get_data_view_summary",
                evidence_key="incurred_on_premium",
                default_args={
                    "metric": "incurred",
                    "view": "cumulative",
                    "denominator": "premium",
                },
                basis_aware=False,
            ),
            WorkflowStepDefinition(
                tool_name="tool_run_ldf_consistency_diagnostics",
                evidence_key="ldf_consistency",
                default_args={},
                basis_aware=True,
            ),
            WorkflowStepDefinition(
                tool_name="tool_run_movement_diagnostics",
                evidence_key="movement_diagnostics",
                default_args={},
                basis_aware=False,
            ),
        ),
        prompt_hint="Selected playbook: Movement Review. Use the Movement Review workflow from AI_PLAYBOOKS.md. Start with data-view summaries and movement-focused evidence, then answer directly.",
        tool_whitelist=(
            "tool_get_data_view_summary",
            "tool_get_data_view",
            "tool_compare_data_views",
            "tool_run_movement_diagnostics",
            "tool_run_ldf_consistency_diagnostics",
            "tool_project_late_emergence_benchmark",
            "tool_get_finding_detail",
        ),
        selection_mode="movement_question",
    ),
    WorkflowDefinition(
        workflow_name="late_emergence_review",
        intent_class="late_emergence",
        goal="Assess how much development may still emerge from later maturities.",
        required_capabilities=("late_emergence", "ldf_consistency", "results_summary"),
        required_evidence=("late_emergence", "ldf_consistency", "results_summary"),
        minimum_evidence_count=2,
        stopping_rule_template="Stop after benchmarking residual emergence and current selection context.",
        basis_behavior=("use_accepted_basis", "proposal_disallowed"),
        answer_contract="review_summary_only",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_project_late_emergence_benchmark",
                evidence_key="late_emergence",
                default_args={},
                basis_aware=True,
            ),
            WorkflowStepDefinition(
                tool_name="tool_run_ldf_consistency_diagnostics",
                evidence_key="ldf_consistency",
                default_args={},
                basis_aware=True,
            ),
            WorkflowStepDefinition(
                tool_name="tool_get_results_summary",
                evidence_key="results_summary",
                default_args={},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Late Emergence Review. Use historical continuation evidence before broad recommendations.",
        tool_whitelist=(
            "tool_project_late_emergence_benchmark",
            "tool_get_results_summary",
            "tool_get_result_for_uwy",
            "tool_get_assumption_context_detail",
        ),
        selection_keywords=(
            "how much more",
            "still emerge",
            "late emergence",
            "still come",
            "still develop",
        ),
    ),
    WorkflowDefinition(
        workflow_name="reserve_change_explanation",
        intent_class="reserve_change_explanation",
        goal="Explain reserve change drivers against the current baseline.",
        required_capabilities=("diagnostics_summary", "results_summary"),
        required_evidence=("diagnostics_summary", "results_summary"),
        minimum_evidence_count=2,
        stopping_rule_template="Stop after collecting baseline diagnostics and current results snapshot.",
        basis_behavior=("use_accepted_basis", "proposal_disallowed"),
        answer_contract="observational_explanation",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_diagnostics_summary",
                evidence_key="diagnostics_summary",
                default_args={},
                basis_aware=True,
            ),
            WorkflowStepDefinition(
                tool_name="tool_get_results_summary",
                evidence_key="results_summary",
                default_args={},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Reserve Change Explanation. Use attribution against baseline before broad scenario discussion.",
        tool_whitelist=(
            "tool_explain_reserve_change",
            "tool_get_last_derived_drop_detail",
            "tool_get_data_view_summary",
            "tool_compare_data_views",
            "tool_get_results_summary",
            "tool_get_result_for_uwy",
        ),
        selection_keywords=(
            "why did reserve",
            "why does reserve",
            "explain reserve change",
            "driver of reserve",
            "reserve change",
            "impact on reserve",
        ),
        selection_mode="reserve_change_question",
        policy_prompt_relevant=True,
    ),
    WorkflowDefinition(
        workflow_name="tail_selection",
        intent_class="assumption_review",
        goal="Review tail suitability with deterministic diagnostics before narrative guidance.",
        required_capabilities=("tail_review",),
        required_evidence=("tail_review",),
        minimum_evidence_count=1,
        stopping_rule_template="Stop after the composite tail review; use segment tail memory as context when available. Current stored tail context: {memory_tail}.",
        basis_behavior=("use_accepted_basis", "proposal_possible"),
        answer_contract="recommendation_with_proposal",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_tail_review",
                evidence_key="tail_review",
                default_args={"candidate_limit": 12},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Tail Selection. Use the composite tail review first before drilldown tail-fit testing. Proactively comment on sub-1 late selected LDFs, whether the tail smooths them from above, and whether the attachment creates too sharp a cut from the previous selected LDF.",
        tool_whitelist=(
            "tool_run_tail_review",
            "tool_evaluate_tail_fit",
            "tool_get_assumption_context_detail",
            "tool_get_scenario_detail",
            "tool_get_finding_detail",
        ),
        selection_keywords=(
            "tail",
            "weibull",
            "inverse power",
            "inverse_power",
            "exponential",
            "r2",
            "fit period",
            "tail fit",
        ),
        policy_prompt_relevant=True,
        requires_continuity=True,
    ),
    WorkflowDefinition(
        workflow_name="method_suitability_review",
        intent_class="method_review",
        goal="Assess CL versus BF suitability using deterministic diagnostics.",
        required_capabilities=("bf_suitability_review",),
        required_evidence=("bf_suitability_review",),
        minimum_evidence_count=1,
        stopping_rule_template="Stop after the composite BF suitability review is gathered unless UWY drilldown is needed.",
        basis_behavior=("use_accepted_basis", "proposal_possible"),
        answer_contract="recommendation_with_proposal",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_bf_suitability_review",
                evidence_key="bf_suitability_review",
                default_args={},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Method Suitability Review. Use the composite BF suitability review first and treat its UWY-level suitability conclusions as the primary evidence base.",
        tool_whitelist=(
            "tool_run_bf_suitability_review",
            "tool_get_results_summary",
            "tool_get_result_for_uwy",
            "tool_get_finding_detail",
            "tool_compare_data_views",
        ),
        selection_keywords=(
            "cl vs bf",
            "chainladder vs bf",
            "bornhuetter",
            "method suitable",
            "bf better",
            "chainladder better",
        ),
        policy_prompt_relevant=True,
        requires_continuity=True,
    ),
    WorkflowDefinition(
        workflow_name="scenario_recommendation",
        intent_class="scenario_recommendation",
        goal="Recommend a tested reserving scenario from deterministic evidence.",
        required_capabilities=("diagnostics_summary", "scenario_iteration", "results_summary"),
        required_evidence=("diagnostics_summary", "scenario_comparison", "results_summary"),
        minimum_evidence_count=2,
        stopping_rule_template="Stop when diagnostics and scenario comparison are complete, or earlier if reviewer hard-fails.",
        basis_behavior=("use_accepted_basis", "proposal_possible"),
        answer_contract="recommendation_with_proposal",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_run_diagnostics_summary",
                evidence_key="diagnostics_summary",
                default_args={},
                basis_aware=True,
            ),
            WorkflowStepDefinition(
                tool_name="tool_iterate_diagnostics_summary",
                evidence_key="scenario_comparison",
                default_args={"max_scenarios": 12},
                basis_aware=True,
            ),
            WorkflowStepDefinition(
                tool_name="tool_get_results_summary",
                evidence_key="results_summary",
                default_args={},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Scenario Recommendation. Use the Scenario Recommendation workflow from AI_PLAYBOOKS.md. Favor diagnostics plus scenario iteration before recommending changes.",
        tool_whitelist=(
            "tool_run_diagnostics_summary",
            "tool_iterate_diagnostics_summary",
            "tool_get_results_summary",
            "tool_get_data_view_summary",
            "tool_get_scenario_detail",
            "tool_get_finding_detail",
            "tool_explain_reserve_change",
            "tool_rank_link_ratios",
            "tool_run_derived_drop_scenario",
            "tool_run_highest_a2a_drop_scenario",
            "tool_recalculate",
        ),
        selection_mode="recommendation_question",
        policy_prompt_relevant=True,
    ),
    WorkflowDefinition(
        workflow_name="data_exploration",
        intent_class="data_exploration",
        goal="Answer the data question with summary-level evidence first.",
        required_capabilities=("results_summary",),
        required_evidence=("results_summary",),
        minimum_evidence_count=1,
        stopping_rule_template="Stop after one relevant summary unless follow-up evidence is required.",
        basis_behavior=("use_accepted_basis", "proposal_disallowed"),
        answer_contract="data_summary",
        steps=(
            WorkflowStepDefinition(
                tool_name="tool_get_results_summary",
                evidence_key="results_summary",
                default_args={},
                basis_aware=True,
            ),
        ),
        prompt_hint="Selected playbook: Data Exploration. Use summary data tools first and only request detailed rows if needed.",
        tool_whitelist=(
            "tool_get_data_view_summary",
            "tool_get_data_view",
            "tool_compare_data_views",
            "tool_get_results_summary",
            "tool_get_result_for_uwy",
        ),
        selection_keywords=(
            "show me",
            "compare data",
            "triangle",
            "data view",
            "ratio",
            "table",
        ),
    ),
)


_WORKFLOW_BY_NAME = {item.workflow_name: item for item in WORKFLOW_DEFINITIONS}


def get_workflow_definition(workflow_name: str) -> WorkflowDefinition | None:
    return _WORKFLOW_BY_NAME.get(str(workflow_name or "").strip())


def select_workflow_definition(prompt: str) -> WorkflowDefinition | None:
    definitions = select_workflow_definitions(prompt)
    return definitions[0] if definitions else None


def select_workflow_definitions(prompt: str) -> tuple[WorkflowDefinition, ...]:
    prompt_text = str(prompt or "").strip().lower()
    if not prompt_text:
        return ()
    matches = tuple(
        definition
        for definition in WORKFLOW_DEFINITIONS
        if _definition_matches_prompt(definition, prompt_text)
    )
    priority = tuple(
        definition
        for definition in matches
        if definition.selection_mode
        in {"movement_question", "reserve_change_question", "additional_drop_request"}
    )
    if priority:
        return (priority[0],)
    specific = tuple(
        definition
        for definition in matches
        if definition.workflow_name not in {"scenario_recommendation", "data_exploration"}
    )
    if len(specific) > 1:
        return specific
    if specific:
        return (specific[0],)
    return matches[:1]


def _definition_matches_prompt(
    definition: WorkflowDefinition,
    prompt_text: str,
) -> bool:
    if definition.selection_mode == "movement_question":
        return _movement_question(prompt_text)
    if definition.selection_mode == "reserve_change_question":
        return _reserve_change_question(prompt_text)
    if definition.selection_mode == "additional_drop_request":
        return _additional_drop_request(prompt_text)
    if definition.selection_mode == "recommendation_question":
        return _recommendation_question(prompt_text)
    if definition.selection_keywords and any(
        keyword in prompt_text for keyword in definition.selection_keywords
    ):
        if definition.selection_requires_any and not any(
            keyword in prompt_text for keyword in definition.selection_requires_any
        ):
            return False
        return True
    return False


def select_workflow_name(prompt: str) -> str:
    definition = select_workflow_definition(prompt)
    return definition.workflow_name if definition is not None else ""
