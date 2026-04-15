from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "v1"

SelectionMethod = Literal["chainladder", "bornhuetter_ferguson"]
SeverityLevel = Literal["low", "medium", "high", "critical"]
TailMode = Literal["inactive", "attached", "reference_fit_only"]
RecommendationClass = Literal[
    "recommend",
    "reasonable_alternative",
    "watch",
    "avoid",
]
SuitabilityClass = Literal[
    "cl_preferred",
    "bf_preferred",
    "mixed",
    "inconclusive",
]
QuarterCloseStatus = Literal[
    "hold_for_review",
    "recommended",
    "reasonable_alternative",
    "watch",
]


class TailConfig(BaseModel):
    curve: str = "weibull"
    attachment_age: int | None = Field(default=None, ge=0)
    projection_period: int = Field(default=0, ge=0)
    fit_period: list[int] = Field(default_factory=list, max_length=2)


class ParamsStore(BaseModel):
    request_id: int = 0
    source: str = "api"
    force_recalc: bool = False
    drop_store: list[list[str | int]] = Field(default_factory=list)
    tail_attachment_age: int | None = Field(default=None, ge=0)
    tail_projection_months: int = Field(default=0, ge=0)
    tail_fit_period_selection: list[int] = Field(default_factory=list)
    average: str = "volume"
    tail_curve: str = "weibull"
    bf_apriori_by_uwy: dict[str, float] = Field(default_factory=dict)
    selected_ultimate_by_uwy: dict[str, SelectionMethod] = Field(default_factory=dict)
    sync_version: int | None = Field(default=None, ge=0)


class ResultsStoreMeta(BaseModel):
    cache_key: str | None = None
    model_cache_key: str | None = None
    figure_version: int | None = None
    sync_version: int | None = None


class WorkflowFromDataframesRequest(BaseModel):
    segment: str
    granularity: str | None = None
    claims_rows: list[dict]
    premium_rows: list[dict]
    config_overrides: dict = Field(default_factory=dict)


class WorkflowInitializationResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    segment: str
    sync_version: int
    initial_params: ParamsStore | None = None
    initial_results_summary: dict = Field(default_factory=dict)


class SessionStateResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    segment: str
    params_store: ParamsStore
    results_store_meta: ResultsStoreMeta = Field(default_factory=ResultsStoreMeta)
    valuation_context: dict = Field(default_factory=dict)
    sync_version: int = 0


class SessionSaveRequest(BaseModel):
    params_store: ParamsStore
    results_store_meta: ResultsStoreMeta = Field(default_factory=ResultsStoreMeta)
    expected_sync_version: int = Field(ge=0)


class SessionSaveResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    segment: str
    sync_version: int
    saved_at: datetime


class RecalculateRequest(BaseModel):
    session_id: str
    average: str
    drop: list[list[str | int]] = Field(default_factory=list)
    drop_valuation: list[list[str | int]] = Field(default_factory=list)
    tail: TailConfig = Field(default_factory=TailConfig)
    bf_apriori: dict[str, float] = Field(default_factory=dict)
    final_ultimate: SelectionMethod = "chainladder"
    selected_ultimate_by_uwy: dict[str, SelectionMethod] = Field(default_factory=dict)
    persist_to_session: bool = True


class RecalculateResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    analysis_basis: dict = Field(default_factory=dict)
    results_table_rows: list[dict[str, str]] = Field(default_factory=list)
    triangle_figure: dict = Field(default_factory=dict)
    emergence_figure: dict = Field(default_factory=dict)
    heatmap_payload: dict = Field(default_factory=dict)
    cache_key: str = ""
    model_cache_key: str = ""
    figure_version: int | None = None
    duration_ms: int = 0


class TailEvaluationRequest(RecalculateRequest):
    pass


class TailEvaluationResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    tail_curve: str
    fit_period: list[int] = Field(default_factory=list)
    attachment_age: int | None = None
    projection_period: int = 0
    r2: float | None = None
    rmse: float | None = None
    point_count: int = 0
    residuals: list[dict] = Field(default_factory=list)
    observed_ldf: list[dict] = Field(default_factory=list)
    fitted_tail_ldf: list[dict] = Field(default_factory=list)
    tail_active: bool = False
    tail_mode: TailMode = "inactive"
    tail_applies_from_age: int | None = None
    attachment_previous_age: int | None = None
    attachment_previous_ldf: float | None = None
    attachment_first_fitted_ldf: float | None = None
    attachment_gap_ratio: float | None = None
    late_subunit_observed_ages: list[int] = Field(default_factory=list)
    input_adjustments: list[str] = Field(default_factory=list)


class AssumptionDetailRequest(BaseModel):
    session_id: str
    start_age: int | None = Field(default=None, ge=0)
    end_age: int | None = Field(default=None, ge=0)
    development_period: int | None = Field(default=None, ge=0)
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class AssumptionDetailResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    metric: Literal["incurred"] = "incurred"
    analysis_basis: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    selected_ldf: list[dict] = Field(default_factory=list)
    fitted_tail_ldf: list[dict] = Field(default_factory=list)
    tail_active: bool = False
    tail_mode: TailMode = "inactive"
    tail_applies_from_age: int | None = None
    observed_a2a: list[dict] = Field(default_factory=list)
    bf_apriori_by_uwy: dict[str, float] = Field(default_factory=dict)
    selected_ultimate_by_uwy: dict[str, SelectionMethod] = Field(default_factory=dict)


class ReviewScoreBreakdown(BaseModel):
    components: dict[str, float] = Field(default_factory=dict)
    penalties: dict[str, float] = Field(default_factory=dict)
    total_score: float = 0.0
    formula_version: str = ""


class ContinuityNote(BaseModel):
    code: str
    severity: SeverityLevel = "low"
    message: str
    context: dict = Field(default_factory=dict)


class PolicyTrace(BaseModel):
    rejected_before: bool = False
    rejected_signatures: list[str] = Field(default_factory=list)
    house_preference_conflicts: list[str] = Field(default_factory=list)
    applied_penalties: dict[str, float] = Field(default_factory=dict)
    governance_tier: Literal["green", "amber", "red"] | None = None


class ReviewCandidate(BaseModel):
    candidate_id: str
    scenario_id: str | None = None
    summary: str
    parameters: dict = Field(default_factory=dict)
    score: float = 0.0
    score_breakdown: ReviewScoreBreakdown = Field(default_factory=ReviewScoreBreakdown)
    recommendation_class: RecommendationClass = "watch"
    metrics: dict = Field(default_factory=dict)
    continuity_notes: list[ContinuityNote] = Field(default_factory=list)
    policy_trace: PolicyTrace = Field(default_factory=PolicyTrace)
    rank: int | None = None


class ReviewRecommendation(BaseModel):
    recommendation_class: RecommendationClass = "watch"
    candidate_id: str | None = None
    scenario_id: str | None = None
    summary: str = ""
    caveats: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    alternative_scenario_ids: list[str] = Field(default_factory=list)


class DropReviewRequest(BaseModel):
    session_id: str
    candidate_limit: int = Field(default=5, ge=1, le=20)
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class DropReviewResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    review_type: Literal["drop_review"] = "drop_review"
    analysis_basis: dict = Field(default_factory=dict)
    baseline: dict = Field(default_factory=dict)
    candidates: list[ReviewCandidate] = Field(default_factory=list)
    recommendation: ReviewRecommendation = Field(default_factory=ReviewRecommendation)
    continuity_notes: list[ContinuityNote] = Field(default_factory=list)
    policy_trace: PolicyTrace = Field(default_factory=PolicyTrace)
    evidence_summary: dict = Field(default_factory=dict)
    run_metadata: dict = Field(default_factory=dict)


class TailReviewRequest(BaseModel):
    session_id: str
    candidate_limit: int = Field(default=12, ge=1, le=30)
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class TailReviewResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    review_type: Literal["tail_review"] = "tail_review"
    analysis_basis: dict = Field(default_factory=dict)
    baseline: dict = Field(default_factory=dict)
    candidates: list[ReviewCandidate] = Field(default_factory=list)
    recommendation: ReviewRecommendation = Field(default_factory=ReviewRecommendation)
    continuity_notes: list[ContinuityNote] = Field(default_factory=list)
    policy_trace: PolicyTrace = Field(default_factory=PolicyTrace)
    evidence_summary: dict = Field(default_factory=dict)
    run_metadata: dict = Field(default_factory=dict)


class BfSuitabilityRequest(BaseModel):
    session_id: str
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class BfSuitabilityResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    review_type: Literal["bf_suitability"] = "bf_suitability"
    analysis_basis: dict = Field(default_factory=dict)
    rows: list[dict] = Field(default_factory=list)
    overall_class: SuitabilityClass = "inconclusive"
    summary: dict = Field(default_factory=dict)
    apriori_guidance: dict = Field(default_factory=dict)
    continuity_notes: list[ContinuityNote] = Field(default_factory=list)
    policy_trace: PolicyTrace = Field(default_factory=PolicyTrace)
    run_metadata: dict = Field(default_factory=dict)


class AnomalyTriageRequest(BaseModel):
    session_id: str
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class AnomalyTriageResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    review_type: Literal["anomaly_triage"] = "anomaly_triage"
    analysis_basis: dict = Field(default_factory=dict)
    triaged_findings: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
    pause_recommendation: bool = False
    run_metadata: dict = Field(default_factory=dict)


class QuarterCloseReviewRequest(BaseModel):
    session_id: str
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class QuarterCloseReviewResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    review_type: Literal["quarter_close"] = "quarter_close"
    analysis_basis: dict = Field(default_factory=dict)
    comparison: dict = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)
    assumption_reviews: dict = Field(default_factory=dict)
    scenario_summary: dict = Field(default_factory=dict)
    continuity: dict = Field(default_factory=dict)
    recommendation: dict = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    run_metadata: dict = Field(default_factory=dict)


class QuarterClosePackRequest(BaseModel):
    session_id: str


class QuarterClosePackResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    review_type: Literal["quarter_close_pack"] = "quarter_close_pack"
    pack: dict = Field(default_factory=dict)
    run_metadata: dict = Field(default_factory=dict)


class DataViewQuery(BaseModel):
    metric: str = "incurred"
    view: str = "cumulative"
    denominator: str | None = None
    denominator_view: str | None = None


class DataViewRequest(BaseModel):
    session_id: str
    query: DataViewQuery = Field(default_factory=DataViewQuery)
    include_summary: bool = True
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class DataViewResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    analysis_basis: dict = Field(default_factory=dict)
    query: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)


class DataCompareRequest(BaseModel):
    session_id: str
    left: DataViewQuery
    right: DataViewQuery
    comparison_mode: str = "difference"


class DataCompareResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    comparison_mode: str = "difference"
    data: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)


class MovementDiagnosticsRequest(BaseModel):
    session_id: str


class MovementDiagnosticsResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    findings: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


class LdfConsistencyRequest(BaseModel):
    session_id: str
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class LdfConsistencyResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    analysis_basis: dict = Field(default_factory=dict)
    findings: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


class LateEmergenceRequest(BaseModel):
    session_id: str
    uwy: str | None = None
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class LateEmergenceResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    analysis_basis: dict = Field(default_factory=dict)
    rows: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


class ResultsRequest(BaseModel):
    session_id: str
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class ReserveChangeRequest(RecalculateRequest):
    basis_type: str | None = None
    scenario_id: str | None = None
    basis_parameters: dict = Field(default_factory=dict)


class ReserveChangeResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    baseline: dict = Field(default_factory=dict)
    candidate: dict = Field(default_factory=dict)
    attribution: dict = Field(default_factory=dict)
    rows: list[dict] = Field(default_factory=list)


class HighestA2ADropRequest(BaseModel):
    session_id: str
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class HighestA2ADropResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    drop: list[list] = Field(default_factory=list)
    top_factors: list[dict] = Field(default_factory=list)
    baseline: dict = Field(default_factory=dict)
    candidate: dict = Field(default_factory=dict)
    scenario: dict = Field(default_factory=dict)


SelectionMode = Literal["max", "min"]
SelectionScope = Literal["per_development_period", "global"]
ThresholdOperator = Literal["lt", "lte", "gt", "gte"]


class LinkRatioRankRequest(BaseModel):
    session_id: str
    selection_mode: SelectionMode = "max"
    scope: SelectionScope = "per_development_period"
    limit: int = Field(default=5, ge=1, le=200)
    threshold_operator: ThresholdOperator | None = None
    threshold_value: float | None = None
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class LinkRatioRankResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    selection_mode: SelectionMode
    scope: SelectionScope
    rows: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


class DerivedDropRule(BaseModel):
    source: Literal["link_ratios"] = "link_ratios"
    selection_mode: SelectionMode = "max"
    scope: SelectionScope = "per_development_period"
    limit: int = Field(default=5, ge=1, le=200)
    include_existing_drops: bool = True
    threshold_operator: ThresholdOperator | None = None
    threshold_value: float | None = None


class DerivedDropScenarioRequest(BaseModel):
    session_id: str
    rule: DerivedDropRule = Field(default_factory=DerivedDropRule)
    rules: list[DerivedDropRule] = Field(default_factory=list)
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class DerivedDropScenarioResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    rule: dict = Field(default_factory=dict)
    drop: list[list] = Field(default_factory=list)
    selected_rows: list[dict] = Field(default_factory=list)
    baseline: dict = Field(default_factory=dict)
    candidate: dict = Field(default_factory=dict)
    scenario: dict = Field(default_factory=dict)


class DiagnosticsRequest(BaseModel):
    session_id: str
    diagnostic_profile: str | None = None
    include_recommendations: bool = True
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class DiagnosticsIterateRequest(BaseModel):
    session_id: str
    max_scenarios: int = Field(default=24, ge=1, le=100)
    include_baseline: bool = True
    basis_type: str | None = None
    scenario_id: str | None = None
    parameters: dict = Field(default_factory=dict)


class RunMetadata(BaseModel):
    run_id: str
    generated_at: datetime
    data_fingerprint: str
    diagnostics_version: str
    scenario_generator_version: str


class DiagnosticEvidence(BaseModel):
    metric_id: str
    value: float
    threshold: float | None = None
    basis: str | None = None
    evidence_id: str | None = None
    diagnostic_id: str | None = None
    diagnostic_version: str | None = None
    unit: str | None = None
    direction: Literal["good", "bad", "neutral"] | None = None
    p_value_or_score: float | None = None
    severity_band: SeverityLevel | None = None
    applicability_conditions: list[str] = Field(default_factory=list)
    alternative_hypotheses: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    required_review_level: Literal["green", "amber", "red"] | None = None


class DiagnosticFinding(BaseModel):
    code: str
    severity: SeverityLevel
    message: str
    evidence: DiagnosticEvidence
    suggested_actions: list[str] = Field(default_factory=list)


class DiagnosticRecommendation(BaseModel):
    code: str
    priority: SeverityLevel
    message: str
    rationale: str
    evidence: DiagnosticEvidence
    proposed_parameters: dict = Field(default_factory=dict)


class ScenarioEvaluation(BaseModel):
    scenario_id: str
    score: float
    summary: str
    parameters: dict = Field(default_factory=dict)
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[DiagnosticRecommendation] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    lineage: dict = Field(default_factory=dict)
    governance: dict = Field(default_factory=dict)
    calibration: dict = Field(default_factory=dict)
    uncertainty: dict = Field(default_factory=dict)
    run_metadata: RunMetadata | None = None


class DiagnosticsResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    analysis_basis: dict = Field(default_factory=dict)
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[DiagnosticRecommendation] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    governance: dict = Field(default_factory=dict)
    calibration: dict = Field(default_factory=dict)
    uncertainty: dict = Field(default_factory=dict)
    run_metadata: RunMetadata | None = None


class DiagnosticsIterateResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    analysis_basis: dict = Field(default_factory=dict)
    baseline: ScenarioEvaluation | None = None
    scenarios: list[ScenarioEvaluation] = Field(default_factory=list)
    iteration_metrics: dict = Field(default_factory=dict)
    governance: dict = Field(default_factory=dict)
    calibration: dict = Field(default_factory=dict)
    uncertainty: dict = Field(default_factory=dict)
    run_metadata: RunMetadata | None = None


class ResultsResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    analysis_basis: dict = Field(default_factory=dict)
    results: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None


class AIChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    fallback_used: bool | None = None


class AIChatToolEvent(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    result_summary: dict = Field(default_factory=dict)


class AIChatCreateRequest(BaseModel):
    segment: str | None = None
    reserving_session_id: str | None = None


class AIChatCreateResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    chat_id: str
    segment: str | None = None
    reserving_session_id: str | None = None
    messages: list[AIChatMessage] = Field(default_factory=list)
    tool_events: list[AIChatToolEvent] = Field(default_factory=list)
    working_memory: dict = Field(default_factory=dict)
    scenario_ledger: list[dict] = Field(default_factory=list)
    deterministic_packet: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str


class AIChatMessageRequest(BaseModel):
    content: str


class AIChatMessageResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    chat_id: str
    segment: str | None = None
    reserving_session_id: str | None = None
    assistant_message: str
    fallback_used: bool = False
    messages: list[AIChatMessage] = Field(default_factory=list)
    tool_events: list[AIChatToolEvent] = Field(default_factory=list)
    working_memory: dict = Field(default_factory=dict)
    scenario_ledger: list[dict] = Field(default_factory=list)
    deterministic_packet: dict = Field(default_factory=dict)
    updated_at: str


class AIChatSessionResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    chat_id: str
    segment: str | None = None
    reserving_session_id: str | None = None
    messages: list[AIChatMessage] = Field(default_factory=list)
    tool_events: list[AIChatToolEvent] = Field(default_factory=list)
    working_memory: dict = Field(default_factory=dict)
    scenario_ledger: list[dict] = Field(default_factory=list)
    deterministic_packet: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str
