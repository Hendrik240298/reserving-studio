from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "v1"

SelectionMethod = Literal["chainladder", "bornhuetter_ferguson"]
SeverityLevel = Literal["low", "medium", "high", "critical"]


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


class RecalculateResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    results_table_rows: list[dict[str, str]] = Field(default_factory=list)
    triangle_figure: dict = Field(default_factory=dict)
    emergence_figure: dict = Field(default_factory=dict)
    heatmap_payload: dict = Field(default_factory=dict)
    cache_key: str = ""
    model_cache_key: str = ""
    figure_version: int | None = None
    duration_ms: int = 0


class DiagnosticsRequest(BaseModel):
    session_id: str
    diagnostic_profile: str | None = None
    include_recommendations: bool = True


class DiagnosticsIterateRequest(BaseModel):
    session_id: str
    max_scenarios: int = Field(default=24, ge=1, le=100)
    include_baseline: bool = True


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
    run_metadata: RunMetadata | None = None


class DiagnosticsResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    findings: list[DiagnosticFinding] = Field(default_factory=list)
    recommendations: list[DiagnosticRecommendation] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    run_metadata: RunMetadata | None = None


class DiagnosticsIterateResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    baseline: ScenarioEvaluation | None = None
    scenarios: list[ScenarioEvaluation] = Field(default_factory=list)
    iteration_metrics: dict = Field(default_factory=dict)
    run_metadata: RunMetadata | None = None


class ResultsResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    results: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
