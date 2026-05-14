from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SeverityLevel = Literal["low", "medium", "high", "critical"]


class DiagnosticEvidence(BaseModel):
    metric_id: str
    value: float
    threshold: float | None = None
    basis: str | None = None
    evidence_id: str
    diagnostic_id: str
    diagnostic_version: str
    unit: str | None = None
    direction: Literal["good", "bad", "neutral"] | None = None
    p_value_or_score: float | None = None
    severity_band: SeverityLevel | None = None
    applicability_conditions: list[str] = Field(default_factory=list)
    alternative_hypotheses: list[str] = Field(default_factory=list)
    confidence: float | None = None
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


class RunMetadata(BaseModel):
    run_id: str
    generated_at: datetime
    data_fingerprint: str
    diagnostics_version: str
    scenario_generator_version: str


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
    run_metadata: RunMetadata | dict = Field(default_factory=dict)
