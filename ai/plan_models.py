from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    evidence_key: str = ""
    basis_aware: bool = True


@dataclass(frozen=True)
class ExecutionPlan:
    playbook: str
    goal: str
    segment: str | None
    session_id: str | None
    workflow_name: str = ""
    intent_class: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    minimum_evidence_count: int = 0
    stopping_rule: str = ""
    basis_behavior: list[str] = field(default_factory=list)
    answer_contract: str = ""
    requires_continuity: bool = False

    def __post_init__(self) -> None:
        if not self.workflow_name:
            object.__setattr__(self, "workflow_name", self.playbook)

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook": self.playbook,
            "workflow_name": self.workflow_name,
            "goal": self.goal,
            "segment": self.segment,
            "session_id": self.session_id,
            "intent_class": self.intent_class,
            "required_capabilities": list(self.required_capabilities),
            "steps": [asdict(step) for step in self.steps],
            "required_evidence": list(self.required_evidence),
            "minimum_evidence_count": self.minimum_evidence_count,
            "stopping_rule": self.stopping_rule,
            "basis_behavior": list(self.basis_behavior),
            "answer_contract": self.answer_contract,
            "requires_continuity": self.requires_continuity,
        }


@dataclass(frozen=True)
class ReviewOutcome:
    status: str
    collected_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendationDecision:
    status: str
    summary: str
    rationale: list[str] = field(default_factory=list)
    recommended_scenario_id: str | None = None
    recommended_basis_id: str | None = None
    alternative_scenario_ids: list[str] = field(default_factory=list)
    alternative_basis_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
