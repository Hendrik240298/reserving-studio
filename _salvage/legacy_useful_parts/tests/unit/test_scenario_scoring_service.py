from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.scenario_scoring_service import ScenarioScoringService


def test_scenario_scoring_service_returns_breakdown_and_total() -> None:
    score = ScenarioScoringService().score(
        findings=[
            {"severity": "high"},
            {"severity": "medium"},
        ],
        drop_count=2,
        continuity_penalty=0.3,
    )

    assert score["formula_version"] == "v1"
    assert score["components"]["diagnostics_severity"] == 7.0
    assert score["penalties"]["drop_count"] == 0.4
    assert score["penalties"]["continuity"] == 0.3
    assert score["score"] == 7.7


def test_review_scoring_classifies_candidate_quality() -> None:
    service = ScenarioScoringService()

    score = service.score_drop_candidate(
        outlier_support=0.8,
        consistency_improvement=0.7,
        late_emergence_support=0.6,
        reserve_impact_penalty=0.1,
        fragility_penalty=0.0,
        governance_penalty=0.0,
        continuity_penalty=0.0,
    )

    assert score["formula_version"] == "v2"
    assert score["score"] == 2.0
    assert (
        service.classify_recommendation(
            total_score=score["score"],
            governance_tier="green",
        )
        == "recommend"
    )
