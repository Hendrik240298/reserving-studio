from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.api.schemas import DiagnosticEvidence


def test_diagnostic_evidence_v2_fields_are_supported() -> None:
    evidence = DiagnosticEvidence(
        metric_id="calendar_residual_slope",
        value=0.031,
        threshold=0.02,
        basis="OLS slope over diagonals",
        evidence_id="ev-1",
        diagnostic_id="CALENDAR_YEAR_DRIFT",
        diagnostic_version="v2.1",
        unit="ratio",
        direction="bad",
        p_value_or_score=0.97,
        severity_band="high",
        applicability_conditions=["min_diagonals>=4"],
        alternative_hypotheses=["large loss year", "claims process change"],
        confidence=0.84,
        required_review_level="amber",
    )

    assert evidence.evidence_id == "ev-1"
    assert evidence.diagnostic_id == "CALENDAR_YEAR_DRIFT"
    assert evidence.diagnostic_version == "v2.1"
    assert evidence.direction == "bad"
    assert evidence.required_review_level == "amber"
