from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.assumption_review_service import AssumptionReviewService


class _ReservingStub:
    def __init__(self):
        self.mode = "baseline"

    def get_results(self):
        return pd.DataFrame({"ultimate": [150.0], "incurred": [100.0]})

    def get_triangle_heatmap_data(self):
        return {"link_ratios": pd.DataFrame({12: [1.5], 24: [1.1]})}


class _EvaluationStub:
    def __init__(self, baseline_eval):
        self._baseline_eval = baseline_eval

    def evaluate_scenario(self, *, reserving, params, **_kwargs):
        reserving.mode = "baseline"
        return self._baseline_eval

    def scenario_totals_for_params(self, *, reserving, params):
        reserving.mode = "baseline"
        return {"total_ibnr": 100.0, "parameters": params}

    def apply_params_to_reserving(self, *, reserving, params):
        reserving.mode = "baseline"


class _MovementDiagnosticsStub:
    def __init__(self, reserving):
        self._reserving = reserving

    def run(self):
        return {
            "findings": [
                {
                    "code": "LARGE_LOSS_PROXY_2022_24",
                    "severity": "high",
                    "message": "large loss proxy",
                    "evidence": {},
                }
            ],
            "summary": {"finding_count": 1},
        }

    def run_ldf_consistency(self):
        return {
            "findings": [{"origin": "2022", "age": "24", "message": "ldf mismatch"}],
            "summary": {"finding_count": 1},
        }

    def run_late_emergence_benchmark(self, *, uwy=None):
        return {"rows": [], "summary": {"row_count": 0}}


def test_anomaly_triage_returns_structured_actionable_findings(monkeypatch) -> None:
    monkeypatch.setattr(
        "source.services.assumption_review_service.MovementDiagnosticsService",
        _MovementDiagnosticsStub,
    )

    baseline_eval = SimpleNamespace(
        scenario_id="baseline",
        score=1.0,
        summary="baseline",
        recommendations=[],
        findings=[
            SimpleNamespace(
                code="DATA_QUALITY_GATE",
                severity="critical",
                message="quality",
                evidence={},
            ),
            SimpleNamespace(
                code="CALENDAR_YEAR_DRIFT",
                severity="high",
                message="calendar",
                evidence={},
            ),
        ],
        governance={"tier": "red"},
        metrics={},
        run_metadata={"run_id": "run-baseline"},
    )
    service = AssumptionReviewService(
        evaluation_service=_EvaluationStub(baseline_eval),
    )

    result = service.triage_anomalies(
        segment="industrial",
        reserving=_ReservingStub(),
        baseline_params={
            "average": "volume",
            "drop": [],
            "drop_valuation": [],
            "tail": {
                "curve": "weibull",
                "attachment_age": 60,
                "projection_period": 0,
                "fit_period": [60, 84],
            },
            "bf_apriori": {},
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": {},
        },
    )

    assert result["pause_recommendation"] is True
    types = {item["type"] for item in result["triaged_findings"]}
    assert {
        "data_quality",
        "calendar_distortion",
        "large_loss_contamination",
        "case_reserve_shift",
    }.issubset(types)
    assert all(item.get("next_diagnostic") for item in result["triaged_findings"])
