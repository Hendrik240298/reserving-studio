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
    def __init__(self, results_df):
        self.mode = "baseline"
        self._results_df = results_df

    def get_results(self):
        return self._results_df.copy()

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
        return {"findings": [], "summary": {"finding_count": 0}}

    def run_ldf_consistency(self):
        return {"findings": [], "summary": {"finding_count": 0}}

    def run_late_emergence_benchmark(self, *, uwy=None):
        return {"rows": [], "summary": {"row_count": 0}}


def test_bf_suitability_returns_mixed_classification(monkeypatch) -> None:
    monkeypatch.setattr(
        "source.services.assumption_review_service.MovementDiagnosticsService",
        _MovementDiagnosticsStub,
    )

    results_df = pd.DataFrame(
        {
            "incurred": [100.0, 85.0],
            "Premium": [100.0, 100.0],
            "cl_ultimate": [100.0, 180.0],
            "bf_ultimate": [100.0, 130.0],
            "ultimate": [100.0, 180.0],
        },
        index=pd.Index(["2020", "2022"]),
    )
    baseline_eval = SimpleNamespace(
        scenario_id="baseline",
        score=1.0,
        summary="baseline",
        recommendations=[
            SimpleNamespace(
                code="RECOMMEND_BF_APRIORI",
                message="Use BF apriori on immature AYs.",
                rationale="Mature loss ratio anchor available.",
                proposed_parameters={"bf_apriori": {"2022": 0.8}},
            )
        ],
        findings=[
            SimpleNamespace(
                code="LATEST_DIAGONAL_DEVIATION_2022",
                severity="high",
                message="volatility",
                evidence={},
            ),
        ],
        governance={"tier": "green"},
        metrics={},
        run_metadata={"run_id": "run-baseline"},
    )
    service = AssumptionReviewService(
        evaluation_service=_EvaluationStub(baseline_eval),
    )

    result = service.review_bf_suitability(
        segment="industrial",
        reserving=_ReservingStub(results_df),
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
        segment_memory={
            "last_selection": {
                "method_by_uwy": {"2020": "chainladder", "2022": "chainladder"}
            }
        },
    )

    assert result["overall_class"] == "mixed"
    assert {row["uwy"]: row["suitability_class"] for row in result["rows"]} == {
        "2020": "cl_preferred",
        "2022": "bf_preferred",
    }
    assert result["apriori_guidance"]["available"] is True
