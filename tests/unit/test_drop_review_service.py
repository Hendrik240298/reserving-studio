from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.assumption_review_service import AssumptionReviewService
from source.services.segment_memory_service import SegmentMemoryService


class _ReservingStub:
    def __init__(self, *, results_by_mode, heatmap_by_mode):
        self.mode = "baseline"
        self._results_by_mode = results_by_mode
        self._heatmap_by_mode = heatmap_by_mode

    def get_results(self):
        return self._results_by_mode[self.mode].copy()

    def get_triangle_heatmap_data(self):
        return self._heatmap_by_mode[self.mode]


class _EvaluationStub:
    def __init__(self, *, evaluations, totals, selector):
        self._evaluations = evaluations
        self._totals = totals
        self._selector = selector

    def evaluate_scenario(self, *, reserving, params, **_kwargs):
        mode = self._selector(params)
        reserving.mode = mode
        return self._evaluations[mode]

    def scenario_totals_for_params(self, *, reserving, params):
        mode = self._selector(params)
        reserving.mode = mode
        payload = dict(self._totals[mode])
        payload["parameters"] = params
        return payload

    def apply_params_to_reserving(self, *, reserving, params):
        reserving.mode = self._selector(params)


class _MovementDiagnosticsStub:
    def __init__(self, reserving):
        self._reserving = reserving

    def run(self):
        return {"findings": [], "summary": {"finding_count": 0}}

    def run_ldf_consistency(self):
        if self._reserving.mode == "baseline":
            findings = [
                {"origin": "2022", "age": "24", "impact_estimate": 20.0},
                {"origin": "2021", "age": "36", "impact_estimate": 10.0},
            ]
        elif self._reserving.mode == "drop_1":
            findings = [{"origin": "2021", "age": "36", "impact_estimate": 3.0}]
        else:
            findings = [{"origin": "2020", "age": "48", "impact_estimate": 18.0}]
        return {
            "findings": findings,
            "summary": {"finding_count": len(findings), "top_findings": findings[:5]},
        }

    def run_late_emergence_benchmark(self, *, uwy=None):
        rows = [
            {
                "origin": uwy or "2022",
                "selected_future_ratio": 0.36,
                "median": 0.15,
                "p75": 0.25,
            }
        ]
        return {"rows": rows, "summary": {"row_count": len(rows)}}


def _evaluation_payload(
    *, scenario_id, recommendations=None, findings=None, tier="green"
):
    return SimpleNamespace(
        scenario_id=scenario_id,
        score=1.0,
        summary=scenario_id,
        recommendations=recommendations or [],
        findings=findings or [],
        governance={"tier": tier},
        metrics={},
        run_metadata={"run_id": f"run-{scenario_id}"},
    )


def test_drop_review_ranks_candidates_and_applies_house_preferences(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "source.services.assumption_review_service.MovementDiagnosticsService",
        _MovementDiagnosticsStub,
    )

    baseline_params = {
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
    }
    baseline_recommendations = [
        SimpleNamespace(
            code="RECOMMEND_DROP_2022_24",
            message="Drop AY 2022 age 24.",
            rationale="Outlier support.",
            priority="high",
            proposed_parameters={"drop": [["2022", 24]]},
            evidence=SimpleNamespace(value=4.8),
        ),
        SimpleNamespace(
            code="RECOMMEND_DROP_2021_36",
            message="Drop AY 2021 age 36.",
            rationale="Secondary outlier.",
            priority="medium",
            proposed_parameters={"drop": [["2021", 36]]},
            evidence=SimpleNamespace(value=3.6),
        ),
    ]

    evaluations = {
        "baseline": _evaluation_payload(
            scenario_id="baseline",
            recommendations=baseline_recommendations,
        ),
        "drop_1": _evaluation_payload(scenario_id="drop_1"),
        "drop_2": _evaluation_payload(scenario_id="drop_2"),
        "drop_combo": _evaluation_payload(scenario_id="drop_combo"),
    }
    totals = {
        "baseline": {"total_ibnr": 100.0},
        "drop_1": {"total_ibnr": 92.0},
        "drop_2": {"total_ibnr": 96.0},
        "drop_combo": {"total_ibnr": 135.0},
    }

    def _selector(params):
        drops = sorted(tuple(item) for item in params.get("drop", []))
        if drops == [("2022", 24)]:
            return "drop_1"
        if drops == [("2021", 36)]:
            return "drop_2"
        if drops == [("2021", 36), ("2022", 24)]:
            return "drop_combo"
        return "baseline"

    reserving = _ReservingStub(
        results_by_mode={
            mode: pd.DataFrame({"ultimate": [100.0], "incurred": [80.0]})
            for mode in evaluations
        },
        heatmap_by_mode={
            mode: {"link_ratios": pd.DataFrame({12: [1.5], 24: [1.1]})}
            for mode in evaluations
        },
    )
    service = AssumptionReviewService(
        evaluation_service=_EvaluationStub(
            evaluations=evaluations,
            totals=totals,
            selector=_selector,
        )
    )

    result = service.review_drops(
        segment="industrial",
        reserving=reserving,
        baseline_params=baseline_params,
        segment_memory={"house_preferences": [{"type": "max_drop_count", "value": 1}]},
        candidate_limit=3,
    )

    assert result["recommendation"]["candidate_id"] == "drop_1"
    assert result["candidates"][0]["candidate_id"] == "drop_1"
    assert result["candidates"][0]["recommendation_class"] == "recommend"
    combo = next(
        item for item in result["candidates"] if item["candidate_id"] == "drop_combo_1"
    )
    assert combo["policy_trace"]["house_preference_conflicts"]


def test_drop_review_marks_rejected_before_candidate_as_avoid(monkeypatch) -> None:
    monkeypatch.setattr(
        "source.services.assumption_review_service.MovementDiagnosticsService",
        _MovementDiagnosticsStub,
    )

    baseline_params = {
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
    }
    candidate_params = {
        **baseline_params,
        "drop": [["2022", 24]],
    }
    signature = SegmentMemoryService.scenario_signature(candidate_params)
    evaluations = {
        "baseline": _evaluation_payload(
            scenario_id="baseline",
            recommendations=[
                SimpleNamespace(
                    code="RECOMMEND_DROP_2022_24",
                    message="Drop AY 2022 age 24.",
                    rationale="Outlier support.",
                    priority="high",
                    proposed_parameters={"drop": [["2022", 24]]},
                    evidence=SimpleNamespace(value=4.8),
                )
            ],
        ),
        "drop_1": _evaluation_payload(scenario_id="drop_1"),
    }
    totals = {"baseline": {"total_ibnr": 100.0}, "drop_1": {"total_ibnr": 95.0}}

    reserving = _ReservingStub(
        results_by_mode={
            mode: pd.DataFrame({"ultimate": [100.0], "incurred": [80.0]})
            for mode in evaluations
        },
        heatmap_by_mode={
            mode: {"link_ratios": pd.DataFrame({12: [1.5], 24: [1.1]})}
            for mode in evaluations
        },
    )
    service = AssumptionReviewService(
        evaluation_service=_EvaluationStub(
            evaluations=evaluations,
            totals=totals,
            selector=lambda params: "drop_1" if params.get("drop") else "baseline",
        )
    )

    result = service.review_drops(
        segment="industrial",
        reserving=reserving,
        baseline_params=baseline_params,
        segment_memory={
            "scenario_dispositions": [
                {"scenario_signature": signature, "decision": "rejected"}
            ]
        },
        candidate_limit=1,
    )

    assert result["candidates"][0]["policy_trace"]["rejected_before"] is True
    assert result["candidates"][0]["recommendation_class"] == "avoid"
