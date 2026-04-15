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
        return {"findings": [], "summary": {"finding_count": 0}}

    def run_late_emergence_benchmark(self, *, uwy=None):
        return {"rows": [], "summary": {"row_count": 0}}


class _DiagnosticsStub:
    def _tail_recommendation(self, _heatmap_data):
        return SimpleNamespace(
            code="RECOMMEND_TAIL_FIT",
            message="Test stable late-age curves.",
            rationale="Late ages are stable above 1.0.",
            evidence={"metric_id": "tail_attachment_age", "value": 60.0},
            proposed_parameters={
                "tail": {
                    "curve_candidates": ["weibull", "exponential"],
                    "attachment_age_candidates": [60, 72],
                    "recommended_attachment_age": 60,
                    "fit_period_candidates": [[60, 84], [72, 84]],
                }
            },
        )


def _evaluation_payload(*, scenario_id, tier="green"):
    return SimpleNamespace(
        scenario_id=scenario_id,
        score=1.0,
        summary=scenario_id,
        recommendations=[],
        findings=[],
        governance={"tier": tier},
        metrics={},
        run_metadata={"run_id": f"run-{scenario_id}"},
    )


def _heatmap(ldf_values, tail_values):
    frame = pd.DataFrame(
        {
            60: [1.06, 1.05, 1.04],
            72: [1.03, 1.02, 1.01],
            84: [1.01, 1.01, 1.0],
        },
        index=pd.Index(["2020", "2021", "2022"]),
    )
    frame.loc["LDF"] = ldf_values
    frame.loc["Tail"] = tail_values
    return {"link_ratios": frame}


def test_tail_review_ranks_stable_candidate_above_unstable_one(monkeypatch) -> None:
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
    evaluations = {
        "baseline": _evaluation_payload(scenario_id="baseline"),
        "tail_good": _evaluation_payload(scenario_id="tail_good"),
        "tail_bad": _evaluation_payload(scenario_id="tail_bad", tier="amber"),
    }
    totals = {
        "baseline": {"total_ibnr": 100.0},
        "tail_good": {"total_ibnr": 103.0},
        "tail_bad": {"total_ibnr": 130.0},
    }

    def _selector(params):
        tail = params.get("tail", {})
        if tail.get("curve") == "weibull" and tail.get("attachment_age") == 60:
            return "tail_good"
        if tail.get("attachment_age") == 72:
            return "tail_bad"
        return "baseline"

    reserving = _ReservingStub(
        results_by_mode={
            mode: pd.DataFrame({"ultimate": [150.0], "incurred": [100.0]})
            for mode in evaluations
        },
        heatmap_by_mode={
            "baseline": _heatmap([1.05, 1.02, 1.01], [1.04, 1.02, 1.01]),
            "tail_good": _heatmap([1.05, 1.02, 1.01], [1.04, 1.02, 1.01]),
            "tail_bad": _heatmap([1.12, 1.06, 1.04], [0.96, 0.95, 0.94]),
        },
    )
    service = AssumptionReviewService(
        evaluation_service=_EvaluationStub(
            evaluations=evaluations,
            totals=totals,
            selector=_selector,
        ),
        diagnostics_service=_DiagnosticsStub(),
    )

    result = service.review_tail(
        segment="industrial",
        reserving=reserving,
        baseline_params=baseline_params,
        segment_memory={
            "house_preferences": [
                {"type": "prefer_stable_tail", "enabled": True},
                {"type": "max_attachment_gap_ratio", "value": 0.1},
            ]
        },
        candidate_limit=4,
    )

    top_signature = SegmentMemoryService.scenario_signature(
        result["candidates"][0]["parameters"]
    )
    assert result["candidates"][0]["candidate_id"].startswith("tail_weibull_60")
    assert result["candidates"][0]["scenario_id"] == f"review_tail_{top_signature}"
    assert (
        result["recommendation"]["candidate_id"]
        == result["candidates"][0]["candidate_id"]
    )
    assert result["recommendation"]["scenario_id"] == f"review_tail_{top_signature}"
    assert result["candidates"][0]["recommendation_class"] == "recommend"
    unstable = next(
        item
        for item in result["candidates"]
        if item["parameters"]["tail"]["attachment_age"] == 72
    )
    assert unstable["policy_trace"]["house_preference_conflicts"]
