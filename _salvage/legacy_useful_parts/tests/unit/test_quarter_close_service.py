from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.assumption_review_service import BaselineContext
from source.services.quarter_close_service import QuarterCloseService


class _ReservingStub:
    def __init__(self) -> None:
        self.active_params = {}
        self.applied_params: list[dict] = []


class _EvaluationStub:
    def apply_params_to_reserving(self, *, reserving, params):
        reserving.active_params = {
            "average": params.get("average"),
            "drop": [list(item) for item in params.get("drop", [])],
            "tail": dict(params.get("tail", {})),
            "final_ultimate": params.get("final_ultimate"),
        }
        reserving.applied_params.append(reserving.active_params)


class _SnapshotStub:
    def build_current_snapshot(
        self, *, reserving, claims_df, premium_df, comparison_basis="current"
    ):
        return {
            "comparison_basis": comparison_basis,
            "valuation_date": "2026-03-31T00:00:00",
            "data_fingerprint": "current-fp",
            "summary": {
                "uwy_count": 2,
                "total_ultimate": 220.0,
                "total_incurred": 180.0,
                "total_ibnr": 40.0,
                "selected_method_counts": {"chainladder": 2},
            },
        }

    def build_prior_proxy_snapshot(
        self,
        *,
        claims_df,
        premium_df,
        params,
        config=None,
        comparison_basis="latest_diagonal_excluded_proxy",
    ):
        return {
            "comparison_basis": comparison_basis,
            "valuation_date": "2025-12-31T00:00:00",
            "data_fingerprint": "prior-fp",
            "summary": {
                "uwy_count": 2,
                "total_ultimate": 210.0,
                "total_incurred": 175.0,
                "total_ibnr": 35.0,
                "selected_method_counts": {"chainladder": 2},
            },
        }

    def build_delta_summary(self, *, current_snapshot, prior_snapshot):
        return {
            "comparison_basis": "latest_diagonal_excluded_proxy",
            "metrics": {"total_ibnr_delta": 5.0},
        }


class _AssumptionReviewStub:
    def __init__(self, *, anomaly_pause: bool = False) -> None:
        self._anomaly_pause = anomaly_pause

    def build_baseline_context(self, *, segment, reserving, baseline_params):
        return BaselineContext(
            evaluation=SimpleNamespace(
                run_metadata={"run_id": "baseline-run"},
                findings=[
                    SimpleNamespace(evidence=SimpleNamespace(evidence_id="ev-1"))
                ],
                recommendations=[
                    SimpleNamespace(evidence=SimpleNamespace(evidence_id="ev-2"))
                ],
            ),
            totals={"total_ibnr": 40.0, "parameters": dict(baseline_params)},
            movement_diagnostics={
                "summary": {"finding_count": 1},
                "findings": [{"code": "INCURRED_SPIKE_2022_12"}],
            },
            ldf_consistency={"summary": {"finding_count": 1}, "findings": []},
            late_emergence={"summary": {"row_count": 1}, "rows": []},
            anomaly_triage={
                "triaged_findings": [
                    {
                        "code": "DATA_QUALITY_GATE",
                        "message": "Resolve missing diagonal before change.",
                    }
                ],
                "summary": {"finding_count": 1},
                "pause_recommendation": self._anomaly_pause,
                "run_metadata": {"run_id": "triage-run"},
            },
        )

    def review_drops(
        self,
        *,
        segment,
        reserving,
        baseline_params,
        segment_memory=None,
        baseline_context=None,
        candidate_limit=5,
    ):
        return {
            "review_type": "drop_review",
            "candidates": [
                {
                    "candidate_id": "drop_1",
                    "summary": "Add drop for AY 2022 age 24",
                    "parameters": {"drop": [["2022", 24]]},
                    "score": 0.82,
                    "score_breakdown": {"total_score": 0.82},
                    "recommendation_class": "recommend",
                    "metrics": {"governance_tier": "green"},
                    "continuity_notes": [],
                    "policy_trace": {"rejected_before": False},
                    "rank": 1,
                }
            ],
            "recommendation": {
                "recommendation_class": "recommend",
                "candidate_id": "drop_1",
                "summary": "Adopt the tested drop candidate.",
            },
            "continuity_notes": [],
            "policy_trace": {"rejected_before": False},
            "run_metadata": {"run_id": "drop-run"},
        }

    def review_tail(
        self,
        *,
        segment,
        reserving,
        baseline_params,
        segment_memory=None,
        baseline_context=None,
        candidate_limit=12,
    ):
        return {
            "review_type": "tail_review",
            "candidates": [
                {
                    "candidate_id": "tail_1",
                    "summary": "Test tail weibull attachment 60 fit 60-84",
                    "parameters": {"tail": {"curve": "weibull", "attachment_age": 60}},
                    "score": 0.61,
                    "score_breakdown": {"total_score": 0.61},
                    "recommendation_class": "reasonable_alternative",
                    "metrics": {"governance_tier": "green"},
                    "continuity_notes": [
                        {
                            "code": "house_preference_conflict",
                            "message": "Candidate conflicts with house preference to prefer stable tail behavior.",
                        }
                    ],
                    "policy_trace": {
                        "house_preference_conflicts": ["prefer stable tail"]
                    },
                    "rank": 1,
                }
            ],
            "recommendation": {
                "recommendation_class": "reasonable_alternative",
                "candidate_id": "tail_1",
                "summary": "Use as alternative tail sensitivity.",
            },
            "continuity_notes": [
                {
                    "code": "house_preference_conflict",
                    "message": "Candidate conflicts with house preference to prefer stable tail behavior.",
                }
            ],
            "policy_trace": {"house_preference_conflicts": ["prefer stable tail"]},
            "run_metadata": {"run_id": "tail-run"},
        }

    def review_bf_suitability(
        self,
        *,
        segment,
        reserving,
        baseline_params,
        segment_memory=None,
        baseline_context=None,
    ):
        return {
            "review_type": "bf_suitability",
            "rows": [{"uwy": "2022", "suitability_class": "bf_preferred"}],
            "overall_class": "mixed",
            "summary": {"row_count": 1},
            "continuity_notes": [
                {
                    "code": "prior_selection_tension",
                    "message": "Current BF suitability assessment conflicts with the prior all-CL segment selection.",
                }
            ],
            "policy_trace": {},
            "run_metadata": {"run_id": "bf-run"},
        }


def _baseline_params() -> dict:
    return {
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


def _source_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    claims_df = pd.DataFrame(
        {
            "uw_year": ["2021", "2021", "2022", "2022"],
            "period": ["2025-09-30", "2025-12-31", "2025-09-30", "2025-12-31"],
        }
    )
    premium_df = claims_df.assign(Premium_selected=100.0)
    return claims_df, premium_df


def test_quarter_close_service_returns_review_packet_and_restores_baseline() -> None:
    claims_df, premium_df = _source_frames()
    reserving = _ReservingStub()
    params = _baseline_params()
    service = QuarterCloseService(
        evaluation_service=_EvaluationStub(),
        assumption_review_service=_AssumptionReviewStub(),
        valuation_snapshot_service=_SnapshotStub(),
    )

    result = service.run_review(
        segment="industrial",
        reserving=reserving,
        claims_df=claims_df,
        premium_df=premium_df,
        baseline_params=params,
        segment_memory={
            "known_issues": ["Large refinery loss in 2021"],
            "house_preferences": ["prefer stable tail"],
        },
    )

    assert result["review_type"] == "quarter_close"
    assert result["comparison"]["delta_summary"]["metrics"]["total_ibnr_delta"] == 5.0
    assert result["recommendation"]["status"] == "recommended"
    assert len(result["recommendation"]["recommended_changes"]) == 2
    assert result["scenario_summary"]["top_ranked"][0]["candidate_id"] == "drop_1"
    assert result["continuity"]["memory_schema_version"] == 3
    assert any(
        item.get("code") == "prior_selection_tension"
        for item in result["continuity"]["continuity_notes"]
    )
    assert "ev-1" in result["evidence_ids"]
    assert reserving.active_params["average"] == "volume"
    assert len(reserving.applied_params) == 2


def test_quarter_close_service_holds_for_review_when_anomalies_pause() -> None:
    claims_df, premium_df = _source_frames()
    service = QuarterCloseService(
        evaluation_service=_EvaluationStub(),
        assumption_review_service=_AssumptionReviewStub(anomaly_pause=True),
        valuation_snapshot_service=_SnapshotStub(),
    )

    result = service.run_review(
        segment="industrial",
        reserving=_ReservingStub(),
        claims_df=claims_df,
        premium_df=premium_df,
        baseline_params=_baseline_params(),
        segment_memory={},
    )

    assert result["recommendation"]["status"] == "hold_for_review"
    assert any(
        "highest-priority anomaly" in item.lower()
        for item in result["recommendation"]["signoff_questions"]
    )
