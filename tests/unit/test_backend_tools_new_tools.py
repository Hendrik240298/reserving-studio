from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.backend_tools import BackendReservingTools
from source.api.schemas import (
    AssumptionDetailResponse,
    AnomalyTriageResponse,
    BfSuitabilityResponse,
    DataCompareResponse,
    DataViewResponse,
    DerivedDropScenarioResponse,
    DropReviewResponse,
    HighestA2ADropResponse,
    LateEmergenceResponse,
    LdfConsistencyResponse,
    LinkRatioRankResponse,
    MovementDiagnosticsResponse,
    QuarterClosePackResponse,
    QuarterCloseReviewResponse,
    ReserveChangeResponse,
    TailEvaluationResponse,
    TailReviewResponse,
)


class _BackendStub:
    def __init__(self) -> None:
        self.last_reserve_change_payload = None

    def get_data_view(self, payload):
        return DataViewResponse(
            session_id=payload.session_id,
            query=payload.query.model_dump(mode="json"),
            data={"records": [{"origin": "2022", "12": 1.0}]},
            summary={
                "latest_age": "12",
                "top_latest_rows": [{"origin": "2022", "value": 1.0}],
            },
        )

    def compare_data_views(self, payload):
        return DataCompareResponse(
            session_id=payload.session_id,
            comparison_mode=payload.comparison_mode,
            data={"records": [{"origin": "2022", "12": 0.1}]},
            summary={
                "latest_age": "12",
                "top_latest_differences": [{"origin": "2022", "value": 0.1}],
            },
        )

    def run_movement_diagnostics(self, payload):
        return MovementDiagnosticsResponse(
            session_id=payload.session_id,
            findings=[{"code": "LARGE_LOSS_PROXY_2022_12"}],
            summary={
                "finding_count": 1,
                "top_findings": [{"code": "LARGE_LOSS_PROXY_2022_12"}],
            },
        )

    def run_ldf_consistency(self, payload):
        return LdfConsistencyResponse(
            session_id=payload.session_id,
            findings=[{"origin": "2022", "age": "12"}],
            summary={
                "finding_count": 1,
                "top_findings": [{"origin": "2022", "age": "12"}],
            },
        )

    def project_late_emergence(self, payload):
        return LateEmergenceResponse(
            session_id=payload.session_id,
            rows=[{"origin": payload.uwy or "2022", "median": 0.15}],
            summary={
                "row_count": 1,
                "top_rows": [{"origin": payload.uwy or "2022", "median": 0.15}],
            },
        )

    def explain_reserve_change(self, payload):
        self.last_reserve_change_payload = payload
        return ReserveChangeResponse(
            session_id=payload.session_id,
            baseline={"total_ibnr": 100.0},
            candidate={"total_ibnr": 120.0},
            attribution={
                "baseline_vs_candidate_delta": 20.0,
                "steps": [{"component": "tail", "delta_ibnr": 20.0}],
            },
            rows=[],
        )

    def run_highest_a2a_drop_scenario(self, payload):
        return HighestA2ADropResponse(
            session_id=payload.session_id,
            drop=[["2022", 12], ["2021", 24]],
            top_factors=[{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            baseline={"score": 10.0},
            candidate={"score": 8.0},
            scenario={
                "scenario_id": "highest_a2a_each_period",
                "summary": "Drop the highest observed a2a factor in each development period",
                "score_delta": -2.0,
            },
        )

    def rank_link_ratios(self, payload):
        return LinkRatioRankResponse(
            session_id=payload.session_id,
            selection_mode=payload.selection_mode,
            scope=payload.scope,
            rows=[{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            summary={
                "row_count": 1,
                "top_rows": [{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            },
        )

    def run_derived_drop_scenario(self, payload):
        return DerivedDropScenarioResponse(
            session_id=payload.session_id,
            rule=payload.rule.model_dump(mode="json"),
            drop=[["2022", 12]],
            selected_rows=[{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            baseline={"score": 10.0},
            candidate={
                "score": 8.0,
                "parameters": {
                    "average": "volume",
                    "drop": [["2022", 12]],
                    "drop_valuation": [],
                    "tail": {
                        "curve": "weibull",
                        "attachment_age": None,
                        "projection_period": 0,
                        "fit_period": [],
                    },
                    "bf_apriori": {},
                    "final_ultimate": "chainladder",
                    "selected_ultimate_by_uwy": {},
                },
                "recommendations": [
                    {
                        "code": "RECOMMEND_DROP_2022_12",
                        "message": "Test dropping AY 2022, age 12 from development selection.",
                        "rationale": "Robust link-ratio outlier by development age.",
                        "proposed_parameters": {
                            "drop": [["2022", 12]],
                            "average": "volume",
                        },
                        "evidence": {"evidence_id": "ev-drop-2022-12"},
                    }
                ],
            },
            scenario={
                "scenario_id": "derived_drop_max_per_development_period",
                "summary": "Drop the highest observed a2a factor in each development period",
                "score_delta": -2.0,
            },
        )

    def evaluate_tail_fit(self, payload):
        return TailEvaluationResponse(
            session_id=payload.session_id,
            tail_curve="weibull",
            fit_period=[12, 45],
            attachment_age=None,
            projection_period=120,
            r2=0.98,
            rmse=0.03,
            point_count=4,
            residuals=[
                {"age": 12, "observed_ldf": 1.2, "fitted_ldf": 1.18, "error": -0.02}
            ],
            observed_ldf=[{"age": 12, "ldf": 1.2}],
            fitted_tail_ldf=[{"age": 12, "ldf": 1.18}],
        )

    def get_assumption_context_detail(self, payload):
        return AssumptionDetailResponse(
            session_id=payload.session_id,
            parameters={
                "average": "volume",
                "tail": {"curve": "weibull", "attachment_age": 30},
                "bf_apriori": {"2005": 0.5988},
                "selected_ultimate_by_uwy": {"2005": "bornhuetter_ferguson"},
            },
            selected_ldf=[{"age": 21, "development_label": "21-24", "ldf": 1.058}],
            fitted_tail_ldf=[{"age": 30, "development_label": "30-33", "ldf": 1.048}],
            observed_a2a=[
                {
                    "origin": "2005",
                    "age": 3,
                    "development_label": "3-6",
                    "a2a": 3.762,
                }
            ],
            bf_apriori_by_uwy={"2005": 0.5988},
            selected_ultimate_by_uwy={"2005": "bornhuetter_ferguson"},
        )

    def run_drop_review(self, payload):
        return DropReviewResponse(
            session_id=payload.session_id,
            candidates=[
                {
                    "candidate_id": "drop_1",
                    "summary": "Drop AY 2022 age 24",
                    "parameters": {"drop": [["2022", 24]]},
                    "score": 0.8,
                    "score_breakdown": {"total_score": 0.8},
                    "recommendation_class": "recommend",
                    "policy_trace": {"rejected_before": False},
                }
            ],
            recommendation={
                "recommendation_class": "recommend",
                "candidate_id": "drop_1",
                "summary": "Adopt tested drop",
            },
            continuity_notes=[],
            policy_trace={},
            evidence_summary={},
            run_metadata={"run_id": "drop-run"},
        )

    def run_tail_review(self, payload):
        return TailReviewResponse(
            session_id=payload.session_id,
            candidates=[
                {
                    "candidate_id": "tail_1",
                    "summary": "Tail Weibull 60",
                    "parameters": {"tail": {"curve": "weibull", "attachment_age": 60}},
                    "score": 0.7,
                    "score_breakdown": {"total_score": 0.7},
                    "recommendation_class": "reasonable_alternative",
                    "policy_trace": {
                        "house_preference_conflicts": ["prefer stable tail"]
                    },
                }
            ],
            recommendation={
                "recommendation_class": "reasonable_alternative",
                "candidate_id": "tail_1",
                "summary": "Use as sensitivity",
            },
            continuity_notes=[
                {"code": "house_preference_conflict", "message": "Prefer stable tail"}
            ],
            policy_trace={"house_preference_conflicts": ["prefer stable tail"]},
            evidence_summary={},
            run_metadata={"run_id": "tail-run"},
        )

    def run_bf_suitability_review(self, payload):
        return BfSuitabilityResponse(
            session_id=payload.session_id,
            rows=[{"uwy": "2022", "suitability_class": "bf_preferred"}],
            overall_class="mixed",
            summary={"row_count": 1},
            apriori_guidance={"available": True},
            continuity_notes=[],
            policy_trace={},
            run_metadata={"run_id": "bf-run"},
        )

    def run_anomaly_triage(self, payload):
        return AnomalyTriageResponse(
            session_id=payload.session_id,
            triaged_findings=[{"type": "data_quality", "message": "check data"}],
            summary={"finding_count": 1},
            pause_recommendation=True,
            run_metadata={"run_id": "triage-run"},
        )

    def run_quarter_close_review(self, payload):
        return QuarterCloseReviewResponse(
            session_id=payload.session_id,
            comparison={
                "current_snapshot": {"valuation_date": "2026-03-31"},
                "delta_summary": {"metrics": {"total_ibnr_delta": 5.0}},
                "limitations": [],
            },
            diagnostics={},
            assumption_reviews={},
            scenario_summary={"top_ranked": [{"candidate_id": "drop_1"}]},
            continuity={
                "memory_schema_version": 2,
                "continuity_notes": [
                    {"code": "prior_selection_tension", "message": "mixed BF view"}
                ],
                "recent_rejected_signatures": ["sig-1"],
            },
            recommendation={
                "status": "recommended",
                "summary": "Quarter-close changes ready.",
                "recommended_changes": [
                    {"candidate_id": "drop_1", "parameters": {"drop": [["2022", 24]]}}
                ],
                "policy_trace": {"selected_candidate_ids": ["drop_1"]},
            },
            evidence_ids=["ev-1"],
            run_metadata={
                "workflow_run_id": "wf-1",
                "current_data_fingerprint": "fp-1",
            },
        )

    def build_quarter_close_pack(self, payload):
        return QuarterClosePackResponse(
            session_id=payload.session_id,
            pack={
                "review_type": "quarter_close_pack",
                "pack_metadata": {"comparison_basis": "latest_diagonal_excluded_proxy"},
                "sections": {
                    "recommended_changes": [{"candidate_id": "drop_1"}],
                    "signoff_questions": ["Approve?"],
                    "policy_trace": {"selected_candidate_ids": ["drop_1"]},
                },
            },
            run_metadata={"workflow_run_id": "wf-1"},
        )


def test_backend_tools_support_new_ai_tools() -> None:
    tools = BackendReservingTools(backend=_BackendStub())

    data_summary = tools.call_tool(
        "tool_get_data_view_summary",
        {"session_id": "s-1", "metric": "incurred", "view": "cumulative"},
    )
    assert data_summary["latest_age"] == "12"

    compare_summary = tools.call_tool(
        "tool_compare_data_views",
        {
            "session_id": "s-1",
            "left_metric": "incurred",
            "left_view": "cumulative",
            "right_metric": "paid",
            "right_view": "cumulative",
        },
    )
    assert compare_summary["latest_age"] == "12"

    movement_summary = tools.call_tool(
        "tool_run_movement_diagnostics",
        {"session_id": "s-1"},
    )
    assert movement_summary["finding_count"] == 1

    ldf_summary = tools.call_tool(
        "tool_run_ldf_consistency_diagnostics",
        {"session_id": "s-1"},
    )
    assert ldf_summary["finding_count"] == 1

    benchmark_summary = tools.call_tool(
        "tool_project_late_emergence_benchmark",
        {"session_id": "s-1", "uwy": "2022"},
    )
    assert benchmark_summary["row_count"] == 1

    reserve_change = tools.call_tool(
        "tool_explain_reserve_change",
        {
            "session_id": "s-1",
            "average": "volume",
            "drop": [],
            "drop_valuation": [],
            "tail": {
                "curve": "weibull",
                "attachment_age": None,
                "projection_period": 0,
                "fit_period": [],
            },
            "bf_apriori": {},
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": {},
        },
    )
    assert reserve_change["delta_ibnr"] == 20.0

    highest_a2a = tools.call_tool(
        "tool_run_highest_a2a_drop_scenario",
        {"session_id": "s-1"},
    )
    assert highest_a2a["scenario_id"] == "highest_a2a_each_period"
    assert highest_a2a["drop_count"] == 2

    rank_summary = tools.call_tool(
        "tool_rank_link_ratios",
        {
            "session_id": "s-1",
            "selection_mode": "max",
            "scope": "per_development_period",
            "limit": 5,
        },
    )
    assert rank_summary["row_count"] == 1

    derived_summary = tools.call_tool(
        "tool_run_derived_drop_scenario",
        {
            "session_id": "s-1",
            "source": "link_ratios",
            "selection_mode": "max",
            "scope": "per_development_period",
            "limit": 5,
            "include_existing_drops": True,
        },
    )
    assert derived_summary["scenario_id"] == "derived_drop_max_per_development_period"
    assert derived_summary["drop_count"] == 1

    threshold_rank = tools.call_tool(
        "tool_rank_link_ratios",
        {
            "session_id": "s-1",
            "selection_mode": "min",
            "scope": "global",
            "limit": 5,
            "threshold_operator": "lt",
            "threshold_value": 1.0,
        },
    )
    assert threshold_rank["row_count"] == 1

    derived_detail = tools.call_tool(
        "tool_get_last_derived_drop_detail",
        {"session_id": "s-1"},
    )
    assert derived_detail["candidate_parameters"]["drop"] == [["2022", 12]]
    assert derived_detail["drop_details"] == [
        {
            "origin": "2022",
            "development_period": 12,
            "observed_a2a": 2.1,
            "support_status": "explicit_recommendation",
            "reason_label": "drop_recommendation",
            "message": "Test dropping AY 2022, age 12 from development selection.",
            "rationale": "Robust link-ratio outlier by development age.",
            "evidence_id": "ev-drop-2022-12",
            "code": "RECOMMEND_DROP_2022_12",
        }
    ]

    tail_eval = tools.call_tool(
        "tool_evaluate_tail_fit",
        {
            "session_id": "s-1",
            "average": "volume",
            "drop": [],
            "drop_valuation": [],
            "tail": {
                "curve": "weibull",
                "attachment_age": None,
                "projection_period": 120,
                "fit_period": [12, 45],
            },
            "bf_apriori": {},
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": {},
        },
    )
    assert tail_eval["r2"] == 0.98

    drop_review = tools.call_tool(
        "tool_run_drop_review",
        {"session_id": "s-1", "candidate_limit": 5},
    )
    assert drop_review["recommendation"]["candidate_id"] == "drop_1"

    tail_review = tools.call_tool(
        "tool_run_tail_review",
        {"session_id": "s-1", "candidate_limit": 8},
    )
    assert tail_review["recommendation"]["candidate_id"] == "tail_1"

    bf_review = tools.call_tool(
        "tool_run_bf_suitability_review",
        {"session_id": "s-1"},
    )
    assert bf_review["overall_class"] == "mixed"

    assumption_detail = tools.call_tool(
        "tool_get_assumption_context_detail",
        {"session_id": "s-1", "start_age": 21, "end_age": 45},
    )
    assert assumption_detail["selected_ldf"][0]["ldf"] == 1.058
    assert assumption_detail["fitted_tail_ldf"][0]["ldf"] == 1.048
    assert assumption_detail["bf_apriori_by_uwy"]["2005"] == 0.5988

    anomaly_review = tools.call_tool(
        "tool_run_anomaly_triage",
        {"session_id": "s-1"},
    )
    assert anomaly_review["pause_recommendation"] is True

    quarter_close_review = tools.call_tool(
        "tool_run_quarter_close_review",
        {"session_id": "s-1"},
    )
    assert quarter_close_review["recommendation"]["status"] == "recommended"
    assert quarter_close_review["continuity"]["memory_schema_version"] == 2

    quarter_close_pack = tools.call_tool(
        "tool_get_quarter_close_pack",
        {"session_id": "s-1"},
    )
    assert (
        quarter_close_pack["pack_metadata"]["comparison_basis"]
        == "latest_diagonal_excluded_proxy"
    )

    reserve_change_sanitized = tools.call_tool(
        "tool_explain_reserve_change",
        {
            "session_id": "s-1",
            "average": "weighted_average_3_year",
            "drop": [],
            "drop_valuation": [],
            "tail": {
                "curve": "power_curve",
                "attachment_age": None,
                "projection_period": 0,
                "fit_period": [12, 24, 36, 48],
            },
            "bf_apriori": {},
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": {
                "1995": 1100,
                "1996": "chainladder",
                "1997": 1200.5,
            },
        },
    )
    assert reserve_change_sanitized["delta_ibnr"] == 20.0
    assert reserve_change_sanitized["input_adjustments"] == [
        "Normalized average 'weighted_average_3_year' to 'volume'.",
        "Normalized tail curve 'power_curve' to 'inverse_power'.",
        "Collapsed tail.fit_period to [12, 48].",
        "Dropped 2 invalid selected_ultimate_by_uwy override(s) and kept only method values.",
    ]


def test_explain_reserve_change_drops_invalid_drop_entries() -> None:
    backend = _BackendStub()
    tools = BackendReservingTools(backend=backend)

    reserve_change = tools.call_tool(
        "tool_explain_reserve_change",
        {
            "session_id": "s-1",
            "average": "volume",
            "drop": [["2001", None], ["2002", 12], [None, 24]],
            "drop_valuation": [["1999", None], ["2000", 12], ["2001"]],
            "tail": {
                "curve": "weibull",
                "attachment_age": None,
                "projection_period": 0,
                "fit_period": [],
            },
            "bf_apriori": {},
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": {},
        },
    )

    assert reserve_change["delta_ibnr"] == 20.0
    assert reserve_change["input_adjustments"] == [
        "Dropped 2 invalid drop entries.",
        "Dropped 2 invalid drop_valuation entries.",
    ]
    assert backend.last_reserve_change_payload is not None
    assert backend.last_reserve_change_payload.drop == [["2002", 12]]
    assert backend.last_reserve_change_payload.drop_valuation == [["2000", 12]]
