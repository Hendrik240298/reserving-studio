from __future__ import annotations

from source.services.quarter_close_service import QuarterCloseService


def test_quarter_close_pack_is_deterministic_and_exportable() -> None:
    service = QuarterCloseService()
    review = {
        "comparison": {
            "delta_summary": {
                "comparison_basis": "latest_diagonal_excluded_proxy",
                "metrics": {"total_ibnr_delta": 5.0},
            },
            "current_snapshot": {"summary": {"total_ibnr": 40.0}},
            "prior_proxy_snapshot": {"summary": {"total_ibnr": 35.0}},
            "limitations": [],
        },
        "diagnostics": {"movement": {"summary": {"finding_count": 1}}},
        "assumption_reviews": {
            "drop_review": {
                "recommendation": {"candidate_id": "drop_1"},
                "candidates": [{"candidate_id": "drop_1"}],
                "continuity_notes": [],
                "policy_trace": {},
            },
            "tail_review": {
                "recommendation": {"candidate_id": "tail_1"},
                "candidates": [{"candidate_id": "tail_1"}],
                "continuity_notes": [],
                "policy_trace": {},
            },
            "bf_suitability": {
                "overall_class": "mixed",
                "summary": {"row_count": 1},
            },
        },
        "scenario_summary": {
            "top_ranked": [{"candidate_id": "drop_1"}],
            "considered": [{"candidate_id": "drop_1"}, {"candidate_id": "tail_1"}],
        },
        "continuity": {
            "continuity_notes": [
                {"code": "prior_selection_tension", "message": "mixed BF view"}
            ]
        },
        "recommendation": {
            "recommended_changes": [{"candidate_id": "drop_1"}],
            "why_reasonable": ["tested candidate improved diagnostics"],
            "caveats": ["peer review required"],
            "judgment_items": ["review BF by UWY"],
            "signoff_questions": ["approve change?"],
            "policy_trace": {"selected_candidate_ids": ["drop_1"]},
        },
        "evidence_ids": ["ev-1", "ev-2"],
        "run_metadata": {
            "workflow_run_id": "wf-1",
            "comparison_basis": "latest_diagonal_excluded_proxy",
            "memory_schema_version": 2,
        },
    }
    pack = service.build_pack(review_result=review)

    assert pack["review_type"] == "quarter_close_pack"
    assert pack["pack_metadata"]["comparison_basis"] == "latest_diagonal_excluded_proxy"
    assert (
        pack["sections"]["data_changes"]["delta_summary"]["metrics"]["total_ibnr_delta"]
        == 5.0
    )
    assert pack["sections"]["recommended_changes"][0]["candidate_id"] == "drop_1"
    assert pack["sections"]["scenarios_considered"]["evidence_ids"] == ["ev-1", "ev-2"]
    assert "policy_trace" in pack["sections"]
