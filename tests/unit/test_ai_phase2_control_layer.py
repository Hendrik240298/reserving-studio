from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.assistant_service import AssistantService
from ai.planner import PlaybookPlanner
from ai.recommendation_policy import RecommendationPolicy
from ai.reviewer import ReviewerGate
from ai.tool_payloads import build_memory_snapshot


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.tool_payloads = []

    def chat_completion(self, **kwargs):
        self.tool_payloads.append(kwargs.get("tools") or [])
        return self._responses.pop(0)


class _CompositeTools:
    tool_specs = [
        {"type": "function", "function": {"name": "tool_run_quarter_close_review"}},
        {"type": "function", "function": {"name": "tool_run_drop_review"}},
        {"type": "function", "function": {"name": "tool_run_tail_review"}},
    ]

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, function_name, args):
        self.calls.append((function_name, dict(args)))
        if function_name == "tool_run_quarter_close_review":
            return {
                "session_id": "s-1",
                "review_type": "quarter_close",
                "comparison": {
                    "current_valuation_date": "2026-03-31",
                    "delta_summary": {"metrics": {"total_ibnr_delta": 5.0}},
                },
                "top_ranked": [
                    {
                        "candidate_id": "drop_1",
                        "score": 0.8,
                        "score_breakdown": {"total_score": 0.8},
                        "policy_trace": {"rejected_before": False},
                    }
                ],
                "recommendation": {
                    "status": "recommended",
                    "summary": "Quarter-close review supports a tested drop change.",
                    "recommended_changes": [
                        {
                            "candidate_id": "drop_1",
                            "parameters": {"drop": [["2022", 24]]},
                            "score_breakdown": {"total_score": 0.8},
                            "policy_trace": {"rejected_before": False},
                        }
                    ],
                    "why_reasonable": ["tested change improved diagnostics"],
                    "policy_trace": {"selected_candidate_ids": ["drop_1"]},
                },
                "continuity": {
                    "memory_schema_version": 2,
                    "continuity_notes": [
                        {
                            "code": "prior_selection_tension",
                            "message": "Prior quarter stayed at baseline assumptions.",
                        }
                    ],
                    "house_preferences": [{"type": "max_drop_count", "value": 1}],
                    "recent_rejected_signatures": [],
                },
                "evidence_ids": ["ev-1"],
                "run_metadata": {
                    "workflow_run_id": "wf-1",
                    "current_data_fingerprint": "fp-1",
                },
            }
        if function_name == "tool_run_drop_review":
            return {
                "session_id": "s-1",
                "review_type": "drop_review",
                "candidate_count": 1,
                "top_candidates": [
                    {
                        "candidate_id": "drop_1",
                        "score": 0.8,
                        "recommendation_class": "recommend",
                        "score_breakdown": {"total_score": 0.8},
                        "policy_trace": {"rejected_before": True},
                    }
                ],
                "recommendation": {
                    "recommendation_class": "recommend",
                    "candidate_id": "drop_1",
                    "summary": "Adopt tested drop",
                },
                "continuity_notes": [
                    {
                        "code": "rejected_before",
                        "message": "Previously rejected.",
                    }
                ],
                "policy_trace": {"rejected_before": True},
                "run_metadata": {"run_id": "drop-run"},
            }
        raise AssertionError(f"Unexpected tool: {function_name}")


def test_playbook_planner_selects_composite_drop_review() -> None:
    plan = PlaybookPlanner().plan(
        user_prompt="Review whether any ratios should be dropped.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "drop_review"
    assert [step.tool_name for step in plan.steps] == ["tool_run_drop_review"]


def test_playbook_planner_selects_quarter_close_review() -> None:
    plan = PlaybookPlanner().plan(
        user_prompt="Please run the quarter-close review pack.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "quarter_close_review"
    assert [step.tool_name for step in plan.steps] == ["tool_run_quarter_close_review"]


def test_reviewer_and_policy_downgrade_rejected_before_drop_review() -> None:
    plan = PlaybookPlanner().plan(
        user_prompt="Review whether any ratios should be dropped.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )
    assert plan is not None
    evidence_packets = [
        {
            "evidence_key": "drop_review",
            "summary": {
                "session_id": "s-1",
                "review_type": "drop_review",
                "top_candidates": [
                    {
                        "candidate_id": "drop_1",
                        "score_breakdown": {"total_score": 0.8},
                        "policy_trace": {"rejected_before": True},
                    }
                ],
                "recommendation": {
                    "recommendation_class": "recommend",
                    "candidate_id": "drop_1",
                    "summary": "Adopt tested drop",
                },
                "continuity_notes": [
                    {"code": "rejected_before", "message": "Previously rejected."}
                ],
                "policy_trace": {"rejected_before": True},
            },
            "governance": {},
            "provenance": {"session_id": "s-1", "evidence_ids": []},
        }
    ]

    review = ReviewerGate().review(plan=plan, evidence_packets=evidence_packets)
    decision = RecommendationPolicy().decide(
        review=review, evidence_packets=evidence_packets
    )

    assert review.status == "pass_with_caveats"
    assert "rejected_before" in review.caveats
    assert decision.status == "reasonable_alternative"


def test_reviewer_and_policy_downgrade_house_preference_conflict_tail_review() -> None:
    plan = PlaybookPlanner().plan(
        user_prompt="Recommend the tail selection for this segment.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )
    assert plan is not None
    evidence_packets = [
        {
            "evidence_key": "tail_review",
            "summary": {
                "session_id": "s-1",
                "review_type": "tail_review",
                "top_candidates": [
                    {
                        "candidate_id": "tail_1",
                        "score_breakdown": {"total_score": 0.7},
                        "policy_trace": {
                            "house_preference_conflicts": ["prefer stable tail"]
                        },
                    }
                ],
                "recommendation": {
                    "recommendation_class": "recommend",
                    "candidate_id": "tail_1",
                    "summary": "Adopt tested tail",
                },
                "continuity_notes": [
                    {
                        "code": "house_preference_conflict",
                        "message": "Conflicts with stable-tail preference.",
                    }
                ],
                "policy_trace": {"house_preference_conflicts": ["prefer stable tail"]},
            },
            "governance": {},
            "provenance": {"session_id": "s-1", "evidence_ids": []},
        }
    ]

    review = ReviewerGate().review(plan=plan, evidence_packets=evidence_packets)
    decision = RecommendationPolicy().decide(
        review=review, evidence_packets=evidence_packets
    )

    assert review.status == "pass_with_caveats"
    assert "house_preference_conflict" in review.caveats
    assert decision.status == "reasonable_alternative"


def test_assistant_builds_phase2_deterministic_packet_from_composite_review() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Quarter-close review ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", _CompositeTools())
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    result = service.run_turn(
        user_prompt="Please run the quarter-close review pack.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    packet = result["deterministic_packet"]
    assert packet["plan"]["playbook"] == "quarter_close_review"
    assert packet["recommendation"]["status"] == "recommended"
    assert packet["continuity_notes"][0]["code"] == "prior_selection_tension"
    assert packet["recommended_changes"][0]["candidate_id"] == "drop_1"
    assert packet["score_breakdown"]["total_score"] == 0.8
    ledger = result["memory_snapshot"].get("scenario_ledger", [])
    assert ledger[0]["scenario_id"] == "drop_1"
    assert ledger[0]["transform"] == "quarter_close"


def test_memory_snapshot_merges_composite_review_candidates_into_ledger() -> None:
    snapshot = build_memory_snapshot(
        review_summary={
            "review_type": "drop_review",
            "top_candidates": [
                {
                    "candidate_id": "drop_3",
                    "summary": "Drop AY 2002 at month 39",
                    "score": 1.25,
                    "recommendation_class": "reasonable_alternative",
                }
            ],
        },
        existing_scenario_ledger=[],
    )

    assert snapshot["scenario_ledger"][0]["scenario_id"] == "drop_3"
    assert snapshot["scenario_ledger"][0]["transform"] == "drop_review"
