from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.assistant_service import AssistantService
from ai.control_plane_types import basis_key_from_parameters
from ai.narration import render_narration_fallback
from ai.planner import PlaybookPlanner
from ai.proposal_manager import ProposalManager
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
        {"type": "function", "function": {"name": "tool_get_assumption_context_detail"}},
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
        if function_name == "tool_run_tail_review":
            base_parameters = args.get("parameters") if isinstance(args.get("parameters"), dict) else {}
            parameters = {
                **base_parameters,
                "average": base_parameters.get("average", "volume"),
                "drop": base_parameters.get("drop", []),
                "drop_valuation": base_parameters.get("drop_valuation", []),
                "tail": {
                    "curve": "weibull",
                    "attachment_age": 45,
                    "projection_period": 600,
                    "fit_period": [45, 132],
                },
                "bf_apriori": base_parameters.get("bf_apriori", {}),
                "final_ultimate": base_parameters.get("final_ultimate", "chainladder"),
                "selected_ultimate_by_uwy": base_parameters.get(
                    "selected_ultimate_by_uwy", {}
                ),
            }
            return {
                "session_id": "s-1",
                "review_type": "tail_review",
                "candidate_count": 1,
                "top_candidates": [
                    {
                        "basis_key": basis_key_from_parameters(parameters),
                        "candidate_id": "tail_1",
                        "score": 0.7,
                        "recommendation_class": "recommend",
                        "score_breakdown": {"total_score": 0.7},
                        "policy_trace": {},
                        "parameters": parameters,
                    }
                ],
                "recommendation": {
                    "recommendation_class": "recommend",
                    "candidate_id": "tail_1",
                    "summary": "Adopt tested tail",
                },
                "continuity_notes": [],
                "policy_trace": {},
                "run_metadata": {"run_id": "tail-run"},
            }
        if function_name == "tool_get_assumption_context_detail":
            return {
                "session_id": "s-1",
                "parameters": args.get("parameters", {}),
                "selected_ldf": [
                    {"age": 42, "development_label": "42-45", "ldf": 1.01}
                ],
                "fitted_tail_ldf": [
                    {"age": 45, "development_label": "45-48", "ldf": 1.005},
                    {"age": 48, "development_label": "48-51", "ldf": 0.999},
                ],
                "tail_active": True,
                "observed_a2a": [],
                "bf_apriori_by_uwy": {},
                "selected_ultimate_by_uwy": {},
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


def test_playbook_planner_composes_multiple_specific_reviews() -> None:
    plan = PlaybookPlanner().plan(
        user_prompt="Show me recommendations for both tail and drops.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "multi_review"
    assert [step.tool_name for step in plan.steps] == [
        "tool_run_drop_review",
        "tool_run_tail_review",
    ]
    assert plan.answer_contract == "review_summary_only"


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


def test_assistant_falls_back_when_model_requests_unavailable_tool() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "tool_get_composite_review",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    tools = _CompositeTools()
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    result = service.run_turn(
        user_prompt="Show me recommendations for both tail and drops.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert result["fallback_used"] is True
    assert result["fallback_reason"] == "unavailable_tool_requested"
    assert result["fallback_detail"] == "tool_get_composite_review"
    assert "AI assistant error" not in result["content"]
    assert [name for name, _args in tools.calls] == [
        "tool_run_drop_review",
        "tool_run_tail_review",
    ]
    assert "### What was reviewed" in result["content"]
    assert "drop_review: drop_1" in result["content"]
    assert "tail_review: tail_1" in result["content"]


def test_broad_tail_review_prompt_uses_composite_review_and_creates_proposal() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Tail review ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    tools = _CompositeTools()
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    result = service.run_turn(
        user_prompt=(
            "Review the tail assumptions. Rank the best tail candidates, explain "
            "attachment continuity, sub-1.0 late factors, and recommend the strongest selection."
        ),
        session_context={"segment": "industrial", "session_id": "s-1"},
        accepted_analysis_basis={
            "basis_type": "review_candidate",
            "parameters": {
                "average": "volume",
                "drop": [["2001", 60], ["2002", 39]],
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
        },
    )

    assert [name for name, _args in tools.calls] == [
        "tool_run_tail_review",
        "tool_get_assumption_context_detail",
    ]
    assumption_args = tools.calls[1][1]
    assert assumption_args["basis_key"] == result["memory_snapshot"]["proposal_basis"]["basis_key"]
    assert assumption_args["parameters"]["tail"]["attachment_age"] == 45
    compact_packet = AssistantService._compact_deterministic_packet(
        result["deterministic_packet"]
    )
    assert compact_packet["assumption_detail"]["fitted_tail_ldf_below_1"] == [
        {"age": 48, "development_label": "48-51", "ldf": 0.999}
    ]
    narration_assumption = result["narration_packet"]["supporting_evidence"][
        "assumption_detail"
    ]
    assert narration_assumption["parameter_tail"]["attachment_age"] == 45
    assert narration_assumption["fitted_tail_ldf_below_1"] == [
        {"age": 48, "development_label": "48-51", "ldf": 0.999}
    ]
    proposal = result["memory_snapshot"]["proposal_basis"]
    assert proposal["status"] == "pending"
    assert proposal["source_review_type"] == "tail_review"
    assert proposal["parameters"]["drop"] == [["2001", 60], ["2002", 39]]
    assert proposal["parameters"]["tail"]["attachment_age"] == 45


def test_drop_review_proposal_combines_eligible_ranked_candidates_up_to_five() -> None:
    candidates = []
    for rank, (origin, age) in enumerate(
        [
            ("2002", 39),
            ("2001", 60),
            ("2001", 12),
            ("2002", 21),
            ("2003", 9),
            ("1996", 87),
        ],
        start=1,
    ):
        parameters = {
            "average": "volume",
            "drop": [[origin, age]],
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
        }
        candidates.append(
            {
                "basis_key": basis_key_from_parameters(parameters),
                "candidate_id": f"drop_{rank}",
                "scenario_id": f"review_drop_{rank}",
                "summary": f"Add drop {origin} age {age}",
                "score": 1.3 - rank / 100,
                "recommendation_class": "reasonable_alternative",
                "rank": rank,
                "policy_trace": {"governance_tier": "amber"},
                "parameters": parameters,
            }
        )

    proposal = ProposalManager.build_from_deterministic_packet(
        deterministic_packet={
            "plan": {
                "playbook": "drop_review",
                "basis_behavior": ["use_accepted_basis", "proposal_possible"],
            },
            "review": {"caveats": []},
            "recommendation": {
                "status": "reasonable_alternative",
                "recommended_basis_key": candidates[0]["basis_key"],
                "rationale": ["reasonable_alternative"],
            },
            "composite_review": {
                "evidence_key": "drop_review",
                "summary": {
                    "session_id": "s-1",
                    "review_type": "drop_review",
                    "analysis_basis": {
                        "parameters": {
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
                        }
                    },
                    "top_candidates": candidates,
                },
                "provenance": {"run_id": "drop-run"},
            },
        },
        accepted_analysis_basis={},
        basis_cache={},
    )

    assert proposal["status"] == "pending"
    assert proposal["candidate_id"] == "combined_drop_review_top_5"
    assert proposal["source_review_type"] == "drop_review"
    assert proposal["parameters"]["drop"] == [
        ["2002", 39],
        ["2001", 60],
        ["2001", 12],
        ["2002", 21],
        ["2003", 9],
    ]


def test_drop_review_combined_proposal_preserves_existing_settings_and_skips_ineligible() -> None:
    accepted_parameters = {
        "average": "volume",
        "drop": [["2002", 39]],
        "drop_valuation": [],
        "tail": {
            "curve": "weibull",
            "attachment_age": 27,
            "projection_period": 0,
            "fit_period": [12, 108],
        },
        "bf_apriori": {"2006": 0.6},
        "final_ultimate": "chainladder",
        "selected_ultimate_by_uwy": {"2006": "bornhuetter_ferguson"},
    }
    candidates = []
    for candidate_id, drop, recommendation_class, policy_trace in [
        ("drop_rejected", ["2001", 60], "reasonable_alternative", {"rejected_before": True}),
        ("drop_watch", ["2001", 12], "watch", {}),
        ("drop_add_1", ["2002", 21], "reasonable_alternative", {}),
        ("drop_add_2", ["2003", 9], "recommend", {}),
    ]:
        parameters = {**accepted_parameters, "drop": [*accepted_parameters["drop"], drop]}
        candidates.append(
            {
                "candidate_id": candidate_id,
                "scenario_id": candidate_id,
                "recommendation_class": recommendation_class,
                "policy_trace": policy_trace,
                "parameters": parameters,
            }
        )

    proposal = ProposalManager.build_from_deterministic_packet(
        deterministic_packet={
            "plan": {"playbook": "drop_review", "basis_behavior": ["proposal_possible"]},
            "review": {"caveats": []},
            "recommendation": {
                "status": "reasonable_alternative",
                "recommended_basis_key": "unused-for-combined",
                "rationale": [],
            },
            "composite_review": {
                "evidence_key": "drop_review",
                "summary": {
                    "session_id": "s-1",
                    "review_type": "drop_review",
                    "top_candidates": candidates,
                },
                "provenance": {},
            },
        },
        accepted_analysis_basis={
            "basis_type": "review_candidate",
            "session_id": "s-1",
            "parameters": accepted_parameters,
        },
        basis_cache={},
    )

    assert proposal["parameters"]["drop"] == [
        ["2002", 39],
        ["2002", 21],
        ["2003", 9],
    ]
    assert proposal["parameters"]["tail"] == accepted_parameters["tail"]
    assert proposal["parameters"]["bf_apriori"] == {"2006": 0.6}


def test_pending_proposal_apply_prompt_does_not_run_tools_or_claim_acceptance() -> None:
    service = AssistantService.__new__(AssistantService)
    client = _FakeClient([])
    tools = _CompositeTools()
    setattr(service, "_client", client)
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    result = service.run_turn(
        user_prompt="please use the strongest selection and apply it",
        session_context={"segment": "industrial", "session_id": "s-1"},
        accepted_analysis_basis={
            "basis_type": "review_candidate",
            "scenario_label": "current drop basis",
            "parameters": {"average": "volume"},
        },
        proposal_basis={
            "proposal_id": "proposal-tail",
            "status": "pending",
            "basis_type": "review_candidate",
            "scenario_label": "tail_1",
            "parameters": {
                "average": "volume",
                "drop": [],
                "drop_valuation": [],
                "tail": {
                    "curve": "weibull",
                    "attachment_age": 45,
                    "projection_period": 600,
                    "fit_period": [45, 132],
                },
                "bf_apriori": {},
                "final_ultimate": "chainladder",
                "selected_ultimate_by_uwy": {},
            },
        },
    )

    assert tools.calls == []
    assert client.tool_payloads == []
    assert "proposal card" in result["content"].lower()
    assert "Analysis Basis is unchanged" in result["content"]


def test_anomaly_fallback_uses_explanatory_sections() -> None:
    content = render_narration_fallback(
        {
            "basis": {"current_basis_label": "Basis used: current baseline session."},
            "reviewed": {"scope": "Triage data-quality or anomaly concerns before parameter recommendations"},
            "recommendation": {
                "status": "hold_for_review",
                "summary": "Anomaly triage indicates recommendations should pause.",
            },
            "supporting_evidence": {
                "reviews": [
                    {
                        "review_type": "anomaly_triage",
                        "finding_count": 2,
                        "pause_recommendation": True,
                        "recommendation": {"recommendation_class": "watch"},
                        "top_findings": [
                            {
                                "severity": "high",
                                "message": "Late premium movement may affect reserve interpretation",
                            }
                        ],
                    }
                ]
            },
            "caveats": ["pause_recommendation"],
            "proposal": {"exists": False},
            "execution": {"status": "executed_exactly"},
        }
    )

    assert "### What was reviewed" in content
    assert "### Evidence Highlights" in content
    assert "anomaly_triage: 2 finding(s) identified" in content
    assert "pause recommendations" in content


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
