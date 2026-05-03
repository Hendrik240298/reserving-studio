from __future__ import annotations

from pathlib import Path
import logging
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.assistant_service import AssistantService
from ai.chat_service import AIChatService
from ai.chat_store import InMemoryChatStore
from ai.control_plane_types import basis_key_from_parameters
from ai.planner import PlaybookPlanner
from ai.reviewer import ReviewerGate
from source.config_manager import ConfigManager


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.last_messages = None
        self.tool_payloads = []

    def chat_completion(self, **kwargs):
        self.last_messages = kwargs.get("messages")
        self.tool_payloads.append(kwargs.get("tools") or [])
        return self._responses.pop(0)


class _DeterministicTools:
    tool_specs = []

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, function_name, args):
        self.calls.append((function_name, dict(args)))
        if function_name == "tool_run_diagnostics_summary":
            return {
                "session_id": "s-1",
                "finding_count": 1,
                "recommendation_count": 1,
                "governance": {
                    "tier": "green",
                    "requires_human_review": False,
                    "escalation_triggers": [],
                },
                "top_findings": [
                    {
                        "code": "LDF_DRIFT_2022",
                        "severity": "medium",
                        "message": "LDF drift noted",
                        "evidence_id": "ev-1",
                    }
                ],
                "top_recommendations": [
                    {
                        "code": "RECOMMEND_DROP_2022_24",
                        "priority": "high",
                        "message": "Test dropping AY 2022 age 24.",
                        "evidence_id": "ev-2",
                        "proposed_parameters": {"drop": [["2022", 24]]},
                    }
                ],
                "metrics": {
                    "assessment_confidence": 0.8,
                    "governance_tier": "green",
                },
            }
        if function_name == "tool_iterate_diagnostics_summary":
            return {
                "session_id": "s-1",
                "analysis_basis": {
                    "basis_type": "baseline",
                    "scenario_id": "baseline",
                    "is_active_session": True,
                },
                "baseline": {
                    "scenario_id": "baseline",
                    "score": 2.0,
                    "governance_tier": "green",
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
                    },
                },
                "scenario_count": 2,
                "top_scenarios": [
                    {
                        "basis_key": basis_key_from_parameters(
                            {
                                "average": "volume",
                                "drop": [["2022", 24]],
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
                        ),
                        "scenario_id": "drop_1",
                        "score": 1.0,
                        "summary": "Apply one tested drop.",
                        "governance_tier": "green",
                        "parameters": {
                            "average": "volume",
                            "drop": [["2022", 24]],
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
                    {
                        "basis_key": basis_key_from_parameters(
                            {
                                "average": "volume",
                                "drop": [["2021", 24]],
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
                        ),
                        "scenario_id": "drop_2",
                        "score": 1.7,
                        "summary": "Alternative tested drop.",
                        "governance_tier": "amber",
                        "parameters": {
                            "average": "volume",
                            "drop": [["2021", 24]],
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
                ],
                "iteration_metrics": {"best_scenario_id": "drop_1"},
            }
        if function_name == "tool_run_tail_review":
            return {
                "session_id": "s-1",
                "analysis_basis": {
                    "basis_type": args.get("basis_type", "review_candidate"),
                    "basis_key": args.get("basis_key"),
                    "scenario_id": args.get("scenario_id"),
                    "is_active_session": False,
                    "parameters": args.get("parameters", {}),
                },
                "review_type": "tail_review",
                "candidate_count": 1,
                "top_candidates": [
                    {
                        "candidate_id": "tail_1",
                        "score": 0.8,
                        "summary": "Attach Weibull at 27.",
                        "parameters": {
                            "average": "volume",
                            "drop": [["2022", 24]],
                            "drop_valuation": [],
                            "tail": {
                                "curve": "weibull",
                                "attachment_age": 27,
                                "projection_period": 0,
                                "fit_period": [12, 108],
                            },
                            "bf_apriori": {},
                            "final_ultimate": "chainladder",
                            "selected_ultimate_by_uwy": {},
                        },
                    }
                ],
                "recommendation": {
                    "candidate_id": "tail_1",
                    "recommendation_class": "recommend",
                    "summary": "Adopt tested tail.",
                },
            }
        if function_name == "tool_run_derived_drop_scenario":
            base_parameters = dict(args.get("parameters") or {})
            base_drops = [list(item) for item in base_parameters.get("drop", [])]
            drop = [["2002", 39], ["2001", 60], ["2004", 3]]
            parameters = {
                **base_parameters,
                "average": base_parameters.get("average", "volume"),
                "drop": [*base_drops, *drop],
                "drop_valuation": base_parameters.get("drop_valuation", []),
                "tail": base_parameters.get(
                    "tail",
                    {
                        "curve": "weibull",
                        "attachment_age": 27,
                        "projection_period": 0,
                        "fit_period": [12, 108],
                    },
                ),
                "bf_apriori": base_parameters.get("bf_apriori", {}),
                "final_ultimate": base_parameters.get(
                    "final_ultimate",
                    "chainladder",
                ),
                "selected_ultimate_by_uwy": base_parameters.get(
                    "selected_ultimate_by_uwy",
                    {},
                ),
            }
            basis_key = basis_key_from_parameters(parameters)
            return {
                "session_id": "s-1",
                "rule": {"selection_mode": "max", "limit": args.get("limit")},
                "drop_count": len(drop),
                "selected_rows": [
                    {"origin": origin, "development_period": age}
                    for origin, age in drop
                ],
                "baseline_score": 2.0,
                "candidate_score": 1.1,
                "score_delta": -0.9,
                "basis_key": basis_key,
                "scenario_id": "drop_combo_1",
                "scenario_label": "drop_combo_1",
                "summary": "Added the highest-impact additional drops.",
                "parameters": parameters,
            }
        if function_name == "tool_recalculate":
            parameters = {
                "average": args.get("average", "volume"),
                "drop": args.get("drop", []),
                "drop_valuation": args.get("drop_valuation", []),
                "tail": args.get(
                    "tail",
                    {
                        "curve": "weibull",
                        "attachment_age": None,
                        "projection_period": 0,
                        "fit_period": [],
                    },
                ),
                "bf_apriori": args.get("bf_apriori", {}),
                "final_ultimate": args.get("final_ultimate", "chainladder"),
                "selected_ultimate_by_uwy": args.get("selected_ultimate_by_uwy", {}),
            }
            return {
                "session_id": "s-1",
                "analysis_basis": {
                    "basis_key": basis_key_from_parameters(parameters),
                    "basis_type": "bespoke",
                    "is_active_session": False,
                    "parameters": parameters,
                },
                "results_table_rows": [],
                "duration_ms": 12,
            }
        if function_name == "tool_get_results_summary":
            return {
                "session_id": "s-1",
                "result_row_count": 4,
                "top_rows": [{"uwy": "2022", "selected_method": "chainladder"}],
                "latest_rows": [
                    {"uwy": "2002", "selected_method": "chainladder"},
                    {"uwy": "2003", "selected_method": "chainladder"},
                    {"uwy": "2004", "selected_method": "chainladder"},
                    {"uwy": "2005", "selected_method": "chainladder"},
                    {"uwy": "2006", "selected_method": "chainladder"},
                ],
            }
        return {"session_id": "s-1"}


def test_playbook_planner_builds_scenario_recommendation_plan() -> None:
    planner = PlaybookPlanner()

    plan = planner.plan(
        user_prompt="What scenario do you recommend?",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "scenario_recommendation"
    assert plan.intent_class == "scenario_recommendation"
    assert plan.answer_contract == "recommendation_with_proposal"
    assert [step.tool_name for step in plan.steps] == [
        "tool_run_diagnostics_summary",
        "tool_iterate_diagnostics_summary",
        "tool_get_results_summary",
    ]


def test_playbook_planner_builds_movement_review_plan() -> None:
    planner = PlaybookPlanner()

    plan = planner.plan(
        user_prompt="Are there unusual claims movements this quarter?",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "movement_review"
    assert plan.answer_contract == "observational_explanation"
    assert "proposal_disallowed" in set(plan.basis_behavior)
    assert [step.evidence_key for step in plan.steps] == [
        "latest_diagonal_incurred_incremental",
        "incurred_on_premium",
        "ldf_consistency",
        "movement_diagnostics",
    ]


def test_playbook_planner_routes_current_basis_baseline_comparison_without_proposal() -> None:
    planner = PlaybookPlanner()

    plan = planner.plan(
        user_prompt="Compare this basis to baseline.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "reserve_change_explanation"
    assert plan.answer_contract == "observational_explanation"
    assert "proposal_disallowed" in set(plan.basis_behavior)


def test_playbook_planner_routes_additional_drop_request_to_derived_drop() -> None:
    planner = PlaybookPlanner()

    plan = planner.plan(
        user_prompt="Please add three additional drops with the most positive impact.",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "derived_drop_expansion"
    assert plan.steps[0].tool_name == "tool_run_derived_drop_scenario"
    assert plan.steps[0].args["limit"] == 3


def test_playbook_planner_binds_analysis_basis_into_tail_review_plan() -> None:
    planner = PlaybookPlanner()

    plan = planner.plan(
        user_prompt="Review the tail assumptions.",
        session_context={"segment": "industrial", "session_id": "s-1"},
        analysis_basis={
            "basis_type": "review_candidate",
            "scenario_id": "drop_1",
            "parameters": {"drop": [["2022", 24]]},
        },
    )

    assert plan is not None
    assert plan.playbook == "tail_selection"
    assert plan.steps[0].args["basis_key"] == basis_key_from_parameters(
        {"drop": [["2022", 24]]}
    )
    assert "scenario_id" not in plan.steps[0].args
    assert plan.steps[0].args["parameters"]["drop"] == [["2022", 24]]


def test_playbook_planner_binds_analysis_basis_into_results_summary_step() -> None:
    planner = PlaybookPlanner()

    plan = planner.plan(
        user_prompt="What scenario do you recommend next?",
        session_context={"segment": "industrial", "session_id": "s-1"},
        analysis_basis={
            "basis_type": "review_candidate",
            "scenario_id": "drop_1",
            "parameters": {"drop": [["2022", 24]]},
        },
    )

    assert plan is not None
    result_step = plan.steps[2]
    assert result_step.tool_name == "tool_get_results_summary"
    assert result_step.args["basis_key"] == basis_key_from_parameters(
        {"drop": [["2022", 24]]}
    )
    assert "scenario_id" not in result_step.args
    assert result_step.args["parameters"]["drop"] == [["2022", 24]]


def test_playbook_planner_builds_data_anomaly_triage_plan() -> None:
    planner = PlaybookPlanner()

    plan = planner.plan(
        user_prompt="Please triage this data quality anomaly before we recommend changes",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert plan is not None
    assert plan.playbook == "data_anomaly_triage"
    assert [step.tool_name for step in plan.steps] == ["tool_run_anomaly_triage"]


def test_reviewer_gate_hard_fails_on_red_governance() -> None:
    planner = PlaybookPlanner()
    plan = planner.plan(
        user_prompt="What scenario do you recommend?",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )
    assert plan is not None

    review = ReviewerGate().review(
        plan=plan,
        evidence_packets=[
            {
                "evidence_key": "diagnostics_summary",
                "summary": {
                    "top_findings": [
                        {
                            "code": "DATA_QUALITY_GATE",
                            "severity": "critical",
                        }
                    ]
                },
                "governance": {"tier": "red"},
                "provenance": {"session_id": "s-1"},
            },
            {
                "evidence_key": "scenario_comparison",
                "summary": {},
                "governance": {},
                "provenance": {"session_id": "s-1"},
            },
        ],
    )

    assert review.status == "hard_fail"
    assert "red_governance_or_data_quality" in review.issues


def test_reviewer_gate_hard_fails_on_recommendation_without_evidence_id() -> None:
    planner = PlaybookPlanner()
    plan = planner.plan(
        user_prompt="What scenario do you recommend?",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )
    assert plan is not None

    review = ReviewerGate().review(
        plan=plan,
        evidence_packets=[
            {
                "evidence_key": "diagnostics_summary",
                "summary": {
                    "top_recommendations": [
                        {
                            "code": "RECOMMEND_DROP_2022_24",
                            "proposed_parameters": {"drop": [["2022", 24]]},
                            "evidence_id": "",
                        }
                    ]
                },
                "governance": {"tier": "green"},
                "provenance": {"session_id": "s-1"},
            },
            {
                "evidence_key": "scenario_comparison",
                "summary": {"top_scenarios": [{"scenario_id": "drop_1", "score": 1.0}]},
                "governance": {"tier": "green"},
                "provenance": {"session_id": "s-1"},
            },
        ],
    )

    assert review.status == "hard_fail"
    assert "recommendation_missing_evidence_id" in review.issues


def test_config_manager_ai_segment_memory_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
paths:
  results: "results/"
  plots: "plots/"
  data: "data/"
  sessions: "sessions/"
last date: "March 2026"
segment: "industrial"
""".strip(),
        encoding="utf-8",
    )
    manager = ConfigManager.from_yaml(config_path)

    manager.save_ai_segment_memory(
        {
            "segment_id": "industrial",
            "known_issues": ["Large refinery loss in 2021"],
        },
        segment="industrial",
    )

    loaded = manager.load_ai_segment_memory(segment="industrial")
    assert loaded["segment_id"] == "industrial"
    assert loaded["known_issues"] == ["Large refinery loss in 2021"]


def test_config_manager_ai_chat_logging_settings_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
paths:
  results: "results/"
  plots: "plots/"
  data: "data/"
  sessions: "sessions/"
last date: "March 2026"
segment: "industrial"
ai:
  chat_logging:
    enabled: true
    path: "chats"
""".strip(),
        encoding="utf-8",
    )

    manager = ConfigManager.from_yaml(config_path)

    assert manager.is_ai_chat_logging_enabled() is True
    assert manager.get_ai_chat_logging_path() == Path("chats")


def test_assistant_runs_deterministic_playbook_before_model_answer() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Recommendation ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", _DeterministicTools())
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    result = service.run_turn(
        user_prompt="What scenario do you recommend?",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert result["content"] == "Recommendation ready."
    tool_names = [name for name, _ in service._tools.calls]
    assert tool_names[:3] == [
        "tool_run_diagnostics_summary",
        "tool_iterate_diagnostics_summary",
        "tool_get_results_summary",
    ]
    packet = result["memory_snapshot"].get("deterministic_packet", {})
    assert packet.get("plan", {}).get("playbook") == "scenario_recommendation"
    assert packet.get("recommendation", {}).get("status") == "recommended"
    assert result["deterministic_packet"]["presentation"]["conclusion"] == "recommended"
    narration_packet = result["narration_packet"]
    assert narration_packet["answer_contract"] == "recommendation_with_proposal"
    assert "proposal_status" in narration_packet["required_answer_sections"]
    assert narration_packet["basis"]["basis_changed"] is False
    assert service._client.last_messages is not None
    assert "deterministic narration packet" in service._client.last_messages[-1]["content"]
    assert "required_answer_sections" in service._client.last_messages[-1]["content"]


def test_assistant_uses_accepted_basis_for_next_tail_review() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Tail recommendation ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    tools = _DeterministicTools()
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    result = service.run_turn(
        user_prompt="Review the tail assumptions.",
        session_context={"segment": "industrial", "session_id": "s-1"},
        working_memory={
            "accepted_analysis_basis": {
                "basis_type": "review_candidate",
                "basis_key": basis_key_from_parameters(
                    {
                        "average": "volume",
                        "drop": [["2022", 24]],
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
                ),
                "scenario_id": "drop_1",
                "is_active_session": False,
                "parameters": {
                    "average": "volume",
                    "drop": [["2022", 24]],
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
            }
        },
    )

    assert result["content"] == "Tail recommendation ready."
    tool_name, args = tools.calls[0]
    assert tool_name == "tool_run_tail_review"
    assert args["basis_key"] == basis_key_from_parameters(
        {
            "average": "volume",
            "drop": [["2022", 24]],
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
    )
    assert "scenario_id" not in args
    assert args["parameters"]["drop"] == [["2022", 24]]


def test_tool_basis_resolution_does_not_execute_from_scenario_label() -> None:
    resolved = AssistantService._resolve_tool_call_basis(
        args={"scenario_id": "drop_1"},
        memory_state={
            "scenario_basis_cache": {
                "basis-123": {
                    "basis_key": "basis-123",
                    "basis_type": "review_candidate",
                    "scenario_id": "drop_1",
                    "parameters": {"drop": [["2022", 24]]},
                }
            }
        },
        workflow_state={"current_user_prompt": "Use drop_1"},
        has_any_basis_arg=True,
    )

    assert resolved == {}


def test_pending_proposal_deterministic_packet_suppresses_follow_up_tools() -> None:
    service = AssistantService.__new__(AssistantService)
    service._observability_enabled = False

    filtered = service._filter_tool_specs_for_turn(
        tool_specs=[
            {"type": "function", "function": {"name": "tool_run_derived_drop_scenario"}},
            {"type": "function", "function": {"name": "tool_get_results_summary"}},
        ],
        deterministic_packet={
            "plan": {"answer_contract": "recommendation_with_proposal"},
            "proposal_basis": {"status": "pending", "proposal_id": "proposal-1"},
        },
        user_prompt="Please add three additional drops.",
        exact_data_required=False,
    )

    assert filtered == []


def test_deterministic_recommendation_creates_proposal_without_accepting_basis() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Recommendation ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", _DeterministicTools())
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    result = service.run_turn(
        user_prompt="What scenario do you recommend?",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    memory_snapshot = result["memory_snapshot"]
    assert memory_snapshot.get("accepted_analysis_basis", {}) == {}
    assert memory_snapshot["proposal_basis"]["status"] == "pending"
    assert memory_snapshot["proposal_basis"]["scenario_id"] == "drop_1"
    assert result["narration_packet"]["proposal"]["exists"] is True
    assert "proposal_changed_basis" in result["narration_packet"]["blocked_claims"]


def test_derived_drop_expansion_creates_proposal_from_derived_result() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Expanded drop recommendation ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    tools = _DeterministicTools()
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    accepted_parameters = {
        "average": "volume",
        "drop": [["2003", 9], ["2002", 21]],
        "drop_valuation": [],
        "tail": {
            "curve": "weibull",
            "attachment_age": 27,
            "projection_period": 0,
            "fit_period": [12, 108],
        },
        "bf_apriori": {},
        "final_ultimate": "chainladder",
        "selected_ultimate_by_uwy": {},
    }

    result = service.run_turn(
        user_prompt="I think just two drops are too few. Could you please add three additional drops with the most positive impact?",
        session_context={"segment": "industrial", "session_id": "s-1"},
        accepted_analysis_basis={
            "basis_type": "review_candidate",
            "scenario_id": "drop_combo_1",
            "parameters": accepted_parameters,
        },
    )

    assert result["content"] == "Expanded drop recommendation ready."
    assert [name for name, _ in tools.calls][:1] == ["tool_run_derived_drop_scenario"]
    tool_args = tools.calls[0][1]
    assert tool_args["limit"] == 3
    assert tool_args["include_existing_drops"] is True
    proposal = result["memory_snapshot"]["proposal_basis"]
    assert proposal["status"] == "pending"
    assert proposal["source_tool"] == "tool_run_derived_drop_scenario"
    assert proposal["parameters"]["drop"] == [
        ["2003", 9],
        ["2002", 21],
        ["2002", 39],
        ["2001", 60],
        ["2004", 3],
    ]
    assert proposal["basis_key"] == basis_key_from_parameters(proposal["parameters"])
    assert result["deterministic_packet"]["recommendation"]["recommended_basis_key"] == proposal["basis_key"]


def test_prior_drop_review_followup_combines_requested_top_drops() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Combined drop recommendation ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    tools = _DeterministicTools()
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    base_parameters = {
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
    result = service.run_turn(
        user_prompt="Use all your three drops.",
        session_context={"segment": "industrial", "session_id": "s-1"},
        working_memory={
            "review_summary": {
                "session_id": "s-1",
                "review_type": "drop_review",
                "analysis_basis": {
                    "basis_type": "baseline",
                    "parameters": base_parameters,
                },
                "top_candidates": [
                    {
                        "candidate_id": "drop_1",
                        "parameters": {**base_parameters, "drop": [["2002", 39]]},
                    },
                    {
                        "candidate_id": "drop_2",
                        "parameters": {**base_parameters, "drop": [["2001", 60]]},
                    },
                    {
                        "candidate_id": "drop_3",
                        "parameters": {**base_parameters, "drop": [["2001", 12]]},
                    },
                ],
            }
        },
    )

    assert result["content"] == "Combined drop recommendation ready."
    assert [name for name, _ in tools.calls][:1] == ["tool_recalculate"]
    assert tools.calls[0][1]["drop"] == [["2002", 39], ["2001", 60], ["2001", 12]]
    proposal = result["memory_snapshot"]["proposal_basis"]
    assert proposal["status"] == "pending"
    assert proposal["source_tool"] == "tool_recalculate"
    assert proposal["source_review_type"] == "combined_drop_recalculation"
    assert proposal["parameters"]["drop"] == [
        ["2002", 39],
        ["2001", 60],
        ["2001", 12],
    ]
    assert result["deterministic_packet"]["recommendation"]["recommended_basis_key"] == proposal["basis_key"]


def test_explicit_bf_request_preserves_current_drop_and_tail_basis() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "BF overview ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    tools = _DeterministicTools()
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

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
        "bf_apriori": {},
        "final_ultimate": "chainladder",
        "selected_ultimate_by_uwy": {},
    }

    result = service.run_turn(
        user_prompt="lets test bf with a priori 60% lr for the three newest ay and chainladder else",
        session_context={"segment": "industrial", "session_id": "s-1"},
        accepted_analysis_basis={
            "basis_type": "review_candidate",
            "scenario_id": "tail_weibull_27_12_108",
            "parameters": accepted_parameters,
        },
    )

    assert result["content"] == "BF overview ready."
    assert [name for name, _args in tools.calls][:2] == [
        "tool_get_results_summary",
        "tool_recalculate",
    ]
    recalc_args = tools.calls[1][1]
    assert recalc_args["drop"] == [["2002", 39]]
    assert recalc_args["tail"] == accepted_parameters["tail"]
    assert recalc_args["bf_apriori"] == {
        "2004": 0.6,
        "2005": 0.6,
        "2006": 0.6,
    }
    assert recalc_args["selected_ultimate_by_uwy"] == {
        "2004": "bornhuetter_ferguson",
        "2005": "bornhuetter_ferguson",
        "2006": "bornhuetter_ferguson",
    }
    assert result["memory_snapshot"]["proposal_basis"]["parameters"]["drop"] == [["2002", 39]]


def test_model_recalculate_partial_override_uses_current_basis_without_stale_selector() -> None:
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
        "bf_apriori": {},
        "final_ultimate": "chainladder",
        "selected_ultimate_by_uwy": {},
    }

    compiled = AssistantService._apply_default_analysis_basis_args(
        function_name="tool_recalculate",
        args={
            "session_id": "s-1",
            "basis_type": "review_candidate",
            "basis_key": "stale-basis-key",
            "bf_apriori": {"2006": 0.6},
            "selected_ultimate_by_uwy": {"2006": "bornhuetter_ferguson"},
        },
        memory_state={
            "accepted_analysis_basis": {
                "basis_type": "review_candidate",
                "basis_key": basis_key_from_parameters(accepted_parameters),
                "parameters": accepted_parameters,
            },
            "scenario_basis_cache": {},
        },
        workflow_state={"current_user_prompt": "test bf 60% for 2006"},
    )

    assert "basis_type" not in compiled
    assert "basis_key" not in compiled
    assert "scenario_id" not in compiled
    assert compiled["drop"] == [["2002", 39]]
    assert compiled["tail"] == accepted_parameters["tail"]
    assert compiled["bf_apriori"] == {"2006": 0.6}
    assert compiled["selected_ultimate_by_uwy"] == {
        "2006": "bornhuetter_ferguson"
    }


def test_explicit_baseline_results_call_drops_current_basis_parameters() -> None:
    current_parameters = {
        "average": "volume",
        "drop": [["2002", 39], ["2003", 9], ["2002", 21]],
        "drop_valuation": [],
        "tail": {
            "curve": "weibull",
            "attachment_age": 27,
            "projection_period": 0,
            "fit_period": [12, 108],
        },
        "bf_apriori": {"2004": 0.6, "2005": 0.6, "2006": 0.6},
        "final_ultimate": "chainladder",
        "selected_ultimate_by_uwy": {
            "2004": "bornhuetter_ferguson",
            "2005": "bornhuetter_ferguson",
            "2006": "bornhuetter_ferguson",
        },
    }

    compiled = AssistantService._apply_default_analysis_basis_args(
        function_name="tool_get_results_summary",
        args={
            "session_id": "s-1",
            "basis_type": "baseline",
            "basis_key": basis_key_from_parameters(current_parameters),
            "parameters": current_parameters,
        },
        memory_state={
            "session_summary": {
                "session_id": "s-1",
                "segment": "industrial",
                "params": {
                    "average": "volume",
                    "drop_store": [],
                    "tail_curve": "weibull",
                    "tail_attachment_age": None,
                    "tail_projection_months": 0,
                    "tail_fit_period_selection": [],
                    "bf_apriori_by_uwy": {},
                    "selected_ultimate_by_uwy": {},
                },
            },
            "accepted_analysis_basis": {
                "basis_type": "bespoke",
                "parameters": current_parameters,
            },
        },
        workflow_state={
            "current_user_prompt": "IBNR comparison between analysis basis and base line at the beginning"
        },
    )

    assert compiled["basis_type"] == "baseline"
    assert compiled["basis_key"] == basis_key_from_parameters(compiled["parameters"])
    assert compiled["parameters"]["drop"] == []
    assert compiled["parameters"]["tail"]["attachment_age"] is None
    assert compiled["parameters"]["bf_apriori"] == {}


def test_reserve_change_compare_uses_current_candidate_and_original_baseline() -> None:
    current_parameters = {
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

    compiled = AssistantService._apply_default_analysis_basis_args(
        function_name="tool_explain_reserve_change",
        args={"session_id": "s-1"},
        memory_state={
            "session_summary": {
                "session_id": "s-1",
                "segment": "industrial",
                "params": {
                    "average": "volume",
                    "drop_store": [],
                    "tail_curve": "weibull",
                    "tail_attachment_age": None,
                    "tail_projection_months": 0,
                    "tail_fit_period_selection": [],
                    "bf_apriori_by_uwy": {},
                    "selected_ultimate_by_uwy": {},
                },
            },
            "accepted_analysis_basis": {
                "basis_type": "bespoke",
                "parameters": current_parameters,
            },
        },
        workflow_state={
            "current_user_prompt": "compare analysis basis to the scenario before all the modifications"
        },
    )

    assert compiled["drop"] == [["2002", 39]]
    assert compiled["tail"]["attachment_age"] == 27
    assert compiled["bf_apriori"] == {"2006": 0.6}
    assert compiled["basis_type"] == "baseline"
    assert compiled["basis_parameters"]["drop"] == []
    assert compiled["basis_parameters"]["tail"]["attachment_age"] is None


def test_recalculate_updates_preview_basis_and_session_summary() -> None:
    service = AssistantService.__new__(AssistantService)

    updated = service._update_memory_state(
        function_name="tool_recalculate",
        tool_result={
            "session_id": "s-1",
            "analysis_basis": {
                "basis_type": "bespoke",
                "scenario_id": None,
                "is_active_session": True,
                "parameters": {
                    "average": "volume",
                    "drop": [["2003", 9], ["2002", 21]],
                    "drop_valuation": [],
                    "tail": {
                        "curve": "weibull",
                        "attachment_age": 27,
                        "projection_period": 0,
                        "fit_period": [12, 108],
                    },
                    "bf_apriori": {"2006": 0.563},
                    "final_ultimate": "chainladder",
                    "selected_ultimate_by_uwy": {"2006": "bornhuetter_ferguson"},
                },
            },
            "results_table_rows": [],
            "duration_ms": 12,
        },
        memory_state={
            "session_summary": {
                "session_id": "s-1",
                "segment": "industrial",
                "params": {
                    "average": "volume",
                    "tail_curve": "weibull",
                    "tail_attachment_age": None,
                    "tail_projection_months": 0,
                    "tail_fit_period_selection": [],
                    "drop_store": [],
                    "drop_count": 0,
                    "bf_apriori_by_uwy": {},
                    "selected_ultimate_by_uwy": {},
                },
            },
            "scenario_basis_cache": {},
        },
    )

    assert updated["preview_basis"] == {}
    assert updated["session_summary"]["params"]["drop_store"] == [
        ["2003", 9],
        ["2002", 21],
    ]


def test_preview_recalculate_keeps_session_summary_unchanged() -> None:
    service = AssistantService.__new__(AssistantService)

    updated = service._update_memory_state(
        function_name="tool_recalculate",
        tool_result={
            "session_id": "s-1",
            "analysis_basis": {
                "basis_type": "bespoke",
                "scenario_id": None,
                "is_active_session": False,
                "parameters": {
                    "average": "volume",
                    "drop": [["2003", 9]],
                    "drop_valuation": [],
                    "tail": {
                        "curve": "weibull",
                        "attachment_age": 27,
                        "projection_period": 0,
                        "fit_period": [12, 108],
                    },
                    "bf_apriori": {},
                    "final_ultimate": "chainladder",
                    "selected_ultimate_by_uwy": {},
                },
            },
            "results_table_rows": [],
            "duration_ms": 12,
        },
        memory_state={
            "session_summary": {
                "session_id": "s-1",
                "segment": "industrial",
                "params": {
                    "average": "volume",
                    "tail_curve": "weibull",
                    "tail_attachment_age": None,
                    "tail_projection_months": 0,
                    "tail_fit_period_selection": [],
                    "drop_store": [],
                    "drop_count": 0,
                    "bf_apriori_by_uwy": {},
                    "selected_ultimate_by_uwy": {},
                },
            },
            "scenario_basis_cache": {},
        },
    )

    assert updated["preview_basis"]["parameters"]["drop"] == [["2003", 9]]
    assert updated["session_summary"]["params"]["drop_store"] == []


def test_assistant_logs_deterministic_orchestration(caplog) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Recommendation ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    setattr(service, "_tools", _DeterministicTools())
    service._observability_enabled = True
    service._deterministic_orchestration_enabled = True

    with caplog.at_level(logging.INFO, logger="ai.assistant_service"):
        service.run_turn(
            user_prompt="What scenario do you recommend?",
            session_context={"segment": "industrial", "session_id": "s-1"},
        )

    merged = "\n".join(record.getMessage() for record in caplog.records)
    assert "deterministic.playbook.selected playbook=scenario_recommendation" in merged
    assert (
        "deterministic.review.completed playbook=scenario_recommendation status=pass"
        in merged
    )
    assert (
        "deterministic.recommendation.completed playbook=scenario_recommendation status=recommended"
        in merged
    )


def test_narrative_guardrails_add_execution_note_for_material_adjustment() -> None:
    guarded = AssistantService._apply_narrative_guardrails(
        "I applied the requested scenario and used those assumptions in the analysis.",
        {},
        [
            {
                "execution_status": "partially_executed",
                "warnings": ["Dropped 1 invalid drop entry."],
            }
        ],
    )

    assert "latest tool run did not execute exactly as requested" in guarded.lower()


def test_narrative_guardrails_add_execution_note_for_rejected_request() -> None:
    guarded = AssistantService._apply_narrative_guardrails(
        "The scenario ran successfully.",
        {},
        [
            {
                "execution_status": "rejected",
                "warnings": ["Request could not be executed."],
            }
        ],
    )

    assert "latest tool request was rejected" in guarded.lower()


def test_assistant_suppresses_redundant_summary_tools_after_deterministic_packet() -> (
    None
):
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Recommendation ready.",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]
    client = _FakeClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    tools = _DeterministicTools()
    tools.tool_specs = [
        {"type": "function", "function": {"name": "tool_run_diagnostics_summary"}},
        {"type": "function", "function": {"name": "tool_iterate_diagnostics_summary"}},
        {"type": "function", "function": {"name": "tool_get_results_summary"}},
        {"type": "function", "function": {"name": "tool_get_data_view_summary"}},
    ]
    setattr(service, "_tools", tools)
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = True

    service.run_turn(
        user_prompt="What scenario do you recommend?",
        session_context={"segment": "industrial", "session_id": "s-1"},
    )

    assert len(client.tool_payloads) == 1
    offered_names = {
        spec.get("function", {}).get("name")
        for spec in client.tool_payloads[0]
        if isinstance(spec, dict)
    }
    assert offered_names == set()
    assert "tool_run_diagnostics_summary" not in offered_names
    assert "tool_iterate_diagnostics_summary" not in offered_names
    assert "tool_get_results_summary" not in offered_names


def test_chat_service_exposes_deterministic_packet_top_level() -> None:
    class _AssistantStub:
        def run_turn(self, **_kwargs):
            return {
                "content": "Recommendation ready.",
                "fallback_used": False,
                "session_id": "s-1",
                "tool_events": [],
                "memory_snapshot": {
                    "scenario_ledger": [{"scenario_id": "drop_1", "score": 1.0}],
                    "deterministic_packet": {
                        "plan": {"playbook": "scenario_recommendation"},
                        "review": {"status": "pass"},
                        "recommendation": {"status": "recommended"},
                    },
                },
            }

    service = AIChatService(
        assistant_factory=lambda: _AssistantStub(),
        store=InMemoryChatStore(),
    )
    session = service.create_chat(segment="industrial", reserving_session_id="s-1")

    response = service.start_message(session.chat_id, "What scenario do you recommend?")
    for _ in range(50):
        current = service.build_chat_response(session.chat_id)
        if not current.get("streaming"):
            break

    final_response = service.build_chat_response(session.chat_id)
    assert (
        final_response["deterministic_packet"]["plan"]["playbook"]
        == "scenario_recommendation"
    )
    assert (
        final_response["deterministic_packet"]["recommendation"]["status"]
        == "recommended"
    )
