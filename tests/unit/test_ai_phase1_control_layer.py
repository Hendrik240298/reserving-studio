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
        if function_name == "tool_get_results_summary":
            return {
                "session_id": "s-1",
                "result_row_count": 4,
                "top_rows": [{"uwy": "2022", "selected_method": "chainladder"}],
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
    assert plan.steps[0].args["scenario_id"] == "drop_1"
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
    assert result_step.args["scenario_id"] == "drop_1"
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
    assert args["scenario_id"] == "drop_1"
    assert args["parameters"]["drop"] == [["2022", 24]]


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
    assert "tool_get_data_view_summary" in offered_names
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
