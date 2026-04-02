from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.assistant_service import AssistantService


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.last_messages = None

    def chat_completion(self, **_kwargs):
        self.last_messages = _kwargs.get("messages")
        return self._responses.pop(0)


class _FailingClient:
    def __init__(self):
        self._calls = 0

    def chat_completion(self, **_kwargs):
        self._calls += 1
        if self._calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "tool_run_diagnostics",
                                        "arguments": '{"session_id": "s-1"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        raise RuntimeError("provider outage")


class _FakeTools:
    tool_specs = []

    def __init__(self):
        self.calls = []

    def call_tool(self, function_name, args):
        self.calls.append((function_name, args))
        if function_name == "tool_get_assumption_context_detail":
            return {
                "session_id": "s-1",
                "parameters": {"average": "volume"},
                "selected_ldf": [
                    {"age": 21, "development_label": "21-24", "ldf": 1.058}
                ],
                "fitted_tail_ldf": [
                    {"age": 30, "development_label": "30-33", "ldf": 1.048}
                ],
                "observed_a2a": [],
                "bf_apriori_by_uwy": {"2005": 0.5988},
                "selected_ultimate_by_uwy": {"2005": "bornhuetter_ferguson"},
            }
        if function_name == "tool_run_diagnostics":
            return {
                "session_id": "s-1",
                "findings": [{"code": "TAIL_SENSITIVITY_HIGH"}],
                "metrics": {"assessment_confidence": 0.7},
            }
        if function_name == "tool_iterate_diagnostics":
            return {
                "session_id": "s-1",
                "scenarios": [{"scenario_id": "drop_1", "findings": []}],
                "iteration_metrics": {"best_scenario_id": "drop_1"},
            }
        return {"session_id": "s-1"}


def test_answer_does_not_force_iteration_before_final_output() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "tool_run_diagnostics",
                                    "arguments": '{"session_id": "s-1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "Final commentary after diagnostics.",
                        "tool_calls": [],
                    }
                }
            ]
        },
    ]

    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    fake_tools = _FakeTools()
    setattr(service, "_tools", fake_tools)
    service._observability_enabled = False

    result = service.answer(user_prompt="Run diagnostics")

    tool_names = [name for name, _ in fake_tools.calls]
    assert "tool_run_diagnostics" in tool_names
    assert "tool_iterate_diagnostics" not in tool_names
    assert "Final commentary" in result


def test_answer_returns_graceful_message_when_provider_fails_after_diagnostics() -> (
    None
):
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FailingClient())
    setattr(service, "_tools", _FakeTools())
    service._observability_enabled = False

    result = service.answer(user_prompt="Run diagnostics", max_steps=2)

    assert "temporarily unavailable" in result.lower()


def test_answer_includes_ai_context_prompt() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    client = _FakeClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    setattr(service, "_tools", _FakeTools())
    service._observability_enabled = False

    result = service.answer(user_prompt="What does current quarter mean?")

    assert result == "ok"
    system_messages = [
        item.get("content", "")
        for item in (client.last_messages or [])
        if item.get("role") == "system"
    ]
    assert any("current quarter" in content.lower() for content in system_messages)
    assert any("latest diagonal" in content.lower() for content in system_messages)
    assert any("movement review" in content.lower() for content in system_messages)
    assert any(
        "claims movement this quarter" in content.lower() for content in system_messages
    )


def test_movement_question_does_not_force_iteration() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "tool_run_diagnostics",
                                    "arguments": '{"session_id": "s-1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "There are unusual incurred movements this quarter.",
                        "tool_calls": [],
                    }
                }
            ]
        },
    ]

    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    fake_tools = _FakeTools()
    setattr(service, "_tools", fake_tools)
    service._observability_enabled = False

    result = service.answer(
        user_prompt="Are there unusual and unexpected incurred claims movements this quarter?"
    )

    tool_names = [name for name, _ in fake_tools.calls]
    assert "tool_run_diagnostics" in tool_names
    assert "tool_iterate_diagnostics" not in tool_names
    assert "unusual incurred movements" in result.lower()


def test_recommendation_question_includes_iteration_hint() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    client = _FakeClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    setattr(service, "_tools", _FakeTools())
    service._observability_enabled = False

    result = service.answer(user_prompt="What scenario do you recommend?")

    assert result == "ok"
    system_messages = [
        item.get("content", "")
        for item in (client.last_messages or [])
        if item.get("role") == "system"
    ]
    assert any(
        "asking for recommendations" in content.lower() for content in system_messages
    )
    assert any(
        "selected playbook: scenario recommendation" in content.lower()
        for content in system_messages
    )


def test_claims_movement_question_includes_incurred_latest_diagonal_hint() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    client = _FakeClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    setattr(service, "_tools", _FakeTools())
    service._observability_enabled = False

    result = service.answer(
        user_prompt="Are there unusual and unexpected claims movements this quarter?"
    )

    assert result == "ok"
    system_messages = [
        item.get("content", "")
        for item in (client.last_messages or [])
        if item.get("role") == "system"
    ]
    merged = "\n".join(system_messages).lower()
    assert "claims without a modifier" in merged
    assert "treat that as incurred" in merged
    assert "latest diagonal" in merged
    assert "selected playbook: movement review" in merged


def test_claims_movement_question_prefetches_incurred_and_a2a_evidence() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    fake_tools = _FakeTools()
    setattr(service, "_tools", fake_tools)
    service._observability_enabled = False

    result = service.run_turn(
        user_prompt="Are there unusual and unexpected claims movements this quarter?",
        session_context={"segment": "seg", "session_id": "s-1"},
    )

    assert result["content"] == "ok"
    tool_names = [name for name, _ in fake_tools.calls]
    assert tool_names[:3] == [
        "tool_get_data_view_summary",
        "tool_get_data_view_summary",
        "tool_run_ldf_consistency_diagnostics",
    ]


def test_tail_selection_prompt_includes_proactive_subunit_and_attachment_checks() -> (
    None
):
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    client = _FakeClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    setattr(service, "_tools", _FakeTools())
    service._observability_enabled = False

    result = service.answer(user_prompt="How should we set the tail?")

    assert result == "ok"
    system_messages = [
        item.get("content", "")
        for item in (client.last_messages or [])
        if item.get("role") == "system"
    ]
    merged = "\n".join(system_messages).lower()
    assert "late selected ldfs below 1.0" in merged
    assert "sharp drop" in merged
    assert "first fitted tail ldf" in merged


def test_drop_reason_prompt_forbids_unsupported_reason_labels() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    client = _FakeClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    setattr(service, "_tools", _FakeTools())
    service._observability_enabled = False

    result = service.answer(user_prompt="List me all drops you have used")

    assert result == "ok"
    system_messages = [
        item.get("content", "")
        for item in (client.last_messages or [])
        if item.get("role") == "system"
    ]
    merged = "\n".join(system_messages).lower()
    assert "load exact scenario or derived-drop detail first" in merged
    assert (
        "only assign a drop reason if the tool output gives explicit support" in merged
    )
    assert "do not relabel a drop as 'below 1.0'" in merged


def test_exact_factor_question_prefetches_assumption_detail() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    fake_tools = _FakeTools()
    setattr(service, "_tools", fake_tools)
    service._observability_enabled = False

    result = service.run_turn(
        user_prompt="Can you show me the fitted LDFs from 21 up to 45?",
        session_context={"segment": "seg", "session_id": "s-1"},
    )

    assert result["content"] == "ok"
    tool_names = [name for name, _ in fake_tools.calls]
    assert tool_names[0] == "tool_get_assumption_context_detail"


def test_exact_follow_up_uses_bound_recommended_scenario_basis() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    fake_tools = _FakeTools()
    setattr(service, "_tools", fake_tools)
    service._observability_enabled = False

    result = service.run_turn(
        user_prompt="What are the fitted tail LDFs from 27 to 45 in your recommended scenario?",
        session_context={"segment": "seg", "session_id": "s-1"},
        working_memory={
            "analysis_basis": {
                "basis_type": "review_candidate",
                "session_id": "s-1",
                "scenario_id": "drop_combo_1",
                "is_active_session": False,
                "parameters": {
                    "average": "volume",
                    "drop": [["2003", 9], ["2002", 21], ["2002", 39]],
                    "drop_valuation": [],
                    "tail": {
                        "curve": "weibull",
                        "attachment_age": 27,
                        "projection_period": 0,
                        "fit_period": [12, 108],
                    },
                    "bf_apriori": {"2005": 0.5988, "2006": 0.5824},
                    "final_ultimate": "chainladder",
                    "selected_ultimate_by_uwy": {
                        "2005": "bornhuetter_ferguson",
                        "2006": "bornhuetter_ferguson",
                    },
                },
            }
        },
    )

    assert result["content"] == "ok"
    tool_name, args = fake_tools.calls[0]
    assert tool_name == "tool_get_assumption_context_detail"
    assert args["scenario_id"] == "drop_combo_1"
    assert args["basis_type"] == "review_candidate"
    assert args["parameters"]["tail"]["attachment_age"] == 27


def test_exact_follow_up_can_switch_back_to_baseline() -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }
    ]

    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", _FakeClient(responses))
    fake_tools = _FakeTools()
    setattr(service, "_tools", fake_tools)
    service._observability_enabled = False

    result = service.run_turn(
        user_prompt="What are the fitted tail LDFs from 27 to 45 in the baseline?",
        session_context={"segment": "seg", "session_id": "s-1"},
        working_memory={
            "session_summary": {
                "session_id": "s-1",
                "segment": "seg",
                "params": {
                    "average": "volume",
                    "tail_curve": "weibull",
                    "tail_attachment_age": 30,
                    "tail_projection_months": 0,
                    "tail_fit_period_selection": [12, 45],
                    "drop_store": [],
                    "drop_count": 0,
                    "bf_apriori_by_uwy": {},
                    "selected_ultimate_by_uwy": {},
                },
            },
            "analysis_basis": {
                "basis_type": "review_candidate",
                "session_id": "s-1",
                "scenario_id": "drop_combo_1",
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
        },
    )

    assert result["content"] == "ok"
    tool_name, args = fake_tools.calls[0]
    assert tool_name == "tool_get_assumption_context_detail"
    assert args["scenario_id"] == "baseline"
    assert args["basis_type"] == "baseline"
    assert args["parameters"]["tail"]["attachment_age"] == 30
