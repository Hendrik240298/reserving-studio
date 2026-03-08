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

    def chat_completion(self, **_kwargs):
        return self._responses.pop(0)


class _FakeTools:
    tool_specs = []

    def __init__(self):
        self.calls = []

    def call_tool(self, function_name, args):
        self.calls.append((function_name, args))
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


def test_answer_forces_iteration_before_final_output() -> None:
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
                        "content": "I will run iterative scenario analysis next.",
                        "tool_calls": [],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "function": {
                                    "name": "tool_iterate_diagnostics",
                                    "arguments": '{"session_id": "s-1", "include_baseline": true}',
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
                        "content": "Final commentary after iteration.",
                        "tool_calls": [],
                    }
                }
            ]
        },
    ]

    service = AssistantService.__new__(AssistantService)
    service._client = _FakeClient(responses)
    fake_tools = _FakeTools()
    service._tools = fake_tools
    service._observability_enabled = False

    result = service.answer(user_prompt="Run diagnostics")

    tool_names = [name for name, _ in fake_tools.calls]
    assert "tool_run_diagnostics" in tool_names
    assert "tool_iterate_diagnostics" in tool_names
    assert "Final commentary" in result
