from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.assistant_service import AssistantService
from ai.tool_payloads import compact_tool_result_for_model


def _assumption_detail_payload() -> dict:
    return {
        "session_id": "s-1",
        "metric": "incurred",
        "analysis_basis": {
            "basis_type": "review_candidate",
            "session_id": "s-1",
            "scenario_id": "drop_combo_1",
            "candidate_id": "drop_1",
            "scenario_signature": "abc123",
            "source_tool": "tool_iterate_diagnostics_summary",
            "source_review_type": "tail_review",
            "is_active_session": False,
            "parameters": {
                "average": "volume",
                "drop": [["2001", 60], ["2002", 39]],
                "drop_valuation": [["2024Q4", 60]],
                "tail": {
                    "curve": "weibull",
                    "attachment_age": 27,
                    "projection_period": 0,
                    "fit_period": [12, 108],
                },
                "bf_apriori": {"2005": 0.61},
                "final_ultimate": "chainladder",
                "selected_ultimate_by_uwy": {"2005": "bornhuetter_ferguson"},
                "raw_basis_blob": "remove-me",
            },
        },
        "parameters": {
            "average": "volume",
            "drop": [["2001", 60], ["2002", 39]],
            "drop_valuation": [["2024Q4", 60]],
            "tail": {
                "curve": "weibull",
                "attachment_age": 27,
                "projection_period": 0,
                "fit_period": [12, 108],
            },
            "bf_apriori": {"2005": 0.61},
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": {"2005": "bornhuetter_ferguson"},
            "raw_parameter_blob": "remove-me",
        },
        "selected_ldf": [
            {"age": 27, "development_label": "27-30", "ldf": 1.017891},
            {"age": 30, "development_label": "30-33", "ldf": 1.012345},
        ],
        "fitted_tail_ldf": [
            {"age": 27, "development_label": "27-30", "ldf": 1.009691},
            {"age": 30, "development_label": "30-33", "ldf": 1.006543},
        ],
        "tail_active": False,
        "tail_mode": "reference_fit_only",
        "tail_applies_from_age": None,
        "observed_a2a": [
            {
                "origin": "2005",
                "age": 27,
                "development_label": "27-30",
                "a2a": 1.0132,
            }
        ],
        "bf_apriori_by_uwy": {"2005": 0.61, "2006": 0.58},
        "selected_ultimate_by_uwy": {
            "2005": "bornhuetter_ferguson",
            "2006": "chainladder",
        },
    }


def _data_view_payload() -> dict:
    return {
        "session_id": "s-1",
        "analysis_basis": {},
        "query": {"metric": "incurred", "view": "cumulative"},
        "summary": {
            "shape": {"rows": 12, "columns": 6},
            "latest_age": 60,
            "latest_diagonal_total": 1234.5,
            "top_latest_rows": [{"uwy": "2005", "value": 210.0}],
            "top_latest_diagonal_rows": [{"uwy": "2005", "value": 75.0}],
            "late_movement_candidates": [{"uwy": "2006", "value": 44.0}],
        },
        "data": {
            "records": [
                {"uwy": str(2000 + index), "12": float(index), "24": float(index + 1)}
                for index in range(12)
            ]
        },
    }


class _RecordingClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.messages_per_call = []

    def chat_completion(self, **kwargs):
        self.messages_per_call.append(kwargs.get("messages") or [])
        return self._responses.pop(0)


class _CompactionTools:
    tool_specs = [
        {
            "type": "function",
            "function": {"name": "tool_get_assumption_context_detail"},
        },
        {"type": "function", "function": {"name": "tool_get_data_view"}},
    ]

    def __init__(self):
        self.calls = []

    def call_tool(self, function_name, args):
        self.calls.append((function_name, dict(args)))
        if function_name == "tool_get_assumption_context_detail":
            return _assumption_detail_payload()
        if function_name == "tool_get_data_view":
            return _data_view_payload()
        return {"session_id": "s-1"}


def test_compact_tool_result_for_assumption_detail_strips_raw_parameters() -> None:
    compacted = compact_tool_result_for_model(
        tool_name="tool_get_assumption_context_detail",
        result=_assumption_detail_payload(),
    )

    assert compacted["parameter_summary"]["drop_count"] == 2
    assert compacted["parameter_summary"]["drop_preview"] == [
        ["2001", 60],
        ["2002", 39],
    ]
    assert compacted["analysis_basis"]["parameter_summary"]["drop_count"] == 2
    assert "parameters" not in compacted
    assert "parameters" not in compacted["analysis_basis"]
    assert "raw_parameter_blob" not in json.dumps(compacted)
    assert "raw_basis_blob" not in json.dumps(compacted)
    assert compacted["selected_ldf"][0]["age"] == 27
    assert compacted["fitted_tail_ldf"][0]["ldf"] == 1.009691


def test_exact_prefetch_system_prompt_uses_compact_assumption_detail_payload() -> None:
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

    client = _RecordingClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    setattr(service, "_tools", _CompactionTools())
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = False

    result = service.run_turn(
        user_prompt="What are the fitted tail LDFs from 27 to 45 in the baseline?",
        session_context={"segment": "seg", "session_id": "s-1"},
    )

    assert result["content"] == "ok"
    system_messages = [
        item.get("content", "")
        for item in client.messages_per_call[0]
        if item.get("role") == "system"
    ]
    merged = "\n".join(system_messages)
    assert "parameter_summary" in merged
    assert "drop_preview" in merged
    assert "raw_parameter_blob" not in merged
    assert "raw_basis_blob" not in merged


def test_assistant_replays_compact_data_view_tool_payload() -> None:
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
                                    "name": "tool_get_data_view",
                                    "arguments": '{"session_id": "s-1", "metric": "incurred", "view": "cumulative"}',
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
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        },
    ]

    client = _RecordingClient(responses)
    service = AssistantService.__new__(AssistantService)
    setattr(service, "_client", client)
    setattr(service, "_tools", _CompactionTools())
    service._observability_enabled = False
    service._deterministic_orchestration_enabled = False

    result = service.run_turn(
        user_prompt="Show me the incurred triangle.",
        session_context={"segment": "seg", "session_id": "s-1"},
    )

    assert result["content"] == "ok"
    tool_messages = [
        item for item in client.messages_per_call[1] if item.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload["record_count"] == 12
    assert len(payload["sample_rows"]) == 5
    assert payload["sample_rows"][0]["uwy"] == "2000"
    assert "data" not in payload
