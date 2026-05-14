from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.api_tools import ReservingApiTools


class _CountingApiTools(ReservingApiTools):
    def __init__(self) -> None:
        super().__init__(base_url="http://example.test")
        self.request_count = 0

    def request_json(self, method, path, body=None):
        self.request_count += 1
        raise AssertionError("Rejected tool requests must not call the API")


def test_recalculate_rejects_invalid_request_without_api_call() -> None:
    tools = _CountingApiTools()

    result = tools.call_tool(
        "tool_recalculate",
        {
            "session_id": "s-1",
            "average": "volume",
            "tail": {"curve": "made_up_curve", "fit_period": [12, 24]},
            "drop": [],
            "drop_valuation": [],
            "selected_ultimate_by_uwy": {},
        },
    )

    assert tools.request_count == 0
    assert result["execution_status"] == "rejected"
    assert result["rejected"] is True
    assert result["rejection_reason"] == "Unsupported tail curve assumption: made_up_curve."
    assert result["execution_record"]["effective_inputs"] == {}
