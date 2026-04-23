from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.execution_records import attach_execution_metadata
from ai.request_validation import build_passthrough_request_validation


def test_attach_execution_metadata_embeds_execution_record() -> None:
    summary = {"session_id": "s-1", "delta_ibnr": 20.0}
    validation = build_passthrough_request_validation(
        {"session_id": "s-1", "basis_type": "baseline"}
    )

    payload = attach_execution_metadata(
        summary,
        tool_name="tool_explain_reserve_change",
        validation=validation,
        session_id="s-1",
    )

    assert payload["execution_status"] == "executed_exactly"
    assert payload["execution_record"]["tool_name"] == "tool_explain_reserve_change"
    assert payload["execution_record"]["requested_inputs"]["basis_type"] == "baseline"
