from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.memory_authoring_service import MemoryAuthoringService


def test_build_context_packet_limits_recent_quarter_notes_only() -> None:
    packet = MemoryAuthoringService().build_context_packet(
        {
            "segment_id": "industrial",
            "segment_overview": "Industrial liability",
            "known_issues": ["issue 1", "issue 2", "issue 3", "issue 4", "issue 5"],
            "house_preferences": ["prefer stable tail"],
            "open_items": ["item 1", "item 2", "item 3", "item 4", "item 5"],
            "recent_quarter_notes": [
                {"period": f"2025Q{i}", "note": f"note {i}"} for i in range(1, 7)
            ],
        }
    )

    assert packet["known_issues"] == [
        "issue 1",
        "issue 2",
        "issue 3",
        "issue 4",
        "issue 5",
    ]
    assert packet["open_items"] == ["item 1", "item 2", "item 3", "item 4", "item 5"]
    assert len(packet["recent_quarter_notes"]) == 4


def test_apply_memory_proposal_appends_open_item() -> None:
    service = MemoryAuthoringService()
    proposal = service.normalize_proposal(
        {
            "field": "open_items",
            "operation": "append",
            "value": ["Review TPA reporting lag"],
            "rationale": "Carry this forward.",
        }
    )

    updates = service._proposal_updates(  # noqa: SLF001
        proposal,
        existing_memory={"open_items": ["Existing item"]},
    )

    assert updates["open_items"] == ["Existing item", "Review TPA reporting lag"]


def test_build_ui_payload_preserves_structured_house_preferences() -> None:
    payload = MemoryAuthoringService().build_ui_payload(
        {
            "segment_id": "industrial",
            "house_preferences": [
                "prefer stable tail",
                {"type": "max_drop_count", "value": 1},
            ],
        }
    )

    assert payload["house_preferences_text"] == "prefer stable tail"
    assert payload["structured_house_preferences"] == [
        {"type": "max_drop_count", "value": 1}
    ]
