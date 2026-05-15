from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.segment_memory_service import SegmentMemoryService


def test_segment_memory_service_migrates_v1_rejections() -> None:
    service = SegmentMemoryService()

    loaded = service.load(
        {
            "segment_id": "industrial",
            "rejected_scenarios": [
                {
                    "scenario_hash": "abc123",
                    "reason": "fragile improvement",
                }
            ],
        },
        segment="industrial",
    )

    assert loaded["schema_version"] == 3
    assert loaded["segment_id"] == "industrial"
    assert loaded["scenario_dispositions"][0]["scenario_signature"] == "abc123"
    assert loaded["scenario_dispositions"][0]["decision"] == "rejected"


def test_segment_memory_merge_preserves_existing_fields_on_partial_update() -> None:
    service = SegmentMemoryService()

    merged = service.merge(
        existing_memory={
            "segment_id": "industrial",
            "house_preferences": ["prefer stable tail"],
            "valuation_history": [{"data_fingerprint": "fp-1"}],
        },
        incoming_memory={"last_recommendation": {"status": "recommended"}},
        segment="industrial",
    )

    assert merged["house_preferences"] == ["prefer stable tail"]
    assert merged["last_recommendation"]["status"] == "recommended"
    assert merged["valuation_history"][0]["data_fingerprint"] == "fp-1"


def test_segment_memory_append_valuation_snapshot_dedupes_same_basis() -> None:
    service = SegmentMemoryService()
    memory = service.load({}, segment="industrial")

    updated = service.append_valuation_snapshot(
        memory=memory,
        snapshot={
            "comparison_basis": "current",
            "data_fingerprint": "fp-1",
        },
    )
    updated = service.append_valuation_snapshot(
        memory=updated,
        snapshot={
            "comparison_basis": "current",
            "data_fingerprint": "fp-1",
        },
    )

    assert len(updated["valuation_history"]) == 1


def test_segment_memory_continuity_summary_surfaces_rejections() -> None:
    summary = SegmentMemoryService().continuity_summary(
        {
            "segment_id": "industrial",
            "scenario_dispositions": [
                {
                    "scenario_signature": "sig-1",
                    "decision": "rejected",
                }
            ],
        }
    )

    assert summary["segment_id"] == "industrial"
    assert summary["recent_rejected_signatures"] == ["sig-1"]


def test_segment_memory_append_scenario_disposition_replaces_same_signature() -> None:
    service = SegmentMemoryService()
    memory = service.load({}, segment="industrial")
    updated = service.append_scenario_disposition(
        memory=memory,
        disposition={
            "scenario_signature": "sig-1",
            "scenario_id": "drop_1",
            "decision": "rejected",
        },
    )
    updated = service.append_scenario_disposition(
        memory=updated,
        disposition={
            "scenario_signature": "sig-1",
            "scenario_id": "drop_1",
            "decision": "accepted",
        },
    )

    assert len(updated["scenario_dispositions"]) == 1
    assert updated["scenario_dispositions"][0]["decision"] == "accepted"


def test_segment_memory_loads_phase3_authoring_fields() -> None:
    loaded = SegmentMemoryService().load(
        {
            "segment_id": "industrial",
            "segment_overview": "US industrial liability book.",
            "open_items": ["Review TPA reporting lag shift"],
            "recent_quarter_notes": [
                {
                    "period": "2026Q1",
                    "note": "Case strengthening followed claims review",
                }
            ],
            "memory_change_log": [
                {
                    "field": "open_items",
                    "action": "approve_proposal",
                    "summary": "Carry forward reporting lag review",
                }
            ],
        },
        segment="industrial",
    )

    assert loaded["schema_version"] == 3
    assert loaded["segment_overview"] == "US industrial liability book."
    assert loaded["open_items"] == ["Review TPA reporting lag shift"]
    assert loaded["recent_quarter_notes"][0]["period"] == "2026Q1"
    assert loaded["memory_change_log"][0]["field"] == "open_items"
