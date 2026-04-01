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

    assert loaded["schema_version"] == 2
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
