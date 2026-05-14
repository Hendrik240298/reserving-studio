from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.markdown import render_drop_review_packet


def test_render_drop_review_packet_includes_result_and_execution_details() -> None:
    packet = render_drop_review_packet(
        review_payload={
            "candidates": [
                {
                    "candidate_id": "drop_1",
                    "recommendation_class": "recommended",
                    "summary": "A tested drop improves diagnostics.",
                },
                {
                    "candidate_id": "drop_2",
                    "recommendation_class": "watch",
                    "summary": "A second candidate is also displayed.",
                },
            ],
        },
        summary_payload={
            "review_type": "drop_review",
            "candidate_count": 2,
            "top_candidates": [
                {
                    "candidate_id": "drop_1",
                    "recommendation_class": "recommended",
                    "summary": "A tested drop improves diagnostics.",
                    "metrics": {"reserve_delta": 10.0},
                }
            ],
            "recommendation": {
                "recommendation_class": "recommended",
                "candidate_id": "drop_1",
                "summary": "Use drop_1 as a review candidate.",
                "caveats": ["Human acceptance required."],
            },
            "evidence_summary": {"tested_candidates": 1},
            "policy_trace": {"governance_tier": "green"},
        },
        requested_inputs={"config_path": "examples/config_quarterly.yml"},
        effective_inputs={
            "segment": "quarterly",
            "granularity": "quarterly",
            "dataset": "quarterly",
            "quarterly_premium_csv": "data/quarterly_premium.csv",
            "candidate_limit": 5,
        },
        command="uv run python -m harness.cli drop-review",
        timestamp="2026-05-14T00:00:00+00:00",
        code_version="abc123",
    )

    assert "# Drop Review Packet" in packet
    assert "## Recommendation / Result" in packet
    assert "Candidate id: drop_1" in packet
    assert "Drop review returned 2 candidates" in packet
    assert "`drop_2`" in packet
    assert "## Execution Details" in packet
    assert "build_workflow_from_dataframes" in packet
    assert "AssumptionReviewService.review_drops" in packet
    assert "Config path: examples/config_quarterly.yml" in packet
    assert "Code version: abc123" in packet
