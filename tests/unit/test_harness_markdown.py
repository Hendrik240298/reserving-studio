from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.markdown import render_drop_review_packet, render_final_report


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
    assert "Triangle.from_claims" in packet
    assert "Reserving.reserve" in packet
    assert "Native candidate selection scanned observed link ratios directly" in packet
    assert "Config path: examples/config_quarterly.yml" in packet
    assert "Code version: abc123" in packet


def test_render_final_report_preserves_body_and_artifacts() -> None:
    packet = render_final_report(
        title="Quarterly Drop Review",
        conversation_id="quarterly-drop-review-20260515",
        body_markdown="## Recommendation\n\nUse the drop review packet and attached triangles.",
        artifact_references=[
            {"label": "Drop review packet", "path": "chats/quarterly_drop_review.md"},
            {"label": "A2A triangle", "path": "chats/quarterly_a2a.md"},
        ],
        requested_inputs={
            "config_path": "examples/config_quarterly.yml",
            "conversation_id": "quarterly-drop-review-20260515",
        },
        effective_inputs={
            "segment": "quarterly",
            "granularity": "quarterly",
            "dataset": "quarterly",
            "final_report_dir": "chats",
        },
        included_inputs=[
            "- drop-review: `chats/quarterly_drop_review.md`",
            "- a2a-triangle: `chats/quarterly_a2a.md`",
        ],
        command="uv run python -m harness.cli final-report",
        warnings=["Human actuarial sign-off required."],
        timestamp="2026-05-15T00:00:00+00:00",
        code_version="abc123",
    )

    assert "# Quarterly Drop Review" in packet
    assert "Conversation id: `quarterly-drop-review-20260515`" in packet
    assert "## Recommendation" in packet
    assert "Drop review packet: `chats/quarterly_drop_review.md`" in packet
    assert "A2A triangle: `chats/quarterly_a2a.md`" in packet
    assert "drop-review: `chats/quarterly_drop_review.md`" in packet
    assert "Human actuarial sign-off required." in packet
    assert "Code version: abc123" in packet
