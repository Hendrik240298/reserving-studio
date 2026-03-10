from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.ai_review import AIReviewService


def test_render_chat_transcript_formats_roles() -> None:
    transcript = AIReviewService.render_chat_transcript(
        [
            {"role": "assistant", "content": "Initial review ready."},
            {"role": "user", "content": "What is the governance tier?"},
        ]
    )

    assert "**Assistant:**" in transcript
    assert "**You:**" in transcript


def test_answer_chat_prompt_governance_response() -> None:
    response = AIReviewService.answer_chat_prompt(
        "Explain governance",
        {
            "governance": {
                "tier": "amber",
                "escalation_triggers": ["negative_development_escalation"],
            }
        },
    )

    assert "AMBER" in response
    assert "negative_development_escalation" in response


def test_build_decision_packet_contains_core_sections() -> None:
    packet = AIReviewService.build_decision_packet(
        {
            "ai_model_meta": {"engine": "deterministic-ai-review"},
            "ai_commentary": "commentary",
            "governance": {"tier": "green"},
            "uncertainty": {"baseline": {}},
            "scenario_matrix": [{"scenario_id": "baseline"}],
            "evidence_trace": [{"evidence_id": "ev1"}],
            "ai_evidence_refs": ["ev1"],
            "ai_override": {"decision": "approve"},
        }
    )

    assert packet["ai_model_meta"]["engine"] == "deterministic-ai-review"
    assert packet["governance"]["tier"] == "green"
    assert packet["scenario_matrix"][0]["scenario_id"] == "baseline"
