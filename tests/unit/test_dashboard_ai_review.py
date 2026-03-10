from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.dashboard import Dashboard


def _dashboard_stub() -> Dashboard:
    return Dashboard.__new__(Dashboard)


def test_render_ai_chat_transcript_formats_messages() -> None:
    history = [
        {"role": "assistant", "content": "Initial review ready."},
        {"role": "user", "content": "What is governance tier?"},
    ]
    transcript = Dashboard._render_ai_chat_transcript(history)

    assert "**Assistant:**" in transcript
    assert "**You:**" in transcript


def test_answer_ai_chat_prompt_returns_governance_text() -> None:
    dashboard = _dashboard_stub()
    reply = dashboard._answer_ai_chat_prompt(
        "Explain governance",
        {
            "governance": {
                "tier": "amber",
                "escalation_triggers": ["negative_development_escalation"],
            }
        },
    )

    assert "AMBER" in reply
    assert "negative_development_escalation" in reply


def test_build_ai_decision_packet_contains_required_sections() -> None:
    packet = Dashboard._build_ai_decision_packet(
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
