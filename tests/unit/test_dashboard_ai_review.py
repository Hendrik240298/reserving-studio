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
            "deterministic_packet": {"recommendation": {"status": "recommended"}},
            "ai_evidence_refs": ["ev1"],
            "ai_override": {"decision": "approve"},
        }
    )

    assert packet["ai_model_meta"]["engine"] == "deterministic-ai-review"
    assert packet["governance"]["tier"] == "green"
    assert packet["scenario_matrix"][0]["scenario_id"] == "baseline"
    assert packet["deterministic_packet"]["recommendation"]["status"] == "recommended"


def test_build_scenario_dispositions_from_review_uses_recommended_changes() -> None:
    dispositions = Dashboard._build_scenario_dispositions_from_review(
        {
            "ai_override": {
                "decision": "reject",
                "approver": "A. Actuary",
                "rationale": "Too fragile",
                "signed_off_at": "2026-04-01T00:00:00Z",
            },
            "deterministic_packet": {
                "recommended_changes": [
                    {
                        "candidate_id": "drop_1",
                        "parameters": {"drop": [["2022", 24]]},
                    }
                ],
                "composite_review": {
                    "summary": {
                        "comparison": {"current_valuation_date": "2026-03-31"},
                        "run_metadata": {"current_data_fingerprint": "fp-1"},
                    }
                },
            },
        }
    )

    assert dispositions[0]["scenario_id"] == "drop_1"
    assert dispositions[0]["decision"] == "rejected"
    assert dispositions[0]["valuation_date"] == "2026-03-31"
    assert dispositions[0]["data_fingerprint"] == "fp-1"


def test_build_ai_decision_packet_preserves_packet_presentation() -> None:
    packet = Dashboard._build_ai_decision_packet(
        {
            "deterministic_packet": {
                "presentation": {
                    "conclusion": "reasonable_alternative",
                    "evidence_used": ["ev1"],
                    "key_caveat": "Peer review required",
                }
            }
        }
    )

    assert (
        packet["deterministic_packet"]["presentation"]["conclusion"]
        == "reasonable_alternative"
    )
    assert (
        packet["deterministic_packet"]["presentation"]["key_caveat"]
        == "Peer review required"
    )


def test_build_ai_recommendation_panel_renders_key_sections() -> None:
    dashboard = _dashboard_stub()

    panel = dashboard._build_ai_recommendation_panel(
        {
            "ai_evidence_refs": ["ev1", "ev2"],
            "deterministic_packet": {
                "review": {
                    "status": "pass_with_caveats",
                    "caveats": [
                        "Actuarial peer review required before parameter adoption"
                    ],
                },
                "recommendation": {
                    "status": "reasonable_alternative",
                    "summary": "A tested scenario improved diagnostics, but review caveats remain.",
                    "recommended_scenario_id": "drop_1",
                    "alternative_scenario_ids": ["drop_2"],
                },
            },
        }
    )

    children = panel.children
    joined = " ".join(
        str(getattr(child, "children", ""))
        for child in children
        if getattr(child, "children", None) is not None
    )
    assert "Conclusion: Reasonable Alternative" in joined
    assert "Review status: Pass With Caveats" in joined
    assert "Evidence used: ev1, ev2" in joined
    assert "Alternative considered: drop_2" in joined
