from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.ai_dashboard import AIDashboard


def test_preset_prompt_specs_define_six_high_value_prompts() -> None:
    specs = AIDashboard._preset_prompt_specs()

    assert len(specs) == 6
    assert [item["label"] for item in specs] == [
        "Quarter-Close Review",
        "Drop Review",
        "Tail Review",
        "Reserve Recommendation",
        "Anomaly Triage",
        "Movement Review",
    ]
    assert (
        specs[0]["prompt"]
        == "Run a quarter-close review pack. Compare the current valuation against the prior proxy, flag any recommendation blockers, rank the strongest assumption changes, and tell me what should go to sign-off."
    )


def test_prompt_text_for_trigger_prefers_preset_prompt_when_card_clicked() -> None:
    prompt = AIDashboard._prompt_text_for_trigger(
        triggered_id="ai-prompt-tail-review",
        typed_prompt="ignored typed value",
    )

    assert (
        prompt
        == "Review the tail assumptions. Rank the best tail candidates, explain attachment continuity, sub-1.0 late factors, and stability risk, and recommend the strongest selection versus sensitivity."
    )


def test_prompt_text_for_trigger_uses_typed_prompt_for_send_button() -> None:
    prompt = AIDashboard._prompt_text_for_trigger(
        triggered_id="ai-chat-send",
        typed_prompt="Custom typed question",
    )

    assert prompt == "Custom typed question"


def test_intro_chat_message_contains_2x3_prompt_grid() -> None:
    intro = AIDashboard._intro_chat_message()
    body = intro.children[0]
    content = body.children[1]
    grid = content.children[2]

    assert grid.style["gridTemplateColumns"] == "repeat(2, minmax(0, 1fr))"
    assert len(grid.children) == 6
    assert grid.children[0].id == "ai-prompt-quarter-close"
    assert (
        content.children[1].children
        == "Hi, I'm Turtuary. I can inspect your current reserving data, run the available workflows and diagnostics, and help you work through the evidence."
    )


def test_render_chat_messages_excludes_intro_shell() -> None:
    rendered = AIDashboard.__new__(AIDashboard)._render_chat_messages([])

    assert rendered == []
