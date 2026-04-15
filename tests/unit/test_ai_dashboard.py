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


def test_analysis_basis_rows_show_bound_scenario_and_session_match() -> None:
    rows = AIDashboard._analysis_basis_rows(
        {
            "basis_type": "review_candidate",
            "scenario_id": "review_drop_abc123",
            "candidate_id": "drop_combo_1",
            "source_tool": "tool_run_tail_review",
            "is_active_session": False,
            "parameters": {
                "average": "volume",
                "drop": [["2003", 9], ["2002", 21], ["2002", 39]],
                "tail": {
                    "curve": "weibull",
                    "attachment_age": 27,
                    "fit_period": [12, 108],
                },
                "bf_apriori": {"2005": 0.5988},
                "selected_ultimate_by_uwy": {"2005": "bornhuetter_ferguson"},
            },
        }
    )

    row_map = {row["field"]: row["value"] for row in rows}
    assert row_map["Basis Type"] == "review_candidate"
    assert row_map["Scenario"] == "drop_combo_1"
    assert row_map["Stable Scenario Key"] == "review_drop_abc123"
    assert row_map["Matches Active Session"] == (
        "no, this basis differs from the current active session"
    )
    assert row_map["Source Tool"] == "tool_run_tail_review"
    assert row_map["Tail Active"] == "yes"
    assert row_map["Tail Mode"] == "attached"


def test_analysis_basis_rows_label_bespoke_basis_without_baseline_scenario_name() -> (
    None
):
    rows = AIDashboard._analysis_basis_rows(
        {
            "basis_type": "bespoke",
            "scenario_id": None,
            "is_active_session": False,
            "parameters": {"average": "volume", "drop": [["2001", 15]]},
        }
    )

    row_map = {row["field"]: row["value"] for row in rows}
    assert row_map["Scenario"] == "custom parameter basis"
    assert row_map["Tail Active"] == "no"
    assert row_map["Tail Mode"] == "reference_fit_only"


def test_memory_proposal_options_render_expected_labels() -> None:
    options = AIDashboard._memory_proposal_options(
        [
            {
                "proposal_id": "mem-1",
                "field": "open_items",
            }
        ]
    )

    assert options == [{"label": "open_items: mem-1", "value": "mem-1"}]


def test_memory_payload_outputs_render_structured_preference_summary() -> None:
    outputs = AIDashboard._memory_payload_outputs(
        {
            "segment_overview": "Industrial liability book.",
            "known_issues_text": "Issue A",
            "house_preferences_text": "Prefer stable tail",
            "recent_quarter_notes_text": "2026Q1 | Case strengthening",
            "open_items_text": "Review reporting lag",
            "structured_house_preferences": [{"type": "max_drop_count", "value": 1}],
            "memory_change_log": [
                {"field": "open_items", "action": "save_manual_update"}
            ],
        },
        status="saved",
    )

    assert outputs[1] == "Industrial liability book."
    assert "max_drop_count=1" in outputs[6]
    assert outputs[8] == "saved"


def test_memory_field_label_adds_question_mark_tooltip() -> None:
    label = AIDashboard._memory_field_label(
        "House Preferences",
        "Tooltip text",
        margin_top="12px",
    )

    assert label.children[0].children == "House Preferences"
    assert label.children[1].children == "?"
    assert label.children[1].title == "Tooltip text"
    assert label.style["marginTop"] == "12px"
