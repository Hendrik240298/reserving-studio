from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.narration import (
    NarrationAssembler,
    build_narration_prompt,
    render_narration_fallback,
)


def test_recommendation_with_pending_proposal_scaffold_preserves_basis() -> None:
    packet = NarrationAssembler.build(
        deterministic_packet={
            "plan": {
                "workflow_name": "drop_review",
                "goal": "Review candidate development drops.",
                "answer_contract": "recommendation_with_proposal",
                "steps": [{"tool_name": "tool_run_drop_review"}],
            },
            "review": {"status": "pass_with_caveats", "caveats": ["Peer review required"]},
            "recommendation": {
                "status": "reasonable_alternative",
                "summary": "A tested drop improves diagnostics.",
                "recommended_basis_id": "basis-1",
                "rationale": ["score_improvement=0.5"],
            },
            "evidence_packets": [
                {
                    "tool_name": "tool_run_drop_review",
                    "evidence_key": "drop_review",
                    "provenance": {"evidence_ids": ["ev-1"]},
                    "execution": {"status": "executed_exactly"},
                }
            ],
            "presentation": {"evidence_used": ["ev-1"]},
        },
        accepted_analysis_basis={
            "basis_type": "baseline",
            "scenario_id": "baseline",
            "parameters": {},
            "is_active_session": True,
        },
        proposal_basis={
            "proposal_id": "proposal-1",
            "status": "pending",
            "basis_key": "basis-1",
            "scenario_label": "drop_1",
            "parameters": {"drop": [["2022", 24]]},
        },
        execution_records=[
            {
                "execution_id": "exec-1",
                "tool_name": "tool_run_drop_review",
                "execution_status": "executed_exactly",
                "warnings": [],
                "material_adjustments": [],
            }
        ],
    )

    assert packet["answer_contract"] == "recommendation_with_proposal"
    assert "proposal_status" in packet["required_answer_sections"]
    assert packet["basis"]["basis_changed"] is False
    assert packet["proposal"]["exists"] is True
    assert packet["proposal"]["label"] == "drop_1"
    assert packet["proposal"]["proposed_drops"] == ["AY 2022 age 24"]
    assert "proposal_changed_basis" in packet["blocked_claims"]
    assert packet["reviewed"]["evidence_ids"] == ["ev-1"]
    assert "Peer review required" in packet["caveats"]


def test_observational_contract_disallows_unsupported_recommendation() -> None:
    packet = NarrationAssembler.build(
        deterministic_packet={
            "plan": {
                "workflow_name": "movement_review",
                "answer_contract": "observational_explanation",
            },
            "review": {"status": "pass"},
            "recommendation": {"status": "watch", "summary": "Movement reviewed."},
            "evidence_packets": [],
        },
        accepted_analysis_basis={},
        proposal_basis={},
        execution_records=[],
    )

    assert packet["proposal"]["exists"] is False
    assert "main_explanation" in packet["required_answer_sections"]
    assert "unsupported_recommendation" in packet["blocked_claims"]
    assert "pending_proposal_when_none" in packet["blocked_claims"]


def test_rejected_execution_blocks_success_style_claims() -> None:
    packet = NarrationAssembler.build(
        deterministic_packet={
            "plan": {"workflow_name": "scenario_recommendation"},
            "review": {"status": "pass"},
            "recommendation": {"status": "watch"},
            "evidence_packets": [],
        },
        accepted_analysis_basis={},
        proposal_basis={},
        execution_records=[
            {
                "execution_status": "rejected",
                "warnings": ["Unsupported tail curve assumption: made_up_curve."],
                "material_adjustments": ["Unsupported tail curve assumption: made_up_curve."],
            }
        ],
    )

    assert packet["execution"]["status"] == "rejected"
    assert "success_style_execution_claim" in packet["blocked_claims"]
    assert "Unsupported tail curve assumption: made_up_curve." in packet["caveats"]


def test_narration_prompt_and_fallback_include_acceptance_boundary() -> None:
    packet = {
        "basis": {"current_basis_label": "Basis used: current baseline session."},
        "reviewed": {"scope": "Drop review"},
        "recommendation": {
            "status": "reasonable_alternative",
            "summary": "A tested drop improves diagnostics.",
        },
        "proposal": {"exists": True, "label": "drop_1"},
        "execution": {"status": "executed_exactly"},
        "caveats": ["Peer review required"],
        "required_answer_sections": ["basis_used", "proposal_status"],
        "blocked_claims": ["proposal_changed_basis"],
    }

    prompt = build_narration_prompt(packet)
    fallback = render_narration_fallback(packet)

    assert "required_answer_sections" in prompt
    assert "blocked_claims" in prompt
    assert "pending explicit Yes/No acceptance" in fallback
    assert "Analysis Basis is unchanged" in fallback


def test_narration_prompt_blocks_pending_status_when_no_proposal_exists() -> None:
    packet = {
        "basis": {
            "current_basis_label": "Basis used: scenario derived_drop_max_global.",
            "accepted_basis": {
                "basis_key": "0b9c05883645ebc2",
                "scenario_label": "derived_drop_max_global",
            },
        },
        "reviewed": {"scope": "Reserve change explanation"},
        "recommendation": {"status": "watch"},
        "proposal": {"exists": False, "status": "accepted"},
        "execution": {"status": "executed_exactly"},
        "caveats": [],
        "required_answer_sections": ["basis_or_data_scope"],
        "blocked_claims": ["pending_proposal_when_none"],
    }

    prompt = build_narration_prompt(packet)

    assert "If proposal.exists is false, do not mention a pending proposal" in prompt
    assert "never through natural-language replies" in prompt


def test_proposal_status_lists_every_proposed_drop() -> None:
    packet = {
        "basis": {"current_basis_label": "Basis used: current baseline session."},
        "reviewed": {"scope": "Drop review"},
        "recommendation": {
            "status": "reasonable_alternative",
            "summary": "Top ranked drops improve diagnostics.",
        },
        "proposal": {
            "exists": True,
            "label": "combined_drop_review_top_5",
            "proposed_drops": [
                "AY 2002 age 39",
                "AY 2001 age 60",
                "AY 2001 age 12",
                "AY 2002 age 21",
                "AY 2003 age 9",
            ],
        },
        "execution": {"status": "executed_exactly"},
        "caveats": [],
        "required_answer_sections": ["basis_used", "proposal_status"],
        "blocked_claims": ["proposal_changed_basis"],
    }

    prompt = build_narration_prompt(packet)
    fallback = render_narration_fallback(packet)

    assert "list every proposed drop" in prompt
    assert "AY 2002 age 39" in fallback
    assert "AY 2003 age 9" in fallback
