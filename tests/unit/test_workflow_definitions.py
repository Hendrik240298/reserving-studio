from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.workflow_definitions import get_workflow_definition, select_workflow_name


def test_select_workflow_name_returns_structured_definition_match() -> None:
    assert select_workflow_name("Run a quarter-close review pack.") == "quarter_close_review"
    assert select_workflow_name("Review the tail assumptions.") == "tail_selection"
    assert select_workflow_name("Explain the biggest claims movement this quarter.") == "movement_review"


def test_quarter_close_workflow_definition_exposes_contract_metadata() -> None:
    definition = get_workflow_definition("quarter_close_review")

    assert definition is not None
    assert definition.intent_class == "quarter_close"
    assert definition.answer_contract == "recommendation_with_proposal"
    assert "proposal_possible" in set(definition.basis_behavior)
    assert definition.requires_continuity is True
    assert definition.steps[0].tool_name == "tool_run_quarter_close_review"


def test_movement_review_definition_disallows_proposals_and_uses_active_session() -> None:
    definition = get_workflow_definition("movement_review")

    assert definition is not None
    assert "proposal_disallowed" in set(definition.basis_behavior)
    assert "use_active_session_only" in set(definition.basis_behavior)
    assert definition.answer_contract == "observational_explanation"
    assert definition.steps[0].basis_aware is False
