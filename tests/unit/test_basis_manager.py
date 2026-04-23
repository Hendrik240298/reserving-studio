from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.basis_manager import BasisManager
from ai.tool_payloads import build_memory_snapshot


def test_build_memory_snapshot_keys_basis_cache_by_basis_key() -> None:
    snapshot = build_memory_snapshot(
        iteration_summary={
            "session_id": "s-1",
            "baseline": {
                "scenario_id": "baseline",
                "score": 2.0,
                "parameters": {
                    "average": "volume",
                    "drop": [],
                    "drop_valuation": [],
                    "tail": {
                        "curve": "weibull",
                        "attachment_age": None,
                        "projection_period": 0,
                        "fit_period": [],
                    },
                    "bf_apriori": {},
                    "final_ultimate": "chainladder",
                    "selected_ultimate_by_uwy": {},
                },
            },
            "top_scenarios": [
                {
                    "scenario_id": "drop_1",
                    "score": 1.0,
                    "parameters": {
                        "average": "volume",
                        "drop": [["2022", 24]],
                        "drop_valuation": [],
                        "tail": {
                            "curve": "weibull",
                            "attachment_age": None,
                            "projection_period": 0,
                            "fit_period": [],
                        },
                        "bf_apriori": {},
                        "final_ultimate": "chainladder",
                        "selected_ultimate_by_uwy": {},
                    },
                }
            ],
        }
    )

    cache = snapshot["scenario_basis_cache"]
    assert "drop_1" not in cache
    assert any(
        isinstance(value, dict) and value.get("scenario_id") == "drop_1"
        for value in cache.values()
    )


def test_lookup_basis_by_requested_id_prefers_basis_key() -> None:
    basis = {
        "basis_key": "basis-123",
        "basis_type": "review_candidate",
        "scenario_id": "review_drop_sig_a",
        "candidate_id": "drop_3",
        "parameters": {"drop": [["2002", 39]]},
    }

    resolved = BasisManager.lookup_basis_by_requested_id(
        requested_id="basis-123",
        current_basis={},
        basis_cache={"basis-123": basis},
    )

    assert resolved["basis_key"] == "basis-123"
    assert resolved["scenario_id"] == "review_drop_sig_a"


def test_scenario_reference_in_prompt_returns_basis_key_for_unique_label() -> None:
    basis_key = BasisManager.scenario_id_mentioned_in_prompt(
        prompt="Show me the fitted tail LDFs for drop_4 from 39 to 60.",
        basis_cache={
            "basis-456": {
                "basis_key": "basis-456",
                "basis_type": "review_candidate",
                "scenario_id": "review_drop_sig_b",
                "candidate_id": "drop_4",
                "parameters": {"drop": [["2001", 60]]},
            }
        },
    )

    assert basis_key == "basis-456"
