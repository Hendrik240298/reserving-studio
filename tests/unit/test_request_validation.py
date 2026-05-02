from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.request_validation import validate_recalculate_like_arguments


def test_validate_recalculate_like_arguments_classifies_non_material_normalization() -> None:
    result = validate_recalculate_like_arguments(
        {
            "session_id": "s-1",
            "average": "weighted_average_3_year",
            "tail": {"curve": "power_curve", "fit_period": [12, 24]},
            "drop": [],
            "drop_valuation": [],
            "selected_ultimate_by_uwy": {},
        }
    )

    assert result.execution_status == "executed_with_non_material_normalization"
    assert result.material_adjustments == []
    assert result.effective_inputs["average"] == "volume"
    assert result.effective_inputs["tail"]["curve"] == "inverse_power"


def test_validate_recalculate_like_arguments_classifies_partial_execution() -> None:
    result = validate_recalculate_like_arguments(
        {
            "session_id": "s-1",
            "average": "volume",
            "tail": {"curve": "weibull", "fit_period": [12, 24, 36, 48]},
            "drop": [["2001", None], ["2002", 12]],
            "drop_valuation": [],
            "selected_ultimate_by_uwy": {"1995": 123, "1996": "chainladder"},
        }
    )

    assert result.execution_status == "partially_executed"
    assert "Collapsed tail.fit_period to [12, 48]." in result.material_adjustments
    assert "Dropped 1 invalid drop entry." in result.material_adjustments
    assert (
        "Dropped 1 invalid selected_ultimate_by_uwy override(s) and kept only method values."
        in result.material_adjustments
    )


def test_validate_recalculate_like_arguments_rejects_invalid_fit_period_values() -> None:
    result = validate_recalculate_like_arguments(
        {
            "session_id": "s-1",
            "average": "volume",
            "tail": {"curve": "weibull", "fit_period": [12, "bad", 48]},
            "drop": [],
            "drop_valuation": [],
            "selected_ultimate_by_uwy": {},
        }
    )

    assert result.execution_status == "rejected"
    assert result.rejection_reason == "Tail fit period contained unsupported values."


def test_validate_recalculate_like_arguments_rejects_unknown_average() -> None:
    result = validate_recalculate_like_arguments(
        {
            "session_id": "s-1",
            "average": "made_up_average",
            "tail": {"curve": "weibull", "fit_period": [12, 24]},
            "drop": [],
            "drop_valuation": [],
            "selected_ultimate_by_uwy": {},
        }
    )

    assert result.execution_status == "rejected"
    assert result.rejection_reason == "Unsupported average assumption: made_up_average."


def test_validate_recalculate_like_arguments_rejects_unknown_tail_curve() -> None:
    result = validate_recalculate_like_arguments(
        {
            "session_id": "s-1",
            "average": "volume",
            "tail": {"curve": "made_up_curve", "fit_period": [12, 24]},
            "drop": [],
            "drop_valuation": [],
            "selected_ultimate_by_uwy": {},
        }
    )

    assert result.execution_status == "rejected"
    assert result.rejection_reason == "Unsupported tail curve assumption: made_up_curve."
