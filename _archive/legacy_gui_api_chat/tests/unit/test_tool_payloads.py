from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.tool_payloads import summarize_assumption_detail_payload


def test_assumption_detail_summary_flags_ldfs_below_one() -> None:
    summary = summarize_assumption_detail_payload(
        {
            "session_id": "s-1",
            "selected_ldf": [
                {"age": 24, "development_label": "24-27", "ldf": 1.02},
                {"age": 27, "development_label": "27-30", "ldf": 0.998},
            ],
            "fitted_tail_ldf": [
                {"age": 30, "development_label": "30-33", "ldf": 1.004},
                {"age": 33, "development_label": "33-36", "ldf": 0.999},
            ],
        }
    )

    assert summary["min_selected_ldf"] == 0.998
    assert summary["min_fitted_tail_ldf"] == 0.999
    assert summary["selected_ldf_below_1"] == [
        {"age": 27, "development_label": "27-30", "ldf": 0.998}
    ]
    assert summary["fitted_tail_ldf_below_1"] == [
        {"age": 33, "development_label": "33-36", "ldf": 0.999}
    ]
