from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.valuation_snapshot_service import ValuationSnapshotService


class _ReservingStub:
    def __init__(self, total_ultimate: float = 150.0, total_incurred: float = 100.0):
        self._results = pd.DataFrame(
            {
                "ultimate": [total_ultimate],
                "incurred": [total_incurred],
                "selected_method": ["chainladder"],
            },
            index=pd.Index(["2022"]),
        )

    def get_results(self):
        return self._results.copy()


class _EvaluationStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_params_to_reserving(self, *, reserving, params):
        self.calls.append({"reserving": reserving, "params": dict(params)})


def test_build_current_snapshot_summarizes_reserving() -> None:
    claims_df = pd.DataFrame(
        {
            "uw_year": ["2020", "2020", "2021", "2021"],
            "period": ["2020-03-31", "2020-06-30", "2021-03-31", "2021-06-30"],
        }
    )
    premium_df = claims_df.assign(Premium_selected=10.0)

    snapshot = ValuationSnapshotService().build_current_snapshot(
        reserving=_ReservingStub(),
        claims_df=claims_df,
        premium_df=premium_df,
    )

    assert snapshot["comparison_basis"] == "current"
    assert snapshot["summary"]["total_ibnr"] == 50.0
    assert snapshot["valuation_date"] is not None


def test_build_prior_proxy_snapshot_excludes_latest_origin_rows(monkeypatch) -> None:
    claims_df = pd.DataFrame(
        {
            "uw_year": ["2020", "2020", "2021", "2021"],
            "period": ["2020-03-31", "2020-06-30", "2021-03-31", "2021-06-30"],
            "paid": [10.0, 20.0, 11.0, 22.0],
            "outstanding": [5.0, 4.0, 6.0, 5.0],
        }
    )
    premium_df = pd.DataFrame(
        {
            "uw_year": ["2020", "2020", "2021", "2021"],
            "period": ["2020-03-31", "2020-06-30", "2021-03-31", "2021-06-30"],
            "Premium_selected": [100.0, 120.0, 110.0, 130.0],
        }
    )
    evaluation = _EvaluationStub()
    captured: dict[str, pd.DataFrame] = {}

    def _fake_build_workflow_from_dataframes(claims_arg, premium_arg, *, config=None):
        captured["claims"] = claims_arg.copy()
        captured["premium"] = premium_arg.copy()
        return _ReservingStub(total_ultimate=140.0, total_incurred=95.0)

    monkeypatch.setattr(
        "source.app.build_workflow_from_dataframes",
        _fake_build_workflow_from_dataframes,
    )

    snapshot = ValuationSnapshotService(
        evaluation_service=evaluation,
    ).build_prior_proxy_snapshot(
        claims_df=claims_df,
        premium_df=premium_df,
        params={"average": "volume"},
        config=None,
    )

    assert snapshot["comparison_basis"] == "latest_diagonal_excluded_proxy"
    assert len(captured["claims"]) == 2
    assert len(captured["premium"]) == 2
    assert evaluation.calls[0]["params"] == {"average": "volume"}
