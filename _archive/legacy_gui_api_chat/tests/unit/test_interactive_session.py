from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.interactive_session import (
    FinalizePayload,
    InteractiveSessionController,
    ParamsStoreSnapshot,
    ResultsStoreSnapshot,
    utc_now,
)


def test_params_store_snapshot_from_dict() -> None:
    payload = {
        "request_id": 7,
        "source": "local",
        "force_recalc": True,
        "drop_store": [["2001", 12]],
        "tail_attachment_age": 60,
        "tail_projection_months": 24,
        "tail_fit_period_selection": [12, 24],
        "average": "volume",
        "tail_curve": "weibull",
        "bf_apriori_by_uwy": {"2001": 0.75},
        "selected_ultimate_by_uwy": {"2001": "bornhuetter_ferguson"},
        "sync_version": 3,
    }

    snapshot = ParamsStoreSnapshot.from_store_dict(payload)

    assert snapshot.request_id == 7
    assert snapshot.drop_store == [["2001", 12]]
    assert snapshot.tail_projection_months == 24
    assert snapshot.selected_ultimate_by_uwy["2001"] == "bornhuetter_ferguson"


def test_results_store_snapshot_from_dict() -> None:
    payload = {
        "triangle_figure": {},
        "emergence_figure": {},
        "drops_display": "2001:12",
        "average": "volume",
        "tail_curve": "weibull",
        "drop_store": [["2001", 12]],
        "tail_attachment_age": None,
        "tail_attachment_display": "None",
        "tail_projection_months": 12,
        "tail_fit_period_selection": [12],
        "tail_fit_period_display": "lower=12, upper=None",
        "selected_ultimate_by_uwy": {"2001": "chainladder"},
        "results_table_rows": [{"uwy": "2001", "ultimate_display": "100.0"}],
        "last_updated": "2026-01-01T00:00:00Z",
        "cache_key": "abc",
        "model_cache_key": "def",
        "figure_version": 2,
        "sync_version": 4,
    }

    snapshot = ResultsStoreSnapshot.from_store_dict(payload)

    assert snapshot.cache_key == "abc"
    assert snapshot.model_cache_key == "def"
    assert snapshot.figure_version == 2
    assert snapshot.tail_projection_months == 12
    assert snapshot.results_table_rows[0]["uwy"] == "2001"


def test_controller_finalize_sets_done_event() -> None:
    controller = InteractiveSessionController()
    payload = FinalizePayload(
        finalized_at_utc=utc_now(),
        segment="quarterly",
        params_store=ParamsStoreSnapshot.from_store_dict({}),
        results_store=ResultsStoreSnapshot.from_store_dict({}),
        results_df=pd.DataFrame({"x": [1.0]}),
    )

    controller.finalize(payload)

    assert controller.done_event.is_set()
    assert controller.finalized
    assert controller.finalized_payload is not None
