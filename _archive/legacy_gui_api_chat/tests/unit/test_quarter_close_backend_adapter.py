from __future__ import annotations

from pathlib import Path
import sys
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.api.adapters.reserving_adapter import (
    InMemoryReservingBackend,
    SessionContext,
)
from source.api.schemas import (
    ParamsStore,
    QuarterClosePackRequest,
    QuarterCloseReviewRequest,
    ResultsStoreMeta,
)


class _QuarterCloseServiceStub:
    def run_review(self, **kwargs):
        assert kwargs["segment"] == "industrial"
        assert kwargs["claims_df"].shape[0] == 1
        return {
            "comparison": {
                "delta_summary": {"comparison_basis": "latest_diagonal_excluded_proxy"}
            },
            "diagnostics": {},
            "assumption_reviews": {},
            "scenario_summary": {},
            "continuity": {"memory_schema_version": 2},
            "recommendation": {"status": "watch"},
            "evidence_ids": ["ev-1"],
            "run_metadata": {"workflow_run_id": "wf-1"},
        }

    def build_pack(self, *, review_result):
        return {"review_type": "quarter_close_pack", "sections": {"policy_trace": {}}}


def _backend_with_session() -> InMemoryReservingBackend:
    backend = InMemoryReservingBackend.__new__(InMemoryReservingBackend)
    backend._lock = threading.RLock()
    backend._config = None
    backend._get_quarter_close_service = lambda: _QuarterCloseServiceStub()
    backend._sessions_by_id = {
        "s-1": SessionContext(
            session_id="s-1",
            segment="industrial",
            reserving=object(),
            sync_version=0,
            params_store=ParamsStore(),
            results_store_meta=ResultsStoreMeta(),
            last_results_payload={},
            source_claims_rows=[{"uw_year": "2022", "period": "2025-12-31"}],
            source_premium_rows=[
                {"uw_year": "2022", "period": "2025-12-31", "Premium_selected": 100.0}
            ],
        )
    }
    backend._sessions_by_segment = {"industrial": "s-1"}
    backend._build_results_payload = lambda reserving: {}
    return backend


def test_backend_runs_quarter_close_review() -> None:
    backend = _backend_with_session()

    response = backend.run_quarter_close_review(
        QuarterCloseReviewRequest(session_id="s-1")
    )

    assert response.session_id == "s-1"
    assert response.recommendation["status"] == "watch"
    assert response.evidence_ids == ["ev-1"]


def test_backend_builds_quarter_close_pack() -> None:
    backend = _backend_with_session()

    response = backend.build_quarter_close_pack(
        QuarterClosePackRequest(session_id="s-1")
    )

    assert response.session_id == "s-1"
    assert response.pack["review_type"] == "quarter_close_pack"
