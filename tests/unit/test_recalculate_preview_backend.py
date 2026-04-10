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
    RecalculateRequest,
    ResultsStoreMeta,
    TailConfig,
)


def test_preview_recalculate_does_not_mutate_active_session() -> None:
    backend = InMemoryReservingBackend.__new__(InMemoryReservingBackend)
    backend._lock = threading.RLock()
    backend._observability_enabled = False

    context = SessionContext(
        session_id="s-1",
        segment="industrial",
        reserving=object(),
        sync_version=0,
        params_store=ParamsStore(
            request_id=7,
            source="test",
            force_recalc=False,
            drop_store=[],
            tail_attachment_age=None,
            tail_projection_months=0,
            tail_fit_period_selection=[],
            average="volume",
            tail_curve="weibull",
            bf_apriori_by_uwy={},
            selected_ultimate_by_uwy={},
            sync_version=0,
        ),
        results_store_meta=ResultsStoreMeta(sync_version=0),
        last_results_payload={"cache_key": "active-cache"},
    )
    backend._sessions_by_id = {"s-1": context}
    backend._sessions_by_segment = {"industrial": "s-1"}

    captured: dict[str, object] = {}

    def _build_results_payload_for_params(*, context, params):
        captured["params"] = params
        return {
            "results_table_rows": [{"uwy": "2022", "ultimate_display": "100.00"}],
            "triangle_figure": {"data": [{"name": "triangle"}]},
            "emergence_figure": {"data": [{"name": "emergence"}]},
            "heatmap_payload": {"shape": [1, 1]},
            "cache_key": "preview-cache",
            "model_cache_key": "preview-model",
            "figure_version": 3,
        }

    backend._build_results_payload_for_params = _build_results_payload_for_params

    response = backend.recalculate(
        RecalculateRequest(
            session_id="s-1",
            average="volume",
            drop=[["2022", 24]],
            drop_valuation=[],
            tail=TailConfig(
                curve="weibull",
                attachment_age=27,
                projection_period=0,
                fit_period=[12, 108],
            ),
            bf_apriori={},
            final_ultimate="chainladder",
            selected_ultimate_by_uwy={},
            persist_to_session=False,
        )
    )

    assert response.analysis_basis["is_active_session"] is False
    assert response.cache_key == "preview-cache"
    assert response.triangle_figure["data"][0]["name"] == "triangle"
    assert captured["params"]["drop"] == [["2022", 24]]
    assert context.params_store.request_id == 7
    assert context.params_store.drop_store == []
    assert context.last_results_payload == {"cache_key": "active-cache"}
    assert context.results_store_meta.sync_version == 0
