from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.api.main import create_app
from source.api.schemas import (
    DiagnosticsIterateResponse,
    DiagnosticsResponse,
    ParamsStore,
    RecalculateResponse,
    ResultsResponse,
    RunMetadata,
    ResultsStoreMeta,
    SessionSaveResponse,
    SessionStateResponse,
    WorkflowInitializationResponse,
)


class FakeBackend:
    def create_workflow_from_dataframes(self, payload):
        return WorkflowInitializationResponse(
            session_id="s-1",
            segment=payload.segment,
            sync_version=0,
        )

    def get_session(self, segment: str):
        return SessionStateResponse(
            session_id="s-1",
            segment=segment,
            params_store=ParamsStore(),
            results_store_meta=ResultsStoreMeta(),
            sync_version=0,
        )

    def save_session(self, segment: str, payload):
        return SessionSaveResponse(
            segment=segment,
            sync_version=payload.expected_sync_version + 1,
            saved_at=datetime.now(timezone.utc),
        )

    def recalculate(self, payload):
        return RecalculateResponse(session_id=payload.session_id)

    def run_diagnostics(self, payload):
        return DiagnosticsResponse(
            session_id=payload.session_id,
            findings=[],
            uncertainty={"version": "v1.1", "total_process_cv": 0.22},
            run_metadata=RunMetadata(
                run_id="run-1",
                generated_at=datetime.now(timezone.utc),
                data_fingerprint="abc123",
                diagnostics_version="v2.1",
                scenario_generator_version="v1.1",
            ),
        )

    def iterate_diagnostics(self, payload):
        return DiagnosticsIterateResponse(
            session_id=payload.session_id,
            baseline=None,
            scenarios=[],
            uncertainty={
                "baseline": {"version": "v1.1", "total_process_cv": 0.22},
                "bootstrap": {"sample_count": 100},
                "tail_model": {"instability_flag": False},
            },
            run_metadata=RunMetadata(
                run_id="run-2",
                generated_at=datetime.now(timezone.utc),
                data_fingerprint="xyz789",
                diagnostics_version="v2.1",
                scenario_generator_version="v1.1",
            ),
        )

    def get_results(self, session_id: str):
        return ResultsResponse(session_id=session_id, results={"ok": True})


def test_api_scaffold_endpoints() -> None:
    app = create_app(backend=FakeBackend())
    client = TestClient(app)

    health_response = client.get("/healthz")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    workflow_response = client.post(
        "/v1/workflows/from-dataframes",
        json={
            "segment": "motor",
            "claims_rows": [{"uw_year": "2020", "period": 12, "incurred": 100.0}],
            "premium_rows": [
                {"uw_year": "2020", "period": 12, "Premium_selected": 200.0}
            ],
        },
    )
    assert workflow_response.status_code == 200
    assert workflow_response.json()["session_id"] == "s-1"

    session_response = client.get("/v1/sessions/motor")
    assert session_response.status_code == 200
    assert session_response.json()["segment"] == "motor"

    recalc_response = client.post(
        "/v1/reserving/recalculate",
        json={
            "session_id": "s-1",
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
    )
    assert recalc_response.status_code == 200
    assert recalc_response.json()["session_id"] == "s-1"

    diagnostics_response = client.post(
        "/v1/diagnostics/run",
        json={"session_id": "s-1"},
    )
    assert diagnostics_response.status_code == 200
    assert diagnostics_response.json()["session_id"] == "s-1"
    assert diagnostics_response.json()["run_metadata"]["run_id"] == "run-1"
    assert diagnostics_response.json()["uncertainty"]["version"] == "v1.1"

    iterate_response = client.post(
        "/v1/diagnostics/iterate",
        json={"session_id": "s-1", "max_scenarios": 5},
    )
    assert iterate_response.status_code == 200
    assert iterate_response.json()["session_id"] == "s-1"
    assert iterate_response.json()["run_metadata"]["run_id"] == "run-2"
    assert "bootstrap" in iterate_response.json()["uncertainty"]

    results_response = client.get("/v1/results/s-1")
    assert results_response.status_code == 200
    assert results_response.json()["results"]["ok"] is True
