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
    DataCompareResponse,
    DataViewResponse,
    DerivedDropScenarioResponse,
    HighestA2ADropResponse,
    LateEmergenceResponse,
    LdfConsistencyResponse,
    LinkRatioRankResponse,
    MovementDiagnosticsResponse,
    ParamsStore,
    RecalculateResponse,
    ReserveChangeResponse,
    TailEvaluationResponse,
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

    def get_data_view(self, payload):
        return DataViewResponse(
            session_id=payload.session_id,
            query=payload.query.model_dump(mode="json"),
            data={"records": [{"origin": "2022", "12": 100.0}]},
            summary={
                "latest_age": "12",
                "top_latest_rows": [{"origin": "2022", "value": 100.0}],
            },
        )

    def compare_data_views(self, payload):
        return DataCompareResponse(
            session_id=payload.session_id,
            comparison_mode=payload.comparison_mode,
            data={"records": [{"origin": "2022", "12": 10.0}]},
            summary={
                "latest_age": "12",
                "top_latest_differences": [{"origin": "2022", "value": 10.0}],
            },
        )

    def run_movement_diagnostics(self, payload):
        return MovementDiagnosticsResponse(
            session_id=payload.session_id,
            findings=[{"code": "INCURRED_SPIKE_2022_12"}],
            summary={"finding_count": 1},
        )

    def run_ldf_consistency(self, payload):
        return LdfConsistencyResponse(
            session_id=payload.session_id,
            findings=[{"origin": "2022", "age": "12"}],
            summary={"finding_count": 1},
        )

    def project_late_emergence(self, payload):
        return LateEmergenceResponse(
            session_id=payload.session_id,
            rows=[{"origin": payload.uwy or "2022", "median": 0.2}],
            summary={"row_count": 1},
        )

    def explain_reserve_change(self, payload):
        return ReserveChangeResponse(
            session_id=payload.session_id,
            baseline={"total_ibnr": 100.0},
            candidate={"total_ibnr": 125.0},
            attribution={"baseline_vs_candidate_delta": 25.0, "steps": []},
            rows=[],
        )

    def run_highest_a2a_drop_scenario(self, payload):
        return HighestA2ADropResponse(
            session_id=payload.session_id,
            drop=[["2022", 12]],
            top_factors=[{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            baseline={"score": 10.0},
            candidate={"score": 8.5},
            scenario={"scenario_id": "highest_a2a_each_period", "score_delta": -1.5},
        )

    def rank_link_ratios(self, payload):
        return LinkRatioRankResponse(
            session_id=payload.session_id,
            selection_mode=payload.selection_mode,
            scope=payload.scope,
            rows=[{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            summary={
                "row_count": 1,
                "top_rows": [{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            },
        )

    def run_derived_drop_scenario(self, payload):
        return DerivedDropScenarioResponse(
            session_id=payload.session_id,
            rule=payload.rule.model_dump(mode="json"),
            drop=[["2022", 12]],
            selected_rows=[{"development_period": 12, "origin": "2022", "a2a": 2.1}],
            baseline={"score": 10.0},
            candidate={"score": 8.0},
            scenario={
                "scenario_id": "derived_drop_max_per_development_period",
                "score_delta": -2.0,
            },
        )

    def evaluate_tail_fit(self, payload):
        return TailEvaluationResponse(
            session_id=payload.session_id,
            tail_curve="weibull",
            fit_period=[12, 45],
            attachment_age=None,
            projection_period=120,
            r2=0.98,
            rmse=0.03,
            point_count=4,
            residuals=[
                {"age": 12, "observed_ldf": 1.2, "fitted_ldf": 1.18, "error": -0.02}
            ],
            observed_ldf=[{"age": 12, "ldf": 1.2}],
            fitted_tail_ldf=[{"age": 12, "ldf": 1.18}],
        )


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

    data_view_response = client.post(
        "/v1/data/view",
        json={
            "session_id": "s-1",
            "query": {"metric": "incurred", "view": "cumulative"},
        },
    )
    assert data_view_response.status_code == 200
    assert data_view_response.json()["summary"]["latest_age"] == "12"

    data_compare_response = client.post(
        "/v1/data/compare",
        json={
            "session_id": "s-1",
            "left": {"metric": "incurred", "view": "cumulative"},
            "right": {"metric": "paid", "view": "cumulative"},
            "comparison_mode": "difference",
        },
    )
    assert data_compare_response.status_code == 200
    assert data_compare_response.json()["comparison_mode"] == "difference"

    movement_response = client.post(
        "/v1/diagnostics/movement",
        json={"session_id": "s-1"},
    )
    assert movement_response.status_code == 200
    assert movement_response.json()["summary"]["finding_count"] == 1

    ldf_response = client.post(
        "/v1/diagnostics/ldf-consistency",
        json={"session_id": "s-1"},
    )
    assert ldf_response.status_code == 200
    assert ldf_response.json()["summary"]["finding_count"] == 1

    late_response = client.post(
        "/v1/diagnostics/late-emergence",
        json={"session_id": "s-1", "uwy": "2022"},
    )
    assert late_response.status_code == 200
    assert late_response.json()["summary"]["row_count"] == 1

    explain_response = client.post(
        "/v1/reserving/explain-change",
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
    assert explain_response.status_code == 200
    assert explain_response.json()["attribution"]["baseline_vs_candidate_delta"] == 25.0

    highest_a2a_response = client.post(
        "/v1/reserving/highest-a2a-drop",
        json={"session_id": "s-1"},
    )
    assert highest_a2a_response.status_code == 200
    assert (
        highest_a2a_response.json()["scenario"]["scenario_id"]
        == "highest_a2a_each_period"
    )

    rank_response = client.post(
        "/v1/link-ratios/rank",
        json={
            "session_id": "s-1",
            "selection_mode": "max",
            "scope": "per_development_period",
            "limit": 5,
        },
    )
    assert rank_response.status_code == 200
    assert rank_response.json()["summary"]["row_count"] == 1

    derived_response = client.post(
        "/v1/reserving/derived-drop-scenario",
        json={
            "session_id": "s-1",
            "rule": {
                "source": "link_ratios",
                "selection_mode": "max",
                "scope": "per_development_period",
                "limit": 5,
                "include_existing_drops": True,
            },
        },
    )
    assert derived_response.status_code == 200
    assert derived_response.json()["scenario"]["score_delta"] == -2.0

    tail_response = client.post(
        "/v1/tail/evaluate",
        json={
            "session_id": "s-1",
            "average": "volume",
            "drop": [],
            "drop_valuation": [],
            "tail": {
                "curve": "weibull",
                "attachment_age": None,
                "projection_period": 120,
                "fit_period": [12, 45],
            },
            "bf_apriori": {},
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": {},
        },
    )
    assert tail_response.status_code == 200
    assert tail_response.json()["r2"] == 0.98
