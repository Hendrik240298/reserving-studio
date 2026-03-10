from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.api.adapters.reserving_adapter import (
    InMemoryReservingBackend,
    SessionContext,
)
from source.api.schemas import (
    DiagnosticEvidence,
    DiagnosticFinding,
    DiagnosticRecommendation,
    ParamsStore,
    ResultsStoreMeta,
    ScenarioEvaluation,
)
from source.services.diagnostics_service import DiagnosticsService


def _backend_stub() -> InMemoryReservingBackend:
    backend = InMemoryReservingBackend.__new__(InMemoryReservingBackend)
    backend._diagnostics_service = DiagnosticsService()
    return backend


def test_backtest_threshold_calibration_uses_segment_and_maturity() -> None:
    backend = _backend_stub()
    results_df = pd.DataFrame(
        {
            "incurred": [90.0, 100.0, 120.0, 110.0],
            "Premium": [1000.0, 1000.0, 1000.0, 1000.0],
            "cl_ultimate": [500.0, 520.0, 540.0, 560.0],
            "bf_ultimate": [480.0, 500.0, 520.0, 540.0],
            "ultimate": [500.0, 520.0, 540.0, 560.0],
        },
        index=pd.Index(["2019", "2020", "2021", "2022"]),
    )
    incurred = pd.DataFrame(
        {
            12: [60.0, 70.0, 80.0, 90.0],
            24: [95.0, 120.0, 130.0, 150.0],
            36: [120.0, 160.0, 200.0, 220.0],
        },
        index=results_df.index,
    )
    link_ratios = pd.DataFrame(
        {
            12: [1.7, 1.75, 1.8, 1.85],
            24: [1.25, 1.3, 1.35, 1.4],
            36: [1.0, 1.0, 1.0, 1.0],
        },
        index=results_df.index,
    )
    link_ratios.loc["LDF"] = [1.8, 1.35, 1.0]

    calibration = backend._calibrate_backtest_thresholds(
        segment="casualty",
        results_df=results_df,
        heatmap_data={"incurred": incurred, "link_ratios": link_ratios},
    )

    assert calibration["maturity_regime"] == "immature"
    assert calibration["residual_count"] >= 8
    assert calibration["backtest_bias_threshold"] >= 0.18
    assert calibration["backtest_mae_threshold"] >= 0.30


def test_governance_assessment_sets_red_with_escalation_triggers() -> None:
    findings = [
        DiagnosticFinding(
            code="DATA_QUALITY_GATE",
            severity="critical",
            message="quality gate",
            evidence=DiagnosticEvidence(metric_id="dq", value=0.1),
            suggested_actions=[],
        ),
        DiagnosticFinding(
            code="PORTFOLIO_SHIFT_SIGNAL_UNCONFIRMED_24",
            severity="medium",
            message="unconfirmed shift",
            evidence=DiagnosticEvidence(metric_id="shift", value=0.2),
            suggested_actions=[],
        ),
    ]

    components = InMemoryReservingBackend._severity_components(findings)
    governance = InMemoryReservingBackend._governance_assessment(
        findings=findings,
        severity_components=components,
    )

    assert governance["tier"] == "red"
    assert "critical_finding_present" in governance["escalation_triggers"]
    assert governance["requires_human_review"] is True


def test_scenario_candidates_include_lineage_rationale_evidence() -> None:
    backend = _backend_stub()
    baseline = {
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
    }
    baseline_eval = ScenarioEvaluation(
        scenario_id="baseline",
        score=1.0,
        summary="baseline",
        recommendations=[
            DiagnosticRecommendation(
                code="RECOMMEND_DROP_2022_24",
                priority="high",
                message="Drop this cell",
                rationale="Outlier",
                evidence=DiagnosticEvidence(
                    metric_id="link_ratio_robust_z_2022_24",
                    value=4.1,
                    evidence_id="ev-drop",
                ),
                proposed_parameters={"drop": [["2022", 24]]},
            )
        ],
    )
    context = SessionContext(
        session_id="s-1",
        segment="motor",
        reserving=cast(Any, None),
        sync_version=0,
        params_store=ParamsStore(),
        results_store_meta=ResultsStoreMeta(),
        last_results_payload={},
    )

    scenarios = backend._build_scenario_candidates(
        context=context,
        baseline=baseline,
        baseline_eval=baseline_eval,
        max_scenarios=3,
    )

    assert scenarios
    assert scenarios[0].parent_scenario_id == "baseline"
    assert scenarios[0].transform == "apply_drop_recommendation"
    assert scenarios[0].rationale_evidence_ids == ["ev-drop"]


def test_serialize_dataframe_converts_period_values_to_strings() -> None:
    frame = pd.DataFrame({"origin": [pd.Period("2020Q1", freq="Q")], "value": [1.0]})
    payload = InMemoryReservingBackend._serialize_dataframe(frame)
    records = payload["records"]

    assert len(records) == 1
    assert records[0]["origin"] == "2020Q1"
