from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.diagnostics_service import DiagnosticsService


def test_diagnostics_v2_returns_findings_and_recommendations() -> None:
    results_df = pd.DataFrame(
        {
            "incurred": [900.0, 880.0, 700.0, 400.0, 150.0],
            "Premium": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "cl_ultimate": [1000.0, 980.0, 875.0, 1100.0, 900.0],
            "bf_ultimate": [990.0, 970.0, 860.0, 800.0, 600.0],
            "ultimate": [1000.0, 980.0, 875.0, 1100.0, 900.0],
        },
        index=pd.Index(["2018", "2019", "2020", "2021", "2022"]),
    )

    link_ratios = pd.DataFrame(
        {
            12: [2.00, 1.98, 2.05, 2.03, 2.01],
            24: [1.30, 1.28, 1.31, 2.40, 1.27],
            36: [1.12, 1.10, 1.09, 1.14, 1.11],
            48: [1.04, 1.05, 1.03, 1.02, 1.04],
        },
        index=pd.Index(["2018", "2019", "2020", "2021", "2022"]),
    )
    heatmap_data = {"link_ratios": link_ratios}

    service = DiagnosticsService()
    run_result = service.run(results_df=results_df, heatmap_data=heatmap_data)

    finding_codes = {item.code for item in run_result.findings}
    recommendation_codes = {item.code for item in run_result.recommendations}

    assert "CL_BF_DIVERGENCE_2021" in finding_codes
    assert "CL_BF_DIVERGENCE_2022" in finding_codes
    assert "RECOMMEND_DROP_2021_24" in recommendation_codes
    assert "RECOMMEND_BF_APRIORI" in recommendation_codes
    assert "RECOMMEND_TAIL_FIT" in recommendation_codes
    assert run_result.metrics["severity_score"] is not None


def test_diagnostics_v2_handles_empty_input() -> None:
    service = DiagnosticsService()
    run_result = service.run(results_df=None, heatmap_data=None)

    assert len(run_result.findings) == 1
    assert run_result.findings[0].code == "NO_MATERIAL_FLAGS"
    assert run_result.recommendations == []


def test_diagnostics_include_latest_diagonal_and_incurred_premium_checks() -> None:
    results_df = pd.DataFrame(
        {
            "incurred": [120.0, 260.0, 430.0, 680.0, 950.0, 1280.0],
            "Premium": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "cl_ultimate": [400.0, 500.0, 700.0, 950.0, 1200.0, 1500.0],
            "bf_ultimate": [420.0, 510.0, 680.0, 900.0, 1100.0, 1300.0],
            "ultimate": [400.0, 500.0, 700.0, 950.0, 1200.0, 1500.0],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )

    incurred = pd.DataFrame(
        {
            12: [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
            24: [180.0, 220.0, 260.0, 300.0, 340.0, 450.0],
            36: [260.0, 320.0, 380.0, 440.0, 500.0, None],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )
    premium = pd.DataFrame(
        {
            12: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            24: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            36: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )
    link_ratios = pd.DataFrame(
        {
            12: [1.8, 1.83, 1.86, 1.88, 1.89, 2.25],
            24: [1.44, 1.45, 1.46, 1.47, 1.48, None],
            36: [1.0, None, None, None, None, None],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )
    link_ratios.loc["LDF"] = [1.85, 1.46, 1.0]
    heatmap_data = {
        "incurred": incurred,
        "premium": premium,
        "link_ratios": link_ratios,
    }

    service = DiagnosticsService(
        latest_diagonal_deviation_threshold=0.12,
        incurred_on_premium_robust_z_threshold=2.0,
        incurred_on_premium_break_threshold=0.15,
    )
    run_result = service.run(results_df=results_df, heatmap_data=heatmap_data)
    codes = {item.code for item in run_result.findings}

    assert any(code.startswith("LATEST_DIAGONAL_DEVIATION_") for code in codes)
    assert any(
        code.startswith("INCURRED_PREMIUM_OUTLIER_")
        or code.startswith("INCURRED_PREMIUM_PORTFOLIO_SHIFT_")
        for code in codes
    )


def test_diagnostics_include_backtest_calendar_tail_paid_and_quality_checks() -> None:
    results_df = pd.DataFrame(
        {
            "incurred": [240.0, 420.0, 700.0, 900.0, 1140.0, 1420.0],
            "Premium": [1000.0, 980.0, 1020.0, 1010.0, 1000.0, 995.0],
            "cl_ultimate": [480.0, 620.0, 860.0, 1150.0, 1450.0, 1780.0],
            "bf_ultimate": [500.0, 640.0, 820.0, 1050.0, 1320.0, 1600.0],
            "ultimate": [480.0, 620.0, 860.0, 1150.0, 1450.0, 1780.0],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )

    incurred = pd.DataFrame(
        {
            12: [120.0, 140.0, 170.0, 220.0, 260.0, 340.0],
            24: [220.0, 250.0, 320.0, 410.0, 520.0, 780.0],
            36: [300.0, 360.0, 460.0, 620.0, 860.0, None],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )
    premium = pd.DataFrame(
        {
            12: [1000.0, 980.0, 1020.0, 1010.0, 1000.0, 995.0],
            24: [1000.0, 980.0, 1020.0, 1010.0, 1000.0, 995.0],
            36: [1000.0, 980.0, 1020.0, 1010.0, 1000.0, 995.0],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )
    paid = pd.DataFrame(
        {
            12: [100.0, 115.0, 140.0, 165.0, 180.0, 210.0],
            24: [180.0, 210.0, 260.0, 330.0, 370.0, 390.0],
            36: [260.0, 300.0, 380.0, 460.0, 500.0, 120.0],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )
    link_ratios = pd.DataFrame(
        {
            12: [1.85, 1.88, 1.90, 1.92, 2.00, 2.30],
            24: [1.35, 1.40, 1.44, 1.51, 1.65, None],
            36: [1.20, None, None, None, None, None],
        },
        index=pd.Index(["2017", "2018", "2019", "2020", "2021", "2022"]),
    )
    link_ratios.loc["LDF"] = [1.9, 1.45, 1.1]
    link_ratios.loc["Tail"] = [1.03, 1.06, 1.08]
    heatmap_data = {
        "incurred": incurred,
        "premium": premium,
        "paid": paid,
        "link_ratios": link_ratios,
    }

    service = DiagnosticsService(
        backtest_bias_threshold=0.01,
        backtest_mae_threshold=0.04,
        calendar_drift_slope_threshold=0.002,
        tail_sensitivity_threshold=0.001,
        paid_incurred_gap_threshold=0.12,
        data_quality_critical_missing_threshold=0.001,
    )
    run_result = service.run(results_df=results_df, heatmap_data=heatmap_data)
    codes = {item.code for item in run_result.findings}

    assert "ROLLING_BACKTEST_BIAS" in codes or "ROLLING_BACKTEST_MAE" in codes
    assert "CALENDAR_YEAR_DRIFT" in codes
    assert "TAIL_SENSITIVITY_HIGH" in codes
    assert any(code.startswith("PAID_INCURRED_COHERENCE_") for code in codes)
    assert run_result.metrics["assessment_confidence"] is not None


def test_data_quality_gate_triggers_for_internal_missing_and_no_premium() -> None:
    results_df = pd.DataFrame(
        {
            "incurred": [100.0, 180.0, 250.0],
            "Premium": [1000.0, 0.0, 0.0],
            "cl_ultimate": [220.0, 300.0, 360.0],
            "bf_ultimate": [210.0, 280.0, 330.0],
            "ultimate": [220.0, 300.0, 360.0],
        },
        index=pd.Index(["2020", "2021", "2022"]),
    )
    incurred = pd.DataFrame(
        {
            12: [80.0, 100.0, 110.0],
            24: [None, 150.0, 190.0],
            36: [220.0, None, None],
        },
        index=pd.Index(["2020", "2021", "2022"]),
    )
    premium = pd.DataFrame(
        {
            12: [1000.0, 0.0, 0.0],
            24: [1000.0, 0.0, 0.0],
            36: [1000.0, 0.0, 0.0],
        },
        index=pd.Index(["2020", "2021", "2022"]),
    )
    link_ratios = pd.DataFrame(
        {
            12: [1.8, 1.9, 2.0],
            24: [1.2, 1.3, None],
            36: [1.0, None, None],
        },
        index=pd.Index(["2020", "2021", "2022"]),
    )
    link_ratios.loc["LDF"] = [1.85, 1.25, 1.0]

    service = DiagnosticsService(data_quality_critical_missing_threshold=0.001)
    run_result = service.run(
        results_df=results_df,
        heatmap_data={
            "incurred": incurred,
            "premium": premium,
            "link_ratios": link_ratios,
        },
    )
    codes = {item.code for item in run_result.findings}
    assert "DATA_QUALITY_GATE" in codes
