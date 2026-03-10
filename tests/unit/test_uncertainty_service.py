from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.uncertainty_service import UncertaintyService


def _sample_inputs() -> tuple[pd.DataFrame, dict]:
    results_df = pd.DataFrame(
        {
            "incurred": [80.0, 95.0, 110.0],
            "ultimate": [120.0, 130.0, 140.0],
            "cl_ultimate": [120.0, 130.0, 140.0],
        },
        index=pd.Index(["2020", "2021", "2022"]),
    )
    incurred = pd.DataFrame(
        {
            12: [40.0, 50.0, 60.0],
            24: [70.0, 85.0, 100.0],
            36: [80.0, 95.0, 110.0],
        },
        index=results_df.index,
    )
    link_ratios = pd.DataFrame(
        {
            12: [1.7, 1.7, 1.7],
            24: [1.2, 1.2, 1.2],
            36: [1.0, 1.0, 1.0],
        },
        index=results_df.index,
    )
    link_ratios.loc["LDF"] = [1.7, 1.2, 1.0]
    return results_df, {"incurred": incurred, "link_ratios": link_ratios}


def test_baseline_uncertainty_returns_expected_structure() -> None:
    service = UncertaintyService()
    results_df, heatmap_data = _sample_inputs()

    uncertainty = service.baseline_uncertainty(
        results_df=results_df,
        heatmap_data=heatmap_data,
    )

    assert uncertainty["version"] == "v1.1"
    assert uncertainty["mack_msep_by_uwy"]
    assert uncertainty["bf_prediction_error_by_uwy"]
    assert uncertainty["mack_total_msep"] > 0
    assert uncertainty["bf_total_prediction_error"] > 0
    assert uncertainty["total_process_cv"] is not None
    assert uncertainty["mack_total_msep"] == 3.438
    assert uncertainty["bf_total_prediction_error"] == 2.9755


def test_bootstrap_distribution_returns_quantiles() -> None:
    service = UncertaintyService()
    results_df, heatmap_data = _sample_inputs()

    distribution = service.bootstrap_predictive_distribution(
        results_df=results_df,
        heatmap_data=heatmap_data,
        sample_count=120,
        seed=3,
    )

    assert distribution["sample_count"] == 120
    assert distribution["p10"] <= distribution["p50"]
    assert distribution["p95"] >= distribution["p50"]
    assert distribution["iqr"] >= 0


def test_tail_model_assessment_flags_instability_for_wide_spread() -> None:
    service = UncertaintyService()
    assessment = service.tail_model_assessment(
        scenarios=[
            {
                "scenario_id": "tail_curve_a",
                "score": 1.0,
                "transform": "tail_curve_fit_period_grid",
            },
            {
                "scenario_id": "tail_curve_b",
                "score": 3.5,
                "transform": "tail_curve_fit_period_grid",
            },
        ]
    )

    assert assessment["tail_scenario_count"] == 2
    assert assessment["instability_flag"] is True
    assert assessment["model_average"]["weighted_average_score"] > 0
