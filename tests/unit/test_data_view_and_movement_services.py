from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.data_view_service import (
    DataViewQuery,
    DataViewService,
    normalize_data_view_query,
)
from source.services.movement_diagnostics_service import MovementDiagnosticsService


class _FrameWrapper:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_frame(self):
        return self._frame.copy()


class _TriangleView:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = {key: value.copy() for key, value in frames.items()}

    def __getitem__(self, key: str):
        return _FrameWrapper(self._frames[key])

    def cum_to_incr(self):
        incremental: dict[str, pd.DataFrame] = {}
        for key, frame in self._frames.items():
            diffs = frame.diff(axis=1)
            diffs.iloc[:, 0] = frame.iloc[:, 0]
            incremental[key] = diffs
        return _TriangleView(incremental)


class _TriangleAccessor:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = frames

    def get_triangle(self, _metric: str = "incurred"):
        return _TriangleView(self._frames)


class _FakeReserving:
    def __init__(self):
        index = pd.Index(["2019", "2020", "2021", "2022"], name="origin")
        columns = pd.Index([12, 24, 36], name="development")
        incurred = pd.DataFrame(
            [
                [100.0, 160.0, 200.0],
                [110.0, 170.0, 220.0],
                [120.0, 185.0, 240.0],
                [140.0, 320.0, 260.0],
            ],
            index=index,
            columns=columns,
        )
        paid = pd.DataFrame(
            [
                [80.0, 130.0, 170.0],
                [85.0, 140.0, 180.0],
                [90.0, 150.0, 190.0],
                [100.0, 150.0, 210.0],
            ],
            index=index,
            columns=columns,
        )
        premium = pd.DataFrame(
            [
                [1000.0, 1000.0, 1000.0],
                [1000.0, 1000.0, 1000.0],
                [1000.0, 1000.0, 1000.0],
                [1000.0, 1020.0, 1200.0],
            ],
            index=index,
            columns=columns,
        )
        self._triangle = _TriangleAccessor(
            {
                "incurred": incurred,
                "paid": paid,
                "Premium_selected": premium,
            }
        )
        self._results = pd.DataFrame(
            {
                "incurred": [200.0, 220.0, 240.0, 260.0],
                "Premium": [1000.0, 1000.0, 1000.0, 1200.0],
                "cl_ultimate": [220.0, 245.0, 280.0, 360.0],
                "bf_ultimate": [225.0, 240.0, 265.0, 310.0],
                "ultimate": [220.0, 245.0, 280.0, 360.0],
                "selected_method": ["chainladder"] * 4,
            },
            index=index,
        )
        link_ratios = pd.DataFrame(
            {
                12: [1.60, 1.55, 1.54, 2.29],
                24: [1.25, 1.29, 1.30, 0.81],
                36: [1.00, 1.00, 1.00, None],
            },
            index=index,
        )
        link_ratios.loc["LDF"] = [1.56, 1.28, 1.0]
        link_ratios.loc["Tail"] = [1.02, 1.01, 1.0]
        self._heatmap = {
            "link_ratios": link_ratios,
            "incurred": incurred,
            "paid": paid,
            "premium": premium,
        }

    def get_triangle_heatmap_data(self):
        return self._heatmap

    def get_results(self):
        return self._results.copy()


def test_data_view_service_supports_ratio_views_and_summary() -> None:
    service = DataViewService(_FakeReserving())
    ratio = service.get_data_view(
        DataViewQuery(metric="incurred", view="cumulative", denominator="premium")
    )
    assert ratio.loc["2022", 24] == 320.0 / 1020.0

    summary = service.summarize_view(
        DataViewQuery(metric="premium", view="incremental")
    )
    assert summary["latest_age"] == "36"
    assert summary["late_movement_candidates"]
    assert summary["top_latest_diagonal_rows"]
    assert summary["latest_diagonal_rows"]


def test_data_view_query_normalization_handles_swapped_metric_and_view() -> None:
    normalized = normalize_data_view_query(
        DataViewQuery(metric="incremental", view="incurred")
    )
    assert normalized.metric == "incurred"
    assert normalized.view == "incremental"


def test_data_view_query_normalization_handles_metric_aliases() -> None:
    normalized_claims = normalize_data_view_query(
        DataViewQuery(metric="claims", view="incremental")
    )
    assert normalized_claims.metric == "incurred"

    normalized = normalize_data_view_query(
        DataViewQuery(metric="incurred_claims", view="incremental")
    )
    assert normalized.metric == "incurred"
    assert normalized.view == "incremental"

    normalized_text = normalize_data_view_query(
        DataViewQuery(metric="incurred claims", view="cumulative")
    )
    assert normalized_text.metric == "incurred"


def test_movement_diagnostics_detects_large_loss_proxy_and_ldf_gap() -> None:
    reserving = _FakeReserving()
    service = MovementDiagnosticsService(reserving)

    movement = service.run()
    finding_codes = {item["code"] for item in movement["findings"]}
    assert any(code.startswith("LARGE_LOSS_PROXY_2022") for code in finding_codes)

    ldf = service.run_ldf_consistency()
    assert ldf["summary"]["finding_count"] >= 1
    assert any(item["origin"] == "2022" for item in ldf["findings"])


def test_movement_diagnostics_prefers_effective_tail_row_over_raw_ldf() -> None:
    reserving = _FakeReserving()
    aligned_link_ratios = pd.DataFrame(
        {
            12: [1.60, 1.60, 1.60, 1.60],
            24: [1.28, 1.28, 1.28, None],
            36: [1.00, 1.00, 1.00, None],
        },
        index=pd.Index(["2019", "2020", "2021", "2022"]),
    )
    aligned_link_ratios.loc["LDF"] = [1.80, 1.50, 1.0]
    aligned_link_ratios.loc["Tail"] = [1.60, 1.28, 1.0]
    reserving._heatmap["link_ratios"] = aligned_link_ratios

    service = MovementDiagnosticsService(reserving)
    ldf = service.run_ldf_consistency()

    assert ldf["summary"]["finding_count"] == 0


def test_late_emergence_benchmark_returns_rows() -> None:
    reserving = _FakeReserving()
    service = MovementDiagnosticsService(reserving)

    benchmark = service.run_late_emergence_benchmark(uwy="2022")
    assert benchmark["summary"]["row_count"] == 1
    assert benchmark["rows"][0]["origin"] == "2022"
