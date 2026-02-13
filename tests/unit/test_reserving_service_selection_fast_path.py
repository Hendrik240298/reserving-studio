from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.cache_service import CacheService
from source.services.params_service import ParamsService
from source.services.reserving_service import ReservingService


class _FakeReserving:
    def __init__(self) -> None:
        self.reserve_calls = 0

    def set_development(self, average: str, drop: list[tuple[str, int]] | None) -> None:
        return None

    def set_tail(
        self,
        curve: str,
        projection_period: int,
        attachment_age: int | None,
        fit_period: tuple[int, int | None] | None,
    ) -> None:
        return None

    def set_bornhuetter_ferguson(self, apriori: float | dict[str, float]) -> None:
        return None

    def reserve(
        self,
        final_ultimate: str,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> None:
        self.reserve_calls += 1


def test_selection_change_reuses_model_without_recalc() -> None:
    fake_reserving = _FakeReserving()
    results_df = pd.DataFrame(
        {
            "incurred": [100.0, 120.0],
            "Premium": [200.0, 240.0],
            "cl_ultimate": [150.0, 180.0],
            "bf_ultimate": [160.0, 175.0],
            "ultimate": [150.0, 180.0],
        },
        index=["2001", "2002"],
    )

    params_service = ParamsService(
        default_average="volume",
        default_tail_curve="weibull",
        default_bf_apriori=0.6,
        get_uwy_labels=lambda: ["2001", "2002"],
        load_session=None,
        get_sync_version=None,
    )
    service = ReservingService(
        reserving=fake_reserving,  # type: ignore[arg-type]
        params_service=params_service,
        cache_service=CacheService(),
        default_average="volume",
        default_tail_curve="weibull",
        default_bf_apriori=0.6,
        segment_key_provider=lambda: "quarterly",
        extract_data=lambda: None,
        get_triangle=lambda: pd.DataFrame(),
        get_emergence=lambda: pd.DataFrame(),
        get_results=lambda: results_df,
        build_triangle_figure=lambda *_args: {"kind": "triangle"},
        build_emergence_figure=lambda *_args: {"kind": "emergence"},
        payload_cache={},
        triangle_cache={},
        emergence_cache={},
    )

    payload_cl = service.get_or_build_results_payload(
        drop_store=[],
        average="volume",
        tail_attachment_age=None,
        tail_curve="weibull",
        tail_fit_period_selection=[],
        bf_apriori_by_uwy={"2001": 0.6, "2002": 0.6},
        selected_ultimate_by_uwy={"2001": "chainladder", "2002": "chainladder"},
        force_recalc=False,
    )
    assert fake_reserving.reserve_calls == 1

    payload_bf = service.get_or_build_results_payload(
        drop_store=[],
        average="volume",
        tail_attachment_age=None,
        tail_curve="weibull",
        tail_fit_period_selection=[],
        bf_apriori_by_uwy={"2001": 0.6, "2002": 0.6},
        selected_ultimate_by_uwy={
            "2001": "bornhuetter_ferguson",
            "2002": "chainladder",
        },
        force_recalc=False,
    )
    assert fake_reserving.reserve_calls == 1

    row_by_uwy = {row["uwy"]: row for row in payload_bf["results_table_rows"]}
    assert (
        row_by_uwy["2001"]["ultimate_display"]
        == row_by_uwy["2001"]["bf_ultimate_display"]
    )
    assert (
        row_by_uwy["2002"]["ultimate_display"]
        == row_by_uwy["2002"]["cl_ultimate_display"]
    )

    assert payload_cl["model_cache_key"] == payload_bf["model_cache_key"]
    assert payload_cl["cache_key"] != payload_bf["cache_key"]
