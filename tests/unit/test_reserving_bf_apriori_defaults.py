from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import build_sample_triangle
from source.reserving import Reserving


def _build_reserving(apriori: float | dict[str, float]) -> Reserving:
    reserving = Reserving(build_sample_triangle())
    reserving.set_development(average="volume", drop=None)
    reserving.set_tail(
        curve="weibull",
        projection_period=0,
        attachment_age=None,
        fit_period=None,
    )
    reserving.set_bornhuetter_ferguson(apriori=apriori)
    reserving.reserve(final_ultimate="chainladder")
    return reserving


def _uwy_labels(reserving: Reserving) -> list[str]:
    return [
        reserving._origin_to_uwy_label(origin)
        for origin in reserving._triangle.get_triangle()[
            "incurred"
        ].latest_diagonal.origin
    ]


def test_empty_bf_apriori_mapping_uses_defaults() -> None:
    scalar = _build_reserving(0.6)
    empty_mapping = _build_reserving({})

    pd.testing.assert_series_equal(
        scalar.df_results["bf_ultimate"],
        empty_mapping.df_results["bf_ultimate"],
        check_names=False,
    )


def test_missing_bf_apriori_entries_fall_back_to_default() -> None:
    partial = _build_reserving({})
    uwy = _uwy_labels(partial)
    assert len(uwy) > 1

    partial_mapping = {uwy[0]: 1.2}
    full_mapping = {label: 0.6 for label in uwy}
    full_mapping[uwy[0]] = 1.2

    reserving_partial = _build_reserving(partial_mapping)
    reserving_full = _build_reserving(full_mapping)

    pd.testing.assert_series_equal(
        reserving_partial.df_results["bf_ultimate"],
        reserving_full.df_results["bf_ultimate"],
        check_names=False,
    )
