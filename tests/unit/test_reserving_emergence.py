from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import build_sample_triangle
from source.reserving import Reserving


def _build_reserving() -> Reserving:
    reserving = Reserving(build_sample_triangle())
    reserving.set_development(average="volume", drop=None)
    reserving.set_tail(
        curve="weibull",
        projection_period=0,
        attachment_age=None,
        fit_period=None,
    )
    reserving.set_bornhuetter_ferguson(apriori=0.6)
    return reserving


def _all_bf_selection(reserving: Reserving) -> dict[str, str]:
    return {
        reserving._origin_to_uwy_label(idx): "bornhuetter_ferguson"
        for idx in reserving.df_results.index
    }


def test_emergence_is_independent_of_selected_results_method() -> None:
    reserving = _build_reserving()
    reserving.reserve(final_ultimate="chainladder")
    baseline = reserving.get_emergence_pattern()

    reserving.reserve(
        final_ultimate="chainladder",
        selected_ultimate_by_uwy=_all_bf_selection(reserving),
    )
    switched = reserving.get_emergence_pattern()

    pd.testing.assert_frame_equal(switched["Actual"], baseline["Actual"])
    pd.testing.assert_frame_equal(switched["Expected"], baseline["Expected"])


def test_emergence_expected_uses_chainladder_tail_cdf() -> None:
    reserving = _build_reserving()
    reserving.reserve(final_ultimate="chainladder")
    reserving.reserve(
        final_ultimate="chainladder",
        selected_ultimate_by_uwy=_all_bf_selection(reserving),
    )

    assert reserving._chainladder_result is not None
    assert reserving._bornhuetter_result is not None
    assert reserving.result is reserving._bornhuetter_result

    emergence = reserving.get_emergence_pattern()
    expected_row = emergence["Expected"].iloc[0]

    cl_cdf = (
        reserving._chainladder_result.named_steps.tail.cdf_["incurred"]
        .to_frame()
        .iloc[0]
    )
    cl_expected = 1 / cl_cdf
    expected_aligned = cl_expected.reindex(expected_row.index)
    if expected_aligned.isna().all():
        expected_aligned = pd.Series(
            cl_expected.values[: len(expected_row.index)],
            index=expected_row.index[: len(cl_expected)],
        )
    expected_aligned = expected_aligned.reindex(expected_row.index)

    pd.testing.assert_series_equal(expected_row, expected_aligned, check_names=False)


def test_emergence_expected_has_values_for_all_development_columns() -> None:
    reserving = _build_reserving()
    reserving.reserve(final_ultimate="chainladder")

    emergence = reserving.get_emergence_pattern()
    expected = emergence["Expected"]

    assert not expected.isna().all(axis=0).any()
