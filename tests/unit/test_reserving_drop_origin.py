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
    return Reserving(build_sample_triangle())


def _origin_drop_tuples(reserving: Reserving, target_uwy: str) -> list[tuple[str, int]]:
    raw_link_ratio = (
        reserving._triangle.get_triangle().link_ratio["incurred"].to_frame()
    )
    for origin in raw_link_ratio.index:
        if reserving._origin_to_uwy_label(origin) != target_uwy:
            continue
        drops: list[tuple[str, int]] = []
        for dev_label, value in raw_link_ratio.loc[origin].items():
            if pd.isna(value):
                continue
            age = reserving._parse_cdf_label_to_age(dev_label)
            if age is None:
                continue
            drops.append((target_uwy, age))
        return drops
    return []


def test_drop_origin_expands_to_row_drop_tuples() -> None:
    reserving = _build_reserving()
    target_uwy = reserving._origin_to_uwy_label(
        reserving._triangle.get_triangle()["incurred"].origin[0]
    )
    expected = _origin_drop_tuples(reserving, target_uwy)

    reserving.set_development(average="volume", drop_origin=target_uwy)

    assert reserving.development is not None
    assert reserving.development.drop == expected


def test_drop_origin_merges_with_explicit_drop() -> None:
    reserving = _build_reserving()
    target_uwy = reserving._origin_to_uwy_label(
        reserving._triangle.get_triangle()["incurred"].origin[0]
    )
    expected_row = _origin_drop_tuples(reserving, target_uwy)
    extra_drop = (target_uwy, 12_345)

    reserving.set_development(
        average="volume",
        drop=[extra_drop],
        drop_origin=target_uwy,
    )

    assert reserving.development is not None
    assert reserving.development.drop == [extra_drop] + expected_row


def test_drop_origin_matches_explicit_drop_results() -> None:
    target_uwy = "1995"

    explicit = _build_reserving()
    expected_row = _origin_drop_tuples(explicit, target_uwy)
    explicit.set_development(average="volume", drop=expected_row)
    explicit.set_tail(
        curve="weibull",
        projection_period=0,
        attachment_age=None,
        fit_period=None,
    )
    explicit.set_bornhuetter_ferguson(apriori=0.6)
    explicit.reserve(final_ultimate="chainladder")

    by_origin = _build_reserving()
    by_origin.set_development(average="volume", drop_origin=target_uwy)
    by_origin.set_tail(
        curve="weibull",
        projection_period=0,
        attachment_age=None,
        fit_period=None,
    )
    by_origin.set_bornhuetter_ferguson(apriori=0.6)
    by_origin.reserve(final_ultimate="chainladder")

    pd.testing.assert_frame_equal(
        explicit.result.named_steps.dev.ldf_.to_frame(),
        by_origin.result.named_steps.dev.ldf_.to_frame(),
    )
    pd.testing.assert_frame_equal(explicit.df_results, by_origin.df_results)


def test_drop_origin_ignores_missing_origin() -> None:
    reserving = _build_reserving()

    reserving.set_development(average="volume", drop_origin="2099")

    assert reserving.development is not None
    assert reserving.development.drop is None
