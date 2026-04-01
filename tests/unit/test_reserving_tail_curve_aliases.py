from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import build_sample_triangle
from source.reserving import Reserving, StableTailCurve


def test_power_curve_alias_maps_to_inverse_power() -> None:
    reserving = Reserving(build_sample_triangle())
    reserving.set_tail(curve="power_curve")

    assert reserving.tail is not None
    assert reserving.tail.curve == "inverse_power"


def test_tail_vector_is_clipped_monotone_and_ends_at_one() -> None:
    raw = np.array([[[1.08, 0.99, 1.03, 1.01, 1.2]]], dtype=float)

    adjusted = Reserving._enforce_monotone_tail_vector(raw)

    assert adjusted.tolist() == [[[1.08, 1.03, 1.03, 1.01, 1.0]]]


def test_tail_cdf_is_recomputed_from_adjusted_ldf_vector() -> None:
    ldf = np.array([[[1.08, 1.03, 1.03, 1.01, 1.0]]], dtype=float)

    cdf = Reserving._cdf_from_tail_ldf(ldf)

    expected = np.array([[[1.15722972, 1.0715089999999998, 1.0403, 1.01, 1.0]]])
    assert np.allclose(cdf, expected)


def test_weibull_tail_prediction_stays_finite_for_large_intercept() -> None:
    tail = StableTailCurve(curve="weibull", extrap_periods=4)
    tail.ldf_ = SimpleNamespace(get_array_module=lambda: np, shape=(1, 1, 1, 3))
    tail._intercept_ = np.array([[[[1000.0]]]], dtype=float)
    tail._slope_ = np.array([[[[2.0]]]], dtype=float)

    predicted = tail._predict_tail(
        np.array([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]], dtype=float)
    )

    assert np.isfinite(predicted).all()
    assert np.all(predicted >= 1.0)
