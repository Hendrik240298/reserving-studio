from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import build_sample_triangle
from source.reserving import Reserving


def test_weighted_average_alias_maps_to_volume() -> None:
    reserving = Reserving(build_sample_triangle())
    reserving.set_development(average="weighted_average_all", drop=None)

    assert reserving.development is not None
    assert reserving.development.average == "volume"

    reserving_two = Reserving(build_sample_triangle())
    reserving_two.set_development(average="volume_weighted_average", drop=None)
    assert reserving_two.development is not None
    assert reserving_two.development.average == "volume"


def test_unknown_average_still_raises_clear_error() -> None:
    reserving = Reserving(build_sample_triangle())

    try:
        reserving.set_development(average="mystery_average", drop=None)
    except ValueError as error:
        assert "average must be 'volume' or 'simple'" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown average alias")
