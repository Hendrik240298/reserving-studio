from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import build_sample_triangle
from source.reserving import Reserving


def test_power_curve_alias_maps_to_inverse_power() -> None:
    reserving = Reserving(build_sample_triangle())
    reserving.set_tail(curve="power_curve")

    assert reserving.tail is not None
    assert reserving.tail.curve == "inverse_power"
