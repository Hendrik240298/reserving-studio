from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import build_sample_triangle


def test_build_sample_triangle_includes_non_zero_premium() -> None:
    triangle = build_sample_triangle().get_triangle("incurred")
    premium_latest = triangle.latest_diagonal["Premium_selected"].to_frame()

    assert float(premium_latest.to_numpy(dtype=float).sum()) > 0.0
