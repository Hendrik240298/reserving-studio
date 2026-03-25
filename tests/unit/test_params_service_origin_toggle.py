from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.params_service import ParamsService


def _build_service() -> ParamsService:
    return ParamsService(
        default_average="volume",
        default_tail_curve="weibull",
        default_tail_projection_months=0,
        default_bf_apriori=0.6,
        get_uwy_labels=lambda: ["1995", "1996"],
        load_session=None,
        get_sync_version=None,
    )


def test_toggle_origin_drops_adds_full_origin_when_partial() -> None:
    service = _build_service()

    updated = service.toggle_origin_drops([["1995", 12]], "1995", [12, 24, 36])

    assert updated == [["1995", 12], ["1995", 24], ["1995", 36]]


def test_toggle_origin_drops_removes_full_origin_when_complete() -> None:
    service = _build_service()

    updated = service.toggle_origin_drops(
        [["1995", 12], ["1995", 24], ["1995", 36], ["1996", 12]],
        "1995",
        [12, 24, 36],
    )

    assert updated == [["1996", 12]]
