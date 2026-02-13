from __future__ import annotations

from source.services.cache_service import CacheService
from source.services.params_service import ParamsService


def _build_params_service() -> ParamsService:
    return ParamsService(
        default_average="volume",
        default_tail_curve="weibull",
        default_bf_apriori=0.6,
        get_uwy_labels=lambda: ["2001", "2002", "2003"],
        load_session=None,
        get_sync_version=None,
    )


def test_selected_ultimate_defaults_to_chainladder() -> None:
    service = _build_params_service()
    selected = service.build_selected_ultimate_by_uwy(None)
    assert selected == {
        "2001": "chainladder",
        "2002": "chainladder",
        "2003": "chainladder",
    }


def test_selected_ultimate_setter_updates_single_uwy() -> None:
    service = _build_params_service()
    selected = service.build_selected_ultimate_by_uwy(None)
    updated = service.set_selected_ultimate_method(
        selected,
        "2002",
        "bornhuetter_ferguson",
    )
    assert updated["2001"] == "chainladder"
    assert updated["2002"] == "bornhuetter_ferguson"
    assert updated["2003"] == "chainladder"


def test_results_cache_key_changes_with_selected_ultimate() -> None:
    cache_service = CacheService()
    key_chainladder = cache_service.build_results_cache_key(
        segment="quarterly",
        default_average="volume",
        default_tail_curve="weibull",
        drop_store=[],
        average="volume",
        tail_attachment_age=None,
        tail_curve="weibull",
        tail_fit_period_selection=[],
        bf_apriori_by_uwy={"2001": 0.6, "2002": 0.6, "2003": 0.6},
        selected_ultimate_by_uwy={
            "2001": "chainladder",
            "2002": "chainladder",
            "2003": "chainladder",
        },
    )
    key_mixed = cache_service.build_results_cache_key(
        segment="quarterly",
        default_average="volume",
        default_tail_curve="weibull",
        drop_store=[],
        average="volume",
        tail_attachment_age=None,
        tail_curve="weibull",
        tail_fit_period_selection=[],
        bf_apriori_by_uwy={"2001": 0.6, "2002": 0.6, "2003": 0.6},
        selected_ultimate_by_uwy={
            "2001": "bornhuetter_ferguson",
            "2002": "chainladder",
            "2003": "chainladder",
        },
    )

    assert key_chainladder != key_mixed
