from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

import pandas as pd

from source.reserving import Reserving
from source.services.cache_service import CacheService
from source.services.params_service import ParamsService


class ReservingService:
    def __init__(
        self,
        *,
        reserving: Reserving,
        params_service: ParamsService,
        cache_service: CacheService,
        default_average: str,
        default_tail_curve: str,
        default_bf_apriori: float,
        segment_key_provider: Callable[[], str],
        extract_data: Callable[[], None],
        get_triangle: Callable[[], pd.DataFrame | None],
        get_emergence: Callable[[], pd.DataFrame | None],
        get_results: Callable[[], pd.DataFrame | None],
        build_triangle_figure: Callable[
            [pd.DataFrame | None, str, int | None, list[int] | None], dict
        ],
        build_emergence_figure: Callable[[pd.DataFrame | None, str], dict],
        build_results_table_figure: Callable[[pd.DataFrame | None, str], dict],
        payload_cache: dict[str, dict],
        triangle_cache: dict[str, dict],
        emergence_cache: dict[str, dict],
        results_table_cache: dict[str, dict],
    ) -> None:
        self._reserving = reserving
        self._params_service = params_service
        self._cache_service = cache_service
        self._default_average = default_average
        self._default_tail_curve = default_tail_curve
        self._default_bf_apriori = default_bf_apriori
        self._segment_key_provider = segment_key_provider
        self._extract_data = extract_data
        self._get_triangle = get_triangle
        self._get_emergence = get_emergence
        self._get_results = get_results
        self._build_triangle_figure = build_triangle_figure
        self._build_emergence_figure = build_emergence_figure
        self._build_results_table_figure = build_results_table_figure
        self._payload_cache = payload_cache
        self._triangle_cache = triangle_cache
        self._emergence_cache = emergence_cache
        self._results_table_cache = results_table_cache

    @staticmethod
    def _build_results_table_rows(
        results_df: pd.DataFrame | None,
    ) -> list[dict[str, str]]:
        if results_df is None or len(results_df) == 0:
            return []

        rows: list[dict[str, str]] = []
        for idx, row in results_df.iterrows():
            if hasattr(idx, "year"):
                uwy = str(idx.year)
            else:
                uwy_text = str(idx)
                uwy = uwy_text[:4] if len(uwy_text) >= 4 else uwy_text

            incurred = float(row.get("incurred", 0.0))
            premium = float(row.get("Premium", 0.0))
            cl_ultimate = float(row.get("cl_ultimate", 0.0))
            bf_ultimate = float(row.get("bf_ultimate", 0.0))
            selected_ultimate = float(row.get("ultimate", 0.0))
            ibnr = selected_ultimate - incurred

            if premium > 0:
                incurred_lr_display = f"{(incurred / premium):.2%}"
                cl_lr_display = f"{(cl_ultimate / premium):.2%}"
                bf_lr_display = f"{(bf_ultimate / premium):.2%}"
                selected_lr_display = f"{(selected_ultimate / premium):.2%}"
            else:
                incurred_lr_display = "N/A"
                cl_lr_display = "N/A"
                bf_lr_display = "N/A"
                selected_lr_display = "N/A"

            rows.append(
                {
                    "uwy": uwy,
                    "incurred_display": f"{incurred:,.0f}",
                    "premium_display": f"{premium:,.0f}",
                    "incurred_loss_ratio_display": incurred_lr_display,
                    "cl_ultimate_display": f"{cl_ultimate:,.2f}",
                    "cl_loss_ratio_display": cl_lr_display,
                    "bf_ultimate_display": f"{bf_ultimate:,.2f}",
                    "bf_loss_ratio_display": bf_lr_display,
                    "ultimate_display": f"{selected_ultimate:,.2f}",
                    "selected_loss_ratio_display": selected_lr_display,
                    "ibnr_display": f"{ibnr:,.2f}",
                }
            )
        return rows

    def _build_results_cache_key(
        self,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> str:
        return self._cache_service.build_results_cache_key(
            segment=self._segment_key_provider(),
            default_average=self._default_average,
            default_tail_curve=self._default_tail_curve,
            drop_store=drop_store,
            average=average,
            tail_attachment_age=tail_attachment_age,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
            selected_ultimate_by_uwy=selected_ultimate_by_uwy,
        )

    def _build_visual_cache_key(
        self,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
    ) -> str:
        return self._cache_service.build_visual_cache_key(
            segment=self._segment_key_provider(),
            default_average=self._default_average,
            default_tail_curve=self._default_tail_curve,
            drop_store=drop_store,
            average=average,
            tail_attachment_age=tail_attachment_age,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
        )

    def apply_recalculation(
        self,
        average: str,
        drops: list[tuple[str, int]] | None,
        tail_attachment_age: int | None,
        tail_curve: str,
        fit_period: tuple[int, int | None] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> None:
        self._reserving.set_development(
            average=average,
            drop=drops,
        )
        self._reserving.set_tail(
            curve=tail_curve,
            projection_period=0,
            attachment_age=tail_attachment_age,
            fit_period=fit_period,
        )
        if bf_apriori_by_uwy:
            self._reserving.set_bornhuetter_ferguson(apriori=bf_apriori_by_uwy)
        else:
            self._reserving.set_bornhuetter_ferguson(apriori=self._default_bf_apriori)
        self._reserving.reserve(
            final_ultimate="chainladder",
            selected_ultimate_by_uwy=selected_ultimate_by_uwy,
        )
        self._extract_data()

    def build_results_payload(
        self,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> dict:
        timestamp = datetime.utcnow().isoformat() + "Z"
        display = "None"
        if drop_store:
            display = ", ".join([f"{item[0]}:{item[1]}" for item in drop_store])
        tail_display = "None"
        if tail_attachment_age is not None:
            tail_display = str(tail_attachment_age)

        fit_period = self._params_service.derive_tail_fit_period(
            tail_fit_period_selection
        )
        if fit_period is None:
            fit_period_display = "lower=None, upper=None"
        else:
            fit_period_display = f"lower={fit_period[0]}, upper={fit_period[1]}"

        visual_cache_key = self._build_visual_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_curve,
            tail_fit_period_selection,
        )
        results_table_cache_key = self._build_results_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
            selected_ultimate_by_uwy=selected_ultimate_by_uwy,
        )

        results_df = self._get_results()

        triangle_figure = self._cache_service.get_or_build(
            cache=self._triangle_cache,
            cache_key=visual_cache_key,
            label="Triangle figure",
            builder=lambda: self._build_triangle_figure(
                self._get_triangle(),
                "Triangle - Link Ratios",
                tail_attachment_age,
                tail_fit_period_selection,
            ),
        )
        emergence_figure = self._cache_service.get_or_build(
            cache=self._emergence_cache,
            cache_key=visual_cache_key,
            label="Emergence figure",
            builder=lambda: self._build_emergence_figure(
                self._get_emergence(),
                "Emergence Pattern",
            ),
        )
        results_figure = self._cache_service.get_or_build(
            cache=self._results_table_cache,
            cache_key=results_table_cache_key,
            label="Results table figure",
            builder=lambda: self._build_results_table_figure(
                results_df,
                "Reserving Results",
            ),
        )

        return {
            "triangle_figure": triangle_figure,
            "emergence_figure": emergence_figure,
            "results_figure": results_figure,
            "drops_display": display,
            "average": average,
            "tail_curve": tail_curve,
            "drop_store": drop_store or [],
            "tail_attachment_age": tail_attachment_age,
            "tail_attachment_display": tail_display,
            "tail_fit_period_selection": tail_fit_period_selection or [],
            "tail_fit_period_display": fit_period_display,
            "selected_ultimate_by_uwy": selected_ultimate_by_uwy or {},
            "results_table_rows": self._build_results_table_rows(results_df),
            "last_updated": timestamp,
        }

    def get_or_build_results_payload(
        self,
        *,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
        force_recalc: bool = False,
    ) -> dict:
        cache_key = self._build_results_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy,
            selected_ultimate_by_uwy,
        )
        cached_payload = (
            None
            if force_recalc
            else self._cache_service.get(self._payload_cache, cache_key)
        )
        if cached_payload is not None:
            logging.info("Using cached reserving payload for current parameters")
            return cached_payload

        fit_period = self._params_service.derive_tail_fit_period(
            tail_fit_period_selection
        )
        started = time.perf_counter()
        self.apply_recalculation(
            average or self._default_average,
            self._params_service.drops_to_tuples(drop_store),
            tail_attachment_age,
            tail_curve or self._default_tail_curve,
            fit_period,
            bf_apriori_by_uwy,
            selected_ultimate_by_uwy,
        )
        recalc_elapsed_ms = (time.perf_counter() - started) * 1000
        logging.info("Recalculation completed in %.0f ms", recalc_elapsed_ms)

        payload_started = time.perf_counter()
        payload = self.build_results_payload(
            drop_store=drop_store,
            average=average,
            tail_attachment_age=tail_attachment_age,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
            selected_ultimate_by_uwy=selected_ultimate_by_uwy,
        )
        payload_elapsed_ms = (time.perf_counter() - payload_started) * 1000
        logging.info("Payload build completed in %.0f ms", payload_elapsed_ms)
        payload["cache_key"] = cache_key
        self._cache_service.set(self._payload_cache, cache_key, payload)
        return payload
