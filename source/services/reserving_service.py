from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
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
        fallback_months_per_dev: int,
        segment_key_provider: Callable[[], str],
        extract_data: Callable[[], None],
        get_triangle: Callable[[], pd.DataFrame | None],
        get_emergence: Callable[[], pd.DataFrame | None],
        get_ave: Callable[
            [dict[str, str] | None], tuple[object, object, object] | None
        ],
        get_results: Callable[[], pd.DataFrame | None],
        build_triangle_figure: Callable[
            [pd.DataFrame | None, str, int | None, list[int] | None], dict
        ],
        build_emergence_figure: Callable[[pd.DataFrame | None, str], dict],
        payload_cache: dict[str, dict],
        triangle_cache: dict[str, dict],
        emergence_cache: dict[str, dict],
        ave_cache: dict[str, dict] | None = None,
    ) -> None:
        self._reserving = reserving
        self._params_service = params_service
        self._cache_service = cache_service
        self._default_average = default_average
        self._default_tail_curve = default_tail_curve
        self._default_bf_apriori = default_bf_apriori
        self._fallback_months_per_dev = max(int(fallback_months_per_dev), 1)
        self._segment_key_provider = segment_key_provider
        self._extract_data = extract_data
        self._get_triangle = get_triangle
        self._get_emergence = get_emergence
        self._get_ave = get_ave
        self._get_results = get_results
        self._build_triangle_figure = build_triangle_figure
        self._build_emergence_figure = build_emergence_figure
        self._payload_cache = payload_cache
        self._triangle_cache = triangle_cache
        self._emergence_cache = emergence_cache
        self._ave_cache = ave_cache if ave_cache is not None else {}

    def set_reserving(self, reserving: Reserving) -> None:
        self._reserving = reserving

    def _infer_months_per_development_period(self, reserving: Reserving) -> int:
        try:
            triangle = reserving._triangle.get_triangle()["incurred"]
            development = [int(value) for value in triangle.development.tolist()]
        except Exception:
            development = []

        if len(development) >= 2:
            spacing = [
                right - left
                for left, right in zip(development[:-1], development[1:])
                if right - left > 0
            ]
            if spacing:
                return min(spacing)
        if len(development) == 1 and development[0] > 0:
            return development[0]
        return self._fallback_months_per_dev

    def derive_tail_projection_settings(
        self,
        *,
        reserving: Reserving,
        tail_projection_months: int,
    ) -> tuple[int, int, int]:
        months_per_dev = self._infer_months_per_development_period(reserving)
        months = max(int(tail_projection_months), 0)
        extrap_periods = months // months_per_dev
        projection_period = extrap_periods * months_per_dev
        return months_per_dev, extrap_periods, projection_period

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

    @staticmethod
    def _normalize_origin_label(value: object) -> str:
        if hasattr(value, "year"):
            return str(value.year)
        text = str(value)
        if len(text) >= 4 and text[:4].isdigit():
            return text[:4]
        return text

    @staticmethod
    def _normalize_origin_sort(value: object) -> int | str:
        label = ReservingService._normalize_origin_label(value)
        if label.isdigit():
            return int(label)
        return label

    @staticmethod
    def _normalize_valuation_label(value: object) -> str:
        if isinstance(value, pd.Period):
            period = value.asfreq("Q")
            return f"{period.year}Q{period.quarter}"
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.notna(timestamp):
            period = timestamp.to_period("Q")
            return f"{period.year}Q{period.quarter}"
        return str(value)

    @staticmethod
    def _normalize_valuation_sort(value: object) -> int | str:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.notna(timestamp):
            period = timestamp.to_period("Q")
            return int(period.year) * 10 + int(period.quarter)
        return ReservingService._normalize_valuation_label(value)

    @staticmethod
    def _flatten_ave_triangle(triangle: object, value_name: str) -> pd.DataFrame:
        if triangle is None:
            return pd.DataFrame(
                columns=[
                    "origin",
                    "origin_label",
                    "origin_sort",
                    "valuation",
                    "valuation_sort",
                    value_name,
                ]
            )

        frame = triangle.to_frame(keepdims=True).reset_index()
        candidate_columns = [
            column
            for column in frame.columns
            if column not in {"index", "level_0", "Total", "origin", "valuation"}
        ]
        if "incurred" in candidate_columns:
            value_column = "incurred"
        else:
            numeric_candidates = (
                frame[candidate_columns]
                .select_dtypes(include=["number"])
                .columns.tolist()
            )
            value_column = numeric_candidates[0] if numeric_candidates else None
            if value_column is None and candidate_columns:
                value_column = candidate_columns[0]
        if value_column is None:
            return pd.DataFrame(
                columns=[
                    "origin",
                    "origin_label",
                    "origin_sort",
                    "valuation",
                    "valuation_sort",
                    value_name,
                ]
            )

        flattened = frame[["origin", "valuation", value_column]].copy()
        flattened["origin_label"] = flattened["origin"].map(
            ReservingService._normalize_origin_label
        )
        flattened["origin_sort"] = flattened["origin"].map(
            ReservingService._normalize_origin_sort
        )
        flattened["valuation"] = flattened["valuation"].map(
            ReservingService._normalize_valuation_label
        )
        flattened["valuation_sort"] = frame["valuation"].map(
            ReservingService._normalize_valuation_sort
        )
        flattened = flattened.rename(columns={value_column: value_name})
        return flattened

    @staticmethod
    def _build_ave_payload(
        ave_triangles: tuple[object, object, object] | None,
    ) -> dict:
        empty_payload = {
            "available": False,
            "options": [],
            "default_valuation": None,
            "origin_views": {},
            "series": {"valuation": [], "expected": [], "actual": [], "diff": []},
        }
        if ave_triangles is None:
            return empty_payload

        ave_triangle, actual_triangle, expected_triangle = ave_triangles
        diff_df = ReservingService._flatten_ave_triangle(ave_triangle, "diff")
        actual_df = ReservingService._flatten_ave_triangle(actual_triangle, "actual")
        expected_df = ReservingService._flatten_ave_triangle(
            expected_triangle, "expected"
        )

        merge_keys = [
            "origin",
            "origin_label",
            "origin_sort",
            "valuation",
            "valuation_sort",
        ]
        merged = diff_df.merge(actual_df, on=merge_keys, how="outer")
        merged = merged.merge(expected_df, on=merge_keys, how="outer")
        if merged.empty:
            return empty_payload

        for column in ["actual", "expected", "diff"]:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged = merged.dropna(subset=["actual", "expected", "diff"], how="all")
        if merged.empty:
            return empty_payload

        merged["diff"] = merged["diff"].where(
            merged["diff"].notna(),
            merged["actual"] - merged["expected"],
        )

        origin_agg = (
            merged.groupby(
                ["valuation", "valuation_sort", "origin_label", "origin_sort"],
                dropna=False,
            )[["actual", "expected", "diff"]]
            .sum(min_count=1)
            .reset_index()
            .sort_values(["valuation_sort", "origin_sort"])
        )
        series_agg = (
            merged.groupby(["valuation", "valuation_sort"], dropna=False)[
                ["actual", "expected", "diff"]
            ]
            .sum(min_count=1)
            .reset_index()
            .sort_values("valuation_sort")
        )

        valuation_options = [
            {"value": str(value), "label": str(value)}
            for value in series_agg["valuation"].tolist()
        ]
        if not valuation_options:
            return empty_payload

        origin_views: dict[str, dict] = {}
        for valuation, group in origin_agg.groupby("valuation", sort=False):
            valuation_key = str(valuation)
            ordered = group.sort_values("origin_sort")
            expected = pd.to_numeric(ordered["expected"], errors="coerce").fillna(0.0)
            actual = pd.to_numeric(ordered["actual"], errors="coerce").fillna(0.0)
            diff = pd.to_numeric(ordered["diff"], errors="coerce").fillna(0.0)
            origin_views[valuation_key] = {
                "origin": [str(value) for value in ordered["origin_label"].tolist()],
                "expected": expected.tolist(),
                "actual": actual.tolist(),
                "diff": diff.tolist(),
                "kpis": {
                    "expected_total": float(expected.sum()),
                    "actual_total": float(actual.sum()),
                    "diff_total": float(diff.sum()),
                },
            }

        return {
            "available": True,
            "options": valuation_options,
            "default_valuation": valuation_options[-1]["value"],
            "origin_views": origin_views,
            "series": {
                "valuation": [str(value) for value in series_agg["valuation"].tolist()],
                "expected": pd.to_numeric(series_agg["expected"], errors="coerce")
                .fillna(0.0)
                .tolist(),
                "actual": pd.to_numeric(series_agg["actual"], errors="coerce")
                .fillna(0.0)
                .tolist(),
                "diff": pd.to_numeric(series_agg["diff"], errors="coerce")
                .fillna(0.0)
                .tolist(),
            },
        }

    def _build_results_cache_key(
        self,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
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
            tail_projection_months=tail_projection_months,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
            selected_ultimate_by_uwy=selected_ultimate_by_uwy,
        )

    def _build_model_cache_key(
        self,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
    ) -> str:
        return self._cache_service.build_model_cache_key(
            segment=self._segment_key_provider(),
            default_average=self._default_average,
            default_tail_curve=self._default_tail_curve,
            drop_store=drop_store,
            average=average,
            tail_attachment_age=tail_attachment_age,
            tail_projection_months=tail_projection_months,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
        )

    def _build_visual_cache_key(
        self,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
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
            tail_projection_months=tail_projection_months,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
        )

    def apply_recalculation(
        self,
        average: str,
        drops: list[tuple[str, int]] | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
        tail_curve: str,
        fit_period: tuple[int, int | None] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> None:
        total_started = time.perf_counter()
        step_started = time.perf_counter()
        self._reserving.set_development(
            average=average,
            drop=drops,
        )
        logging.info(
            "Recalc step set_development completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        _months_per_dev, extrap_periods, projection_period = (
            self.derive_tail_projection_settings(
                reserving=self._reserving,
                tail_projection_months=tail_projection_months,
            )
        )
        logging.info(
            "Recalc step derive_tail_projection completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        self._reserving.set_tail(
            curve=tail_curve,
            extrap_periods=extrap_periods,
            projection_period=projection_period,
            attachment_age=tail_attachment_age,
            fit_period=fit_period,
        )
        logging.info(
            "Recalc step set_tail completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        if bf_apriori_by_uwy:
            self._reserving.set_bornhuetter_ferguson(apriori=bf_apriori_by_uwy)
        else:
            self._reserving.set_bornhuetter_ferguson(apriori=self._default_bf_apriori)
        logging.info(
            "Recalc step set_bornhuetter_ferguson completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        self._reserving.reserve(
            final_ultimate="chainladder",
            selected_ultimate_by_uwy=selected_ultimate_by_uwy,
        )
        logging.info(
            "Recalc step reserve completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        self._extract_data()
        logging.info(
            "Recalc step extract_data completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        logging.info(
            "Recalc apply_recalculation total completed in %.0f ms",
            (time.perf_counter() - total_started) * 1000,
        )

    def apply_preview_recalculation(
        self,
        average: str,
        drops: list[tuple[str, int]] | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
        tail_curve: str,
        fit_period: tuple[int, int | None] | None,
    ) -> None:
        total_started = time.perf_counter()
        step_started = time.perf_counter()
        self._reserving.set_development(
            average=average,
            drop=drops,
        )
        logging.info(
            "Preview step set_development completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        _months_per_dev, extrap_periods, projection_period = (
            self.derive_tail_projection_settings(
                reserving=self._reserving,
                tail_projection_months=tail_projection_months,
            )
        )
        logging.info(
            "Preview step derive_tail_projection completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        self._reserving.set_tail(
            curve=tail_curve,
            extrap_periods=extrap_periods,
            projection_period=projection_period,
            attachment_age=tail_attachment_age,
            fit_period=fit_period,
        )
        logging.info(
            "Preview step set_tail completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        self._reserving.compute_chainladder_preview()
        logging.info(
            "Preview step compute_chainladder completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        logging.info(
            "Preview total completed in %.0f ms",
            (time.perf_counter() - total_started) * 1000,
        )

    def apply_triangle_preview_recalculation(
        self,
        average: str,
        drops: list[tuple[str, int]] | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
        tail_curve: str,
        fit_period: tuple[int, int | None] | None,
    ) -> None:
        total_started = time.perf_counter()
        step_started = time.perf_counter()
        self._reserving.set_development(
            average=average,
            drop=drops,
        )
        logging.info(
            "Triangle preview step set_development completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        _months_per_dev, extrap_periods, projection_period = (
            self.derive_tail_projection_settings(
                reserving=self._reserving,
                tail_projection_months=tail_projection_months,
            )
        )
        logging.info(
            "Triangle preview step derive_tail_projection completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        self._reserving.set_tail(
            curve=tail_curve,
            extrap_periods=extrap_periods,
            projection_period=projection_period,
            attachment_age=tail_attachment_age,
            fit_period=fit_period,
        )
        logging.info(
            "Triangle preview step set_tail completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        step_started = time.perf_counter()
        self._reserving.compute_triangle_preview()
        logging.info(
            "Triangle preview step compute_triangle completed in %.0f ms",
            (time.perf_counter() - step_started) * 1000,
        )
        logging.info(
            "Triangle preview total completed in %.0f ms",
            (time.perf_counter() - total_started) * 1000,
        )

    @staticmethod
    def _apply_selected_ultimate_to_results(
        results_df: pd.DataFrame | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> pd.DataFrame | None:
        if results_df is None or len(results_df) == 0:
            return results_df

        selected_mapping: dict[str, str] = {}
        for key, value in (selected_ultimate_by_uwy or {}).items():
            method = str(value).strip().lower()
            if method in {"chainladder", "bornhuetter_ferguson"}:
                selected_mapping[str(key)] = method

        next_results = results_df.copy()
        selected_methods: list[str] = []
        selected_ultimate_values: list[float] = []
        for idx, row in next_results.iterrows():
            uwy_text = str(idx)
            uwy = uwy_text[:4] if len(uwy_text) >= 4 else uwy_text

            method = selected_mapping.get(uwy, "chainladder")
            if method == "bornhuetter_ferguson":
                selected_ultimate_values.append(float(row.get("bf_ultimate", 0.0)))
            else:
                method = "chainladder"
                selected_ultimate_values.append(float(row.get("cl_ultimate", 0.0)))
            selected_methods.append(method)

        next_results["ultimate"] = selected_ultimate_values
        next_results["selected_method"] = selected_methods
        return next_results

    def _build_selection_only_payload(
        self,
        *,
        base_payload: dict,
        selected_ultimate_by_uwy: dict[str, str] | None,
        results_cache_key: str,
    ) -> dict:
        payload = base_payload.copy()
        selection_started = time.perf_counter()
        selected_results_df = self._apply_selected_ultimate_to_results(
            self._get_results(),
            selected_ultimate_by_uwy,
        )
        select_elapsed_ms = (time.perf_counter() - selection_started) * 1000
        rows_started = time.perf_counter()
        rows = self._build_results_table_rows(selected_results_df)
        rows_elapsed_ms = (time.perf_counter() - rows_started) * 1000
        payload["selected_ultimate_by_uwy"] = selected_ultimate_by_uwy or {}
        payload["results_table_rows"] = rows
        payload["cache_key"] = results_cache_key
        payload["last_updated"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        logging.info(
            "Selection-only payload steps selected_results_ms=%.0f rows_ms=%.0f",
            select_elapsed_ms,
            rows_elapsed_ms,
        )
        return payload

    def build_results_payload(
        self,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> dict:
        payload_started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
            tail_projection_months,
            tail_curve,
            tail_fit_period_selection,
        )
        model_cache_key = self._build_model_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_projection_months,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
        )

        results_df = self._get_results()
        selected_results_df = self._apply_selected_ultimate_to_results(
            results_df,
            selected_ultimate_by_uwy,
        )

        logging.info(
            "Payload step apply_selected_results completed in %.0f ms",
            (time.perf_counter() - payload_started) * 1000,
        )
        rows_started = time.perf_counter()
        rows = self._build_results_table_rows(selected_results_df)
        logging.info(
            "Payload step build_results_rows completed in %.0f ms",
            (time.perf_counter() - rows_started) * 1000,
        )
        payload = {
            "drops_display": display,
            "average": average,
            "tail_curve": tail_curve,
            "drop_store": drop_store or [],
            "tail_attachment_age": tail_attachment_age,
            "tail_attachment_display": tail_display,
            "tail_projection_months": int(tail_projection_months),
            "tail_fit_period_selection": tail_fit_period_selection or [],
            "tail_fit_period_display": fit_period_display,
            "selected_ultimate_by_uwy": selected_ultimate_by_uwy or {},
            "results_table_rows": rows,
            "visual_cache_key": visual_cache_key,
            "model_cache_key": model_cache_key,
            "last_updated": timestamp,
        }
        logging.info(
            "Payload build_results_payload total completed in %.0f ms",
            (time.perf_counter() - payload_started) * 1000,
        )
        return payload

    def get_or_build_visual_payload(
        self,
        *,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
    ) -> tuple[dict, dict]:
        visual_started = time.perf_counter()
        visual_cache_key = self._build_visual_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_projection_months,
            tail_curve,
            tail_fit_period_selection,
        )
        triangle_started = time.perf_counter()
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
            copy_mode="none",
        )
        logging.info(
            "Payload step triangle_figure completed in %.0f ms",
            (time.perf_counter() - triangle_started) * 1000,
        )
        emergence_started = time.perf_counter()
        emergence_figure = self._cache_service.get_or_build(
            cache=self._emergence_cache,
            cache_key=visual_cache_key,
            label="Emergence figure",
            builder=lambda: self._build_emergence_figure(
                self._get_emergence(),
                "Emergence Pattern",
            ),
            copy_mode="none",
        )
        logging.info(
            "Payload step emergence_figure completed in %.0f ms",
            (time.perf_counter() - emergence_started) * 1000,
        )
        logging.info(
            "Visual payload total completed in %.0f ms",
            (time.perf_counter() - visual_started) * 1000,
        )
        return triangle_figure, emergence_figure

    def get_or_build_ave_payload(
        self,
        *,
        results_cache_key: str,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> dict:
        cached_payload = self._cache_service.get(
            self._ave_cache,
            results_cache_key,
            copy_mode="none",
            label="AvE payload",
        )
        if cached_payload is not None:
            return cached_payload

        ave_started = time.perf_counter()
        triangles_started = time.perf_counter()
        ave_triangles = self._get_ave(selected_ultimate_by_uwy or None)
        logging.info(
            "Payload step get_ave_triangles completed in %.0f ms",
            (time.perf_counter() - triangles_started) * 1000,
        )
        build_started = time.perf_counter()
        ave_payload = self._build_ave_payload(ave_triangles)
        logging.info(
            "Payload step build_ave_payload completed in %.0f ms",
            (time.perf_counter() - build_started) * 1000,
        )
        ave_payload["cache_key"] = results_cache_key
        self._cache_service.set(
            self._ave_cache,
            results_cache_key,
            ave_payload,
            copy_mode="none",
            label="AvE payload",
        )
        logging.info(
            "AvE payload total completed in %.0f ms",
            (time.perf_counter() - ave_started) * 1000,
        )
        return ave_payload

    @staticmethod
    def build_empty_ave_payload() -> dict:
        return {
            "available": False,
            "options": [],
            "default_valuation": None,
            "origin_views": {},
            "series": {"valuation": [], "expected": [], "actual": [], "diff": []},
            "cache_key": None,
        }

    def get_or_build_results_payload(
        self,
        *,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_projection_months: int,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
        force_recalc: bool = False,
    ) -> dict:
        model_cache_key = self._build_model_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_projection_months,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy,
        )
        results_cache_key = self._build_results_cache_key(
            drop_store,
            average,
            tail_attachment_age,
            tail_projection_months,
            tail_curve,
            tail_fit_period_selection,
            bf_apriori_by_uwy,
            selected_ultimate_by_uwy,
        )
        cached_payload = (
            None
            if force_recalc
            else self._cache_service.get(
                self._payload_cache,
                model_cache_key,
                copy_mode="shallow",
                label="Results metadata",
            )
        )
        if cached_payload is not None:
            logging.info("Using cached reserving model payload for current parameters")
            return self._build_selection_only_payload(
                base_payload=cached_payload,
                selected_ultimate_by_uwy=selected_ultimate_by_uwy,
                results_cache_key=results_cache_key,
            )

        fit_period = self._params_service.derive_tail_fit_period(
            tail_fit_period_selection
        )
        started = time.perf_counter()
        self.apply_recalculation(
            average or self._default_average,
            self._params_service.drops_to_tuples(drop_store),
            tail_attachment_age,
            tail_projection_months,
            tail_curve or self._default_tail_curve,
            fit_period,
            bf_apriori_by_uwy,
            None,
        )
        recalc_elapsed_ms = (time.perf_counter() - started) * 1000
        logging.info("Recalculation completed in %.0f ms", recalc_elapsed_ms)

        payload_started = time.perf_counter()
        payload = self.build_results_payload(
            drop_store=drop_store,
            average=average,
            tail_attachment_age=tail_attachment_age,
            tail_projection_months=tail_projection_months,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
            selected_ultimate_by_uwy=selected_ultimate_by_uwy,
        )
        payload_elapsed_ms = (time.perf_counter() - payload_started) * 1000
        logging.info("Payload build completed in %.0f ms", payload_elapsed_ms)
        payload["cache_key"] = results_cache_key
        payload["model_cache_key"] = model_cache_key
        self._cache_service.set(
            self._payload_cache,
            model_cache_key,
            payload,
            copy_mode="shallow",
            label="Results metadata",
        )
        return payload
