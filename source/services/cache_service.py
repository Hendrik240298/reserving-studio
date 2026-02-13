from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from typing import Callable


class CacheService:
    def __init__(self, max_entries: int = 32):
        self._max_entries = max_entries

    def get(self, cache: dict[str, dict], key: str) -> dict | None:
        cached_value = cache.get(key)
        if cached_value is None:
            return None
        cache.pop(key, None)
        cache[key] = cached_value
        return deepcopy(cached_value)

    def set(self, cache: dict[str, dict], key: str, value: dict) -> None:
        cache.pop(key, None)
        cache[key] = deepcopy(value)
        while len(cache) > self._max_entries:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)

    def get_or_build(
        self,
        *,
        cache: dict[str, dict],
        cache_key: str,
        label: str,
        builder: Callable[[], dict],
    ) -> dict:
        started = time.perf_counter()
        cached_figure = self.get(cache, cache_key)
        if cached_figure is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logging.info("%s reused in %.0f ms", label, elapsed_ms)
            return cached_figure

        figure_dict = builder()
        elapsed_ms = (time.perf_counter() - started) * 1000
        logging.info("%s built in %.0f ms", label, elapsed_ms)
        self.set(cache, cache_key, figure_dict)
        return figure_dict

    def build_results_cache_key(
        self,
        *,
        segment: str,
        default_average: str,
        default_tail_curve: str,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        selected_ultimate_by_uwy: dict[str, str] | None,
    ) -> str:
        normalized_drops: list[list[str | int]] = []
        for item in drop_store or []:
            if not isinstance(item, list) or len(item) != 2:
                continue
            try:
                normalized_drops.append([str(item[0]), int(item[1])])
            except (TypeError, ValueError):
                continue
        normalized_drops.sort(key=lambda item: (str(item[0]), int(item[1])))

        normalized_fit_period: list[int] = []
        for item in tail_fit_period_selection or []:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in normalized_fit_period:
                normalized_fit_period.append(value)
        normalized_fit_period.sort()

        normalized_bf: dict[str, float] = {}
        for key, value in sorted((bf_apriori_by_uwy or {}).items()):
            try:
                normalized_bf[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        normalized_selected_method: dict[str, str] = {}
        for key, value in sorted((selected_ultimate_by_uwy or {}).items()):
            method = str(value).strip().lower()
            if method not in {"chainladder", "bornhuetter_ferguson"}:
                continue
            normalized_selected_method[str(key)] = method

        payload = {
            "segment": segment,
            "average": average or default_average,
            "tail_curve": tail_curve or default_tail_curve,
            "tail_attachment_age": tail_attachment_age,
            "tail_fit_period_selection": normalized_fit_period,
            "drops": normalized_drops,
            "bf_apriori_by_uwy": normalized_bf,
            "selected_ultimate_by_uwy": normalized_selected_method,
        }
        return json.dumps(payload, sort_keys=True)

    def build_visual_cache_key(
        self,
        *,
        segment: str,
        default_average: str,
        default_tail_curve: str,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
    ) -> str:
        return self.build_results_cache_key(
            segment=segment,
            default_average=default_average,
            default_tail_curve=default_tail_curve,
            drop_store=drop_store,
            average=average,
            tail_attachment_age=tail_attachment_age,
            tail_curve=tail_curve,
            tail_fit_period_selection=tail_fit_period_selection,
            bf_apriori_by_uwy=None,
            selected_ultimate_by_uwy=None,
        )
