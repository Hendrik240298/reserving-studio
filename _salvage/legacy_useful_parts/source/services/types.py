from __future__ import annotations

from typing import TypedDict


class ParamsState(TypedDict):
    request_id: int
    source: str
    force_recalc: bool
    drop_store: list[list[str | int]]
    tail_attachment_age: int | None
    tail_projection_months: int
    tail_fit_period_selection: list[int]
    average: str
    tail_curve: str
    bf_apriori_by_uwy: dict[str, float]
    selected_ultimate_by_uwy: dict[str, str]
    sync_version: int | None


class ResultsPayload(TypedDict, total=False):
    triangle_figure: dict
    emergence_figure: dict
    drops_display: str
    average: str
    tail_curve: str
    drop_store: list[list[str | int]]
    tail_attachment_age: int | None
    tail_attachment_display: str
    tail_projection_months: int
    tail_fit_period_selection: list[int]
    tail_fit_period_display: str
    selected_ultimate_by_uwy: dict[str, str]
    results_table_rows: list[dict[str, str]]
    last_updated: str
    cache_key: str
    model_cache_key: str
    figure_version: int
    sync_version: int
