from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Literal

import pandas as pd


SelectionMethod = Literal["chainladder", "bornhuetter_ferguson"]


def _normalize_selection_mapping(raw: object) -> dict[str, SelectionMethod]:
    mapping: dict[str, SelectionMethod] = {}
    if not isinstance(raw, dict):
        return mapping
    for key, value in raw.items():
        method = str(value).strip().lower()
        if method not in {"chainladder", "bornhuetter_ferguson"}:
            continue
        mapping[str(key)] = method  # type: ignore[assignment]
    return mapping


@dataclass(frozen=True)
class ParamsStoreSnapshot:
    request_id: int
    source: str
    force_recalc: bool
    drop_store: list[list[str | int]]
    tail_attachment_age: int | None
    tail_fit_period_selection: list[int]
    average: str
    tail_curve: str
    bf_apriori_by_uwy: dict[str, float]
    selected_ultimate_by_uwy: dict[str, SelectionMethod]
    sync_version: int | None

    @classmethod
    def from_store_dict(cls, payload: object) -> "ParamsStoreSnapshot":
        if not isinstance(payload, dict):
            raise ValueError("params-store payload must be a dict.")
        request_id = int(payload.get("request_id", 0))
        source = str(payload.get("source", "unknown"))
        force_recalc = bool(payload.get("force_recalc", False))

        drop_store: list[list[str | int]] = []
        raw_drop_store = payload.get("drop_store", [])
        if isinstance(raw_drop_store, list):
            for item in raw_drop_store:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        drop_store.append([str(item[0]), int(item[1])])
                    except (TypeError, ValueError):
                        continue

        tail_attachment_age = payload.get("tail_attachment_age")
        if tail_attachment_age is not None:
            try:
                tail_attachment_age = int(tail_attachment_age)
            except (TypeError, ValueError):
                tail_attachment_age = None

        tail_fit_period_selection: list[int] = []
        raw_tail_fit = payload.get("tail_fit_period_selection", [])
        if isinstance(raw_tail_fit, list):
            for item in raw_tail_fit:
                try:
                    tail_fit_period_selection.append(int(item))
                except (TypeError, ValueError):
                    continue

        average = str(payload.get("average", "volume"))
        tail_curve = str(payload.get("tail_curve", "weibull"))

        bf_apriori_by_uwy: dict[str, float] = {}
        raw_bf = payload.get("bf_apriori_by_uwy", {})
        if isinstance(raw_bf, dict):
            for key, value in raw_bf.items():
                try:
                    bf_apriori_by_uwy[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        raw_sync_version = payload.get("sync_version")
        sync_version: int | None
        if raw_sync_version is None:
            sync_version = None
        else:
            try:
                sync_version = int(raw_sync_version)
            except (TypeError, ValueError):
                sync_version = None

        return cls(
            request_id=request_id,
            source=source,
            force_recalc=force_recalc,
            drop_store=drop_store,
            tail_attachment_age=tail_attachment_age,
            tail_fit_period_selection=tail_fit_period_selection,
            average=average,
            tail_curve=tail_curve,
            bf_apriori_by_uwy=bf_apriori_by_uwy,
            selected_ultimate_by_uwy=_normalize_selection_mapping(
                payload.get("selected_ultimate_by_uwy")
            ),
            sync_version=sync_version,
        )


@dataclass(frozen=True)
class ResultsStoreSnapshot:
    triangle_figure: dict
    emergence_figure: dict
    drops_display: str
    average: str
    tail_curve: str
    drop_store: list[list[str | int]]
    tail_attachment_age: int | None
    tail_attachment_display: str
    tail_fit_period_selection: list[int]
    tail_fit_period_display: str
    selected_ultimate_by_uwy: dict[str, SelectionMethod]
    results_table_rows: list[dict[str, str]]
    last_updated: str
    cache_key: str | None = None
    model_cache_key: str | None = None
    figure_version: int | None = None
    sync_version: int | None = None

    @classmethod
    def from_store_dict(cls, payload: object) -> "ResultsStoreSnapshot":
        if not isinstance(payload, dict):
            raise ValueError("results-store payload must be a dict.")

        drop_store: list[list[str | int]] = []
        raw_drop_store = payload.get("drop_store", [])
        if isinstance(raw_drop_store, list):
            for item in raw_drop_store:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        drop_store.append([str(item[0]), int(item[1])])
                    except (TypeError, ValueError):
                        continue

        tail_attachment_age = payload.get("tail_attachment_age")
        if tail_attachment_age is not None:
            try:
                tail_attachment_age = int(tail_attachment_age)
            except (TypeError, ValueError):
                tail_attachment_age = None

        tail_fit_period_selection: list[int] = []
        raw_tail_fit = payload.get("tail_fit_period_selection", [])
        if isinstance(raw_tail_fit, list):
            for item in raw_tail_fit:
                try:
                    tail_fit_period_selection.append(int(item))
                except (TypeError, ValueError):
                    continue

        results_table_rows: list[dict[str, str]] = []
        raw_rows = payload.get("results_table_rows", [])
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, dict):
                    results_table_rows.append({str(k): str(v) for k, v in row.items()})

        cache_key_raw = payload.get("cache_key")
        model_cache_key_raw = payload.get("model_cache_key")

        figure_version_raw = payload.get("figure_version")
        figure_version: int | None
        if figure_version_raw is None:
            figure_version = None
        else:
            try:
                figure_version = int(figure_version_raw)
            except (TypeError, ValueError):
                figure_version = None

        sync_version_raw = payload.get("sync_version")
        sync_version: int | None
        if sync_version_raw is None:
            sync_version = None
        else:
            try:
                sync_version = int(sync_version_raw)
            except (TypeError, ValueError):
                sync_version = None

        return cls(
            triangle_figure=payload.get("triangle_figure")
            if isinstance(payload.get("triangle_figure"), dict)
            else {},
            emergence_figure=payload.get("emergence_figure")
            if isinstance(payload.get("emergence_figure"), dict)
            else {},
            drops_display=str(payload.get("drops_display", "")),
            average=str(payload.get("average", "volume")),
            tail_curve=str(payload.get("tail_curve", "weibull")),
            drop_store=drop_store,
            tail_attachment_age=tail_attachment_age,
            tail_attachment_display=str(payload.get("tail_attachment_display", "None")),
            tail_fit_period_selection=tail_fit_period_selection,
            tail_fit_period_display=str(
                payload.get("tail_fit_period_display", "lower=None, upper=None")
            ),
            selected_ultimate_by_uwy=_normalize_selection_mapping(
                payload.get("selected_ultimate_by_uwy")
            ),
            results_table_rows=results_table_rows,
            last_updated=str(payload.get("last_updated", "")),
            cache_key=str(cache_key_raw) if cache_key_raw is not None else None,
            model_cache_key=(
                str(model_cache_key_raw) if model_cache_key_raw is not None else None
            ),
            figure_version=figure_version,
            sync_version=sync_version,
        )


@dataclass(frozen=True)
class FinalizePayload:
    finalized_at_utc: datetime
    segment: str
    params_store: ParamsStoreSnapshot
    results_store: ResultsStoreSnapshot
    results_df: pd.DataFrame
    emergence_df: pd.DataFrame | None = None
    triangle_df: pd.DataFrame | None = None
    run_metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class InteractiveSessionController:
    done_event: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    latest_params_store: ParamsStoreSnapshot | None = None
    latest_results_store: ResultsStoreSnapshot | None = None
    finalized_payload: FinalizePayload | None = None
    finalized: bool = False
    canceled: bool = False
    error: str | None = None

    def publish_latest(
        self,
        *,
        params_store: ParamsStoreSnapshot | None,
        results_store: ResultsStoreSnapshot | None,
    ) -> None:
        with self.lock:
            self.latest_params_store = params_store
            self.latest_results_store = results_store

    def finalize(self, payload: FinalizePayload) -> None:
        with self.lock:
            self.finalized_payload = payload
            self.finalized = True
            self.canceled = False
            self.error = None
            self.done_event.set()

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.done_event.set()

    def cancel(self) -> None:
        with self.lock:
            self.canceled = True
            self.done_event.set()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
