from __future__ import annotations

from typing import Callable

import pandas as pd

from source.services.types import ParamsState


class ParamsService:
    def __init__(
        self,
        *,
        default_average: str,
        default_tail_curve: str,
        default_bf_apriori: float,
        get_uwy_labels: Callable[[], list[str]],
        load_session: Callable[[], dict] | None,
        get_sync_version: Callable[[], int] | None,
    ) -> None:
        self._default_average = default_average
        self._default_tail_curve = default_tail_curve
        self._default_bf_apriori = default_bf_apriori
        self._get_uwy_labels = get_uwy_labels
        self._load_session = load_session
        self._get_sync_version = get_sync_version

    def normalize_drop_store(self, raw: object) -> list[list[str | int]]:
        normalized: list[list[str | int]] = []
        if not isinstance(raw, list):
            return normalized
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            origin, dev = item[0], item[1]
            try:
                normalized.append([str(origin), int(dev)])
            except (TypeError, ValueError):
                continue
        return normalized

    def parse_dev_label(self, dev_label: object) -> int | None:
        if dev_label is None:
            return None
        try:
            dev_str = str(dev_label)
            if "-" in dev_str:
                return int(dev_str.split("-")[0])
            return int(dev_str)
        except (TypeError, ValueError):
            return None

    def normalize_tail_fit_selection(self, raw: object) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, (int, float, str)):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        normalized: list[int] = []
        for item in raw:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in normalized:
                normalized.append(value)
        return normalized

    def toggle_tail_fit_selection(self, existing: list[int], dev: int) -> list[int]:
        normalized = []
        for item in existing or []:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value not in normalized:
                normalized.append(value)
        if dev in normalized:
            normalized = [value for value in normalized if value != dev]
        else:
            normalized.append(dev)
        return normalized

    def build_bf_apriori_rows(self, raw: object = None) -> list[dict[str, object]]:
        uwy_labels = self._get_uwy_labels()
        mapping: dict[str, float] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if pd.isna(numeric) or numeric < 0:
                    continue
                mapping[str(key)] = numeric

        if isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict):
                    continue
                uwy = str(row.get("uwy", "")).strip()
                if not uwy:
                    continue
                try:
                    numeric = float(row.get("apriori"))
                except (TypeError, ValueError):
                    continue
                if pd.isna(numeric) or numeric < 0:
                    continue
                mapping[uwy] = numeric

        rows: list[dict[str, object]] = []
        for uwy in uwy_labels:
            factor = mapping.get(uwy, self._default_bf_apriori)
            rows.append({"uwy": uwy, "apriori": float(factor)})
        return rows

    def bf_rows_to_mapping(
        self,
        rows: list[dict[str, object]] | None,
    ) -> dict[str, float]:
        normalized_rows = self.build_bf_apriori_rows(rows)
        mapping: dict[str, float] = {}
        for row in normalized_rows:
            uwy = str(row.get("uwy", "")).strip()
            if not uwy:
                continue
            try:
                value = float(row.get("apriori"))
            except (TypeError, ValueError):
                value = self._default_bf_apriori
            if pd.isna(value) or value < 0:
                value = self._default_bf_apriori
            mapping[uwy] = value
        return mapping

    def derive_tail_fit_period(
        self,
        selection: list[int] | None,
    ) -> tuple[int, int | None] | None:
        if not selection:
            return None
        normalized = self.normalize_tail_fit_selection(selection)
        if not normalized:
            return None
        sorted_values = sorted(set(normalized))
        if len(sorted_values) == 1:
            return (sorted_values[0], None)
        return (sorted_values[0], sorted_values[-1])

    def toggle_drop(
        self,
        existing: list[list[str | int]],
        origin: str,
        dev: int,
    ) -> list[list[str | int]]:
        normalized = []
        for item in existing or []:
            if not isinstance(item, list) or len(item) != 2:
                continue
            try:
                normalized.append((str(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue

        entry = (str(origin), int(dev))
        if entry in normalized:
            normalized = [item for item in normalized if item != entry]
        else:
            normalized.append(entry)
        return [[item[0], item[1]] for item in normalized]

    def drops_to_tuples(
        self,
        drops: list[list[str | int]] | None,
    ) -> list[tuple[str, int]] | None:
        if not drops:
            return None
        parsed: list[tuple[str, int]] = []
        for item in drops:
            if not isinstance(item, list) or len(item) != 2:
                continue
            origin, dev = item[0], item[1]
            try:
                parsed.append((str(origin), int(dev)))
            except (TypeError, ValueError):
                continue
        return parsed or None

    def build_params_state(
        self,
        *,
        drop_store: list[list[str | int]] | None,
        average: str | None,
        tail_attachment_age: int | None,
        tail_curve: str | None,
        tail_fit_period_selection: list[int] | None,
        bf_apriori_by_uwy: dict[str, float] | None,
        request_id: int,
        source: str,
        force_recalc: bool,
        sync_version: int | None = None,
    ) -> ParamsState:
        normalized_drop_store = self.normalize_drop_store(drop_store)
        normalized_tail_fit = self.normalize_tail_fit_selection(
            tail_fit_period_selection
        )

        parsed_tail = None
        if tail_attachment_age is not None:
            try:
                parsed_tail = int(tail_attachment_age)
            except (TypeError, ValueError):
                parsed_tail = None

        normalized_bf = {
            key: float(value)
            for key, value in self.bf_rows_to_mapping(
                self.build_bf_apriori_rows(bf_apriori_by_uwy)
            ).items()
        }

        return {
            "request_id": int(request_id),
            "source": source,
            "force_recalc": bool(force_recalc),
            "drop_store": normalized_drop_store,
            "tail_attachment_age": parsed_tail,
            "tail_fit_period_selection": normalized_tail_fit,
            "average": average or self._default_average,
            "tail_curve": tail_curve or self._default_tail_curve,
            "bf_apriori_by_uwy": normalized_bf,
            "sync_version": sync_version,
        }

    def load_session_params_state(
        self,
        *,
        request_id: int,
        force_recalc: bool,
        default_drop_store: list[list[str | int]],
        default_average: str,
        default_tail_attachment_age: int | None,
        default_tail_curve: str,
        default_tail_fit_period_selection: list[int],
        default_bf_apriori_rows: list[dict[str, object]],
    ) -> ParamsState:
        if self._load_session is None:
            return self.build_params_state(
                drop_store=default_drop_store,
                average=default_average,
                tail_attachment_age=default_tail_attachment_age,
                tail_curve=default_tail_curve,
                tail_fit_period_selection=default_tail_fit_period_selection,
                bf_apriori_by_uwy=self.bf_rows_to_mapping(default_bf_apriori_rows),
                request_id=request_id,
                source="load",
                force_recalc=force_recalc,
                sync_version=None,
            )

        session = self._load_session()
        sync_version = self._get_sync_version() if self._get_sync_version else None
        return self.build_params_state(
            drop_store=session.get("drops"),
            average=session.get("average", default_average),
            tail_attachment_age=session.get("tail_attachment_age"),
            tail_curve=session.get("tail_curve", default_tail_curve),
            tail_fit_period_selection=session.get("tail_fit_period"),
            bf_apriori_by_uwy=session.get("bf_apriori_by_uwy"),
            request_id=request_id,
            source="load",
            force_recalc=force_recalc,
            sync_version=sync_version,
        )
