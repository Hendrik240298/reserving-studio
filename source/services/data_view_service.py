from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import pandas as pd

from source.reserving import Reserving


@dataclass(frozen=True)
class DataViewQuery:
    metric: str = "incurred"
    view: str = "cumulative"
    denominator: str | None = None
    denominator_view: str | None = None


VALID_DATA_METRICS = {"incurred", "paid", "outstanding", "premium"}
VALID_DATA_VIEWS = {"cumulative", "incremental"}
METRIC_ALIASES = {
    "claims": "incurred",
    "incurred claims": "incurred",
    "incurred_claims": "incurred",
    "claims incurred": "incurred",
    "paid claims": "paid",
    "paid_claims": "paid",
    "outstanding claims": "outstanding",
    "outstanding_claims": "outstanding",
    "earned premium": "premium",
    "written premium": "premium",
    "gross written premium": "premium",
    "gwp": "premium",
}


def normalize_data_view_query(query: DataViewQuery) -> DataViewQuery:
    metric = _normalize_metric_token(query.metric, default="incurred")
    view = str(query.view or "cumulative").strip().lower()
    denominator = _normalize_metric_name(query.denominator)
    denominator_view = _normalize_view_name(query.denominator_view)

    # Be tolerant when the caller swaps metric and view, which can happen
    # in model-generated tool arguments from prompts like "incurred incremental".
    if metric in VALID_DATA_VIEWS and view in VALID_DATA_METRICS:
        metric, view = view, metric
    elif metric in VALID_DATA_VIEWS and view not in VALID_DATA_VIEWS:
        view = metric
        metric = "incurred"

    if metric not in VALID_DATA_METRICS:
        raise ValueError(f"Unsupported metric '{query.metric}'")
    if view not in VALID_DATA_VIEWS:
        raise ValueError(f"Unsupported view '{query.view}'")

    return DataViewQuery(
        metric=metric,
        view=view,
        denominator=denominator,
        denominator_view=denominator_view,
    )


class DataViewService:
    def __init__(self, reserving: Reserving) -> None:
        self._reserving = reserving

    def get_data_view(self, query: DataViewQuery) -> pd.DataFrame:
        query = normalize_data_view_query(query)
        numerator = self._get_triangle(metric=query.metric, triangle_view=query.view)
        denominator = (query.denominator or "none").strip().lower()
        if denominator in {"", "none"}:
            return numerator

        denominator_view = query.denominator_view or query.view
        denominator_df = self._get_triangle(
            metric=denominator,
            triangle_view=denominator_view,
        )
        numerator_aligned, denominator_aligned = numerator.align(
            denominator_df,
            join="outer",
        )
        safe_denominator = denominator_aligned.where(denominator_aligned > 0)
        return numerator_aligned.div(safe_denominator)

    def summarize_view(self, query: DataViewQuery) -> dict[str, Any]:
        query = normalize_data_view_query(query)
        frame = self.get_data_view(query)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        latest_col = self._latest_column(numeric)
        latest_non_null_count = 0
        latest_total = None
        top_rows: list[dict[str, Any]] = []
        latest_diagonal_rows: list[dict[str, Any]] = []
        late_movers: list[dict[str, Any]] = []
        if latest_col is not None:
            latest_series = numeric[latest_col].dropna()
            latest_non_null_count = int(latest_series.shape[0])
            if len(latest_series) > 0:
                latest_total = float(latest_series.sum())
                top_rows = self._top_series_entries(latest_series, limit=5)

        for origin, row in numeric.iterrows():
            series = row.dropna()
            if len(series) == 0:
                continue
            tail_age = series.index[-1]
            tail_value = float(series.iloc[-1])
            latest_diagonal_rows.append(
                {
                    "origin": self._origin_label(origin),
                    "age": self._column_label(tail_age),
                    "value": round(tail_value, 6),
                }
            )
            if len(series) < 3:
                continue
            prior_scale = float(series.abs().iloc[:-1].max() or 0.0)
            if prior_scale <= 0:
                continue
            if abs(tail_value) >= prior_scale * 0.15:
                late_movers.append(
                    {
                        "origin": self._origin_label(origin),
                        "age": self._column_label(tail_age),
                        "value": round(tail_value, 6),
                        "relative_to_peak": round(abs(tail_value) / prior_scale, 6),
                    }
                )

        diagonal_total = (
            round(sum(float(item["value"]) for item in latest_diagonal_rows), 6)
            if latest_diagonal_rows
            else None
        )
        top_latest_diagonal_rows = sorted(
            latest_diagonal_rows,
            key=lambda item: abs(float(item.get("value", 0.0))),
            reverse=True,
        )[:5]

        summary = {
            "query": {
                "metric": query.metric,
                "view": query.view,
                "denominator": query.denominator,
                "denominator_view": query.denominator_view,
            },
            "shape": {
                "origin_count": int(numeric.shape[0]),
                "development_count": int(numeric.shape[1]),
            },
            "latest_age": self._column_label(latest_col)
            if latest_col is not None
            else None,
            "latest_non_null_count": latest_non_null_count,
            "latest_total": round(latest_total, 6)
            if latest_total is not None
            else None,
            "top_latest_rows": top_rows,
            "latest_diagonal_total": diagonal_total,
            "latest_diagonal_rows": latest_diagonal_rows,
            "top_latest_diagonal_rows": top_latest_diagonal_rows,
            "late_movement_candidates": late_movers[:5],
        }
        return summary

    def compare_views(
        self,
        left: DataViewQuery,
        right: DataViewQuery,
        *,
        comparison_mode: str = "difference",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        left = normalize_data_view_query(left)
        right = normalize_data_view_query(right)
        left_frame = self.get_data_view(left).apply(pd.to_numeric, errors="coerce")
        right_frame = self.get_data_view(right).apply(pd.to_numeric, errors="coerce")
        left_aligned, right_aligned = left_frame.align(right_frame, join="outer")

        normalized_mode = str(comparison_mode or "difference").strip().lower()
        if normalized_mode == "ratio":
            compare_frame = left_aligned.div(right_aligned.where(right_aligned != 0))
        else:
            normalized_mode = "difference"
            compare_frame = left_aligned.subtract(right_aligned)

        latest_col = self._latest_column(compare_frame)
        top_rows: list[dict[str, Any]] = []
        if latest_col is not None:
            latest_series = compare_frame[latest_col].dropna()
            if len(latest_series) > 0:
                top_rows = self._top_series_entries(latest_series, limit=5)

        summary = {
            "comparison_mode": normalized_mode,
            "left": self.summarize_view(left)["query"],
            "right": self.summarize_view(right)["query"],
            "latest_age": self._column_label(latest_col)
            if latest_col is not None
            else None,
            "top_latest_differences": top_rows,
        }
        return compare_frame, summary

    def _get_triangle(self, *, metric: str, triangle_view: str) -> pd.DataFrame:
        triangle_obj = self._reserving._triangle
        normalized_view = (triangle_view or "cumulative").strip().lower()

        incurred_triangle = triangle_obj.get_triangle("incurred")
        paid_triangle = triangle_obj.get_triangle("paid")
        if normalized_view == "incremental":
            incurred_triangle = incurred_triangle.cum_to_incr()
            paid_triangle = paid_triangle.cum_to_incr()

        incurred_df = incurred_triangle["incurred"].to_frame()

        metric_key = (metric or "incurred").strip().lower()
        if metric_key == "incurred":
            return incurred_df
        if metric_key == "paid":
            return paid_triangle["paid"].to_frame()
        if metric_key == "premium":
            return (
                incurred_triangle["Premium_selected"]
                .to_frame()
                .rename(columns={"Premium_selected": "premium"})
            )
        if metric_key == "outstanding":
            paid_df = paid_triangle["paid"].to_frame()
            incurred_aligned, paid_aligned = incurred_df.align(paid_df, join="outer")
            return incurred_aligned - paid_aligned
        raise ValueError(f"Unsupported metric '{metric}'")

    @staticmethod
    def _latest_column(frame: pd.DataFrame) -> object | None:
        columns = [col for col in frame.columns if frame[col].notna().any()]
        if not columns:
            return None
        return columns[-1]

    @staticmethod
    def _top_series_entries(series: pd.Series, *, limit: int) -> list[dict[str, Any]]:
        ordered = series.reindex(series.abs().sort_values(ascending=False).index)
        rows: list[dict[str, Any]] = []
        for origin, value in ordered.iloc[:limit].items():
            rows.append(
                {
                    "origin": DataViewService._origin_label(origin),
                    "value": round(float(value), 6),
                }
            )
        return rows

    @staticmethod
    def _origin_label(origin: object) -> str:
        if hasattr(origin, "year"):
            return str(origin.year)
        text = str(origin)
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else text

    @staticmethod
    def _column_label(value: object) -> str:
        if value is None:
            return ""
        return str(value)


def serialize_dataframe(frame: pd.DataFrame | None) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        return {"records": []}
    serializable = frame.copy()
    if isinstance(serializable.columns, pd.MultiIndex):
        serializable.columns = [
            "|".join(str(part) for part in col if part is not None)
            for col in serializable.columns
        ]
    serializable = serializable.reset_index()
    records = serializable.to_dict(orient="records")
    output: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        output.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return {"records": output}


def summarize_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    series = pd.Series(values, dtype="float64")
    return {
        "count": int(series.shape[0]),
        "median": round(float(median(values)), 6),
        "p75": round(float(series.quantile(0.75)), 6),
        "p90": round(float(series.quantile(0.90)), 6),
        "max": round(float(series.max()), 6),
    }


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return str(value)
    return value


def _normalize_metric_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = _normalize_metric_token(value, default=None)
    if not normalized or normalized == "none":
        return None
    if normalized not in VALID_DATA_METRICS:
        raise ValueError(f"Unsupported denominator '{value}'")
    return normalized


def _normalize_view_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in VALID_DATA_VIEWS:
        raise ValueError(f"Unsupported denominator view '{value}'")
    return normalized


def _normalize_metric_token(value: object, *, default: str | None) -> str | None:
    if value is None:
        return default
    normalized = str(value).strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    if not normalized:
        return default
    return METRIC_ALIASES.get(
        normalized,
        normalized.replace(" ", "_" if normalized in VALID_DATA_METRICS else " "),
    )
