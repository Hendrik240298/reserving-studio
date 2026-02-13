# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import logging
import os
from pathlib import Path

import chainladder as cl
import pandas as pd

from source.config_manager import ConfigManager
from source.dashboard import Dashboard
from source.reserving import Reserving
from source.triangle import Triangle


logging.basicConfig(level=logging.INFO)


def _coerce_uw_year(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp("Y")
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    values = series.astype(str)
    digits = values.str.fullmatch(r"\d{4}")
    if digits.any():
        return pd.to_datetime(values.where(digits), format="%Y", errors="ignore")
    return series.astype(str)


def _coerce_period(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp("Q")
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series
    values = series.astype(str)
    numeric = pd.to_numeric(values, errors="coerce")
    if not isinstance(numeric, pd.Series):
        numeric = pd.Series(numeric, index=values.index)
    if numeric.notna().any():
        return numeric
    if values.str.contains("Q").any():
        try:
            return pd.Series(
                pd.PeriodIndex(values, freq="Q").to_timestamp("Q"),
                index=values.index,
            )
        except Exception:
            pass
    converted = pd.to_datetime(values, errors="ignore")
    if isinstance(converted, pd.Series):
        return converted
    return pd.Series(converted, index=values.index)


def _normalize_periods(df: pd.DataFrame) -> pd.DataFrame:
    numeric_periods = df["period"]
    if not pd.api.types.is_numeric_dtype(numeric_periods):
        numeric_periods = pd.to_numeric(numeric_periods, errors="coerce")
        if not isinstance(numeric_periods, pd.Series):
            numeric_periods = pd.Series(numeric_periods, index=df.index)
        if numeric_periods.notna().sum() == 0:
            return df
        df["period"] = numeric_periods
    if not pd.api.types.is_datetime64_any_dtype(df["uw_year"]):
        df["uw_year"] = pd.to_datetime(df["uw_year"], errors="coerce")

    def _to_period_date(row):
        if pd.isna(row["uw_year"]) or pd.isna(row["period"]):
            return row["period"]
        try:
            steps = int(row["period"]) // 3
        except (TypeError, ValueError):
            return row["period"]
        return row["uw_year"] + pd.offsets.QuarterEnd(steps)

    df["period"] = df.apply(_to_period_date, axis=1)
    return df


def _triangle_to_dataframe(triangle: cl.Triangle) -> pd.DataFrame:
    if triangle.is_cumulative:
        incremental = triangle.cum_to_incr()
    else:
        incremental = triangle

    df = incremental.to_frame(keepdims=True)
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()

    origin_col = next((col for col in ["origin", "uw_year"] if col in df.columns), None)
    dev_col = next(
        (col for col in ["development", "period"] if col in df.columns), None
    )
    if origin_col is None or dev_col is None:
        level_cols = [col for col in df.columns if str(col).startswith("level_")]
        if len(level_cols) >= 2:
            origin_col, dev_col = level_cols[0], level_cols[1]
        elif len(df.columns) >= 3:
            origin_col, dev_col = df.columns[0], df.columns[1]

    if origin_col is None or dev_col is None:
        raise ValueError("Sample triangle is missing origin/development columns")

    value_cols = [col for col in df.columns if col not in {origin_col, dev_col}]
    if "incurred" in value_cols:
        value_col = "incurred"
    elif "paid" in value_cols:
        value_col = "paid"
    else:
        numeric_cols = df[value_cols].select_dtypes(include=["number"]).columns
        if len(numeric_cols) >= 1:
            value_col = numeric_cols[0]
        elif len(value_cols) == 1:
            value_col = value_cols[0]
        else:
            raise ValueError("Sample triangle value column could not be inferred")

    df = df.rename(
        columns={origin_col: "uw_year", dev_col: "period", value_col: "incurred"}
    )
    df["uw_year"] = _coerce_uw_year(df["uw_year"])
    df["period"] = _coerce_period(df["period"])
    df = _normalize_periods(df)
    df["paid"] = df["incurred"].astype(float)
    df["outstanding"] = 0.0

    return df


def _load_quarterly_csv() -> pd.DataFrame:
    if cl.__file__ is None:
        raise ValueError("chainladder package path not available")
    base_path = Path(cl.__file__).parent
    csv_path = base_path / "utils" / "data" / "quarterly.csv"
    df = pd.read_csv(csv_path)
    origin_col = next((col for col in ["origin", "uw_year"] if col in df.columns), None)
    dev_col = next(
        (col for col in ["development", "period"] if col in df.columns), None
    )
    if origin_col is None or dev_col is None:
        candidate_cols = [col for col in df.columns if col not in {"incurred", "paid"}]
        if len(candidate_cols) >= 2:
            origin_col, dev_col = candidate_cols[0], candidate_cols[1]
    value_cols = [col for col in df.columns if col not in {origin_col, dev_col}]
    if "incurred" in value_cols:
        value_col = "incurred"
    elif "paid" in value_cols:
        value_col = "paid"
    else:
        numeric_cols = df[value_cols].select_dtypes(include=["number"]).columns
        if len(numeric_cols) >= 1:
            value_col = numeric_cols[0]
        elif len(value_cols) == 1:
            value_col = value_cols[0]
        else:
            raise ValueError("Sample csv value column could not be inferred")
    df = df.rename(
        columns={origin_col: "uw_year", dev_col: "period", value_col: "incurred"}
    )
    df["uw_year"] = _coerce_uw_year(df["uw_year"])
    df["period"] = _coerce_period(df["period"])
    df = _normalize_periods(df)
    df["paid"] = df["incurred"].astype(float)
    df["outstanding"] = 0.0
    return df


def build_sample_triangle() -> Triangle:
    sample = cl.load_sample("quarterly")["incurred"]
    try:
        data = _triangle_to_dataframe(sample)
    except ValueError:
        data = _load_quarterly_csv()
    return Triangle(data)


def build_reserving(
    triangle: Triangle, config: ConfigManager | None = None
) -> Reserving:
    reserving = Reserving(triangle)
    average = "volume"
    drop = None
    tail_attachment_age = None
    tail_curve = "weibull"
    tail_fit_period_selection: list[int] = []
    bf_apriori_by_uwy: dict[str, float] | None = None
    selected_ultimate_by_uwy: dict[str, str] | None = None
    if config is not None:
        session = config.load_session()
        average = session.get("average", average)
        drop = _normalize_drops(session.get("drops"))
        tail_curve = session.get("tail_curve", tail_curve)
        tail_attachment_age = session.get("tail_attachment_age")
        tail_fit_period_selection = _normalize_tail_fit_period_selection(
            session.get("tail_fit_period")
        )
        bf_apriori_by_uwy = _normalize_bf_apriori_by_uwy(
            session.get("bf_apriori_by_uwy")
        )
        selected_ultimate_by_uwy = _normalize_selected_ultimate_by_uwy(
            session.get("selected_ultimate_by_uwy")
        )
        if tail_attachment_age is not None:
            try:
                tail_attachment_age = int(tail_attachment_age)
            except (TypeError, ValueError):
                tail_attachment_age = None
    fit_period = _derive_tail_fit_period(tail_fit_period_selection)

    reserving.set_development(average=average, drop=drop)
    reserving.set_tail(
        curve=tail_curve,
        projection_period=0,
        attachment_age=tail_attachment_age,
        fit_period=fit_period,
    )
    if bf_apriori_by_uwy:
        reserving.set_bornhuetter_ferguson(apriori=bf_apriori_by_uwy)
    else:
        reserving.set_bornhuetter_ferguson(apriori=0.6)
    reserving.reserve(
        final_ultimate="chainladder",
        selected_ultimate_by_uwy=selected_ultimate_by_uwy,
    )
    return reserving


def _normalize_drops(raw: object) -> list[tuple[str, int]] | None:
    if not raw:
        return None
    normalized: list[tuple[str, int]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                origin, dev = item[0], item[1]
                try:
                    normalized.append((str(origin), int(dev)))
                except (TypeError, ValueError):
                    continue
    return normalized or None


def _normalize_tail_fit_period_selection(raw: object) -> list[int]:
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


def _derive_tail_fit_period(
    selection: list[int] | None,
) -> tuple[int, int | None] | None:
    if not selection:
        return None
    normalized = _normalize_tail_fit_period_selection(selection)
    if not normalized:
        return None
    sorted_values = sorted(set(normalized))
    if len(sorted_values) == 1:
        return (sorted_values[0], None)
    return (sorted_values[0], sorted_values[-1])


def _normalize_bf_apriori_by_uwy(raw: object) -> dict[str, float] | None:
    if not raw:
        return None

    normalized: dict[str, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        return None

    for key, value in items:
        try:
            factor = float(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(factor) or factor < 0:
            continue
        normalized[str(key)] = factor

    return normalized or None


def _normalize_selected_ultimate_by_uwy(raw: object) -> dict[str, str] | None:
    if not raw or not isinstance(raw, dict):
        return None

    normalized: dict[str, str] = {}
    for key, value in raw.items():
        method = str(value).strip().lower()
        if method not in {"chainladder", "bornhuetter_ferguson"}:
            continue
        normalized[str(key)] = method
    return normalized or None


def load_config() -> ConfigManager | None:
    config_path = os.environ.get("RESERVING_CONFIG", "config.yml")
    path = Path(config_path)
    if not path.exists():
        return None
    return ConfigManager.from_yaml(path)


def main() -> None:
    config = load_config()
    triangle = build_sample_triangle()
    reserving = build_reserving(triangle, config=config)
    dashboard = Dashboard(reserving, config=config)
    dashboard.show()


if __name__ == "__main__":
    main()
