from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Literal

import pandas as pd

from source.claims_collection import ClaimsCollection
from source.config_manager import ConfigManager
from source.premium_repository import PremiumRepository
from source.reserving import Reserving
from source.triangle import Triangle


DEFAULT_METHOD: Literal["chainladder", "bornhuetter_ferguson"] = "chainladder"


@dataclass(frozen=True)
class NativeDropAnalysisOptions:
    average: str
    base_drops: list[tuple[str, int]]
    method: Literal["chainladder", "bornhuetter_ferguson"]
    use_tail: bool
    enforce_monotone_tail: bool
    tail_curve: str
    tail_attachment_age: int | None
    tail_projection_months: int
    tail_fit_period: list[int]
    bf_apriori_by_uwy: dict[str, float]


@dataclass(frozen=True)
class NativeDropSnapshot:
    reserving: Reserving
    results: pd.DataFrame
    link_ratios: pd.DataFrame
    parameters: dict[str, Any]
    total_ultimate: float
    total_incurred: float
    total_ibnr: float


def build_options_from_config(
    config: ConfigManager,
    *,
    method: Literal["chainladder", "bornhuetter_ferguson"] = DEFAULT_METHOD,
    use_tail: bool = False,
    enforce_monotone_tail: bool = False,
) -> NativeDropAnalysisOptions:
    session = config.load_session()
    return NativeDropAnalysisOptions(
        average=str(session.get("average", "volume")),
        base_drops=_normalize_drop_pairs(session.get("drops")),
        method=method,
        use_tail=use_tail,
        enforce_monotone_tail=enforce_monotone_tail,
        tail_curve=str(session.get("tail_curve", "weibull")),
        tail_attachment_age=_optional_int(session.get("tail_attachment_age")),
        tail_projection_months=max(int(session.get("tail_projection_months", 0) or 0), 0),
        tail_fit_period=_normalize_int_list(session.get("tail_fit_period")),
        bf_apriori_by_uwy=_normalize_float_map(session.get("bf_apriori_by_uwy")),
    )


def build_triangle(
    claims_df: pd.DataFrame,
    premium_df: pd.DataFrame,
    *,
    config: ConfigManager | None,
) -> Triangle:
    claims = ClaimsCollection(
        claims_df,
        values_are_cumulative=bool(claims_df.attrs.get("values_are_cumulative", False)),
    )
    premium = PremiumRepository.from_dataframe(
        config_manager=config,
        dataframe=premium_df,
    )
    return Triangle.from_claims(claims, premium)


def run_native_drop_analysis(
    triangle: Triangle,
    *,
    options: NativeDropAnalysisOptions,
    candidate_limit: int,
) -> dict[str, Any]:
    baseline = _snapshot(triangle, options=options, drops=options.base_drops)
    signals = _candidate_signals(
        baseline.link_ratios,
        baseline_drops=options.base_drops,
        candidate_limit=candidate_limit,
    )
    candidates = [
        _analyze_candidate(
            triangle,
            options=options,
            baseline=baseline,
            signal=signal,
        )
        for signal in signals
    ]
    ordered = sorted(candidates, key=_candidate_sort_key, reverse=True)
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index

    recommendation = _recommendation(ordered, options=options)
    return {
        "review_type": "native_drop_analysis",
        "baseline": {
            "parameters": baseline.parameters,
            "metrics": {
                "total_ultimate": round(baseline.total_ultimate, 6),
                "total_incurred": round(baseline.total_incurred, 6),
                "total_ibnr": round(baseline.total_ibnr, 6),
                "analysis_mode": {
                    "method": options.method,
                    "use_tail": options.use_tail,
                    "enforce_monotone_tail": options.enforce_monotone_tail,
                },
            },
        },
        "candidates": ordered,
        "recommendation": recommendation,
        "continuity_notes": [],
        "policy_trace": {
            "analysis_mode": "native_reserving_direct",
            "method": options.method,
            "use_tail": options.use_tail,
            "enforce_monotone_tail": options.enforce_monotone_tail,
        },
        "evidence_summary": {
            "baseline_candidate_signals": signals[:5],
        },
        "run_metadata": {
            "analysis_mode": {
                "method": options.method,
                "use_tail": options.use_tail,
                "enforce_monotone_tail": options.enforce_monotone_tail,
            },
            "baseline_total_ibnr": round(baseline.total_ibnr, 6),
            "baseline_total_ultimate": round(baseline.total_ultimate, 6),
        },
    }


def _snapshot(
    triangle: Triangle,
    *,
    options: NativeDropAnalysisOptions,
    drops: list[tuple[str, int]],
) -> NativeDropSnapshot:
    parameters = _analysis_parameters(options, drops=drops)
    reserving = _build_reserving(triangle, parameters=parameters)
    results = reserving.get_results().copy()
    total_ultimate = float(results["ultimate"].sum()) if len(results) else 0.0
    total_incurred = float(results["incurred"].sum()) if len(results) else 0.0
    total_ibnr = total_ultimate - total_incurred
    return NativeDropSnapshot(
        reserving=reserving,
        results=results,
        link_ratios=reserving.get_triangle_heatmap_data()["link_ratios"].copy(),
        parameters=parameters,
        total_ultimate=total_ultimate,
        total_incurred=total_incurred,
        total_ibnr=total_ibnr,
    )


def _candidate_signals(
    link_ratios: pd.DataFrame,
    *,
    baseline_drops: list[tuple[str, int]],
    candidate_limit: int,
) -> list[dict[str, Any]]:
    triangle_only = link_ratios.loc[
        ~link_ratios.index.astype(str).isin(["LDF", "Tail"])
    ].apply(pd.to_numeric, errors="coerce")
    selected_row = _row_named(link_ratios, "LDF")
    baseline_drop_set = set(baseline_drops)
    signals: list[dict[str, Any]] = []
    for column in triangle_only.columns:
        age = _column_age(column)
        selected_value = _to_optional_float(selected_row.get(column) if selected_row is not None else None)
        if age is None or selected_value is None or selected_value <= 0:
            continue
        series = triangle_only[column].dropna()
        if len(series) < 4:
            continue
        values = [float(value) for value in series.tolist()]
        center = float(median(values))
        abs_dev = [abs(value - center) for value in values]
        mad = float(median(abs_dev))
        if mad <= 0:
            continue
        scale = max(1.4826 * mad, 1e-9)
        for origin, raw_value in series.items():
            origin_label = _origin_label(origin)
            pair = (origin_label, age)
            if pair in baseline_drop_set:
                continue
            observed = float(raw_value)
            robust_z = abs((observed - center) / scale)
            relative_gap = abs(observed - selected_value) / abs(selected_value)
            if robust_z < 3.0 and relative_gap < 0.15:
                continue
            score = round(robust_z + relative_gap * 4.0, 6)
            signals.append(
                {
                    "drop_pair": pair,
                    "origin": origin_label,
                    "age": age,
                    "observed_a2a": round(observed, 6),
                    "selected_ldf": round(selected_value, 6),
                    "robust_z": round(robust_z, 6),
                    "relative_gap_to_selected": round(relative_gap, 6),
                    "signal_score": score,
                    "priority": "high" if robust_z >= 4.5 or relative_gap >= 0.25 else "medium",
                }
            )
    signals.sort(
        key=lambda item: (
            _priority_weight(item.get("priority")),
            float(item.get("signal_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return signals[:candidate_limit]


def _analyze_candidate(
    triangle: Triangle,
    *,
    options: NativeDropAnalysisOptions,
    baseline: NativeDropSnapshot,
    signal: dict[str, Any],
) -> dict[str, Any]:
    origin = str(signal["origin"])
    age = int(signal["age"])
    drops = sorted(set(options.base_drops) | {(origin, age)})
    candidate = _snapshot(triangle, options=options, drops=drops)
    raw_before = _factor_value(baseline.link_ratios, row_name="LDF", age=age)
    raw_after = _factor_value(candidate.link_ratios, row_name="LDF", age=age)
    effective_before = _effective_factor_value(baseline.link_ratios, age=age)
    effective_after = _effective_factor_value(candidate.link_ratios, age=age)
    ibnr_delta = candidate.total_ibnr - baseline.total_ibnr
    ultimate_delta = candidate.total_ultimate - baseline.total_ultimate
    per_origin_delta = _per_origin_ibnr_delta(baseline.results, candidate.results)
    recommendation_class = _classify_candidate(
        baseline_total_ibnr=baseline.total_ibnr,
        ibnr_delta=ibnr_delta,
        raw_selection_changed=_changed(raw_before, raw_after),
        effective_projection_changed=_changed(effective_before, effective_after),
    )
    summary = _candidate_summary(
        origin=origin,
        age=age,
        options=options,
        ibnr_delta=ibnr_delta,
        raw_selection_changed=_changed(raw_before, raw_after),
        effective_projection_changed=_changed(effective_before, effective_after),
    )
    return {
        "candidate_id": _drop_candidate_id(origin, age),
        "summary": summary,
        "parameters": candidate.parameters,
        "score": round(abs(ibnr_delta), 6),
        "recommendation_class": recommendation_class,
        "metrics": {
            "drop_pairs": [[origin, age]],
            "baseline_total_ibnr": round(baseline.total_ibnr, 6),
            "candidate_total_ibnr": round(candidate.total_ibnr, 6),
            "ibnr_delta": round(ibnr_delta, 6),
            "baseline_total_ultimate": round(baseline.total_ultimate, 6),
            "candidate_total_ultimate": round(candidate.total_ultimate, 6),
            "ultimate_delta": round(ultimate_delta, 6),
            "observed_a2a": signal.get("observed_a2a"),
            "raw_ldf_before": _round_or_none(raw_before),
            "raw_ldf_after": _round_or_none(raw_after),
            "effective_ldf_before": _round_or_none(effective_before),
            "effective_ldf_after": _round_or_none(effective_after),
            "raw_selection_changed": _changed(raw_before, raw_after),
            "effective_projection_changed": _changed(effective_before, effective_after),
            "robust_z": signal.get("robust_z"),
            "relative_gap_to_selected": signal.get("relative_gap_to_selected"),
            "signal_score": signal.get("signal_score"),
            "changed_uwys": _changed_uwys(per_origin_delta),
            "analysis_mode": {
                "method": options.method,
                "use_tail": options.use_tail,
                "enforce_monotone_tail": options.enforce_monotone_tail,
            },
        },
        "continuity_notes": [],
        "policy_trace": {
            "analysis_mode": "native_reserving_direct",
            "priority": signal.get("priority"),
        },
    }


def _recommendation(
    ordered: list[dict[str, Any]],
    *,
    options: NativeDropAnalysisOptions,
) -> dict[str, Any]:
    if not ordered:
        return {
            "recommendation_class": "no_candidates",
            "candidate_id": None,
            "summary": "No deterministic drop candidates were found directly from the triangle under the selected analysis mode.",
            "caveats": ["missing_candidates"],
            "alternatives": [],
        }
    top = ordered[0]
    caveats: list[str] = []
    metrics = top.get("metrics", {}) if isinstance(top.get("metrics"), dict) else {}
    if not options.use_tail:
        caveats.append("Analysis ran with no tail effect applied.")
    if not options.enforce_monotone_tail:
        caveats.append("Monotone tail correction was disabled for this analysis.")
    if metrics.get("raw_selection_changed") and not metrics.get("effective_projection_changed"):
        caveats.append(
            "Raw development selection changed, but the effective factor path used for ultimates did not."
        )
    return {
        "recommendation_class": top.get("recommendation_class"),
        "candidate_id": top.get("candidate_id"),
        "summary": top.get("summary"),
        "caveats": caveats,
        "alternatives": [
            str(item.get("candidate_id"))
            for item in ordered[1:3]
            if str(item.get("candidate_id") or "").strip()
        ],
    }


def _analysis_parameters(
    options: NativeDropAnalysisOptions,
    *,
    drops: list[tuple[str, int]],
) -> dict[str, Any]:
    tail = {
        "curve": options.tail_curve if options.use_tail else "weibull",
        "attachment_age": options.tail_attachment_age if options.use_tail else None,
        "projection_period": options.tail_projection_months if options.use_tail else 0,
        "fit_period": list(options.tail_fit_period) if options.use_tail else [],
    }
    return {
        "average": options.average,
        "drop": [list(item) for item in drops],
        "tail": tail,
        "bf_apriori": dict(options.bf_apriori_by_uwy),
        "final_ultimate": options.method,
        "analysis_mode": {
            "method": options.method,
            "use_tail": options.use_tail,
            "enforce_monotone_tail": options.enforce_monotone_tail,
        },
    }


def _build_reserving(triangle: Triangle, *, parameters: dict[str, Any]) -> Reserving:
    reserving = Reserving(triangle)
    drops = _normalize_drop_pairs(parameters.get("drop", []))
    tail = parameters.get("tail", {}) if isinstance(parameters.get("tail"), dict) else {}
    fit_period = _derive_tail_fit_period(_normalize_int_list(tail.get("fit_period", [])))
    months_per_dev = _infer_months_per_dev(triangle)
    extrap_periods, projection_period = _derive_tail_projection_settings(
        tail_projection_months=int(tail.get("projection_period", 0) or 0),
        months_per_dev=months_per_dev,
    )
    reserving.set_development(
        average=str(parameters.get("average", "volume")),
        drop=drops or None,
    )
    reserving.set_tail(
        curve=str(tail.get("curve", "weibull")),
        attachment_age=_optional_int(tail.get("attachment_age")),
        extrap_periods=extrap_periods,
        projection_period=projection_period,
        fit_period=fit_period,
    )
    bf_apriori = parameters.get("bf_apriori")
    reserving.set_bornhuetter_ferguson(apriori=bf_apriori or 0.6)
    reserving.reserve(
        final_ultimate=str(parameters.get("final_ultimate", DEFAULT_METHOD)),
        selected_ultimate_by_uwy={},
        enforce_monotone_tail=bool(
            parameters.get("analysis_mode", {}).get("enforce_monotone_tail", False)
        ),
    )
    return reserving


def _per_origin_ibnr_delta(baseline_results: pd.DataFrame, candidate_results: pd.DataFrame) -> pd.Series:
    baseline_ibnr = baseline_results["ultimate"] - baseline_results["incurred"]
    candidate_ibnr = candidate_results["ultimate"] - candidate_results["incurred"]
    base_aligned, cand_aligned = baseline_ibnr.align(candidate_ibnr, join="outer")
    return cand_aligned.fillna(0.0) - base_aligned.fillna(0.0)


def _changed_uwys(per_origin_delta: pd.Series, *, limit: int = 6) -> list[dict[str, Any]]:
    changed = per_origin_delta[per_origin_delta.abs() > 1e-9]
    ordered = changed.reindex(changed.abs().sort_values(ascending=False).index)
    return [
        {"uwy": _origin_label(origin), "ibnr_delta": round(float(value), 6)}
        for origin, value in ordered.iloc[:limit].items()
    ]


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, float]:
    metrics = candidate.get("metrics", {}) if isinstance(candidate.get("metrics"), dict) else {}
    return (
        1 if metrics.get("effective_projection_changed") else 0,
        abs(float(metrics.get("ibnr_delta", 0.0) or 0.0)),
        float(metrics.get("signal_score", 0.0) or 0.0),
    )


def _classify_candidate(
    *,
    baseline_total_ibnr: float,
    ibnr_delta: float,
    raw_selection_changed: bool,
    effective_projection_changed: bool,
) -> str:
    if abs(ibnr_delta) < 1e-9:
        if raw_selection_changed and not effective_projection_changed:
            return "diagnostic_only"
        if effective_projection_changed:
            return "shape_change"
        return "no_change"
    relative_change = abs(ibnr_delta) / max(abs(baseline_total_ibnr), 1.0)
    if relative_change >= 0.05 or abs(ibnr_delta) >= 100.0:
        return "material_change"
    return "reserve_change"


def _candidate_summary(
    *,
    origin: str,
    age: int,
    options: NativeDropAnalysisOptions,
    ibnr_delta: float,
    raw_selection_changed: bool,
    effective_projection_changed: bool,
) -> str:
    method_label = options.method.replace("_", " ")
    tail_label = "with tail enabled" if options.use_tail else "with no tail effect"
    if abs(ibnr_delta) < 1e-9:
        if raw_selection_changed and not effective_projection_changed:
            return (
                f"Add drop for AY {origin} age {age}. The raw development factor changes, but the effective projection path and total IBNR stay unchanged under {method_label} {tail_label}."
            )
        return (
            f"Add drop for AY {origin} age {age}. Total IBNR stays unchanged under {method_label} {tail_label}."
        )
    direction = "increase" if ibnr_delta > 0 else "decrease"
    return (
        f"Add drop for AY {origin} age {age}. Total IBNR would {direction} by {abs(ibnr_delta):,.2f} under {method_label} {tail_label}."
    )


def _drop_candidate_id(origin: str, age: int) -> str:
    return f"drop_ay{origin}_age{age}"


def _row_named(frame: pd.DataFrame, row_name: str) -> pd.Series | None:
    rows = frame.loc[frame.index.astype(str).isin([row_name])]
    if rows.empty:
        return None
    return rows.iloc[0].apply(pd.to_numeric, errors="coerce")


def _factor_value(frame: pd.DataFrame, *, row_name: str, age: int) -> float | None:
    row = _row_named(frame, row_name)
    if row is None:
        return None
    column = _column_for_age(frame.columns, age)
    if column is None:
        return None
    return _to_optional_float(row.get(column))


def _effective_factor_value(frame: pd.DataFrame, *, age: int) -> float | None:
    for row_name in ["Tail", "LDF"]:
        value = _factor_value(frame, row_name=row_name, age=age)
        if value is not None:
            return value
    return None


def _column_for_age(columns: Any, age: int) -> Any | None:
    for column in columns:
        if Reserving._parse_cdf_label_to_age(column) == age:
            return column
    return None


def _priority_weight(priority: object) -> int:
    normalized = str(priority or "").strip().lower()
    if normalized == "high":
        return 3
    if normalized == "medium":
        return 2
    if normalized == "low":
        return 1
    return 0


def _changed(left: float | None, right: float | None, *, tolerance: float = 1e-12) -> bool:
    if left is None and right is None:
        return False
    if left is None or right is None:
        return True
    return abs(left - right) > tolerance


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _normalize_drop_pairs(raw: object) -> list[tuple[str, int]]:
    if not isinstance(raw, list):
        return []
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in raw:
        pair = _normalize_drop_pair(item)
        if pair is None or pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _normalize_drop_pair(raw: object) -> tuple[str, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    origin = str(raw[0]).strip()
    try:
        age = int(raw[1])
    except (TypeError, ValueError):
        return None
    return (origin, age) if origin else None


def _normalize_int_list(raw: object) -> list[int]:
    values = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    out: list[int] = []
    for item in values:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value not in out:
            out.append(value)
    return out


def _normalize_float_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(numeric) or numeric < 0:
            continue
        out[str(key)] = numeric
    return out


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _infer_months_per_dev(triangle: Triangle) -> int:
    try:
        incurred = triangle.get_triangle()["incurred"]
        development = [int(value) for value in incurred.development.tolist()]
    except Exception:
        development = []
    if len(development) >= 2:
        deltas = [
            right - left
            for left, right in zip(development[:-1], development[1:])
            if right - left > 0
        ]
        if deltas:
            return min(deltas)
    if len(development) == 1 and development[0] > 0:
        return development[0]
    return 12


def _derive_tail_projection_settings(
    *,
    tail_projection_months: int,
    months_per_dev: int,
) -> tuple[int, int]:
    months = max(int(tail_projection_months), 0)
    safe_months_per_dev = max(int(months_per_dev), 1)
    extrap_periods = months // safe_months_per_dev
    projection_period = extrap_periods * safe_months_per_dev
    return extrap_periods, projection_period


def _derive_tail_fit_period(selection: list[int] | None) -> tuple[int, int | None] | None:
    if not selection:
        return None
    values = sorted(set(int(item) for item in selection))
    if len(values) == 1:
        return (values[0], None)
    return (values[0], values[-1])


def _origin_label(origin: object) -> str:
    if hasattr(origin, "year"):
        return str(origin.year)
    text = str(origin)
    return text[:4] if len(text) >= 4 and text[:4].isdigit() else text


def _column_age(column: object) -> int | None:
    return Reserving._parse_cdf_label_to_age(column)


def _to_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = pd.to_numeric([value], errors="coerce")[0]
    except Exception:
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)
