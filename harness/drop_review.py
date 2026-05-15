from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Literal

import pandas as pd

from source.claims_collection import ClaimsCollection
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config
from source.premium_repository import PremiumRepository
from source.reserving import Reserving
from source.services.diagnostics_service import DiagnosticsService
from source.services.movement_diagnostics_service import MovementDiagnosticsService
from source.triangle import Triangle

from harness.markdown import render_drop_review_packet


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG_PATH = Path("examples/config_quarterly.yml")
DEFAULT_ARTIFACT_DIR = Path("harness/artifacts")
DEFAULT_METHOD: Literal["chainladder", "bornhuetter_ferguson"] = "chainladder"


@dataclass(frozen=True)
class DropReviewPacketResult:
    output_path: Path
    review_type: str
    recommendation_class: str | None
    candidate_id: str | None
    candidate_count: int
    warnings: list[str] = field(default_factory=list)


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
class NativeDropAnalysisSnapshot:
    reserving: Reserving
    results: pd.DataFrame
    heatmap: dict[str, Any]
    diagnostics: Any
    ldf_consistency: dict[str, Any]
    parameters: dict[str, Any]
    total_ultimate: float
    total_incurred: float
    total_ibnr: float


def run_drop_review_packet(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
    candidate_limit: int = 5,
    method: Literal["chainladder", "bornhuetter_ferguson"] = DEFAULT_METHOD,
    use_tail: bool = False,
    enforce_monotone_tail: bool = False,
) -> DropReviewPacketResult:
    """Run native drop analysis and write a markdown review packet."""

    resolved_config_path = _resolve_repo_path(config_path)
    if output_path is None:
        output_path = _default_output_path()
    resolved_output_path = _resolve_repo_path(output_path)

    requested_inputs = {
        "config_path": _display_path(resolved_config_path),
        "candidate_limit": candidate_limit,
        "method": method,
        "use_tail": use_tail,
        "enforce_monotone_tail": enforce_monotone_tail,
    }

    config = ConfigManager.from_yaml(resolved_config_path)
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)
    triangle = _build_triangle(claims_df, premium_df, config=config)
    options = _analysis_options_from_config(
        config,
        method=method,
        use_tail=use_tail,
        enforce_monotone_tail=enforce_monotone_tail,
    )
    review_payload = _run_native_drop_review(
        triangle,
        options=options,
        candidate_limit=candidate_limit,
    )
    summary_payload = _summarize_drop_review_payload(review_payload)

    effective_inputs = {
        "segment": config.get_segment(),
        "granularity": config.get_granularity(),
        "dataset": config.get_workflow_dataset(),
        "quarterly_premium_csv": config.get_workflow_quarterly_premium_csv(),
        "candidate_limit": candidate_limit,
        "method": options.method,
        "use_tail": options.use_tail,
        "enforce_monotone_tail": options.enforce_monotone_tail,
        "average": options.average,
        "base_drop_count": len(options.base_drops),
    }
    command = _command_for_packet(
        config_path=config_path,
        output_path=output_path,
        candidate_limit=candidate_limit,
        method=method,
        use_tail=use_tail,
        enforce_monotone_tail=enforce_monotone_tail,
    )

    packet = render_drop_review_packet(
        review_payload=review_payload,
        summary_payload=summary_payload,
        requested_inputs=requested_inputs,
        effective_inputs=effective_inputs,
        command=command,
        timestamp=datetime.now(timezone.utc).isoformat(),
        code_version=_git_revision(),
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(packet, encoding="utf-8")

    recommendation = summary_payload.get("recommendation") or {}
    candidates = summary_payload.get("top_candidates") or []
    return DropReviewPacketResult(
        output_path=resolved_output_path,
        review_type=str(summary_payload.get("review_type") or "native_drop_analysis"),
        recommendation_class=_optional_str(recommendation.get("recommendation_class")),
        candidate_id=_optional_str(recommendation.get("candidate_id")),
        candidate_count=int(summary_payload.get("candidate_count") or len(candidates)),
        warnings=_collect_warnings(review_payload),
    )


def _run_native_drop_review(
    triangle: Triangle,
    *,
    options: NativeDropAnalysisOptions,
    candidate_limit: int,
) -> dict[str, Any]:
    baseline = _analysis_snapshot(triangle, options=options, drops=options.base_drops)
    recommendation_map = _drop_recommendation_map(baseline.diagnostics)
    candidate_inputs = _build_native_drop_candidates(
        recommendation_map,
        baseline_drops=options.base_drops,
        candidate_limit=candidate_limit,
    )
    candidates = [
        _analyze_drop_candidate(
            triangle,
            options=options,
            baseline=baseline,
            candidate=item,
        )
        for item in candidate_inputs
    ]
    ordered = sorted(candidates, key=_candidate_sort_key, reverse=True)
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
    recommendation = _build_native_recommendation(ordered, options=options)
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
            "baseline_drop_recommendations": recommendation_map,
            "baseline_ldf_findings": baseline.ldf_consistency.get("findings", [])[:5],
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


def _analysis_snapshot(
    triangle: Triangle,
    *,
    options: NativeDropAnalysisOptions,
    drops: list[tuple[str, int]],
) -> NativeDropAnalysisSnapshot:
    parameters = _analysis_parameters(options, drops=drops)
    reserving = _build_reserving_for_analysis(triangle, parameters=parameters)
    results = reserving.get_results().copy()
    total_ultimate = float(results["ultimate"].sum()) if len(results) else 0.0
    total_incurred = float(results["incurred"].sum()) if len(results) else 0.0
    total_ibnr = total_ultimate - total_incurred
    heatmap = reserving.get_triangle_heatmap_data()
    diagnostics = DiagnosticsService().run(results_df=results, heatmap_data=heatmap)
    ldf_consistency = MovementDiagnosticsService(reserving).run_ldf_consistency()
    return NativeDropAnalysisSnapshot(
        reserving=reserving,
        results=results,
        heatmap=heatmap,
        diagnostics=diagnostics,
        ldf_consistency=ldf_consistency,
        parameters=parameters,
        total_ultimate=total_ultimate,
        total_incurred=total_incurred,
        total_ibnr=total_ibnr,
    )


def _build_native_drop_candidates(
    recommendation_map: dict[tuple[str, int], dict[str, Any]],
    *,
    baseline_drops: list[tuple[str, int]],
    candidate_limit: int,
) -> list[dict[str, Any]]:
    baseline_drop_set = set(baseline_drops)
    candidates: list[dict[str, Any]] = []
    for pair, payload in recommendation_map.items():
        if pair in baseline_drop_set:
            continue
        candidates.append(
            {
                "drop_pair": pair,
                "evidence_value": float(payload.get("evidence_value", 0.0) or 0.0),
                "message": str(payload.get("message") or "").strip(),
                "priority": str(payload.get("priority") or "").strip().lower() or None,
            }
        )
    candidates.sort(
        key=lambda item: (
            _priority_weight(item.get("priority")),
            float(item.get("evidence_value", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return candidates[:candidate_limit]


def _analyze_drop_candidate(
    triangle: Triangle,
    *,
    options: NativeDropAnalysisOptions,
    baseline: NativeDropAnalysisSnapshot,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    pair = candidate["drop_pair"]
    drops = sorted(set(options.base_drops) | {pair})
    snapshot = _analysis_snapshot(triangle, options=options, drops=drops)
    origin, age = pair
    raw_before = _row_age_value(baseline.heatmap.get("link_ratios"), row_name="LDF", age=age)
    raw_after = _row_age_value(snapshot.heatmap.get("link_ratios"), row_name="LDF", age=age)
    effective_before = _effective_factor_value(baseline.heatmap.get("link_ratios"), age=age)
    effective_after = _effective_factor_value(snapshot.heatmap.get("link_ratios"), age=age)
    observed_a2a = _observed_link_ratio_value(
        baseline.heatmap.get("link_ratios"),
        origin=origin,
        age=age,
    )
    total_ibnr_delta = snapshot.total_ibnr - baseline.total_ibnr
    total_ultimate_delta = snapshot.total_ultimate - baseline.total_ultimate
    per_origin_ibnr_delta = _per_origin_ibnr_delta(baseline.results, snapshot.results)
    changed_uwys = _changed_uwys(per_origin_ibnr_delta)
    raw_selection_changed = _changed(raw_before, raw_after)
    effective_projection_changed = _changed(effective_before, effective_after)
    recommendation_class = _classify_candidate(
        baseline_total_ibnr=baseline.total_ibnr,
        ibnr_delta=total_ibnr_delta,
        raw_selection_changed=raw_selection_changed,
        effective_projection_changed=effective_projection_changed,
    )
    summary = _candidate_summary(
        origin=origin,
        age=age,
        options=options,
        ibnr_delta=total_ibnr_delta,
        raw_selection_changed=raw_selection_changed,
        effective_projection_changed=effective_projection_changed,
    )
    return {
        "candidate_id": _drop_candidate_id(origin, age),
        "summary": summary,
        "parameters": snapshot.parameters,
        "score": round(abs(total_ibnr_delta), 6),
        "recommendation_class": recommendation_class,
        "metrics": {
            "drop_pairs": [[origin, age]],
            "baseline_total_ibnr": round(baseline.total_ibnr, 6),
            "candidate_total_ibnr": round(snapshot.total_ibnr, 6),
            "ibnr_delta": round(total_ibnr_delta, 6),
            "baseline_total_ultimate": round(baseline.total_ultimate, 6),
            "candidate_total_ultimate": round(snapshot.total_ultimate, 6),
            "ultimate_delta": round(total_ultimate_delta, 6),
            "observed_a2a": _round_or_none(observed_a2a),
            "raw_ldf_before": _round_or_none(raw_before),
            "raw_ldf_after": _round_or_none(raw_after),
            "effective_ldf_before": _round_or_none(effective_before),
            "effective_ldf_after": _round_or_none(effective_after),
            "raw_selection_changed": raw_selection_changed,
            "effective_projection_changed": effective_projection_changed,
            "outlier_evidence_value": round(
                float(candidate.get("evidence_value", 0.0) or 0.0),
                6,
            ),
            "changed_uwys": changed_uwys,
            "analysis_mode": {
                "method": options.method,
                "use_tail": options.use_tail,
                "enforce_monotone_tail": options.enforce_monotone_tail,
            },
            "ldf_finding_count": int(
                snapshot.ldf_consistency.get("summary", {}).get("finding_count", 0) or 0
            ),
        },
        "continuity_notes": [],
        "policy_trace": {
            "analysis_mode": "native_reserving_direct",
            "priority": candidate.get("priority"),
            "diagnostics_message": candidate.get("message"),
        },
    }


def _build_native_recommendation(
    ordered: list[dict[str, Any]],
    *,
    options: NativeDropAnalysisOptions,
) -> dict[str, Any]:
    if not ordered:
        return {
            "recommendation_class": "no_candidates",
            "candidate_id": None,
            "summary": "No deterministic drop candidates were available under the selected native analysis mode.",
            "caveats": ["missing_candidates"],
            "alternatives": [],
        }
    top = ordered[0]
    metrics = top.get("metrics", {}) if isinstance(top.get("metrics"), dict) else {}
    caveats: list[str] = []
    if top.get("recommendation_class") == "diagnostic_only":
        caveats.append(
            "Top candidate changes raw development selection but leaves effective projection and total IBNR unchanged under the selected analysis mode."
        )
    if not options.use_tail:
        caveats.append("Analysis ran with no tail effect applied.")
    if not options.enforce_monotone_tail:
        caveats.append("Monotone tail correction was disabled for this analysis.")
    if metrics.get("effective_projection_changed") is False and metrics.get(
        "raw_selection_changed"
    ):
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


def _analysis_options_from_config(
    config: ConfigManager,
    *,
    method: Literal["chainladder", "bornhuetter_ferguson"],
    use_tail: bool,
    enforce_monotone_tail: bool,
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


def _analysis_parameters(
    options: NativeDropAnalysisOptions,
    *,
    drops: list[tuple[str, int]],
) -> dict[str, Any]:
    if options.use_tail:
        tail = {
            "curve": options.tail_curve,
            "attachment_age": options.tail_attachment_age,
            "projection_period": options.tail_projection_months,
            "fit_period": list(options.tail_fit_period),
        }
    else:
        tail = {
            "curve": "weibull",
            "attachment_age": None,
            "projection_period": 0,
            "fit_period": [],
        }
    return {
        "average": options.average,
        "drop": [list(item) for item in drops],
        "drop_valuation": [],
        "tail": tail,
        "bf_apriori": dict(options.bf_apriori_by_uwy),
        "final_ultimate": options.method,
        "selected_ultimate_by_uwy": {},
        "analysis_mode": {
            "method": options.method,
            "use_tail": options.use_tail,
            "enforce_monotone_tail": options.enforce_monotone_tail,
        },
    }


def _build_reserving_for_analysis(
    triangle: Triangle,
    *,
    parameters: dict[str, Any],
) -> Reserving:
    reserving = Reserving(triangle)
    average = str(parameters.get("average", "volume"))
    drops = [
        tuple(item)
        for item in parameters.get("drop", [])
        if isinstance(item, list) and len(item) == 2
    ]
    drop_pairs = [(str(origin), int(age)) for origin, age in drops]
    tail = parameters.get("tail", {}) if isinstance(parameters.get("tail"), dict) else {}
    fit_period = _derive_tail_fit_period(
        _normalize_int_list(tail.get("fit_period", []))
    )
    months_per_dev = _infer_months_per_development_period(triangle)
    extrap_periods, projection_period = _derive_tail_projection_settings(
        tail_projection_months=int(tail.get("projection_period", 0) or 0),
        months_per_dev=months_per_dev,
    )
    reserving.set_development(average=average, drop=drop_pairs or None)
    reserving.set_tail(
        curve=str(tail.get("curve", "weibull")),
        attachment_age=_optional_int(tail.get("attachment_age")),
        extrap_periods=extrap_periods,
        projection_period=projection_period,
        fit_period=fit_period,
    )
    bf_apriori = parameters.get("bf_apriori")
    if isinstance(bf_apriori, dict) and bf_apriori:
        reserving.set_bornhuetter_ferguson(apriori=bf_apriori)
    else:
        reserving.set_bornhuetter_ferguson(apriori=0.6)
    reserving.reserve(
        final_ultimate=str(parameters.get("final_ultimate", DEFAULT_METHOD)),
        selected_ultimate_by_uwy={},
        enforce_monotone_tail=bool(
            parameters.get("analysis_mode", {}).get("enforce_monotone_tail", False)
        ),
    )
    return reserving


def _build_triangle(
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


def _drop_recommendation_map(diagnostics: Any) -> dict[tuple[str, int], dict[str, Any]]:
    recommendations = list(getattr(diagnostics, "recommendations", []) or [])
    mapped: dict[tuple[str, int], dict[str, Any]] = {}
    for item in recommendations:
        code = str(getattr(item, "code", "") or "")
        if not code.startswith("RECOMMEND_DROP_"):
            continue
        proposed = getattr(item, "proposed_parameters", {}) or {}
        raw_drop = proposed.get("drop", []) if isinstance(proposed, dict) else []
        if not isinstance(raw_drop, list) or not raw_drop:
            continue
        pair = _normalize_drop_pair(raw_drop[0])
        if pair is None:
            continue
        evidence = getattr(item, "evidence", {}) or {}
        priority = str(getattr(item, "priority", "") or "").strip().lower() or None
        evidence_value = _to_optional_float(
            evidence.get("value") if isinstance(evidence, dict) else None
        )
        mapped[pair] = {
            "message": str(getattr(item, "message", "") or "").strip(),
            "priority": priority,
            "evidence_value": evidence_value if evidence_value is not None else 0.0,
        }
    return dict(
        sorted(
            mapped.items(),
            key=lambda row: float(row[1].get("evidence_value", 0.0) or 0.0),
            reverse=True,
        )
    )


def _per_origin_ibnr_delta(
    baseline_results: pd.DataFrame,
    candidate_results: pd.DataFrame,
) -> pd.Series:
    baseline_ibnr = baseline_results["ultimate"] - baseline_results["incurred"]
    candidate_ibnr = candidate_results["ultimate"] - candidate_results["incurred"]
    baseline_aligned, candidate_aligned = baseline_ibnr.align(candidate_ibnr, join="outer")
    baseline_aligned = baseline_aligned.fillna(0.0)
    candidate_aligned = candidate_aligned.fillna(0.0)
    return candidate_aligned - baseline_aligned


def _changed_uwys(per_origin_delta: pd.Series, *, limit: int = 6) -> list[dict[str, Any]]:
    changed = per_origin_delta[per_origin_delta.abs() > 1e-9]
    if changed.empty:
        return []
    ordered = changed.reindex(changed.abs().sort_values(ascending=False).index)
    rows: list[dict[str, Any]] = []
    for origin, value in ordered.iloc[:limit].items():
        rows.append({"uwy": _origin_label(origin), "ibnr_delta": round(float(value), 6)})
    return rows


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, float]:
    metrics = candidate.get("metrics", {}) if isinstance(candidate.get("metrics"), dict) else {}
    return (
        1 if metrics.get("effective_projection_changed") else 0,
        abs(float(metrics.get("ibnr_delta", 0.0) or 0.0)),
        float(metrics.get("outlier_evidence_value", 0.0) or 0.0),
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
    if abs(ibnr_delta) < 1e-9:
        if raw_selection_changed and not effective_projection_changed:
            return (
                f"Add drop for AY {origin} age {age}. The raw development factor changes, but the effective projection path and total IBNR stay unchanged under {method_label}"
                + (" with tail enabled." if options.use_tail else " with no tail effect.")
            )
        return (
            f"Add drop for AY {origin} age {age}. Total IBNR stays unchanged under {method_label}"
            + (" with tail enabled." if options.use_tail else " with no tail effect.")
        )
    direction = "increase" if ibnr_delta > 0 else "decrease"
    return (
        f"Add drop for AY {origin} age {age}. Total IBNR would {direction} by {abs(ibnr_delta):,.2f} under {method_label}"
        + (" with tail enabled." if options.use_tail else " with no tail effect.")
    )


def _drop_candidate_id(origin: str, age: int) -> str:
    return f"drop_ay{origin}_age{age}"


def _effective_factor_value(link_ratios: Any, *, age: int) -> float | None:
    frame = link_ratios if isinstance(link_ratios, pd.DataFrame) else None
    if frame is None or frame.empty:
        return None
    for row_name in ["Tail", "LDF"]:
        value = _row_age_value(frame, row_name=row_name, age=age)
        if value is not None:
            return value
    return None


def _row_age_value(link_ratios: Any, *, row_name: str, age: int) -> float | None:
    frame = link_ratios if isinstance(link_ratios, pd.DataFrame) else None
    if frame is None or frame.empty:
        return None
    rows = frame.loc[frame.index.astype(str).isin([row_name])]
    if rows.empty:
        return None
    column = _column_for_age(frame.columns, age)
    if column is None:
        return None
    return _to_optional_float(rows.iloc[0].get(column))


def _observed_link_ratio_value(
    link_ratios: Any,
    *,
    origin: str,
    age: int,
) -> float | None:
    frame = link_ratios if isinstance(link_ratios, pd.DataFrame) else None
    if frame is None or frame.empty:
        return None
    triangle_only = frame.loc[~frame.index.astype(str).isin(["LDF", "Tail"])]
    rows = triangle_only.loc[triangle_only.index.astype(str).str.startswith(origin)]
    if rows.empty:
        return None
    column = _column_for_age(triangle_only.columns, age)
    if column is None:
        return None
    return _to_optional_float(rows.iloc[0].get(column))


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
    if value is None:
        return None
    return round(float(value), 6)


def _normalize_drop_pairs(raw: object) -> list[tuple[str, int]]:
    if not isinstance(raw, list):
        return []
    normalized: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for item in raw:
        pair = _normalize_drop_pair(item)
        if pair is None or pair in seen:
            continue
        seen.add(pair)
        normalized.append(pair)
    return normalized


def _normalize_drop_pair(raw: object) -> tuple[str, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    origin = str(raw[0]).strip()
    try:
        age = int(raw[1])
    except (TypeError, ValueError):
        return None
    if not origin:
        return None
    return (origin, age)


def _normalize_int_list(raw: object) -> list[int]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    normalized: list[int] = []
    for item in values:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_float_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(numeric) or numeric < 0:
            continue
        normalized[str(key)] = numeric
    return normalized


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_months_per_development_period(triangle: Triangle) -> int:
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


def _derive_tail_fit_period(
    selection: list[int] | None,
) -> tuple[int, int | None] | None:
    if not selection:
        return None
    sorted_values = sorted(set(int(value) for value in selection))
    if len(sorted_values) == 1:
        return (sorted_values[0], None)
    return (sorted_values[0], sorted_values[-1])


def _origin_label(origin: object) -> str:
    if hasattr(origin, "year"):
        return str(origin.year)
    text = str(origin)
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return text


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists() or candidate.parent.exists():
        return candidate
    return REPO_ROOT / path


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_ARTIFACT_DIR / f"drop_review_quarterly_{stamp}.md"


def _command_for_packet(
    *,
    config_path: Path,
    output_path: Path,
    candidate_limit: int,
    method: Literal["chainladder", "bornhuetter_ferguson"],
    use_tail: bool,
    enforce_monotone_tail: bool,
) -> str:
    parts = [
        "uv run python -m harness.cli drop-review",
        f"--config {config_path}",
        f"--candidate-limit {candidate_limit}",
        f"--method {method}",
        f"--output {output_path}",
    ]
    if use_tail:
        parts.append("--use-tail")
    if enforce_monotone_tail:
        parts.append("--enforce-monotone-tail")
    return " ".join(parts)


def _summarize_drop_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = _as_list(payload.get("candidates"))
    recommendation = _as_dict(payload.get("recommendation"))
    return {
        "review_type": payload.get("review_type"),
        "candidate_count": len(candidates),
        "top_candidates": [_compact_review_candidate(item) for item in candidates[:5]],
        "recommendation": _compact_review_recommendation(recommendation),
        "continuity_notes": _as_list(payload.get("continuity_notes"))[:5],
        "policy_trace": _as_dict(payload.get("policy_trace")),
        "evidence_summary": _as_dict(payload.get("evidence_summary")),
        "run_metadata": _as_dict(payload.get("run_metadata")),
    }


def _compact_review_candidate(item: Any) -> dict[str, Any]:
    candidate = _as_dict(item)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "recommendation_class": candidate.get("recommendation_class"),
        "summary": candidate.get("summary"),
        "score": candidate.get("score"),
        "metrics": candidate.get("metrics", {}),
    }


def _compact_review_recommendation(recommendation: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_class": recommendation.get("recommendation_class"),
        "candidate_id": recommendation.get("candidate_id"),
        "summary": recommendation.get("summary"),
        "caveats": _as_list(recommendation.get("caveats")),
        "alternatives": _as_list(recommendation.get("alternatives")),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    revision = result.stdout.strip()
    return revision or "unknown"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _collect_warnings(review_payload: dict) -> list[str]:
    warnings: list[str] = []
    recommendation = review_payload.get("recommendation")
    if isinstance(recommendation, dict):
        warnings.extend(str(item) for item in recommendation.get("caveats", []) if item)
    if not review_payload.get("candidates"):
        warnings.append(
            "Native drop analysis returned no deterministic drop candidates for the selected mode."
        )
    return warnings
