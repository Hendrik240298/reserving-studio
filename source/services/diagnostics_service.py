from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, cast

import pandas as pd
from pandas.api.types import is_scalar


@dataclass(frozen=True)
class DiagnosticFinding:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any]
    suggested_actions: list[str]


@dataclass(frozen=True)
class DiagnosticRecommendation:
    code: str
    priority: str
    message: str
    rationale: str
    evidence: dict[str, Any]
    proposed_parameters: dict


@dataclass(frozen=True)
class DiagnosticsRunResult:
    findings: list[DiagnosticFinding]
    recommendations: list[DiagnosticRecommendation]
    metrics: dict[str, float | str | None]


@dataclass(frozen=True)
class ShiftSignal:
    age: int
    basis_label: str
    shift: float
    effect_size: float
    window: int


class DiagnosticsService:
    DIAGNOSTICS_VERSION = "v2.2"

    def __init__(
        self,
        *,
        cl_bf_diff_threshold: float = 0.1,
        immature_gap_threshold: float = 0.25,
        mature_threshold: float = 0.75,
        link_ratio_robust_z_threshold: float = 3.5,
        link_ratio_iqr_threshold: float = 0.35,
        elr_robust_z_threshold: float = 2.5,
        latest_diagonal_deviation_threshold: float = 0.15,
        incurred_on_premium_robust_z_threshold: float = 3.0,
        incurred_on_premium_break_threshold: float = 0.2,
        backtest_bias_threshold: float = 0.12,
        backtest_mae_threshold: float = 0.2,
        calendar_drift_slope_threshold: float = 0.02,
        tail_sensitivity_threshold: float = 0.03,
        paid_incurred_gap_threshold: float = 0.2,
        data_quality_critical_missing_threshold: float = 0.01,
        incurred_decrease_materiality_threshold: float = 0.15,
        incurred_decrease_alert_count: int = 10,
        paid_negative_increment_alert_count: int = 1,
        portfolio_shift_persistence_ages: int = 2,
        portfolio_shift_corroboration_effect_size: float = 1.25,
        negative_development_mad_multiplier: float = 3.0,
        negative_development_pct_latest: float = 0.01,
    ) -> None:
        self._cl_bf_diff_threshold = float(cl_bf_diff_threshold)
        self._immature_gap_threshold = float(immature_gap_threshold)
        self._mature_threshold = float(mature_threshold)
        self._link_ratio_robust_z_threshold = float(link_ratio_robust_z_threshold)
        self._link_ratio_iqr_threshold = float(link_ratio_iqr_threshold)
        self._elr_robust_z_threshold = float(elr_robust_z_threshold)
        self._latest_diagonal_deviation_threshold = float(
            latest_diagonal_deviation_threshold
        )
        self._incurred_on_premium_robust_z_threshold = float(
            incurred_on_premium_robust_z_threshold
        )
        self._incurred_on_premium_break_threshold = float(
            incurred_on_premium_break_threshold
        )
        self._backtest_bias_threshold = float(backtest_bias_threshold)
        self._backtest_mae_threshold = float(backtest_mae_threshold)
        self._calendar_drift_slope_threshold = float(calendar_drift_slope_threshold)
        self._tail_sensitivity_threshold = float(tail_sensitivity_threshold)
        self._paid_incurred_gap_threshold = float(paid_incurred_gap_threshold)
        self._data_quality_critical_missing_threshold = float(
            data_quality_critical_missing_threshold
        )
        self._incurred_decrease_materiality_threshold = float(
            incurred_decrease_materiality_threshold
        )
        self._incurred_decrease_alert_count = int(incurred_decrease_alert_count)
        self._paid_negative_increment_alert_count = int(
            paid_negative_increment_alert_count
        )
        self._portfolio_shift_persistence_ages = int(portfolio_shift_persistence_ages)
        self._portfolio_shift_corroboration_effect_size = float(
            portfolio_shift_corroboration_effect_size
        )
        self._negative_development_mad_multiplier = float(
            negative_development_mad_multiplier
        )
        self._negative_development_pct_latest = float(negative_development_pct_latest)

    def run(
        self,
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> DiagnosticsRunResult:
        findings: list[DiagnosticFinding] = []
        recommendations: list[DiagnosticRecommendation] = []

        maturity = self._build_maturity_map(results_df)
        findings.extend(self._maturity_weighted_cl_vs_bf(results_df, maturity))
        findings.extend(self._loss_ratio_consistency(results_df, maturity))
        findings.extend(self._latest_diagonal_actual_vs_expected(heatmap_data))
        findings.extend(self._incurred_on_premium_development(heatmap_data))
        findings.extend(self._rolling_origin_backtest(heatmap_data))
        findings.extend(self._calendar_year_drift(heatmap_data))
        findings.extend(
            self._tail_sensitivity_check(results_df, heatmap_data, maturity)
        )
        findings.extend(self._paid_incurred_coherence(heatmap_data, maturity))
        findings.extend(self._data_quality_gate(heatmap_data))
        findings.extend(self._negative_development_triage(heatmap_data))

        link_findings, drop_recommendations = self._link_ratio_diagnostics(heatmap_data)
        findings.extend(link_findings)
        recommendations.extend(drop_recommendations)

        tail_recommendation = self._tail_recommendation(heatmap_data)
        if tail_recommendation is not None:
            recommendations.append(tail_recommendation)

        bf_recommendation = self._bf_apriori_recommendation(results_df, maturity)
        if bf_recommendation is not None:
            recommendations.append(bf_recommendation)

        if not findings:
            findings.append(
                DiagnosticFinding(
                    code="NO_MATERIAL_FLAGS",
                    severity="low",
                    message="No material diagnostic flags triggered by the rule set.",
                    evidence={
                        "metric_id": "diagnostics_rule_set",
                        "value": 0.0,
                        "threshold": None,
                        "basis": "default_profile_v2",
                    },
                    suggested_actions=[
                        "Run scenario search to test alternative drops and tail configurations",
                    ],
                )
            )

        return DiagnosticsRunResult(
            findings=findings,
            recommendations=recommendations,
            metrics=self._summary_metrics(results_df, maturity, findings),
        )

    def _maturity_weighted_cl_vs_bf(
        self,
        results_df: pd.DataFrame | None,
        maturity: dict[str, float],
    ) -> list[DiagnosticFinding]:
        if results_df is None or len(results_df) == 0:
            return []

        findings: list[DiagnosticFinding] = []
        for idx, row in results_df.iterrows():
            uwy = self._uwy_label(idx)
            premium = float(row.get("Premium", 0.0) or 0.0)
            cl_ultimate = float(row.get("cl_ultimate", 0.0) or 0.0)
            bf_ultimate = float(row.get("bf_ultimate", 0.0) or 0.0)
            if premium <= 0:
                continue
            gap = abs(cl_ultimate - bf_ultimate) / premium
            maturity_value = maturity.get(uwy, 0.0)
            threshold = (
                self._cl_bf_diff_threshold
                if maturity_value >= self._mature_threshold
                else self._immature_gap_threshold
            )
            if gap <= threshold:
                continue
            severity = "high" if gap > threshold * 1.5 else "medium"
            findings.append(
                DiagnosticFinding(
                    code=f"CL_BF_DIVERGENCE_{uwy}",
                    severity=severity,
                    message=(
                        f"AY {uwy} CL vs BF divergence is {gap:.1%} of premium at "
                        f"maturity {maturity_value:.1%}, above threshold {threshold:.1%}."
                    ),
                    evidence={
                        "metric_id": f"cl_bf_gap_{uwy}",
                        "value": gap,
                        "threshold": threshold,
                        "basis": "abs(cl_ultimate-bf_ultimate)/premium with maturity-aware tolerance",
                    },
                    suggested_actions=[
                        "If AY is immature, stress BF apriori factors",
                        "If AY is mature, validate selected drops and calendar effects",
                    ],
                )
            )
        return findings

    def _loss_ratio_consistency(
        self,
        results_df: pd.DataFrame | None,
        maturity: dict[str, float],
    ) -> list[DiagnosticFinding]:
        if results_df is None or len(results_df) < 3:
            return []

        mature_lrs: list[float] = []
        ay_lrs: list[tuple[str, float]] = []
        for idx, row in results_df.iterrows():
            uwy = self._uwy_label(idx)
            premium = float(row.get("Premium", 0.0) or 0.0)
            cl_ultimate = float(row.get("cl_ultimate", 0.0) or 0.0)
            if premium <= 0 or cl_ultimate <= 0:
                continue
            lr = cl_ultimate / premium
            ay_lrs.append((uwy, lr))
            if maturity.get(uwy, 0.0) >= self._mature_threshold:
                mature_lrs.append(lr)

        if len(mature_lrs) < 3:
            return []

        center = float(median(mature_lrs))
        mad = self._median_abs_dev(mature_lrs)
        if mad <= 0:
            return []
        scale = 1.4826 * mad

        findings: list[DiagnosticFinding] = []
        for uwy, lr in ay_lrs:
            z_score = abs((lr - center) / scale)
            if z_score < self._elr_robust_z_threshold:
                continue
            findings.append(
                DiagnosticFinding(
                    code=f"LOSS_RATIO_OUTLIER_{uwy}",
                    severity=(
                        "high"
                        if z_score >= self._elr_robust_z_threshold * 1.4
                        else "medium"
                    ),
                    message=(
                        f"AY {uwy} implied CL loss ratio is atypical versus mature AY baseline "
                        f"(robust z {z_score:.2f})."
                    ),
                    evidence={
                        "metric_id": f"cl_lr_robust_z_{uwy}",
                        "value": z_score,
                        "threshold": self._elr_robust_z_threshold,
                        "basis": "robust z of cl_ultimate/premium vs mature AY median",
                    },
                    suggested_actions=[
                        "Check premium/exposure quality and earning pattern",
                        "Validate AY-specific BF apriori if AY is immature",
                    ],
                )
            )
        return findings

    def _link_ratio_diagnostics(
        self,
        heatmap_data: dict | None,
    ) -> tuple[list[DiagnosticFinding], list[DiagnosticRecommendation]]:
        if not isinstance(heatmap_data, dict):
            return ([], [])

        link_ratios = self._to_dataframe(heatmap_data.get("link_ratios"))
        if link_ratios is None or link_ratios.empty:
            return ([], [])

        triangle_only = link_ratios.loc[
            ~link_ratios.index.astype(str).isin(["LDF", "Tail"])
        ]
        if triangle_only.empty:
            return ([], [])

        numeric = triangle_only.apply(pd.to_numeric, errors="coerce")
        if numeric.empty:
            return ([], [])

        findings: list[DiagnosticFinding] = []
        recommendations: list[DiagnosticRecommendation] = []

        for col in numeric.columns:
            column = numeric[col].dropna()
            if len(column) < 4:
                continue

            values = [float(value) for value in column.tolist()]
            col_median = float(median(values))
            mad = self._median_abs_dev(values)
            if col_median == 0:
                continue

            iqr = float(column.quantile(0.75) - column.quantile(0.25))
            dispersion_ratio = abs(iqr / col_median)
            if dispersion_ratio > self._link_ratio_iqr_threshold:
                findings.append(
                    DiagnosticFinding(
                        code=f"LINK_RATIO_INSTABILITY_{col}",
                        severity="medium",
                        message=(
                            f"Development age {col} shows elevated cross-origin dispersion "
                            f"(IQR/median {dispersion_ratio:.2f})."
                        ),
                        evidence={
                            "metric_id": f"link_ratio_iqr_ratio_{col}",
                            "value": dispersion_ratio,
                            "threshold": self._link_ratio_iqr_threshold,
                            "basis": "IQR(link ratios)/median(link ratios) by development age",
                        },
                        suggested_actions=[
                            "Review large claims and valuation jumps around this development age",
                            "Test targeted drop candidates for this age",
                        ],
                    )
                )

            if mad <= 0:
                continue
            scale = 1.4826 * mad
            age = self._parse_int(col)
            if age is None:
                continue

            for origin, raw_value in column.items():
                value = float(raw_value)
                robust_z = abs((value - col_median) / scale)
                if robust_z <= self._link_ratio_robust_z_threshold:
                    continue
                origin_label = self._uwy_label(origin)
                findings.append(
                    DiagnosticFinding(
                        code=f"LINK_RATIO_OUTLIER_{origin_label}_{age}",
                        severity=(
                            "high"
                            if robust_z > self._link_ratio_robust_z_threshold * 1.4
                            else "medium"
                        ),
                        message=(
                            f"AY {origin_label} at age {age} is a link-ratio outlier "
                            f"(robust z {robust_z:.2f})."
                        ),
                        evidence={
                            "metric_id": f"link_ratio_robust_z_{origin_label}_{age}",
                            "value": robust_z,
                            "threshold": self._link_ratio_robust_z_threshold,
                            "basis": "abs(value-median)/(1.4826*MAD) by development age",
                        },
                        suggested_actions=[
                            "Run scenario excluding this AY-age factor",
                            "Validate data integrity for this AY and valuation",
                        ],
                    )
                )
                recommendations.append(
                    DiagnosticRecommendation(
                        code=f"RECOMMEND_DROP_{origin_label}_{age}",
                        priority=(
                            "high"
                            if robust_z > self._link_ratio_robust_z_threshold * 1.4
                            else "medium"
                        ),
                        message=f"Test dropping AY {origin_label}, age {age} from development selection.",
                        rationale="Robust link-ratio outlier by development age.",
                        evidence={
                            "metric_id": f"link_ratio_robust_z_{origin_label}_{age}",
                            "value": robust_z,
                            "threshold": self._link_ratio_robust_z_threshold,
                            "basis": "MAD-based outlier detection",
                        },
                        proposed_parameters={
                            "drop": [[origin_label, age]],
                            "average": "volume",
                        },
                    )
                )

        return findings, self._dedupe_drop_recommendations(recommendations)

    def _latest_diagonal_actual_vs_expected(
        self,
        heatmap_data: dict | None,
    ) -> list[DiagnosticFinding]:
        if not isinstance(heatmap_data, dict):
            return []

        incurred = self._to_dataframe(heatmap_data.get("incurred"))
        link_ratios = self._to_dataframe(heatmap_data.get("link_ratios"))
        if incurred is None or link_ratios is None:
            return []
        if incurred.empty or link_ratios.empty:
            return []

        ldf_row = link_ratios.loc[link_ratios.index.astype(str).isin(["LDF"])]
        if ldf_row.empty:
            return []
        ldf = ldf_row.iloc[0].apply(pd.to_numeric, errors="coerce")
        ldf_by_age = self._ldf_factor_by_age(ldf)
        numeric_incurred = incurred.apply(pd.to_numeric, errors="coerce")
        age_map: dict[int, object] = {}
        for col in numeric_incurred.columns:
            age = self._parse_int(col)
            if age is not None:
                age_map[age] = col
        if len(age_map) < 2:
            return []

        findings: list[DiagnosticFinding] = []
        for origin, row in numeric_incurred.iterrows():
            row_series = pd.Series(row)
            available_ages = [
                age
                for age, col in age_map.items()
                if self._scalar_or_none(row_series.get(col)) is not None
            ]
            if len(available_ages) < 2:
                continue
            available_ages.sort()
            prev_age = available_ages[-2]
            latest_age = available_ages[-1]
            prev_col = age_map[prev_age]
            latest_col = age_map[latest_age]
            prev_scalar = self._scalar_or_none(row_series.get(prev_col))
            latest_scalar = self._scalar_or_none(row_series.get(latest_col))
            prev_value = float(prev_scalar or 0.0)
            actual_latest = float(latest_scalar or 0.0)
            ldf_factor = float(ldf_by_age.get(prev_age, 1.0))
            if prev_value <= 0 or ldf_factor <= 0:
                continue

            expected_latest = prev_value * ldf_factor
            if expected_latest <= 0:
                continue
            deviation = (actual_latest - expected_latest) / expected_latest
            abs_dev = abs(deviation)
            if abs_dev < self._latest_diagonal_deviation_threshold:
                continue

            origin_label = self._uwy_label(origin)
            findings.append(
                DiagnosticFinding(
                    code=f"LATEST_DIAGONAL_DEVIATION_{origin_label}",
                    severity=(
                        "high"
                        if abs_dev > self._latest_diagonal_deviation_threshold * 1.6
                        else "medium"
                    ),
                    message=(
                        f"AY {origin_label} latest diagonal actual differs from one-step expected by "
                        f"{deviation:.1%} (age {prev_age}->{latest_age})."
                    ),
                    evidence={
                        "metric_id": f"latest_diag_dev_{origin_label}",
                        "value": abs_dev,
                        "threshold": self._latest_diagonal_deviation_threshold,
                        "basis": "abs((actual_latest-prev*ldf_prev)/(prev*ldf_prev))",
                    },
                    suggested_actions=[
                        "Investigate latest valuation movement and large claim activity",
                        "Stress-test CL/BF selections with iterative scenarios",
                    ],
                )
            )
        return findings

    def _incurred_on_premium_development(
        self,
        heatmap_data: dict | None,
    ) -> list[DiagnosticFinding]:
        if not isinstance(heatmap_data, dict):
            return []

        incurred = self._to_dataframe(heatmap_data.get("incurred"))
        premium = self._to_dataframe(heatmap_data.get("premium"))
        if incurred is None or premium is None or incurred.empty or premium.empty:
            return []

        numeric_incurred = incurred.apply(pd.to_numeric, errors="coerce")
        numeric_premium = premium.apply(pd.to_numeric, errors="coerce")
        premium_by_origin: dict[object, float] = {}
        for origin, row in numeric_premium.iterrows():
            row_series = pd.Series(row)
            values = [
                float(value)
                for value in row_series.dropna().tolist()
                if float(value) > 0
            ]
            if values:
                premium_by_origin[origin] = values[0]

        findings: list[DiagnosticFinding] = []
        age_to_ratios: dict[int, list[tuple[object, float]]] = {}
        for col in numeric_incurred.columns:
            age = self._parse_int(col)
            if age is None:
                continue
            ratios: list[tuple[object, float]] = []
            column_values = numeric_incurred[col]
            if isinstance(column_values, pd.DataFrame):
                column_series = pd.Series(column_values.iloc[:, 0])
            else:
                column_series = pd.Series(column_values)
            for origin, value in column_series.dropna().items():
                premium_value = premium_by_origin.get(origin)
                if premium_value is None or premium_value <= 0:
                    continue
                ratio = float(value) / premium_value
                ratios.append((origin, ratio))
            if ratios:
                age_to_ratios[age] = ratios

        if not isinstance(numeric_incurred, pd.DataFrame):
            return findings
        cumulative_df = cast(pd.DataFrame, numeric_incurred)
        incremental = self._incremental_from_cumulative(cumulative_df)
        age_to_ratios_incremental: dict[int, list[tuple[object, float]]] = {}
        for col in incremental.columns:
            age = self._parse_int(col)
            if age is None:
                continue
            column_series = pd.Series(incremental[col])
            ratios: list[tuple[object, float]] = []
            for origin, value in column_series.dropna().items():
                premium_value = premium_by_origin.get(origin)
                if premium_value is None or premium_value <= 0:
                    continue
                ratios.append((origin, float(value) / premium_value))
            if ratios:
                age_to_ratios_incremental[age] = ratios

        cumulative_findings, cumulative_shift_signals = (
            self._incurred_premium_basis_diagnostics(
                age_to_ratios=age_to_ratios,
                basis_label="cumulative",
                outlier_prefix="INCURRED_PREMIUM_OUTLIER",
                shift_prefix="INCURRED_PREMIUM_PORTFOLIO_SHIFT",
            )
        )
        findings.extend(cumulative_findings)

        incremental_findings, incremental_shift_signals = (
            self._incurred_premium_basis_diagnostics(
                age_to_ratios=age_to_ratios_incremental,
                basis_label="incremental",
                outlier_prefix="INCURRED_PREMIUM_OUTLIER_INC",
                shift_prefix="INCURRED_PREMIUM_PORTFOLIO_SHIFT_INC",
            )
        )
        findings.extend(incremental_findings)
        findings.extend(
            self._portfolio_shift_corroboration_findings(
                cumulative_shift_signals=cumulative_shift_signals,
                incremental_shift_signals=incremental_shift_signals,
                heatmap_data=heatmap_data,
            )
        )
        return findings

    def _incurred_premium_basis_diagnostics(
        self,
        *,
        age_to_ratios: dict[int, list[tuple[object, float]]],
        basis_label: str,
        outlier_prefix: str,
        shift_prefix: str,
    ) -> tuple[list[DiagnosticFinding], list[ShiftSignal]]:
        findings: list[DiagnosticFinding] = []
        shift_signals: list[ShiftSignal] = []
        for age, ratios in age_to_ratios.items():
            if len(ratios) < 4:
                continue
            ratio_values = [value for _, value in ratios]
            center = float(median(ratio_values))
            mad = self._median_abs_dev(ratio_values)
            if mad <= 0:
                continue
            scale = 1.4826 * mad
            for origin, ratio in ratios:
                robust_z = abs((ratio - center) / scale)
                if robust_z < self._incurred_on_premium_robust_z_threshold:
                    continue
                origin_label = self._uwy_label(origin)
                findings.append(
                    DiagnosticFinding(
                        code=f"{outlier_prefix}_{origin_label}_{age}",
                        severity=(
                            "high"
                            if robust_z
                            > self._incurred_on_premium_robust_z_threshold * 1.4
                            else "medium"
                        ),
                        message=(
                            f"AY {origin_label} {basis_label} incurred/premium at age {age} is atypical "
                            f"vs peer origins (robust z {robust_z:.2f})."
                        ),
                        evidence={
                            "metric_id": f"incurred_premium_{basis_label}_robust_z_{origin_label}_{age}",
                            "value": robust_z,
                            "threshold": self._incurred_on_premium_robust_z_threshold,
                            "basis": f"robust z of {basis_label} incurred/premium by development age",
                        },
                        suggested_actions=[
                            "Check for portfolio mix or pricing shift in this origin",
                            "Validate premium denominator alignment for this AY",
                        ],
                    )
                )

        sorted_ages = sorted(age_to_ratios.keys())
        for age in sorted_ages:
            ratios = age_to_ratios[age]
            if len(ratios) < 8:
                continue
            ordered = sorted(ratios, key=lambda item: self._origin_sort_key(item[0]))
            window = max(3, min(5, len(ordered) // 3))
            if window * 2 > len(ordered):
                window = len(ordered) // 2
            if window < 3:
                continue
            older = [value for _, value in ordered[:window]]
            recent = [value for _, value in ordered[-window:]]
            older_med = float(median(older))
            recent_med = float(median(recent))
            if older_med <= 0:
                continue
            shift = (recent_med - older_med) / older_med
            if abs(shift) < self._incurred_on_premium_break_threshold:
                continue

            all_values = [value for _, value in ordered]
            mad = self._median_abs_dev(all_values)
            scale = 1.4826 * mad if mad > 0 else 0.0
            if scale <= 0:
                continue
            effect_size = abs(recent_med - older_med) / scale
            if effect_size < 1.25:
                continue

            slope = self._linear_slope(
                [float(idx) for idx in range(len(ordered))],
                [float(value) for _, value in ordered],
            )
            if shift > 0 and slope <= 0:
                continue
            if shift < 0 and slope >= 0:
                continue
            shift_signals.append(
                ShiftSignal(
                    age=age,
                    basis_label=basis_label,
                    shift=shift,
                    effect_size=effect_size,
                    window=window,
                )
            )
        return findings, shift_signals

    def _portfolio_shift_corroboration_findings(
        self,
        *,
        cumulative_shift_signals: list[ShiftSignal],
        incremental_shift_signals: list[ShiftSignal],
        heatmap_data: dict | None,
    ) -> list[DiagnosticFinding]:
        if not cumulative_shift_signals and not incremental_shift_signals:
            return []

        by_age_cumulative = {item.age: item for item in cumulative_shift_signals}
        by_age_incremental = {item.age: item for item in incremental_shift_signals}
        all_ages = sorted(
            set(by_age_cumulative.keys()) | set(by_age_incremental.keys())
        )
        if not all_ages:
            return []

        calendar_drift_flag = bool(self._calendar_year_drift(heatmap_data))
        corroborated_ages: list[int] = []
        findings: list[DiagnosticFinding] = []

        for age in all_ages:
            cumulative = by_age_cumulative.get(age)
            incremental = by_age_incremental.get(age)
            same_direction = (
                cumulative is not None
                and incremental is not None
                and cumulative.shift * incremental.shift > 0
            )
            effect_size_ok = (
                cumulative is not None
                and incremental is not None
                and min(cumulative.effect_size, incremental.effect_size)
                >= self._portfolio_shift_corroboration_effect_size
            )
            corroborated = same_direction and effect_size_ok and not calendar_drift_flag
            if corroborated:
                corroborated_ages.append(age)
                continue

            signal = cumulative or incremental
            if signal is None:
                continue
            findings.append(
                DiagnosticFinding(
                    code=f"PORTFOLIO_SHIFT_SIGNAL_UNCONFIRMED_{age}",
                    severity="medium",
                    message=(
                        f"Portfolio shift signal at age {age} is not fully corroborated; "
                        "treat as possible signal rather than confirmed driver."
                    ),
                    evidence={
                        "metric_id": f"portfolio_shift_signal_{age}",
                        "value": abs(signal.shift),
                        "threshold": self._incurred_on_premium_break_threshold,
                        "basis": (
                            "shift signal requires corroboration across cumulative/incremental channels "
                            "and no strong calendar-drift alternative"
                        ),
                        "diagnostic_id": "PORTFOLIO_SHIFT_GUARDRAIL",
                        "direction": "bad",
                        "severity_band": "medium",
                        "alternative_hypotheses": [
                            "calendar year inflation/process drift",
                            "claims handling or case reserving change",
                        ],
                        "confidence": 0.45,
                        "required_review_level": "amber",
                        "applicability_conditions": [
                            "min_origins_per_age>=8",
                            "robust_effect_size>=threshold",
                        ],
                    },
                    suggested_actions=[
                        "Run calendar-year and influence diagnostics before causal attribution",
                        "Use hedged language: possible shift signal, not confirmed",
                        "Segment by product/region to isolate mix effects",
                    ],
                )
            )

        if corroborated_ages:
            persistence = self._max_consecutive_ages(corroborated_ages)
            for age in corroborated_ages:
                cumulative = by_age_cumulative.get(age)
                incremental = by_age_incremental.get(age)
                if cumulative is None or incremental is None:
                    continue
                severity = (
                    "high"
                    if persistence >= self._portfolio_shift_persistence_ages
                    and min(cumulative.effect_size, incremental.effect_size)
                    >= self._portfolio_shift_corroboration_effect_size
                    else "medium"
                )
                findings.append(
                    DiagnosticFinding(
                        code=f"PORTFOLIO_SHIFT_CORROBORATED_{age}",
                        severity=severity,
                        message=(
                            f"Portfolio shift signal is corroborated at age {age} across cumulative and incremental views "
                            f"(shift {cumulative.shift:.1%}/{incremental.shift:.1%})."
                        ),
                        evidence={
                            "metric_id": f"portfolio_shift_corroborated_{age}",
                            "value": abs(cumulative.shift),
                            "threshold": self._incurred_on_premium_break_threshold,
                            "basis": (
                                "directionally consistent cumulative+incremental shift with no strong calendar drift "
                                f"and persistence={persistence}"
                            ),
                            "diagnostic_id": "PORTFOLIO_SHIFT_GUARDRAIL",
                            "direction": "bad",
                            "severity_band": severity,
                            "alternative_hypotheses": [
                                "calendar year effects (not dominant)",
                                "case reserving process shifts",
                            ],
                            "confidence": 0.7 if severity == "high" else 0.6,
                            "required_review_level": "amber",
                            "applicability_conditions": [
                                "corroborated_channels=2",
                                "calendar_drift_flag=false",
                            ],
                        },
                        suggested_actions=[
                            "Validate frequency/severity decomposition where claim count data is available",
                            "Document alternative hypotheses alongside shift interpretation",
                        ],
                    )
                )

        return findings

    @staticmethod
    def _max_consecutive_ages(ages: list[int]) -> int:
        if not ages:
            return 0
        ordered = sorted(set(ages))
        best = 1
        run = 1
        for prev, current in zip(ordered[:-1], ordered[1:]):
            if current - prev <= 12:
                run += 1
                if run > best:
                    best = run
            else:
                run = 1
        return best

    def _rolling_origin_backtest(
        self,
        heatmap_data: dict | None,
    ) -> list[DiagnosticFinding]:
        residual_points = self._residual_points(heatmap_data)
        if len(residual_points) < 8:
            return []

        values = [item["residual"] for item in residual_points]
        bias = float(sum(values) / len(values))
        mae = float(sum(abs(value) for value in values) / len(values))

        findings: list[DiagnosticFinding] = []
        if abs(bias) > self._backtest_bias_threshold:
            findings.append(
                DiagnosticFinding(
                    code="ROLLING_BACKTEST_BIAS",
                    severity="high"
                    if abs(bias) > self._backtest_bias_threshold * 1.5
                    else "medium",
                    message=(
                        f"One-step emergence backtest indicates bias of {bias:.1%} "
                        f"across observed AY-age transitions."
                    ),
                    evidence={
                        "metric_id": "rolling_backtest_bias",
                        "value": abs(bias),
                        "threshold": self._backtest_bias_threshold,
                        "basis": "mean((actual-expected)/expected) over observed transitions",
                    },
                    suggested_actions=[
                        "Review development selections and calendar adjustments",
                        "Cross-check against BF for immature origins",
                    ],
                )
            )
        if mae > self._backtest_mae_threshold:
            findings.append(
                DiagnosticFinding(
                    code="ROLLING_BACKTEST_MAE",
                    severity="high"
                    if mae > self._backtest_mae_threshold * 1.5
                    else "medium",
                    message=(
                        f"One-step emergence backtest error is elevated (MAE {mae:.1%})."
                    ),
                    evidence={
                        "metric_id": "rolling_backtest_mae",
                        "value": mae,
                        "threshold": self._backtest_mae_threshold,
                        "basis": "mean(abs((actual-expected)/expected)) over observed transitions",
                    },
                    suggested_actions=[
                        "Test targeted drops and tail refits via scenario iteration",
                        "Investigate volatile AY-age cells with large residuals",
                    ],
                )
            )
        return findings

    def _calendar_year_drift(
        self, heatmap_data: dict | None
    ) -> list[DiagnosticFinding]:
        residual_points = self._residual_points(heatmap_data)
        if len(residual_points) < 8:
            return []

        grouped: dict[int, list[float]] = {}
        for point in residual_points:
            calendar_year = int(point["calendar_year"])
            residual_value = float(point["residual"])
            if calendar_year not in grouped:
                grouped[calendar_year] = []
            grouped[calendar_year].append(residual_value)
        if len(grouped) < 4:
            return []

        ordered = sorted(
            (year, sum(vals) / len(vals)) for year, vals in grouped.items()
        )
        x_values = [float(idx) for idx, _ in enumerate(ordered)]
        y_values = [float(value) for _, value in ordered]
        slope = self._linear_slope(x_values, y_values)
        if abs(slope) <= self._calendar_drift_slope_threshold:
            return []

        return [
            DiagnosticFinding(
                code="CALENDAR_YEAR_DRIFT",
                severity="high"
                if abs(slope) > self._calendar_drift_slope_threshold * 1.5
                else "medium",
                message=(
                    f"Residual emergence by calendar year shows drift (slope {slope:.3f} per period), "
                    "suggesting inflation/process trend effects."
                ),
                evidence={
                    "metric_id": "calendar_residual_slope",
                    "value": abs(slope),
                    "threshold": self._calendar_drift_slope_threshold,
                    "basis": "OLS slope of mean residuals across calendar years",
                },
                suggested_actions=[
                    "Consider explicit calendar-year adjustment or segmentation",
                    "Review claim handling and inflation changes in affected periods",
                ],
            )
        ]

    def _tail_sensitivity_check(
        self,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
        maturity: dict[str, float],
    ) -> list[DiagnosticFinding]:
        if (
            results_df is None
            or len(results_df) == 0
            or not isinstance(heatmap_data, dict)
        ):
            return []
        link_ratios = self._to_dataframe(heatmap_data.get("link_ratios"))
        if link_ratios is None or link_ratios.empty:
            return []

        tail_row = link_ratios.loc[link_ratios.index.astype(str).isin(["Tail"])]
        if tail_row.empty:
            return []
        tail_values = [
            float(value)
            for value in tail_row.iloc[0]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .tolist()
            if float(value) > 0
        ]
        if not tail_values:
            return []
        tail_factor = max(tail_values)
        if tail_factor <= 1.0:
            return []

        total_ultimate = 0.0
        stressed_delta = 0.0
        for idx, row in results_df.iterrows():
            uwy = self._uwy_label(idx)
            cl_ultimate = float(row.get("cl_ultimate", 0.0) or 0.0)
            incurred = float(row.get("incurred", 0.0) or 0.0)
            if cl_ultimate <= 0:
                continue
            total_ultimate += cl_ultimate
            unpaid = max(cl_ultimate - incurred, 0.0)
            immature_weight = max(1.0 - maturity.get(uwy, 0.0), 0.0)
            stressed_delta += unpaid * immature_weight * (tail_factor - 1.0)

        if total_ultimate <= 0:
            return []
        sensitivity = stressed_delta / total_ultimate
        if sensitivity <= self._tail_sensitivity_threshold:
            return []

        return [
            DiagnosticFinding(
                code="TAIL_SENSITIVITY_HIGH",
                severity="high"
                if sensitivity > self._tail_sensitivity_threshold * 1.6
                else "medium",
                message=(
                    f"Portfolio appears tail-sensitive (approximate stressed ultimate impact {sensitivity:.1%})."
                ),
                evidence={
                    "metric_id": "tail_sensitivity_ratio",
                    "value": sensitivity,
                    "threshold": self._tail_sensitivity_threshold,
                    "basis": "weighted unpaid sensitivity to current tail factor",
                },
                suggested_actions=[
                    "Run multi-curve tail scenarios and compare ranked diagnostics",
                    "Increase governance focus on tail start and fit-period assumptions",
                ],
            )
        ]

    def _paid_incurred_coherence(
        self,
        heatmap_data: dict | None,
        maturity: dict[str, float],
    ) -> list[DiagnosticFinding]:
        if not isinstance(heatmap_data, dict):
            return []
        incurred = self._to_dataframe(heatmap_data.get("incurred"))
        paid = self._to_dataframe(heatmap_data.get("paid"))
        if incurred is None or paid is None or incurred.empty or paid.empty:
            return []

        numeric_incurred = incurred.apply(pd.to_numeric, errors="coerce")
        numeric_paid = paid.apply(pd.to_numeric, errors="coerce")
        findings: list[DiagnosticFinding] = []
        ratios: list[tuple[str, float]] = []

        common_columns = [
            col for col in numeric_incurred.columns if col in numeric_paid.columns
        ]
        for origin in numeric_incurred.index:
            origin_label = self._uwy_label(origin)
            observed_cols = [
                col
                for col in common_columns
                if pd.notna(numeric_incurred.loc[origin, col])
                and pd.notna(numeric_paid.loc[origin, col])
            ]
            if not observed_cols:
                continue
            latest_col = max(observed_cols, key=lambda col: self._parse_int(col) or 0)
            incurred_value = float(numeric_incurred.loc[origin, latest_col] or 0.0)
            paid_value = float(numeric_paid.loc[origin, latest_col] or 0.0)
            if incurred_value <= 0:
                continue
            ratio = paid_value / incurred_value
            if maturity.get(origin_label, 0.0) >= 0.6:
                ratios.append((origin_label, ratio))

        if len(ratios) < 4:
            return []
        ratio_values = [ratio for _, ratio in ratios]
        center = float(median(ratio_values))
        mad = self._median_abs_dev(ratio_values)
        if mad <= 0:
            return []
        scale = 1.4826 * mad
        for origin_label, ratio in ratios:
            robust_z = abs((ratio - center) / scale)
            if robust_z < self._paid_incurred_gap_threshold * 10.0:
                continue
            findings.append(
                DiagnosticFinding(
                    code=f"PAID_INCURRED_COHERENCE_{origin_label}",
                    severity="medium"
                    if robust_z < self._paid_incurred_gap_threshold * 15.0
                    else "high",
                    message=(
                        f"AY {origin_label} paid/incurred ratio appears atypical at latest maturity "
                        f"(robust z {robust_z:.2f})."
                    ),
                    evidence={
                        "metric_id": f"paid_incurred_robust_z_{origin_label}",
                        "value": robust_z,
                        "threshold": self._paid_incurred_gap_threshold * 10.0,
                        "basis": "robust z of latest paid/incurred ratio for mature AYs",
                    },
                    suggested_actions=[
                        "Review case reserve strength and settlement speed assumptions",
                        "Compare paid and incurred views under Munich-style reconciliation",
                    ],
                )
            )
        return findings

    def _data_quality_gate(self, heatmap_data: dict | None) -> list[DiagnosticFinding]:
        if not isinstance(heatmap_data, dict):
            return []
        incurred = self._to_dataframe(heatmap_data.get("incurred"))
        premium = self._to_dataframe(heatmap_data.get("premium"))
        if incurred is None or premium is None or incurred.empty or premium.empty:
            return []

        numeric_incurred = incurred.apply(pd.to_numeric, errors="coerce")
        numeric_premium = premium.apply(pd.to_numeric, errors="coerce")
        paid = self._to_dataframe(heatmap_data.get("paid"))
        numeric_paid = (
            paid.apply(pd.to_numeric, errors="coerce")
            if isinstance(paid, pd.DataFrame) and not paid.empty
            else None
        )
        ordered_cols = sorted(
            numeric_incurred.columns,
            key=lambda c: self._parse_int(c) or 0,
        )
        expected_cells = 0
        missing_cells = 0
        for _, row in numeric_incurred.iterrows():
            row_series = pd.Series(row)
            observed_ages = [
                self._parse_int(col) or 0
                for col in ordered_cols
                if self._scalar_or_none(row_series.get(col)) is not None
            ]
            if not observed_ages:
                continue
            latest_age = max(observed_ages)
            for col in ordered_cols:
                col_age = self._parse_int(col)
                if col_age is None or col_age > latest_age:
                    continue
                expected_cells += 1
                if self._scalar_or_none(row_series.get(col)) is None:
                    missing_cells += 1
        total_cells = float(max(expected_cells, 1))
        missing_ratio = float(missing_cells) / total_cells

        negative_increment_count = 0
        material_incurred_decrease_count = 0
        for _, row in numeric_incurred.iterrows():
            row_series = pd.Series(row)
            previous = None
            for col in sorted(
                numeric_incurred.columns, key=lambda c: self._parse_int(c) or 0
            ):
                value = self._scalar_or_none(row_series.get(col))
                if value is None:
                    continue
                value_float = float(value)
                if previous is not None and value_float < previous:
                    negative_increment_count += 1
                    decrease_ratio = (previous - value_float) / max(previous, 1.0)
                    if decrease_ratio >= self._incurred_decrease_materiality_threshold:
                        material_incurred_decrease_count += 1
                previous = value_float

        paid_negative_increment_count = 0
        if isinstance(numeric_paid, pd.DataFrame):
            ordered_paid_cols = sorted(
                numeric_paid.columns,
                key=lambda c: self._parse_int(c) or 0,
            )
            for _, row in numeric_paid.iterrows():
                row_series = pd.Series(row)
                previous = None
                for col in ordered_paid_cols:
                    value = self._scalar_or_none(row_series.get(col))
                    if value is None:
                        continue
                    value_float = float(value)
                    if previous is not None and value_float < previous:
                        paid_negative_increment_count += 1
                    previous = value_float

        nonpositive_premium_count = 0
        for _, row in numeric_premium.iterrows():
            row_series = pd.Series(row)
            values = [
                value
                for value in [
                    self._scalar_or_none(item) for item in row_series.tolist()
                ]
                if value is not None and value > 0
            ]
            if not values:
                nonpositive_premium_count += 1
        data_quality_score = max(
            0.0,
            1.0
            - missing_ratio
            - min(material_incurred_decrease_count / 25.0, 0.35)
            - min(paid_negative_increment_count / 15.0, 0.35),
        )

        findings: list[DiagnosticFinding] = []
        if (
            missing_ratio > self._data_quality_critical_missing_threshold
            or material_incurred_decrease_count > self._incurred_decrease_alert_count
            or paid_negative_increment_count
            >= self._paid_negative_increment_alert_count
            or nonpositive_premium_count > 0
        ):
            severity = (
                "critical"
                if missing_ratio > self._data_quality_critical_missing_threshold * 3.0
                else "high"
            )
            findings.append(
                DiagnosticFinding(
                    code="DATA_QUALITY_GATE",
                    severity=severity,
                    message=(
                        "Data quality exceptions detected (missing values, non-monotonic cumulative cells, "
                        "or non-positive premium cells) that may reduce diagnostic reliability."
                    ),
                    evidence={
                        "metric_id": "data_quality_score",
                        "value": data_quality_score,
                        "threshold": 0.95,
                        "basis": (
                            f"missing_ratio={missing_ratio:.2%}, "
                            f"incurred_negative_increment_count={negative_increment_count}, "
                            f"material_incurred_decrease_count={material_incurred_decrease_count}, "
                            f"paid_negative_increment_count={paid_negative_increment_count}, "
                            f"nonpositive_premium_count={nonpositive_premium_count}"
                        ),
                    },
                    suggested_actions=[
                        "Resolve data quality issues before acting on high-impact model recommendations",
                        "Track diagnostic confidence as reduced while quality gate is failing",
                    ],
                )
            )
        return findings

    def _negative_development_triage(
        self,
        heatmap_data: dict | None,
    ) -> list[DiagnosticFinding]:
        if not isinstance(heatmap_data, dict):
            return []

        incurred = self._to_dataframe(heatmap_data.get("incurred"))
        paid = self._to_dataframe(heatmap_data.get("paid"))
        if incurred is None or incurred.empty:
            return []

        numeric_incurred = cast(
            pd.DataFrame,
            incurred.apply(pd.to_numeric, errors="coerce"),
        )
        incurred_incremental = self._incremental_from_cumulative(numeric_incurred)
        paid_incremental = None
        if isinstance(paid, pd.DataFrame) and not paid.empty:
            numeric_paid = cast(
                pd.DataFrame,
                paid.apply(pd.to_numeric, errors="coerce"),
            )
            paid_incremental = self._incremental_from_cumulative(numeric_paid)

        events: list[tuple[str, int, float, float, str]] = []
        diagonal_counts: dict[int, int] = {}
        for origin in incurred_incremental.index:
            origin_label = self._uwy_label(origin)
            origin_year = self._parse_int(origin_label)
            latest_cumulative = self._latest_non_null(numeric_incurred.loc[origin])
            latest_cumulative_value = float(latest_cumulative or 0.0)

            for col in incurred_incremental.columns:
                age = self._parse_int(col)
                if age is None:
                    continue
                incurred_delta = self._scalar_or_none(
                    incurred_incremental.loc[origin, col]
                )
                if incurred_delta is None or incurred_delta >= 0:
                    continue

                abs_delta = abs(float(incurred_delta))
                threshold = self._negative_threshold_for_age(
                    frame=incurred_incremental,
                    age_col=col,
                    latest_cumulative_value=latest_cumulative_value,
                )
                if abs_delta < threshold:
                    continue

                paid_delta = (
                    self._scalar_or_none(paid_incremental.loc[origin, col])
                    if isinstance(paid_incremental, pd.DataFrame)
                    and col in paid_incremental.columns
                    and origin in paid_incremental.index
                    else None
                )
                channel = self._classify_negative_channel(
                    incurred_delta=incurred_delta,
                    paid_delta=paid_delta,
                )
                events.append((origin_label, age, abs_delta, threshold, channel))

                if origin_year is not None:
                    diagonal_key = int(origin_year + (age / 12.0))
                    diagonal_counts[diagonal_key] = (
                        diagonal_counts.get(diagonal_key, 0) + 1
                    )

        if not events:
            return []

        max_diagonal_hits = max(diagonal_counts.values()) if diagonal_counts else 0
        diagonal_cluster = max_diagonal_hits >= 2
        severity = "high" if len(events) >= 4 or diagonal_cluster else "medium"
        top_event = max(events, key=lambda item: item[2])
        driver_label, driver_age, driver_abs_delta, _, driver_channel = top_event
        review_level = "red" if diagonal_cluster or len(events) >= 6 else "amber"
        confidence = 0.45 if diagonal_cluster else 0.55

        return [
            DiagnosticFinding(
                code="NEGATIVE_DEVELOPMENT_TRIAGE",
                severity=severity,
                message=(
                    f"Negative development triage detected {len(events)} material reversal events; "
                    f"largest at AY {driver_label} age {driver_age} ({driver_abs_delta:,.0f}) in {driver_channel}."
                ),
                evidence={
                    "metric_id": "negative_development_event_count",
                    "value": float(len(events)),
                    "threshold": 1.0,
                    "basis": (
                        "material negative incremental movement threshold=max("
                        f"{self._negative_development_mad_multiplier:.1f}*MAD, "
                        f"{self._negative_development_pct_latest:.1%}*latest cumulative)"
                    ),
                    "diagnostic_id": "NEGATIVE_DEVELOPMENT_TRIAGE",
                    "direction": "bad",
                    "severity_band": severity,
                    "alternative_hypotheses": [
                        "recoveries, salvage/subrogation, or commutation",
                        "calendar-year operational or inflation effects",
                        "case reserve strengthening or weakening",
                    ],
                    "confidence": confidence,
                    "required_review_level": review_level,
                    "applicability_conditions": [
                        "incremental view available",
                        "minimum non-null by age >= 4",
                    ],
                    "diagonal_cluster_count": float(max_diagonal_hits),
                    "driver_origin": driver_label,
                    "driver_age": float(driver_age),
                },
                suggested_actions=[
                    "Open negative-development triage drilldown and classify likely cause",
                    "Run scenarios with and without affected origins and compare stability",
                    "Suppress strong causal shift language until triage review is completed",
                ],
            )
        ]

    def _negative_threshold_for_age(
        self,
        *,
        frame: pd.DataFrame,
        age_col: object,
        latest_cumulative_value: float,
    ) -> float:
        column_values = [
            abs(float(value))
            for value in pd.Series(frame[age_col]).dropna().tolist()
            if float(value) != 0.0
        ]
        mad = self._median_abs_dev(column_values)
        robust_component = self._negative_development_mad_multiplier * 1.4826 * mad
        latest_component = self._negative_development_pct_latest * max(
            latest_cumulative_value,
            1.0,
        )
        return max(robust_component, latest_component)

    @staticmethod
    def _latest_non_null(row: pd.Series) -> float | None:
        values = [
            DiagnosticsService._scalar_or_none(item) for item in pd.Series(row).tolist()
        ]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return float(values[-1])

    @staticmethod
    def _classify_negative_channel(
        *,
        incurred_delta: float,
        paid_delta: float | None,
    ) -> str:
        if incurred_delta < 0 and paid_delta is not None and paid_delta < 0:
            return "paid+incurred"
        if incurred_delta < 0 and paid_delta is not None and paid_delta >= 0:
            return "incurred-only"
        return "incurred"

    def _bf_apriori_recommendation(
        self,
        results_df: pd.DataFrame | None,
        maturity: dict[str, float],
    ) -> DiagnosticRecommendation | None:
        if results_df is None or len(results_df) < 4:
            return None

        mature_cl_lrs: list[float] = []
        immature_rows: list[tuple[str, float, float]] = []
        for idx, row in results_df.iterrows():
            uwy = self._uwy_label(idx)
            premium = float(row.get("Premium", 0.0) or 0.0)
            cl_ultimate = float(row.get("cl_ultimate", 0.0) or 0.0)
            if premium <= 0 or cl_ultimate <= 0:
                continue
            cl_lr = cl_ultimate / premium
            maturity_value = maturity.get(uwy, 0.0)
            if maturity_value >= self._mature_threshold:
                mature_cl_lrs.append(cl_lr)
            elif maturity_value <= 0.5:
                immature_rows.append((uwy, maturity_value, cl_lr))

        if len(mature_cl_lrs) < 3 or not immature_rows:
            return None

        anchor = float(median(mature_cl_lrs))
        proposed: dict[str, float] = {}
        for uwy, maturity_value, cl_lr in immature_rows:
            blended = anchor * (1.0 - maturity_value) + cl_lr * maturity_value
            proposed[uwy] = round(max(blended, 0.0), 4)

        return DiagnosticRecommendation(
            code="RECOMMEND_BF_APRIORI",
            priority="high",
            message="Use maturity-weighted BF apriori factors for immature AYs.",
            rationale="Anchors immature AY expectations to mature AY implied loss ratios.",
            evidence={
                "metric_id": "bf_apriori_anchor_lr",
                "value": anchor,
                "threshold": self._mature_threshold,
                "basis": "median mature CL loss ratio blended by AY maturity",
            },
            proposed_parameters={"bf_apriori": proposed},
        )

    def _tail_recommendation(
        self,
        heatmap_data: dict | None,
    ) -> DiagnosticRecommendation | None:
        if not isinstance(heatmap_data, dict):
            return None

        link_ratios = self._to_dataframe(heatmap_data.get("link_ratios"))
        if link_ratios is None or link_ratios.empty:
            return None

        triangle_only = link_ratios.loc[
            ~link_ratios.index.astype(str).isin(["LDF", "Tail"])
        ]
        numeric = triangle_only.apply(pd.to_numeric, errors="coerce")
        if numeric.empty:
            return None

        age_scores: list[tuple[int, float]] = []
        for col in numeric.columns:
            series = numeric[col].dropna()
            if len(series) < 3:
                continue
            age = self._parse_int(col)
            if age is None:
                continue
            med = float(series.median())
            if med == 0:
                continue
            iqr_ratio = abs(float(series.quantile(0.75) - series.quantile(0.25)) / med)
            age_scores.append((age, iqr_ratio))

        if len(age_scores) < 3:
            return None

        age_scores.sort(key=lambda item: item[0])
        stable_tail_ages = [
            age for age, score in age_scores if score <= self._link_ratio_iqr_threshold
        ]
        if len(stable_tail_ages) >= 3:
            start_age = stable_tail_ages[-3]
        else:
            start_age = age_scores[-3][0]
        upper_age = age_scores[-1][0]

        candidate_fit_periods: list[list[int]] = [[start_age, upper_age]]
        if start_age > 1:
            candidate_fit_periods.append([start_age - 1, upper_age])
        if len(stable_tail_ages) >= 4:
            candidate_fit_periods.append([stable_tail_ages[-4], upper_age])

        scored_periods: list[tuple[list[int], float]] = []
        age_score_map = {age: score for age, score in age_scores}
        for period in candidate_fit_periods:
            period_ages = [
                age
                for age in sorted(age_score_map.keys())
                if period[0] <= age <= period[1]
            ]
            if not period_ages:
                continue
            avg_dispersion = sum(age_score_map[age] for age in period_ages) / len(
                period_ages
            )
            scored_periods.append((period, float(avg_dispersion)))

        if not scored_periods:
            return None
        scored_periods.sort(key=lambda item: item[1])
        recommended_fit_period = scored_periods[0][0]

        return DiagnosticRecommendation(
            code="RECOMMEND_TAIL_FIT",
            priority="medium",
            message="Test Weibull and inverse-power tail fits anchored on stable late development ages.",
            rationale="Late-age link ratios with lower dispersion improve tail fit stability.",
            evidence={
                "metric_id": "tail_fit_start_age",
                "value": float(recommended_fit_period[0]),
                "threshold": float(upper_age),
                "basis": "fit-period selected by minimum mean IQR/median dispersion across candidate intervals",
            },
            proposed_parameters={
                "tail": {
                    "curve_candidates": ["weibull", "inverse_power", "exponential"],
                    "fit_period_candidates": candidate_fit_periods,
                    "recommended_fit_period": recommended_fit_period,
                }
            },
        )

    @staticmethod
    def _incremental_from_cumulative(cumulative_df: pd.DataFrame) -> pd.DataFrame:
        ordered_cols = sorted(
            cumulative_df.columns,
            key=lambda col: DiagnosticsService._parse_int(col) or 0,
        )
        incremental = cumulative_df[ordered_cols].copy()
        if isinstance(incremental, pd.Series):
            incremental = incremental.to_frame()
        for idx, col in enumerate(ordered_cols):
            if idx == 0:
                continue
            prev_col = ordered_cols[idx - 1]
            incremental[col] = cumulative_df[col] - cumulative_df[prev_col]
        return cast(pd.DataFrame, incremental)

    def _summary_metrics(
        self,
        results_df: pd.DataFrame | None,
        maturity: dict[str, float],
        findings: list[DiagnosticFinding],
    ) -> dict[str, float | str | None]:
        if results_df is None or len(results_df) == 0:
            return {
                "average_maturity": None,
                "immature_ay_count": 0.0,
                "finding_count": float(len(findings)),
                "severity_score": float(self.compute_severity_score(findings)),
            }

        maturity_values = list(maturity.values())
        average_maturity = (
            float(sum(maturity_values) / len(maturity_values))
            if maturity_values
            else 0.0
        )
        immature_count = float(sum(1 for value in maturity_values if value < 0.5))
        return {
            "average_maturity": average_maturity,
            "immature_ay_count": immature_count,
            "finding_count": float(len(findings)),
            "severity_score": float(self.compute_severity_score(findings)),
            "assessment_confidence": self._assessment_confidence(findings),
        }

    @staticmethod
    def _assessment_confidence(findings: list[DiagnosticFinding]) -> float:
        penalty = 0.0
        for finding in findings:
            if finding.severity == "critical":
                penalty += 0.25
            elif finding.severity == "high":
                penalty += 0.07
            elif finding.severity == "medium":
                penalty += 0.03
        return max(0.1, round(1.0 - penalty, 3))

    @staticmethod
    def compute_severity_score(findings: list[DiagnosticFinding]) -> float:
        weights = {"low": 0.5, "medium": 2.0, "high": 5.0, "critical": 8.0}
        return float(sum(weights.get(item.severity, 1.0) for item in findings))

    @staticmethod
    def _build_maturity_map(results_df: pd.DataFrame | None) -> dict[str, float]:
        if results_df is None or len(results_df) == 0:
            return {}
        maturity: dict[str, float] = {}
        for idx, row in results_df.iterrows():
            uwy = DiagnosticsService._uwy_label(idx)
            incurred = float(row.get("incurred", 0.0) or 0.0)
            cl_ultimate = float(row.get("cl_ultimate", 0.0) or 0.0)
            if cl_ultimate <= 0:
                maturity[uwy] = 0.0
                continue
            maturity[uwy] = max(0.0, min(1.0, incurred / cl_ultimate))
        return maturity

    @staticmethod
    def _dedupe_drop_recommendations(
        recommendations: list[DiagnosticRecommendation],
    ) -> list[DiagnosticRecommendation]:
        by_code: dict[str, DiagnosticRecommendation] = {}
        for item in recommendations:
            existing = by_code.get(item.code)
            if existing is None:
                by_code[item.code] = item
                continue
            existing_value = float(existing.evidence.get("value", 0.0) or 0.0)
            next_value = float(item.evidence.get("value", 0.0) or 0.0)
            if next_value > existing_value:
                by_code[item.code] = item

        ordered = sorted(
            by_code.values(),
            key=lambda rec: float(rec.evidence.get("value", 0.0) or 0.0),
            reverse=True,
        )
        return ordered[:6]

    @staticmethod
    def _to_dataframe(raw: object) -> pd.DataFrame | None:
        if isinstance(raw, pd.DataFrame):
            return raw
        if not isinstance(raw, dict):
            return None
        records = raw.get("records")
        if not isinstance(records, list):
            return None
        frame = pd.DataFrame(records)
        if frame.empty:
            return None
        index_col = next(
            (
                col
                for col in ["origin", "index", "uw_year", "Unnamed: 0"]
                if col in frame.columns
            ),
            None,
        )
        if index_col is not None:
            frame = frame.set_index(index_col)
        return frame

    @staticmethod
    def _median_abs_dev(values: list[float]) -> float:
        if not values:
            return 0.0
        center = float(median(values))
        abs_dev = [abs(value - center) for value in values]
        return float(median(abs_dev))

    @staticmethod
    def _parse_int(value: object) -> int | None:
        try:
            text = str(value)
            if "-" in text:
                text = text.split("-")[0]
            return int(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _uwy_label(index_value: object) -> str:
        year_value = getattr(index_value, "year", None)
        if year_value is not None:
            return str(year_value)
        text = str(index_value)
        if len(text) >= 4 and text[:4].isdigit():
            return text[:4]
        return text

    def _residual_points(
        self, heatmap_data: dict | None
    ) -> list[dict[str, float | int]]:
        if not isinstance(heatmap_data, dict):
            return []
        incurred = self._to_dataframe(heatmap_data.get("incurred"))
        link_ratios = self._to_dataframe(heatmap_data.get("link_ratios"))
        if (
            incurred is None
            or link_ratios is None
            or incurred.empty
            or link_ratios.empty
        ):
            return []

        ldf_row = link_ratios.loc[link_ratios.index.astype(str).isin(["LDF"])]
        if ldf_row.empty:
            return []
        ldf = ldf_row.iloc[0].apply(pd.to_numeric, errors="coerce")
        ldf_by_age = self._ldf_factor_by_age(ldf)
        numeric_incurred = incurred.apply(pd.to_numeric, errors="coerce")
        ordered_cols = sorted(
            numeric_incurred.columns, key=lambda col: self._parse_int(col) or 0
        )

        points: list[dict[str, float | int]] = []
        for origin in numeric_incurred.index:
            origin_label = self._uwy_label(origin)
            origin_year = self._parse_int(origin_label)
            for prev_col, next_col in zip(ordered_cols[:-1], ordered_cols[1:]):
                prev_age = self._parse_int(prev_col)
                next_age = self._parse_int(next_col)
                if prev_age is None or next_age is None:
                    continue
                prev_val = numeric_incurred.loc[origin, prev_col]
                next_val = numeric_incurred.loc[origin, next_col]
                if pd.isna(prev_val) or pd.isna(next_val):
                    continue
                prev_float = float(prev_val)
                next_float = float(next_val)
                ldf_factor = float(ldf_by_age.get(prev_age, 1.0))
                expected = prev_float * ldf_factor
                if expected <= 0:
                    continue
                residual = (next_float - expected) / expected
                calendar_year = (
                    int(origin_year + (next_age / 12.0))
                    if origin_year is not None
                    else next_age
                )
                points.append(
                    {
                        "residual": residual,
                        "calendar_year": calendar_year,
                    }
                )
        return points

    @staticmethod
    def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        mean_x = sum(x_values) / len(x_values)
        mean_y = sum(y_values) / len(y_values)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        denominator = sum((x - mean_x) ** 2 for x in x_values)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _ldf_factor_by_age(self, ldf_series: pd.Series) -> dict[int, float]:
        mapping: dict[int, float] = {}
        for label, raw_value in ldf_series.items():
            age = self._parse_int(label)
            value = self._scalar_or_none(raw_value)
            if age is None or value is None or value <= 0:
                continue
            mapping[age] = float(value)
        return mapping

    @staticmethod
    def _origin_sort_key(origin: object) -> tuple[int, str]:
        label = DiagnosticsService._uwy_label(origin)
        try:
            return (0, str(int(label)))
        except (TypeError, ValueError):
            return (1, label)

    @staticmethod
    def _scalar_or_none(value: object) -> float | None:
        if value is None:
            return None
        if not is_scalar(value) and not isinstance(value, (pd.Series, pd.DataFrame)):
            return None
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return None
            return DiagnosticsService._scalar_or_none(value.iloc[0, 0])
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            return DiagnosticsService._scalar_or_none(value.iloc[0])
        try:
            parsed = float(cast(Any, value))
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return parsed
