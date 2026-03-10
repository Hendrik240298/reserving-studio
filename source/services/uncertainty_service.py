from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class UncertaintyService:
    VERSION = "v1.1"

    def baseline_uncertainty(
        self,
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> dict[str, Any]:
        if results_df is None or results_df.empty:
            return {
                "version": self.VERSION,
                "mack_msep_by_uwy": {},
                "bf_prediction_error_by_uwy": {},
                "mack_total_msep": 0.0,
                "bf_total_prediction_error": 0.0,
                "total_process_sd": 0.0,
                "total_process_cv": None,
            }

        maturity = self._build_maturity_map(results_df)
        residuals = self._residual_series(heatmap_data)
        residual_sigma = self._residual_sigma(residuals)

        mack_msep: dict[str, float] = {}
        bf_error: dict[str, float] = {}
        total_ibnr = 0.0
        total_mack_msep = 0.0
        total_bf_error = 0.0

        for idx, row in results_df.iterrows():
            key = self._uwy_key(idx)
            incurred = float(row.get("incurred", 0.0) or 0.0)
            ultimate = float(row.get("ultimate", row.get("cl_ultimate", 0.0)) or 0.0)
            ibnr = max(0.0, ultimate - incurred)
            total_ibnr += ibnr

            mat = float(maturity.get(key, 0.5))
            process_loading = np.sqrt(max(0.15, 1.0 - mat))
            parameter_loading = max(0.12, 0.65 - 0.5 * mat)
            process_sd = ibnr * residual_sigma * process_loading
            parameter_sd = ibnr * residual_sigma * parameter_loading
            msep_value = float(process_sd**2 + parameter_sd**2)

            credibility = min(max(mat, 0.05), 0.98)
            bf_scale = max(0.1, 1.0 - 0.6 * credibility)
            bf_prediction_error = float(ibnr * residual_sigma * bf_scale)

            mack_msep[key] = round(msep_value, 4)
            bf_error[key] = round(bf_prediction_error, 4)
            total_mack_msep += msep_value
            total_bf_error += bf_prediction_error

        process_sd_total = float(np.sqrt(max(0.0, total_mack_msep)))
        process_cv = process_sd_total / total_ibnr if total_ibnr > 0 else None
        return {
            "version": self.VERSION,
            "residual_sigma": round(residual_sigma, 6),
            "mack_msep_by_uwy": mack_msep,
            "bf_prediction_error_by_uwy": bf_error,
            "mack_total_msep": round(total_mack_msep, 4),
            "bf_total_prediction_error": round(total_bf_error, 4),
            "total_process_sd": round(process_sd_total, 4),
            "total_process_cv": round(float(process_cv), 4)
            if process_cv is not None
            else None,
        }

    def bootstrap_predictive_distribution(
        self,
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
        sample_count: int = 800,
        seed: int = 17,
    ) -> dict[str, Any]:
        if results_df is None or results_df.empty:
            return {"sample_count": 0}

        ibnr_values: list[float] = []
        for _, row in results_df.iterrows():
            incurred = float(row.get("incurred", 0.0) or 0.0)
            ultimate = float(row.get("ultimate", row.get("cl_ultimate", 0.0)) or 0.0)
            ibnr_values.append(max(0.0, ultimate - incurred))
        if not ibnr_values:
            return {"sample_count": 0}

        residuals = self._residual_series(heatmap_data)
        if not residuals:
            residuals = [0.0]

        rng = np.random.default_rng(seed)
        residual_array = np.array(residuals, dtype=float)
        ibnr_array = np.array(ibnr_values, dtype=float)
        simulations: list[float] = []
        for _ in range(int(sample_count)):
            shocks = rng.choice(residual_array, size=len(ibnr_array), replace=True)
            simulated = np.maximum(0.0, ibnr_array * (1.0 + shocks))
            simulations.append(float(simulated.sum()))

        values = np.array(simulations, dtype=float)
        p10 = float(np.quantile(values, 0.1))
        p50 = float(np.quantile(values, 0.5))
        p90 = float(np.quantile(values, 0.9))
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        downside = max(0.0, p50 - p10)
        upside = max(0.0, p90 - p50)
        skew_ratio = upside / downside if downside > 0 else None
        return {
            "sample_count": int(sample_count),
            "seed": int(seed),
            "p10": round(p10, 4),
            "p50": round(p50, 4),
            "p75": round(float(np.quantile(values, 0.75)), 4),
            "p90": round(p90, 4),
            "p95": round(float(np.quantile(values, 0.95)), 4),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std(ddof=0)), 4),
            "iqr": round(iqr, 4),
            "right_tail_skew_ratio": round(float(skew_ratio), 4)
            if skew_ratio is not None
            else None,
        }

    def tail_model_assessment(
        self, *, scenarios: list[dict[str, Any]]
    ) -> dict[str, Any]:
        tail_items = [
            item
            for item in scenarios
            if str(item.get("scenario_id", "")).startswith("tail_")
            or "tail" in str(item.get("transform", "")).lower()
        ]
        if len(tail_items) < 2:
            return {
                "tail_scenario_count": len(tail_items),
                "instability_flag": False,
                "reason": "insufficient_tail_scenarios",
                "model_average": {},
            }

        scores = np.array([float(item.get("score", 0.0) or 0.0) for item in tail_items])
        score_std = float(scores.std(ddof=0))
        score_range = float(scores.max() - scores.min())
        instability = score_std >= 0.75 or score_range >= 2.0

        model_average = self._tail_model_average(tail_items)
        return {
            "tail_scenario_count": len(tail_items),
            "score_std": round(score_std, 4),
            "score_range": round(score_range, 4),
            "instability_flag": instability,
            "model_average": model_average,
        }

    def _tail_model_average(self, tail_items: list[dict[str, Any]]) -> dict[str, Any]:
        weighted_scores: list[float] = []
        weights: list[float] = []
        curve_weights: dict[str, float] = {}
        for item in tail_items:
            score = float(item.get("score", 0.0) or 0.0)
            weight = 1.0 / max(0.05, score)
            weights.append(weight)
            weighted_scores.append(score * weight)

            params = item.get("parameters")
            curve = None
            if isinstance(params, dict):
                tail = params.get("tail")
                if isinstance(tail, dict):
                    curve_raw = tail.get("curve")
                    if curve_raw is not None:
                        curve = str(curve_raw)
            if curve is None:
                curve = "unknown"
            curve_weights[curve] = curve_weights.get(curve, 0.0) + weight

        total_weight = float(sum(weights))
        if total_weight <= 0:
            return {}
        sorted_curve_items = sorted(
            [(key, float(value)) for key, value in curve_weights.items()],
            key=lambda item: item[1],
            reverse=True,
        )
        normalized_curve_weights = {
            key: round(value / total_weight, 4) for key, value in sorted_curve_items
        }
        preferred_curve = (
            next(iter(normalized_curve_weights.keys()))
            if normalized_curve_weights
            else None
        )
        weighted_score = float(sum(weighted_scores) / total_weight)
        return {
            "weighted_average_score": round(weighted_score, 4),
            "curve_weights": normalized_curve_weights,
            "preferred_curve": preferred_curve,
        }

    @staticmethod
    def _build_maturity_map(results_df: pd.DataFrame | None) -> dict[str, float]:
        if results_df is None or results_df.empty:
            return {}
        maturity: dict[str, float] = {}
        for idx, row in results_df.iterrows():
            incurred = float(row.get("incurred", 0.0) or 0.0)
            ultimate = float(row.get("ultimate", row.get("cl_ultimate", 0.0)) or 0.0)
            if ultimate <= 0:
                continue
            maturity[UncertaintyService._uwy_key(idx)] = float(
                max(0.0, min(1.0, incurred / ultimate))
            )
        return maturity

    @staticmethod
    def _uwy_key(index_value: object) -> str:
        year_value = getattr(index_value, "year", None)
        if year_value is not None:
            return str(year_value)
        return str(index_value)

    def _residual_sigma(self, residuals: list[float]) -> float:
        if len(residuals) < 2:
            return 0.12
        sigma = float(np.std(np.array(residuals, dtype=float), ddof=0))
        return max(0.05, min(0.65, sigma))

    @staticmethod
    def _residual_series(heatmap_data: dict | None) -> list[float]:
        if not isinstance(heatmap_data, dict):
            return []
        incurred = heatmap_data.get("incurred")
        link_ratios = heatmap_data.get("link_ratios")
        if not isinstance(incurred, pd.DataFrame) or not isinstance(
            link_ratios, pd.DataFrame
        ):
            return []
        if "LDF" not in link_ratios.index:
            return []

        age_steps = [
            int(col)
            for col in incurred.columns
            if str(col).isdigit() or isinstance(col, int)
        ]
        age_steps = sorted(age_steps)
        if len(age_steps) < 2:
            return []
        age_pairs = list(zip(age_steps[:-1], age_steps[1:]))

        ldf_row = link_ratios.loc["LDF"]
        ldf_map: dict[int, object] = {}
        for key, value in ldf_row.items():
            try:
                ldf_map[int(key)] = value
            except (TypeError, ValueError):
                continue
        residuals: list[float] = []
        for _, row in incurred.iterrows():
            for age, next_age in age_pairs:
                actual = UncertaintyService._to_float(row.get(next_age))
                expected_base = UncertaintyService._to_float(row.get(age))
                ldf = UncertaintyService._to_float(ldf_map.get(age))
                if actual is None or expected_base is None or ldf is None:
                    continue
                if (
                    not np.isfinite(actual)
                    or not np.isfinite(expected_base)
                    or not np.isfinite(ldf)
                ):
                    continue
                expected = expected_base * ldf
                if expected <= 0:
                    continue
                residuals.append((actual - expected) / expected)
        return residuals

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            parsed = pd.to_numeric([value], errors="coerce")[0]
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return float(parsed)
