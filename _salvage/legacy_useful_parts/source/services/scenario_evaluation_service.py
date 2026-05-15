from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Literal, cast
import uuid

import pandas as pd

from source.services.review_models import (
    DiagnosticEvidence,
    DiagnosticFinding,
    DiagnosticRecommendation,
    RunMetadata,
    ScenarioEvaluation,
)
from source.reserving import Reserving
from source.services.diagnostics_service import DiagnosticsService
from source.services.scenario_scoring_service import ScenarioScoringService
from source.services.uncertainty_service import UncertaintyService


class ScenarioEvaluationService:
    _DEFAULT_BACKTEST_BIAS_THRESHOLD = 0.12
    _DEFAULT_BACKTEST_MAE_THRESHOLD = 0.2
    _SEGMENT_MULTIPLIER_BY_KEY = {
        "motor": 0.9,
        "property": 1.05,
        "casualty": 1.2,
        "liability": 1.2,
    }
    _MATURITY_MULTIPLIER_BY_REGIME = {
        "immature": 1.25,
        "mixed": 1.0,
        "mature": 0.85,
    }

    def __init__(
        self,
        *,
        diagnostics_service: DiagnosticsService | None = None,
        uncertainty_service: UncertaintyService | None = None,
        scoring_service: ScenarioScoringService | None = None,
        scenario_generator_version: str = "v1.2",
    ) -> None:
        self._diagnostics_service = diagnostics_service or DiagnosticsService()
        self._uncertainty_service = uncertainty_service or UncertaintyService()
        self._scoring_service = scoring_service or ScenarioScoringService()
        self._scenario_generator_version = str(scenario_generator_version)

    def evaluate_scenario(
        self,
        *,
        segment: str,
        reserving: Reserving,
        params: dict[str, Any],
        scenario_id: str,
        summary: str,
        parent_scenario_id: str | None,
        transform: str,
        rationale_evidence_ids: list[str],
        continuity_penalty: float = 0.0,
        governance_penalty: float = 0.0,
    ) -> ScenarioEvaluation:
        self.apply_params_to_reserving(reserving=reserving, params=params)
        results_df = reserving.get_results()
        heatmap_data = reserving.get_triangle_heatmap_data()
        diagnostics_service, calibration = self.calibrated_diagnostics_service(
            segment=segment,
            results_df=results_df,
            heatmap_data=heatmap_data,
        )
        run_result = diagnostics_service.run(
            results_df=results_df,
            heatmap_data=heatmap_data,
        )
        run_metadata = self.build_run_metadata(
            results_df=results_df,
            heatmap_data=heatmap_data,
        )
        findings = [
            self.map_finding(item, run_metadata=run_metadata)
            for item in run_result.findings
        ]
        recommendations = [
            self.map_recommendation(item, run_metadata=run_metadata)
            for item in run_result.recommendations
        ]
        severity_components = self.severity_components(findings)
        governance = self.governance_assessment(
            findings=findings,
            severity_components=severity_components,
        )
        drop_count = len(params.get("drop", []))
        score_payload = self._scoring_service.score(
            findings=run_result.findings,
            drop_count=drop_count,
            continuity_penalty=continuity_penalty,
            governance_penalty=governance_penalty,
        )

        scenario_metrics = cast(dict[str, Any], dict(run_result.metrics))
        scenario_metrics["severity_components"] = severity_components
        scenario_metrics["governance_tier"] = governance["tier"]
        scenario_metrics["governance_escalation_triggers"] = governance[
            "escalation_triggers"
        ]
        scenario_metrics["governance_requires_human_review"] = governance[
            "requires_human_review"
        ]
        scenario_metrics["threshold_calibration"] = calibration
        uncertainty = self._uncertainty_service.baseline_uncertainty(
            results_df=results_df,
            heatmap_data=heatmap_data,
        )
        scenario_metrics["uncertainty"] = uncertainty
        scenario_metrics["score_breakdown"] = score_payload["components"]
        scenario_metrics["score_penalties"] = score_payload["penalties"]
        scenario_metrics["score_formula_version"] = score_payload["formula_version"]

        return ScenarioEvaluation(
            scenario_id=scenario_id,
            score=float(score_payload["score"]),
            summary=summary,
            parameters=dict(params),
            findings=findings,
            recommendations=recommendations,
            metrics=scenario_metrics,
            lineage={
                "parent_scenario_id": parent_scenario_id,
                "transform": transform,
                "rationale_evidence_ids": list(rationale_evidence_ids),
            },
            governance=governance,
            calibration=calibration,
            uncertainty=uncertainty,
            run_metadata=run_metadata,
        )

    def scenario_totals_for_params(
        self,
        *,
        reserving: Reserving,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.apply_params_to_reserving(reserving=reserving, params=params)
        results_df = reserving.get_results()
        total_ultimate = float(results_df["ultimate"].sum()) if len(results_df) else 0.0
        total_incurred = float(results_df["incurred"].sum()) if len(results_df) else 0.0
        total_ibnr = total_ultimate - total_incurred
        by_uwy = []
        for idx, row in results_df.iterrows():
            uwy = getattr(idx, "year", None)
            label = str(uwy) if uwy is not None else str(idx)[:4]
            by_uwy.append(
                {
                    "uwy": label,
                    "ultimate": round(float(row.get("ultimate", 0.0) or 0.0), 6),
                    "ibnr": round(
                        float(row.get("ultimate", 0.0) or 0.0)
                        - float(row.get("incurred", 0.0) or 0.0),
                        6,
                    ),
                    "selected_method": str(row.get("selected_method", "chainladder")),
                }
            )
        return {
            "parameters": dict(params),
            "total_ultimate": round(total_ultimate, 6),
            "total_incurred": round(total_incurred, 6),
            "total_ibnr": round(total_ibnr, 6),
            "rows": by_uwy,
        }

    def apply_params_to_reserving(
        self,
        *,
        reserving: Reserving,
        params: dict[str, Any],
    ) -> None:
        drops = self.filter_valid_drop_pairs(
            reserving=reserving,
            drops=self.normalize_drop_pairs(params.get("drop", [])),
        )
        drop_valuation = self.normalize_drop_valuation(params.get("drop_valuation", []))
        tail_config = params.get("tail", {})
        tail_fit_period = self.normalize_fit_period(tail_config.get("fit_period", []))
        tail_projection_months = int(tail_config.get("projection_period", 0) or 0)
        months_per_dev = self.infer_months_per_dev(reserving)
        extrap_periods = tail_projection_months // months_per_dev
        projection_period = extrap_periods * months_per_dev

        reserving.set_development(
            average=Reserving._normalize_average(params.get("average", "volume")),
            drop=drops,
            drop_valuation=drop_valuation,
        )
        reserving.set_tail(
            curve=str(tail_config.get("curve", "weibull")),
            attachment_age=tail_config.get("attachment_age"),
            extrap_periods=extrap_periods,
            projection_period=projection_period,
            fit_period=tail_fit_period,
        )

        bf_apriori = params.get("bf_apriori", {})
        if isinstance(bf_apriori, dict) and bf_apriori:
            reserving.set_bornhuetter_ferguson(
                apriori=self.autocomplete_bf_apriori(
                    reserving=reserving,
                    bf_apriori=bf_apriori,
                )
            )
        else:
            reserving.set_bornhuetter_ferguson(apriori=0.6)

        reserving.reserve(
            final_ultimate=self.normalize_final_ultimate(
                params.get("final_ultimate", "chainladder")
            ),
            selected_ultimate_by_uwy=dict(params.get("selected_ultimate_by_uwy", {})),
        )

    def calibrated_diagnostics_service(
        self,
        *,
        segment: str,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> tuple[DiagnosticsService, dict[str, Any]]:
        calibration = self.calibrate_backtest_thresholds(
            segment=segment,
            results_df=results_df,
            heatmap_data=heatmap_data,
        )
        diagnostics_service = DiagnosticsService(
            backtest_bias_threshold=float(calibration["backtest_bias_threshold"]),
            backtest_mae_threshold=float(calibration["backtest_mae_threshold"]),
        )
        return diagnostics_service, calibration

    def calibrate_backtest_thresholds(
        self,
        *,
        segment: str,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> dict[str, Any]:
        maturity_regime = self.maturity_regime(results_df)
        residual_points = self._diagnostics_service._residual_points(heatmap_data)
        abs_residuals = sorted(
            abs(float(item.get("residual", 0.0) or 0.0)) for item in residual_points
        )

        segment_key = str(segment).strip().lower()
        segment_multiplier = float(
            self._SEGMENT_MULTIPLIER_BY_KEY.get(segment_key, 1.0)
        )
        maturity_multiplier = float(
            self._MATURITY_MULTIPLIER_BY_REGIME.get(maturity_regime, 1.0)
        )
        floor_bias = (
            self._DEFAULT_BACKTEST_BIAS_THRESHOLD
            * segment_multiplier
            * maturity_multiplier
        )
        floor_mae = (
            self._DEFAULT_BACKTEST_MAE_THRESHOLD
            * segment_multiplier
            * maturity_multiplier
        )

        if len(abs_residuals) < 8:
            return {
                "segment": segment,
                "maturity_regime": maturity_regime,
                "residual_count": len(abs_residuals),
                "backtest_bias_threshold": round(float(floor_bias), 4),
                "backtest_mae_threshold": round(float(floor_mae), 4),
                "method": "segment_maturity_floor",
            }

        empirical_bias = self.quantile(abs_residuals, 0.55)
        empirical_mae = self.quantile(abs_residuals, 0.8)
        calibrated_bias = min(0.45, max(floor_bias, empirical_bias))
        calibrated_mae = min(0.65, max(floor_mae, empirical_mae))
        return {
            "segment": segment,
            "maturity_regime": maturity_regime,
            "residual_count": len(abs_residuals),
            "backtest_bias_threshold": round(float(calibrated_bias), 4),
            "backtest_mae_threshold": round(float(calibrated_mae), 4),
            "method": "backtest_quantile_blend",
        }

    def maturity_regime(self, results_df: pd.DataFrame | None) -> str:
        maturity_map = self._diagnostics_service._build_maturity_map(results_df)
        if not maturity_map:
            return "mixed"
        average_maturity = sum(maturity_map.values()) / len(maturity_map)
        if average_maturity < 0.4:
            return "immature"
        if average_maturity >= 0.75:
            return "mature"
        return "mixed"

    def build_run_metadata(
        self,
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> RunMetadata:
        return RunMetadata(
            run_id=str(uuid.uuid4()),
            generated_at=datetime.now(timezone.utc),
            data_fingerprint=self.data_fingerprint(
                results_df=results_df,
                heatmap_data=heatmap_data,
            ),
            diagnostics_version=DiagnosticsService.DIAGNOSTICS_VERSION,
            scenario_generator_version=self._scenario_generator_version,
        )

    @staticmethod
    def map_finding(item, *, run_metadata: RunMetadata) -> DiagnosticFinding:
        evidence = dict(item.evidence)
        value = float(evidence.get("value", 0.0) or 0.0)
        threshold_raw = evidence.get("threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None
        basis_raw = evidence.get("basis")
        basis = str(basis_raw) if basis_raw is not None else None
        metric_id = str(evidence.get("metric_id", "metric"))
        diagnostic_id = str(evidence.get("diagnostic_id", item.code))
        evidence_id = str(
            evidence.get(
                "evidence_id",
                ScenarioEvaluationService.make_evidence_id(
                    run_id=run_metadata.run_id,
                    diagnostic_id=diagnostic_id,
                    metric_id=metric_id,
                ),
            )
        )
        diagnostic_version = str(
            evidence.get(
                "diagnostic_version",
                run_metadata.diagnostics_version,
            )
        )
        applicability_conditions = ScenarioEvaluationService.to_string_list(
            evidence.get("applicability_conditions")
        )
        alternative_hypotheses = ScenarioEvaluationService.to_string_list(
            evidence.get("alternative_hypotheses")
        )
        severity = str(item.severity)
        if severity not in {"low", "medium", "high", "critical"}:
            severity = "medium"
        return DiagnosticFinding(
            code=str(item.code),
            severity=cast(Literal["low", "medium", "high", "critical"], severity),
            message=str(item.message),
            evidence=DiagnosticEvidence(
                metric_id=metric_id,
                value=value,
                threshold=threshold,
                basis=basis,
                evidence_id=evidence_id,
                diagnostic_id=diagnostic_id,
                diagnostic_version=diagnostic_version,
                unit=ScenarioEvaluationService.to_optional_string(evidence.get("unit")),
                direction=ScenarioEvaluationService.normalize_direction(
                    evidence.get("direction")
                ),
                p_value_or_score=ScenarioEvaluationService.to_optional_float(
                    evidence.get("p_value_or_score")
                ),
                severity_band=cast(
                    Literal["low", "medium", "high", "critical"] | None,
                    ScenarioEvaluationService.normalize_severity_level(
                        evidence.get("severity_band"),
                        fallback=severity,
                    ),
                ),
                applicability_conditions=applicability_conditions,
                alternative_hypotheses=alternative_hypotheses,
                confidence=ScenarioEvaluationService.to_optional_float(
                    evidence.get("confidence")
                ),
                required_review_level=ScenarioEvaluationService.normalize_review_level(
                    evidence.get("required_review_level")
                ),
            ),
            suggested_actions=list(item.suggested_actions),
        )

    @staticmethod
    def map_recommendation(
        item, *, run_metadata: RunMetadata
    ) -> DiagnosticRecommendation:
        evidence = dict(item.evidence)
        value = float(evidence.get("value", 0.0) or 0.0)
        threshold_raw = evidence.get("threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None
        basis_raw = evidence.get("basis")
        basis = str(basis_raw) if basis_raw is not None else None
        priority = str(item.priority)
        if priority not in {"low", "medium", "high", "critical"}:
            priority = "medium"
        metric_id = str(evidence.get("metric_id", "metric"))
        diagnostic_id = str(evidence.get("diagnostic_id", item.code))
        evidence_id = str(
            evidence.get(
                "evidence_id",
                ScenarioEvaluationService.make_evidence_id(
                    run_id=run_metadata.run_id,
                    diagnostic_id=diagnostic_id,
                    metric_id=metric_id,
                ),
            )
        )
        return DiagnosticRecommendation(
            code=str(item.code),
            priority=cast(Literal["low", "medium", "high", "critical"], priority),
            message=str(item.message),
            rationale=str(item.rationale),
            evidence=DiagnosticEvidence(
                metric_id=metric_id,
                value=value,
                threshold=threshold,
                basis=basis,
                evidence_id=evidence_id,
                diagnostic_id=diagnostic_id,
                diagnostic_version=str(
                    evidence.get(
                        "diagnostic_version",
                        run_metadata.diagnostics_version,
                    )
                ),
                unit=ScenarioEvaluationService.to_optional_string(evidence.get("unit")),
                direction=ScenarioEvaluationService.normalize_direction(
                    evidence.get("direction")
                ),
                p_value_or_score=ScenarioEvaluationService.to_optional_float(
                    evidence.get("p_value_or_score")
                ),
                severity_band=cast(
                    Literal["low", "medium", "high", "critical"] | None,
                    ScenarioEvaluationService.normalize_severity_level(
                        evidence.get("severity_band"),
                        fallback=priority,
                    ),
                ),
                applicability_conditions=ScenarioEvaluationService.to_string_list(
                    evidence.get("applicability_conditions")
                ),
                alternative_hypotheses=ScenarioEvaluationService.to_string_list(
                    evidence.get("alternative_hypotheses")
                ),
                confidence=ScenarioEvaluationService.to_optional_float(
                    evidence.get("confidence")
                ),
                required_review_level=ScenarioEvaluationService.normalize_review_level(
                    evidence.get("required_review_level")
                ),
            ),
            proposed_parameters=dict(item.proposed_parameters),
        )

    @staticmethod
    def severity_components(findings: list[DiagnosticFinding]) -> dict[str, float]:
        buckets = {
            "data_quality": 0.0,
            "stability": 0.0,
            "backtest": 0.0,
            "coherence": 0.0,
            "tail": 0.0,
            "other": 0.0,
        }
        weights = {"low": 0.5, "medium": 2.0, "high": 5.0, "critical": 8.0}
        for finding in findings:
            weight = float(weights.get(finding.severity, 1.0))
            code = finding.code
            if "DATA_QUALITY" in code or "NEGATIVE_DEVELOPMENT" in code:
                buckets["data_quality"] += weight
            elif "BACKTEST" in code:
                buckets["backtest"] += weight
            elif "TAIL" in code:
                buckets["tail"] += weight
            elif "COHERENCE" in code or "PAID_INCURRED" in code:
                buckets["coherence"] += weight
            elif any(
                token in code
                for token in [
                    "LINK_RATIO",
                    "CALENDAR",
                    "PORTFOLIO_SHIFT",
                    "LATEST_DIAGONAL",
                ]
            ):
                buckets["stability"] += weight
            else:
                buckets["other"] += weight
        return {key: round(value, 4) for key, value in buckets.items()}

    @staticmethod
    def governance_assessment(
        *,
        findings: list[DiagnosticFinding],
        severity_components: dict[str, float],
    ) -> dict[str, Any]:
        triggers: list[str] = []
        critical_present = any(item.severity == "critical" for item in findings)
        high_present = any(item.severity == "high" for item in findings)
        unconfirmed_shift = any(
            item.code.startswith("PORTFOLIO_SHIFT_SIGNAL_UNCONFIRMED")
            for item in findings
        )
        severe_negative_development = any(
            item.code == "NEGATIVE_DEVELOPMENT_TRIAGE"
            and item.severity in {"high", "critical"}
            for item in findings
        )

        if critical_present:
            triggers.append("critical_finding_present")
        if severity_components.get("data_quality", 0.0) >= 8.0:
            triggers.append("data_quality_gate_block")
        if (
            severity_components.get("backtest", 0.0) >= 5.0
            and severity_components.get("stability", 0.0) >= 5.0
        ):
            triggers.append("backtest_stability_joint_stress")
        if severe_negative_development:
            triggers.append("negative_development_escalation")
        if unconfirmed_shift:
            triggers.append("unconfirmed_portfolio_shift_signal")
        if (
            severity_components.get("tail", 0.0) >= 5.0
            and severity_components.get("backtest", 0.0) >= 2.0
        ):
            triggers.append("tail_backtest_joint_stress")

        if any(
            token in triggers
            for token in [
                "critical_finding_present",
                "data_quality_gate_block",
                "backtest_stability_joint_stress",
            ]
        ):
            tier = "red"
        elif high_present or bool(triggers):
            tier = "amber"
        else:
            tier = "green"

        if tier == "red":
            actions = [
                "Mandatory actuarial lead review before sign-off",
                "Record override rationale and approval chain",
            ]
        elif tier == "amber":
            actions = ["Actuarial peer review required before parameter adoption"]
        else:
            actions = ["Standard reviewer sign-off"]

        return {
            "tier": tier,
            "escalation_triggers": triggers,
            "requires_human_review": tier in {"amber", "red"},
            "required_actions": actions,
        }

    @staticmethod
    def data_fingerprint(
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> str:
        payload: dict[str, object] = {
            "results": ScenarioEvaluationService.safe_df_records(results_df),
            "heatmap": {},
        }
        if isinstance(heatmap_data, dict):
            serialized: dict[str, object] = {}
            for key in ["incurred", "paid", "premium", "link_ratios"]:
                serialized[key] = ScenarioEvaluationService.safe_df_records(
                    ScenarioEvaluationService.to_dataframe(heatmap_data.get(key))
                )
            payload["heatmap"] = serialized
        canonical = json.dumps(
            payload,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def safe_df_records(dataframe: pd.DataFrame | None) -> list[dict[str, Any]]:
        if dataframe is None or dataframe.empty:
            return []
        frame = dataframe.copy()
        if frame.index.name is not None or not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index()
        records = frame.to_dict(orient="records")
        normalized: list[dict[str, Any]] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    str(key): ScenarioEvaluationService.json_safe_value(value)
                    for key, value in row.items()
                }
            )
        return normalized

    @staticmethod
    def json_safe_value(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [ScenarioEvaluationService.json_safe_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): ScenarioEvaluationService.json_safe_value(item)
                for key, item in value.items()
            }
        if value is pd.NA:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, pd.Period):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return isoformat()
            except Exception:
                pass
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def make_evidence_id(*, run_id: str, diagnostic_id: str, metric_id: str) -> str:
        raw = f"{run_id}|{diagnostic_id}|{metric_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        clipped_q = min(max(float(q), 0.0), 1.0)
        index = (len(values) - 1) * clipped_q
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return float(values[lower] * (1.0 - weight) + values[upper] * weight)

    @staticmethod
    def normalize_drop_pairs(
        raw_pairs: object,
    ) -> list[tuple[str, int]] | None:
        if not isinstance(raw_pairs, list):
            return None
        parsed: list[tuple[str, int]] = []
        for pair in raw_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            try:
                parsed.append((str(pair[0]), int(pair[1])))
            except (TypeError, ValueError):
                continue
        return parsed or None

    @staticmethod
    def normalize_drop_valuation(raw_pairs: object) -> list[str] | None:
        if not isinstance(raw_pairs, list):
            return None
        years: list[str] = []
        for pair in raw_pairs:
            if not isinstance(pair, list) or len(pair) < 1:
                continue
            years.append(str(pair[0]))
        return years or None

    @staticmethod
    def normalize_fit_period(raw_fit_period: object) -> tuple[int, int | None] | None:
        if not isinstance(raw_fit_period, list) or not raw_fit_period:
            return None
        normalized = sorted({int(value) for value in raw_fit_period})
        if len(normalized) == 1:
            return (normalized[0], None)
        return (normalized[0], normalized[-1])

    @staticmethod
    def infer_months_per_dev(reserving: Reserving) -> int:
        try:
            triangle = reserving._triangle.get_triangle()["incurred"]
            development = [int(value) for value in triangle.development.tolist()]
        except Exception:
            return 3
        if len(development) < 2:
            return development[0] if development else 3
        deltas = [
            right - left
            for left, right in zip(development[:-1], development[1:])
            if right - left > 0
        ]
        if not deltas:
            return 3
        return min(deltas)

    @staticmethod
    def filter_valid_drop_pairs(
        *,
        reserving: Reserving,
        drops: list[tuple[str, int]] | None,
    ) -> list[tuple[str, int]] | None:
        if not drops:
            return None
        try:
            heatmap_data = reserving.get_triangle_heatmap_data()
            link_ratios_raw = heatmap_data.get("link_ratios")
        except Exception:
            return drops
        if not isinstance(link_ratios_raw, pd.DataFrame) or link_ratios_raw.empty:
            return drops

        valid_origins = {
            ScenarioEvaluationService.origin_label(origin)
            for origin in link_ratios_raw.index
            if str(origin) not in {"LDF", "Tail"}
        }
        valid_periods = {
            age
            for age in (
                Reserving._parse_cdf_label_to_age(col)
                for col in link_ratios_raw.columns
            )
            if age is not None
        }
        filtered: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for origin, age in drops:
            pair = (str(origin), int(age))
            if pair in seen:
                continue
            if pair[0] not in valid_origins or pair[1] not in valid_periods:
                continue
            seen.add(pair)
            filtered.append(pair)
        return filtered or None

    @staticmethod
    def origin_label(origin: object) -> str:
        if hasattr(origin, "year"):
            return str(origin.year)
        text = str(origin)
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else text

    @staticmethod
    def normalize_final_ultimate(
        value: object,
    ) -> Literal["chainladder", "bornhuetter_ferguson"]:
        normalized = str(value).strip().lower()
        if normalized == "bornhuetter_ferguson":
            return "bornhuetter_ferguson"
        return "chainladder"

    @staticmethod
    def autocomplete_bf_apriori(
        *,
        reserving: Reserving,
        bf_apriori: dict,
    ) -> dict[str, float]:
        completed = {str(key): float(value) for key, value in bf_apriori.items()}
        fallback = 0.6
        try:
            incurred_triangle = reserving._triangle.get_triangle()["incurred"]
            origins = list(incurred_triangle.origin)
        except Exception:
            return completed

        for origin in origins:
            year_value = getattr(origin, "year", None)
            key = str(year_value) if year_value is not None else str(origin)
            if key in completed:
                continue
            completed[key] = fallback
        return completed

    @staticmethod
    def to_optional_float(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def to_optional_string(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def to_string_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def normalize_direction(value: object) -> Literal["good", "bad", "neutral"] | None:
        normalized = str(value).strip().lower()
        if normalized in {"good", "bad", "neutral"}:
            return cast(Literal["good", "bad", "neutral"], normalized)
        return None

    @staticmethod
    def normalize_severity_level(
        value: object,
        *,
        fallback: str | None = None,
    ) -> str | None:
        normalized = str(value).strip().lower()
        if normalized in {"low", "medium", "high", "critical"}:
            return normalized
        if fallback is None:
            return None
        fallback_norm = str(fallback).strip().lower()
        if fallback_norm in {"low", "medium", "high", "critical"}:
            return fallback_norm
        return None

    @staticmethod
    def normalize_review_level(
        value: object,
    ) -> Literal["green", "amber", "red"] | None:
        normalized = str(value).strip().lower()
        if normalized in {"green", "amber", "red"}:
            return cast(Literal["green", "amber", "red"], normalized)
        return None

    @staticmethod
    def to_dataframe(raw: object) -> pd.DataFrame | None:
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
