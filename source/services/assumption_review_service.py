from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

from source.reserving import Reserving
from source.services.diagnostics_service import DiagnosticsService
from source.services.movement_diagnostics_service import MovementDiagnosticsService
from source.services.scenario_evaluation_service import ScenarioEvaluationService
from source.services.scenario_scoring_service import ScenarioScoringService
from source.services.segment_memory_service import SegmentMemoryService


@dataclass(frozen=True)
class _BaselineContext:
    evaluation: Any
    totals: dict[str, Any]
    ldf_consistency: dict[str, Any]
    late_emergence: dict[str, Any]
    anomaly_triage: dict[str, Any]


class AssumptionReviewService:
    _SUPPORTED_TAIL_CURVES = ("weibull", "inverse_power", "exponential")

    def __init__(
        self,
        *,
        evaluation_service: ScenarioEvaluationService | None = None,
        scoring_service: ScenarioScoringService | None = None,
        diagnostics_service: DiagnosticsService | None = None,
        segment_memory_service: SegmentMemoryService | None = None,
    ) -> None:
        self._evaluation_service = evaluation_service or ScenarioEvaluationService()
        self._scoring_service = scoring_service or ScenarioScoringService()
        self._diagnostics_service = diagnostics_service or DiagnosticsService()
        self._segment_memory_service = segment_memory_service or SegmentMemoryService()

    def review_drops(
        self,
        *,
        segment: str,
        reserving: Reserving,
        baseline_params: dict[str, Any],
        segment_memory: dict[str, Any] | None = None,
        candidate_limit: int = 5,
    ) -> dict[str, Any]:
        baseline = self._build_baseline_context(
            segment=segment,
            reserving=reserving,
            baseline_params=baseline_params,
        )
        candidates = self._build_drop_candidates(
            baseline_eval=baseline.evaluation,
            baseline_params=baseline_params,
            candidate_limit=candidate_limit,
        )
        reviewed: list[dict[str, Any]] = []
        for item in candidates:
            reviewed.append(
                self._score_drop_candidate(
                    segment=segment,
                    reserving=reserving,
                    baseline=baseline,
                    candidate=item,
                    segment_memory=segment_memory,
                )
            )

        ordered = sorted(
            reviewed, key=lambda row: float(row.get("score", 0.0)), reverse=True
        )
        for index, row in enumerate(ordered, start=1):
            row["rank"] = index

        best = ordered[0] if ordered else None
        recommendation = self._build_candidate_recommendation(best, ordered)
        return {
            "review_type": "drop_review",
            "baseline": {
                "scenario_id": baseline.evaluation.scenario_id,
                "score": baseline.evaluation.score,
                "governance": baseline.evaluation.governance,
                "parameters": dict(baseline_params),
                "metrics": {
                    "total_ibnr": baseline.totals.get("total_ibnr"),
                    "ldf_finding_count": baseline.ldf_consistency.get(
                        "summary", {}
                    ).get("finding_count", 0),
                    "anomaly_pause": baseline.anomaly_triage.get(
                        "pause_recommendation", False
                    ),
                },
            },
            "candidates": ordered,
            "recommendation": recommendation,
            "continuity_notes": best.get("continuity_notes", [])
            if isinstance(best, dict)
            else [],
            "policy_trace": best.get("policy_trace", {})
            if isinstance(best, dict)
            else {},
            "evidence_summary": {
                "baseline_drop_recommendations": self._drop_recommendation_map(
                    baseline.evaluation
                ),
                "baseline_late_emergence_rows": baseline.late_emergence.get("rows", [])[
                    :5
                ],
                "baseline_ldf_findings": baseline.ldf_consistency.get("findings", [])[
                    :5
                ],
            },
            "run_metadata": self._run_metadata(baseline.evaluation),
        }

    def review_tail(
        self,
        *,
        segment: str,
        reserving: Reserving,
        baseline_params: dict[str, Any],
        segment_memory: dict[str, Any] | None = None,
        candidate_limit: int = 12,
    ) -> dict[str, Any]:
        baseline = self._build_baseline_context(
            segment=segment,
            reserving=reserving,
            baseline_params=baseline_params,
        )
        candidates = self._build_tail_candidates(
            reserving=reserving,
            baseline_params=baseline_params,
            candidate_limit=candidate_limit,
        )
        reviewed: list[dict[str, Any]] = []
        for item in candidates:
            reviewed.append(
                self._score_tail_candidate(
                    segment=segment,
                    reserving=reserving,
                    baseline=baseline,
                    candidate=item,
                    segment_memory=segment_memory,
                )
            )

        ordered = sorted(
            reviewed, key=lambda row: float(row.get("score", 0.0)), reverse=True
        )
        for index, row in enumerate(ordered, start=1):
            row["rank"] = index
        best = ordered[0] if ordered else None
        recommendation = self._build_candidate_recommendation(best, ordered)
        return {
            "review_type": "tail_review",
            "baseline": {
                "scenario_id": baseline.evaluation.scenario_id,
                "score": baseline.evaluation.score,
                "governance": baseline.evaluation.governance,
                "parameters": dict(baseline_params),
                "metrics": self._evaluate_tail_metrics(
                    reserving=reserving,
                    params=baseline_params,
                ),
            },
            "candidates": ordered,
            "recommendation": recommendation,
            "continuity_notes": best.get("continuity_notes", [])
            if isinstance(best, dict)
            else [],
            "policy_trace": best.get("policy_trace", {})
            if isinstance(best, dict)
            else {},
            "evidence_summary": {
                "baseline_tail_recommendation": self._heuristic_tail_recommendation(
                    reserving
                ),
                "baseline_anomaly_summary": baseline.anomaly_triage.get("summary", {}),
            },
            "run_metadata": self._run_metadata(baseline.evaluation),
        }

    def review_bf_suitability(
        self,
        *,
        segment: str,
        reserving: Reserving,
        baseline_params: dict[str, Any],
        segment_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        baseline = self._build_baseline_context(
            segment=segment,
            reserving=reserving,
            baseline_params=baseline_params,
        )
        results_df = reserving.get_results()
        maturity = self._maturity_by_uwy(results_df)
        diagnostics_by_uwy = self._diagnostics_flags_by_uwy(
            baseline.evaluation.findings
        )
        apriori_guidance = self._bf_apriori_guidance(baseline.evaluation)

        rows: list[dict[str, Any]] = []
        for idx, row in results_df.iterrows():
            uwy = self._uwy_label(idx)
            premium = float(row.get("Premium", 0.0) or 0.0)
            incurred = float(row.get("incurred", 0.0) or 0.0)
            cl_ultimate = float(row.get("cl_ultimate", 0.0) or 0.0)
            bf_ultimate = float(row.get("bf_ultimate", 0.0) or 0.0)
            maturity_value = maturity.get(uwy, 0.0)
            cl_gap = abs(cl_ultimate - bf_ultimate) / premium if premium > 0 else 0.0
            percent_reported = incurred / cl_ultimate if cl_ultimate > 0 else 0.0
            flags = diagnostics_by_uwy.get(uwy, [])
            has_volatility = any(
                token in code
                for code in flags
                for token in [
                    "LATEST_DIAGONAL_DEVIATION",
                    "PORTFOLIO_SHIFT",
                    "NEGATIVE_DEVELOPMENT",
                    "LOSS_RATIO_OUTLIER",
                ]
            )
            maturity_support = 0.6 - maturity_value
            volatility_support = 0.25 if has_volatility else -0.05
            cl_sensitivity_support = min(cl_gap / 0.25, 0.35)
            percent_reported_support = self._percent_reported_support(percent_reported)
            apriori_readiness_support = (
                0.2 if apriori_guidance.get("available") else -0.15
            )
            score_breakdown = self._scoring_service.score_bf_suitability(
                maturity_support=maturity_support,
                volatility_support=volatility_support,
                cl_sensitivity_support=cl_sensitivity_support,
                percent_reported_support=percent_reported_support,
                apriori_readiness_support=apriori_readiness_support,
            )
            suitability_class = self._scoring_service.classify_bf_suitability(
                total_score=float(score_breakdown["score"])
            )
            rows.append(
                {
                    "uwy": uwy,
                    "maturity": round(maturity_value, 4),
                    "cl_bf_gap_on_premium": round(cl_gap, 4),
                    "percent_reported": round(percent_reported, 4),
                    "diagnostic_flags": flags,
                    "score_breakdown": {
                        "components": score_breakdown["components"],
                        "penalties": score_breakdown["penalties"],
                        "total_score": score_breakdown["score"],
                        "formula_version": score_breakdown["formula_version"],
                    },
                    "suitability_class": suitability_class,
                }
            )

        overall_class = self._overall_bf_class(rows)
        continuity_notes, policy_trace = self._bf_continuity_context(
            segment_memory=segment_memory,
            overall_class=overall_class,
        )
        return {
            "review_type": "bf_suitability",
            "rows": rows,
            "overall_class": overall_class,
            "summary": {
                "row_count": len(rows),
                "class_counts": self._class_counts(rows, key="suitability_class"),
                "anomaly_pause": baseline.anomaly_triage.get(
                    "pause_recommendation", False
                ),
            },
            "apriori_guidance": apriori_guidance,
            "continuity_notes": continuity_notes,
            "policy_trace": policy_trace,
            "run_metadata": self._run_metadata(baseline.evaluation),
        }

    def triage_anomalies(
        self,
        *,
        segment: str,
        reserving: Reserving,
        baseline_params: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_eval = self._evaluation_service.evaluate_scenario(
            segment=segment,
            reserving=reserving,
            params=dict(baseline_params),
            scenario_id="baseline",
            summary="Current session configuration",
            parent_scenario_id=None,
            transform="baseline",
            rationale_evidence_ids=[],
        )
        movement = MovementDiagnosticsService(reserving)
        movement_payload = movement.run()
        ldf_payload = movement.run_ldf_consistency()
        classified: list[dict[str, Any]] = []
        classified.extend(self._classify_diagnostic_findings(baseline_eval.findings))
        classified.extend(
            self._classify_movement_findings(movement_payload.get("findings", []))
        )
        classified.extend(self._classify_ldf_findings(ldf_payload.get("findings", [])))
        classified = self._dedupe_triaged_findings(classified)
        pause = any(bool(item.get("pause_recommendation")) for item in classified)
        return {
            "review_type": "anomaly_triage",
            "triaged_findings": classified,
            "summary": {
                "finding_count": len(classified),
                "type_counts": self._class_counts(classified, key="type"),
                "top_findings": classified[:5],
            },
            "pause_recommendation": pause,
            "run_metadata": self._run_metadata(baseline_eval),
        }

    def _build_baseline_context(
        self,
        *,
        segment: str,
        reserving: Reserving,
        baseline_params: dict[str, Any],
    ) -> _BaselineContext:
        baseline_eval = self._evaluation_service.evaluate_scenario(
            segment=segment,
            reserving=reserving,
            params=dict(baseline_params),
            scenario_id="baseline",
            summary="Current session configuration",
            parent_scenario_id=None,
            transform="baseline",
            rationale_evidence_ids=[],
        )
        totals = self._evaluation_service.scenario_totals_for_params(
            reserving=reserving,
            params=dict(baseline_params),
        )
        movement = MovementDiagnosticsService(reserving)
        ldf_consistency = movement.run_ldf_consistency()
        late_emergence = movement.run_late_emergence_benchmark()
        anomaly_triage = self.triage_anomalies(
            segment=segment,
            reserving=reserving,
            baseline_params=baseline_params,
        )
        self._evaluation_service.apply_params_to_reserving(
            reserving=reserving,
            params=baseline_params,
        )
        return _BaselineContext(
            evaluation=baseline_eval,
            totals=totals,
            ldf_consistency=ldf_consistency,
            late_emergence=late_emergence,
            anomaly_triage=anomaly_triage,
        )

    def _build_drop_candidates(
        self,
        *,
        baseline_eval: Any,
        baseline_params: dict[str, Any],
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        baseline_drop_pairs = {
            self._normalize_drop_pair(item) for item in baseline_params.get("drop", [])
        }
        baseline_drop_pairs.discard(None)
        recommendation_map = self._drop_recommendation_map(baseline_eval)
        ranked_pairs = [
            pair
            for pair in recommendation_map.keys()
            if pair not in baseline_drop_pairs
        ]
        candidates: list[dict[str, Any]] = []
        for index, pair in enumerate(ranked_pairs[:candidate_limit], start=1):
            params = self._clone_params(baseline_params)
            drops = {item for item in baseline_drop_pairs if item is not None}
            drops.add(pair)
            params["drop"] = [list(item) for item in sorted(drops)]
            candidates.append(
                {
                    "candidate_id": f"drop_{index}",
                    "summary": f"Add drop for AY {pair[0]} age {pair[1]}",
                    "parameters": params,
                    "drop_pairs": [pair],
                }
            )
        if len(ranked_pairs) >= 2 and len(candidates) < candidate_limit:
            combo_pairs = ranked_pairs[:2]
            params = self._clone_params(baseline_params)
            drops = {item for item in baseline_drop_pairs if item is not None}
            for pair in combo_pairs:
                drops.add(pair)
            params["drop"] = [list(item) for item in sorted(drops)]
            candidates.append(
                {
                    "candidate_id": "drop_combo_1",
                    "summary": "Combine the top two supported drop candidates",
                    "parameters": params,
                    "drop_pairs": combo_pairs,
                }
            )
        return candidates[:candidate_limit]

    def _score_drop_candidate(
        self,
        *,
        segment: str,
        reserving: Reserving,
        baseline: _BaselineContext,
        candidate: dict[str, Any],
        segment_memory: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = self._clone_params(candidate.get("parameters", {}))
        scenario_eval = self._evaluation_service.evaluate_scenario(
            segment=segment,
            reserving=reserving,
            params=params,
            scenario_id=str(candidate.get("candidate_id")),
            summary=str(candidate.get("summary", "")),
            parent_scenario_id="baseline",
            transform="drop_review",
            rationale_evidence_ids=[],
        )
        totals = self._evaluation_service.scenario_totals_for_params(
            reserving=reserving,
            params=params,
        )
        movement = MovementDiagnosticsService(reserving)
        ldf_payload = movement.run_ldf_consistency()
        drop_pairs = [
            pair
            for pair in (
                self._normalize_drop_pair(item)
                for item in candidate.get("drop_pairs", [])
            )
            if pair is not None
        ]
        outlier_support = self._drop_outlier_support(
            baseline_eval=baseline.evaluation,
            drop_pairs=drop_pairs,
        )
        consistency_improvement = self._consistency_improvement(
            baseline_payload=baseline.ldf_consistency,
            candidate_payload=ldf_payload,
        )
        late_emergence_support = self._late_emergence_support(
            baseline_payload=baseline.late_emergence,
            drop_pairs=drop_pairs,
        )
        reserve_impact_penalty = self._reserve_impact_penalty(
            baseline_total=float(baseline.totals.get("total_ibnr", 0.0) or 0.0),
            candidate_total=float(totals.get("total_ibnr", 0.0) or 0.0),
        )
        fragility_penalty = self._drop_fragility_penalty(
            consistency_improvement=consistency_improvement,
            reserve_impact_penalty=reserve_impact_penalty,
            drop_count=len(drop_pairs),
        )
        continuity_notes, policy_trace = self._candidate_policy_trace(
            params=params,
            segment_memory=segment_memory,
            governance_tier=scenario_eval.governance.get("tier"),
            review_type="drop_review",
            reserve_context={
                "drop_count": len(params.get("drop", [])),
            },
        )
        anomaly_penalty = self._anomaly_penalty(baseline.anomaly_triage)
        score_breakdown = self._scoring_service.score_drop_candidate(
            outlier_support=outlier_support,
            consistency_improvement=consistency_improvement,
            late_emergence_support=late_emergence_support,
            reserve_impact_penalty=reserve_impact_penalty,
            fragility_penalty=fragility_penalty,
            governance_penalty=self._governance_penalty(
                scenario_eval.governance.get("tier")
            )
            + anomaly_penalty,
            continuity_penalty=sum(
                float(value) for value in policy_trace["applied_penalties"].values()
            ),
        )
        recommendation_class = self._scoring_service.classify_recommendation(
            total_score=float(score_breakdown["score"]),
            governance_tier=scenario_eval.governance.get("tier"),
            rejected_before=bool(policy_trace["rejected_before"]),
        )
        self._evaluation_service.apply_params_to_reserving(
            reserving=reserving,
            params=self._clone_params(baseline.totals.get("parameters", {}))
            if isinstance(baseline.totals.get("parameters"), dict)
            else self._clone_params(candidate.get("parameters", {})),
        )
        return {
            "candidate_id": str(candidate.get("candidate_id")),
            "summary": str(candidate.get("summary", "")),
            "parameters": params,
            "score": score_breakdown["score"],
            "score_breakdown": {
                "components": score_breakdown["components"],
                "penalties": score_breakdown["penalties"],
                "total_score": score_breakdown["score"],
                "formula_version": score_breakdown["formula_version"],
            },
            "recommendation_class": recommendation_class,
            "metrics": {
                "drop_pairs": [list(item) for item in drop_pairs],
                "baseline_total_ibnr": baseline.totals.get("total_ibnr"),
                "candidate_total_ibnr": totals.get("total_ibnr"),
                "ibnr_delta": round(
                    float(totals.get("total_ibnr", 0.0) or 0.0)
                    - float(baseline.totals.get("total_ibnr", 0.0) or 0.0),
                    6,
                ),
                "ldf_finding_count": int(
                    ldf_payload.get("summary", {}).get("finding_count", 0) or 0
                ),
                "governance_tier": scenario_eval.governance.get("tier"),
            },
            "continuity_notes": continuity_notes,
            "policy_trace": policy_trace,
        }

    def _build_tail_candidates(
        self,
        *,
        reserving: Reserving,
        baseline_params: dict[str, Any],
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        heuristic = self._heuristic_tail_recommendation(reserving)
        tail_params = heuristic.get("proposed_parameters", {}).get("tail", {})
        curve_candidates = [
            curve
            for curve in tail_params.get(
                "curve_candidates", self._SUPPORTED_TAIL_CURVES
            )
            if curve in self._SUPPORTED_TAIL_CURVES
        ]
        attachment_candidates = self._unique_ints(
            [
                baseline_params.get("tail", {}).get("attachment_age"),
                tail_params.get("recommended_attachment_age"),
                *tail_params.get("attachment_age_candidates", []),
            ]
        )
        fit_period_candidates = [
            list(item)
            for item in tail_params.get("fit_period_candidates", [])
            if isinstance(item, list) and len(item) >= 2
        ]
        baseline_fit = list(baseline_params.get("tail", {}).get("fit_period", []))
        if len(baseline_fit) >= 2:
            fit_period_candidates.insert(0, baseline_fit)
        if not fit_period_candidates:
            fit_period_candidates = self._default_fit_period_candidates(reserving)
        if not attachment_candidates:
            attachment_candidates = self._default_attachment_candidates(reserving)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for curve in curve_candidates:
            for attachment_age in attachment_candidates:
                for fit_period in fit_period_candidates:
                    params = self._clone_params(baseline_params)
                    params["tail"]["curve"] = curve
                    params["tail"]["attachment_age"] = attachment_age
                    params["tail"]["fit_period"] = list(fit_period)
                    key = SegmentMemoryService.scenario_signature(params)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "candidate_id": f"tail_{curve}_{attachment_age}_{fit_period[0]}_{fit_period[-1]}",
                            "summary": f"Test tail {curve} attachment {attachment_age} fit {fit_period[0]}-{fit_period[-1]}",
                            "parameters": params,
                        }
                    )
        return candidates[:candidate_limit]

    def _score_tail_candidate(
        self,
        *,
        segment: str,
        reserving: Reserving,
        baseline: _BaselineContext,
        candidate: dict[str, Any],
        segment_memory: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = self._clone_params(candidate.get("parameters", {}))
        scenario_eval = self._evaluation_service.evaluate_scenario(
            segment=segment,
            reserving=reserving,
            params=params,
            scenario_id=str(candidate.get("candidate_id")),
            summary=str(candidate.get("summary", "")),
            parent_scenario_id="baseline",
            transform="tail_review",
            rationale_evidence_ids=[],
        )
        totals = self._evaluation_service.scenario_totals_for_params(
            reserving=reserving,
            params=params,
        )
        tail_metrics = self._evaluate_tail_metrics(reserving=reserving, params=params)
        reserve_impact_penalty = self._reserve_impact_penalty(
            baseline_total=float(baseline.totals.get("total_ibnr", 0.0) or 0.0),
            candidate_total=float(totals.get("total_ibnr", 0.0) or 0.0),
        )
        continuity_notes, policy_trace = self._candidate_policy_trace(
            params=params,
            segment_memory=segment_memory,
            governance_tier=scenario_eval.governance.get("tier"),
            review_type="tail_review",
            reserve_context=tail_metrics,
        )
        fit_quality = self._tail_fit_quality(tail_metrics)
        continuity_score = self._tail_continuity_score(
            tail_metrics=tail_metrics,
            segment_memory=segment_memory,
        )
        stability_score = self._tail_stability_score(tail_metrics)
        reserve_reasonableness = max(0.0, 1.0 - reserve_impact_penalty)
        score_breakdown = self._scoring_service.score_tail_candidate(
            fit_quality=fit_quality,
            continuity_score=continuity_score,
            stability_score=stability_score,
            reserve_reasonableness=reserve_reasonableness,
            continuity_gap_penalty=min(
                float(tail_metrics.get("attachment_gap_ratio") or 0.0) * 2.0, 1.0
            ),
            subunit_penalty=min(
                len(tail_metrics.get("late_subunit_observed_ages", [])) * 0.15, 0.6
            ),
            instability_penalty=self._tail_instability_penalty(tail_metrics),
            governance_penalty=self._governance_penalty(
                scenario_eval.governance.get("tier")
            )
            + self._anomaly_penalty(baseline.anomaly_triage),
            continuity_penalty=sum(
                float(value) for value in policy_trace["applied_penalties"].values()
            ),
        )
        recommendation_class = self._scoring_service.classify_recommendation(
            total_score=float(score_breakdown["score"]),
            governance_tier=scenario_eval.governance.get("tier"),
            rejected_before=bool(policy_trace["rejected_before"]),
        )
        self._evaluation_service.apply_params_to_reserving(
            reserving=reserving,
            params=self._clone_params(baseline.totals.get("parameters", {}))
            if isinstance(baseline.totals.get("parameters"), dict)
            else self._clone_params(candidate.get("parameters", {})),
        )
        return {
            "candidate_id": str(candidate.get("candidate_id")),
            "summary": str(candidate.get("summary", "")),
            "parameters": params,
            "score": score_breakdown["score"],
            "score_breakdown": {
                "components": score_breakdown["components"],
                "penalties": score_breakdown["penalties"],
                "total_score": score_breakdown["score"],
                "formula_version": score_breakdown["formula_version"],
            },
            "recommendation_class": recommendation_class,
            "metrics": {
                **tail_metrics,
                "baseline_total_ibnr": baseline.totals.get("total_ibnr"),
                "candidate_total_ibnr": totals.get("total_ibnr"),
                "ibnr_delta": round(
                    float(totals.get("total_ibnr", 0.0) or 0.0)
                    - float(baseline.totals.get("total_ibnr", 0.0) or 0.0),
                    6,
                ),
                "governance_tier": scenario_eval.governance.get("tier"),
            },
            "continuity_notes": continuity_notes,
            "policy_trace": policy_trace,
        }

    def _heuristic_tail_recommendation(self, reserving: Reserving) -> dict[str, Any]:
        heatmap_data = reserving.get_triangle_heatmap_data()
        recommendation = self._diagnostics_service._tail_recommendation(heatmap_data)
        if recommendation is None:
            return {}
        return {
            "code": recommendation.code,
            "message": recommendation.message,
            "rationale": recommendation.rationale,
            "evidence": recommendation.evidence,
            "proposed_parameters": recommendation.proposed_parameters,
        }

    def _evaluate_tail_metrics(
        self,
        *,
        reserving: Reserving,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self._evaluation_service.apply_params_to_reserving(
            reserving=reserving, params=params
        )
        heatmap_data = reserving.get_triangle_heatmap_data()
        link_ratios_raw = heatmap_data.get("link_ratios")
        if not isinstance(link_ratios_raw, pd.DataFrame) or link_ratios_raw.empty:
            return {}
        observed_row = link_ratios_raw.loc[link_ratios_raw.index.astype(str) == "LDF"]
        fitted_row = link_ratios_raw.loc[link_ratios_raw.index.astype(str) == "Tail"]
        if observed_row.empty or fitted_row.empty:
            return {}

        fit_period = self._normalize_fit_period(
            params.get("tail", {}).get("fit_period", [])
        )
        residuals: list[float] = []
        observed_values: list[float] = []
        fitted_values: list[float] = []
        observed_by_age: dict[int, float] = {}
        fitted_by_age: dict[int, float] = {}
        for col in link_ratios_raw.columns:
            age = Reserving._parse_cdf_label_to_age(col)
            if age is None:
                continue
            observed = self._to_optional_float(observed_row.iloc[0].get(col))
            fitted = self._to_optional_float(fitted_row.iloc[0].get(col))
            if observed is not None:
                observed_by_age[age] = observed
            if fitted is not None:
                fitted_by_age[age] = fitted
            if fit_period is not None:
                start, end = fit_period
                if age < start:
                    continue
                if end is not None and age > end:
                    continue
            if observed is None or fitted is None:
                continue
            observed_values.append(observed)
            fitted_values.append(fitted)
            residuals.append(fitted - observed)

        rmse = None
        r2 = None
        if observed_values:
            observed_series = pd.Series(observed_values, dtype=float)
            fitted_series = pd.Series(fitted_values, dtype=float)
            rmse = float((((fitted_series - observed_series) ** 2).mean()) ** 0.5)
            variance = float(((observed_series - observed_series.mean()) ** 2).sum())
            if variance > 0:
                r2 = float(
                    1.0 - (((observed_series - fitted_series) ** 2).sum() / variance)
                )

        attachment_age = params.get("tail", {}).get("attachment_age")
        attachment_previous_age = None
        attachment_previous_ldf = None
        attachment_first_fitted_ldf = None
        attachment_gap_ratio = None
        late_subunit_observed_ages: list[int] = []
        if attachment_age is not None:
            prior_ages = [age for age in observed_by_age if age < int(attachment_age)]
            if prior_ages:
                attachment_previous_age = max(prior_ages)
                attachment_previous_ldf = observed_by_age.get(attachment_previous_age)
            attachment_first_fitted_ldf = fitted_by_age.get(int(attachment_age))
            if (
                attachment_previous_ldf is not None
                and attachment_previous_ldf > 0
                and attachment_first_fitted_ldf is not None
            ):
                attachment_gap_ratio = (
                    max(attachment_previous_ldf - attachment_first_fitted_ldf, 0.0)
                    / attachment_previous_ldf
                )
            late_subunit_observed_ages = [
                age
                for age, value in sorted(observed_by_age.items())
                if age >= int(attachment_age) and value < 1.0
            ]
        return {
            "tail_curve": params.get("tail", {}).get("curve"),
            "fit_period": list(params.get("tail", {}).get("fit_period", [])),
            "attachment_age": attachment_age,
            "r2": round(r2, 6) if r2 is not None else None,
            "rmse": round(rmse, 6) if rmse is not None else None,
            "point_count": len(observed_values),
            "attachment_previous_age": attachment_previous_age,
            "attachment_previous_ldf": round(attachment_previous_ldf, 6)
            if attachment_previous_ldf is not None
            else None,
            "attachment_first_fitted_ldf": round(attachment_first_fitted_ldf, 6)
            if attachment_first_fitted_ldf is not None
            else None,
            "attachment_gap_ratio": round(attachment_gap_ratio, 6)
            if attachment_gap_ratio is not None
            else None,
            "late_subunit_observed_ages": late_subunit_observed_ages,
            "residual_sample": [round(value, 6) for value in residuals[:5]],
        }

    def _classify_diagnostic_findings(
        self, findings: list[Any]
    ) -> list[dict[str, Any]]:
        triaged: list[dict[str, Any]] = []
        for finding in findings:
            code = str(getattr(finding, "code", "") or "")
            severity = str(getattr(finding, "severity", "low") or "low")
            message = str(getattr(finding, "message", "") or "")
            evidence = getattr(finding, "evidence", {}) or {}
            classification = self._diagnostic_classification(code)
            if classification is None:
                continue
            triaged.append(
                {
                    "type": classification["type"],
                    "source": "diagnostics",
                    "code": code,
                    "severity": severity,
                    "message": message,
                    "reserve_relevance": classification["reserve_relevance"],
                    "next_diagnostic": classification["next_diagnostic"],
                    "pause_recommendation": classification["pause_recommendation"],
                    "evidence": self._evidence_payload(evidence),
                }
            )
        return triaged

    def _classify_movement_findings(
        self, findings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        triaged: list[dict[str, Any]] = []
        for finding in findings:
            code = str(finding.get("code", ""))
            classification = self._movement_classification(code)
            if classification is None:
                continue
            triaged.append(
                {
                    "type": classification["type"],
                    "source": "movement",
                    "code": code,
                    "severity": str(finding.get("severity", "low")),
                    "message": str(finding.get("message", "")),
                    "reserve_relevance": classification["reserve_relevance"],
                    "next_diagnostic": classification["next_diagnostic"],
                    "pause_recommendation": classification["pause_recommendation"],
                    "evidence": dict(finding.get("evidence", {}))
                    if isinstance(finding.get("evidence"), dict)
                    else {},
                }
            )
        return triaged

    def _classify_ldf_findings(
        self, findings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        triaged: list[dict[str, Any]] = []
        for finding in findings[:10]:
            triaged.append(
                {
                    "type": "case_reserve_shift",
                    "source": "ldf_consistency",
                    "code": f"LDF_CONSISTENCY_{finding.get('origin', '')}_{finding.get('age', '')}",
                    "severity": "medium",
                    "message": str(finding.get("message", "")),
                    "reserve_relevance": "medium",
                    "next_diagnostic": "Review reserve change attribution for the affected AY and development age.",
                    "pause_recommendation": False,
                    "evidence": dict(finding),
                }
            )
        return triaged

    def _dedupe_triaged_findings(
        self, findings: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for item in findings:
            key = f"{item.get('type')}|{item.get('code')}"
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        severity_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        ordered.sort(
            key=lambda item: severity_rank.get(
                str(item.get("severity", "low")).lower(), 0
            ),
            reverse=True,
        )
        return ordered

    def _drop_recommendation_map(
        self, baseline_eval: Any
    ) -> dict[tuple[str, int], dict[str, Any]]:
        mapped: dict[tuple[str, int], dict[str, Any]] = {}
        for rec in getattr(baseline_eval, "recommendations", []):
            code = str(getattr(rec, "code", ""))
            if not code.startswith("RECOMMEND_DROP_"):
                continue
            proposed = getattr(rec, "proposed_parameters", {}) or {}
            for item in proposed.get("drop", []):
                pair = self._normalize_drop_pair(item)
                if pair is None:
                    continue
                mapped[pair] = {
                    "code": code,
                    "message": getattr(rec, "message", ""),
                    "rationale": getattr(rec, "rationale", ""),
                    "priority": getattr(rec, "priority", "medium"),
                    "evidence_value": self._evidence_value(
                        getattr(rec, "evidence", {})
                    ),
                }
        return dict(
            sorted(
                mapped.items(),
                key=lambda item: item[1].get("evidence_value", 0.0),
                reverse=True,
            )
        )

    def _drop_outlier_support(
        self, *, baseline_eval: Any, drop_pairs: list[tuple[str, int]]
    ) -> float:
        recommendation_map = self._drop_recommendation_map(baseline_eval)
        if not drop_pairs:
            return 0.0
        supports: list[float] = []
        for pair in drop_pairs:
            evidence_value = float(
                recommendation_map.get(pair, {}).get("evidence_value", 0.0) or 0.0
            )
            supports.append(
                min(evidence_value / 6.0, 1.0) if evidence_value > 0 else 0.1
            )
        return round(sum(supports) / len(supports), 4)

    def _consistency_improvement(
        self,
        *,
        baseline_payload: dict[str, Any],
        candidate_payload: dict[str, Any],
    ) -> float:
        baseline_findings = (
            baseline_payload.get("findings", [])
            if isinstance(baseline_payload, dict)
            else []
        )
        candidate_findings = (
            candidate_payload.get("findings", [])
            if isinstance(candidate_payload, dict)
            else []
        )
        baseline_count = max(len(baseline_findings), 1)
        candidate_count = len(candidate_findings)
        count_improvement = max(baseline_count - candidate_count, 0) / baseline_count
        baseline_impact = self._impact_sum(baseline_findings)
        candidate_impact = self._impact_sum(candidate_findings)
        impact_improvement = max(baseline_impact - candidate_impact, 0.0) / max(
            baseline_impact, 1.0
        )
        return round(0.6 * count_improvement + 0.4 * impact_improvement, 4)

    def _late_emergence_support(
        self,
        *,
        baseline_payload: dict[str, Any],
        drop_pairs: list[tuple[str, int]],
    ) -> float:
        rows = (
            baseline_payload.get("rows", [])
            if isinstance(baseline_payload, dict)
            else []
        )
        by_origin = {
            str(item.get("origin")): item
            for item in rows
            if isinstance(item, dict) and str(item.get("origin", "")).strip()
        }
        if not drop_pairs:
            return 0.0
        supports: list[float] = []
        for origin, _age in drop_pairs:
            row = by_origin.get(origin)
            if not isinstance(row, dict):
                supports.append(0.15)
                continue
            selected = self._to_optional_float(row.get("selected_future_ratio"))
            median_value = self._to_optional_float(row.get("median"))
            p75 = self._to_optional_float(row.get("p75"))
            if selected is None or median_value is None:
                supports.append(0.15)
                continue
            if p75 is not None and selected >= p75:
                supports.append(1.0)
            elif selected > median_value:
                supports.append(0.65)
            else:
                supports.append(0.2)
        return round(sum(supports) / len(supports), 4)

    def _drop_fragility_penalty(
        self,
        *,
        consistency_improvement: float,
        reserve_impact_penalty: float,
        drop_count: int,
    ) -> float:
        penalty = 0.0
        if consistency_improvement < 0.1:
            penalty += 0.25
        if reserve_impact_penalty > 0.25 and consistency_improvement < 0.25:
            penalty += 0.25
        if drop_count > 1:
            penalty += min((drop_count - 1) * 0.1, 0.3)
        return round(penalty, 4)

    def _reserve_impact_penalty(
        self, *, baseline_total: float, candidate_total: float
    ) -> float:
        delta = abs(candidate_total - baseline_total)
        return round(min(delta / max(abs(baseline_total), 1.0), 1.0), 4)

    def _tail_fit_quality(self, tail_metrics: dict[str, Any]) -> float:
        r2 = self._to_optional_float(tail_metrics.get("r2"))
        rmse = self._to_optional_float(tail_metrics.get("rmse"))
        point_count = int(tail_metrics.get("point_count", 0) or 0)
        quality = 0.0
        if r2 is not None:
            quality += max(min(r2, 1.0), -1.0) * 0.7
        if rmse is not None:
            quality += max(0.0, 0.3 - min(rmse, 0.3))
        if point_count >= 3:
            quality += 0.15
        return round(max(quality, 0.0), 4)

    def _tail_continuity_score(
        self,
        *,
        tail_metrics: dict[str, Any],
        segment_memory: dict[str, Any] | None,
    ) -> float:
        gap_ratio = (
            self._to_optional_float(tail_metrics.get("attachment_gap_ratio")) or 0.0
        )
        score = max(0.0, 0.5 - gap_ratio)
        prior_tail = {}
        if isinstance(segment_memory, dict):
            prior_tail = (
                segment_memory.get("last_selection", {}).get("tail", {})
                if isinstance(segment_memory.get("last_selection"), dict)
                else {}
            )
        prior_attachment = self._to_optional_float(prior_tail.get("attachment_age"))
        current_attachment = self._to_optional_float(tail_metrics.get("attachment_age"))
        if prior_attachment is not None and current_attachment is not None:
            delta = abs(current_attachment - prior_attachment)
            score += max(0.0, 0.35 - min(delta / 48.0, 0.35))
        return round(max(score, 0.0), 4)

    def _tail_stability_score(self, tail_metrics: dict[str, Any]) -> float:
        score = 0.5
        late_subunit = tail_metrics.get("late_subunit_observed_ages", [])
        if isinstance(late_subunit, list) and late_subunit:
            score -= min(len(late_subunit) * 0.15, 0.4)
        rmse = self._to_optional_float(tail_metrics.get("rmse"))
        if rmse is not None and rmse <= 0.05:
            score += 0.2
        return round(max(score, 0.0), 4)

    def _tail_instability_penalty(self, tail_metrics: dict[str, Any]) -> float:
        penalty = 0.0
        gap_ratio = (
            self._to_optional_float(tail_metrics.get("attachment_gap_ratio")) or 0.0
        )
        if gap_ratio > 0.1:
            penalty += min(gap_ratio * 2.0, 0.6)
        late_subunit = tail_metrics.get("late_subunit_observed_ages", [])
        if isinstance(late_subunit, list):
            penalty += min(len(late_subunit) * 0.1, 0.4)
        return round(penalty, 4)

    def _candidate_policy_trace(
        self,
        *,
        params: dict[str, Any],
        segment_memory: dict[str, Any] | None,
        governance_tier: object,
        review_type: str,
        reserve_context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        scenario_signature = self._segment_memory_service.scenario_signature(params)
        dispositions = (
            segment_memory.get("scenario_dispositions", [])
            if isinstance(segment_memory, dict)
            else []
        )
        rejected_signatures = []
        for item in dispositions:
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision", "")).strip().lower()
            signature = str(
                item.get("scenario_signature", "") or item.get("scenario_hash", "")
            ).strip()
            if decision == "rejected" and signature:
                rejected_signatures.append(signature)

        notes: list[dict[str, Any]] = []
        penalties: dict[str, float] = {}
        if scenario_signature in set(rejected_signatures):
            notes.append(
                {
                    "code": "rejected_before",
                    "severity": "high",
                    "message": "This scenario signature was previously rejected for the segment.",
                    "context": {"scenario_signature": scenario_signature},
                }
            )
            penalties["rejected_before"] = 0.45

        conflicts = self._house_preference_conflicts(
            preferences=(
                segment_memory.get("house_preferences", [])
                if isinstance(segment_memory, dict)
                else []
            ),
            review_type=review_type,
            reserve_context=reserve_context,
        )
        for conflict in conflicts:
            notes.append(
                {
                    "code": "house_preference_conflict",
                    "severity": "medium",
                    "message": conflict,
                    "context": {},
                }
            )
        if conflicts:
            penalties["house_preferences"] = min(len(conflicts) * 0.15, 0.45)

        return notes, {
            "rejected_before": scenario_signature in set(rejected_signatures),
            "rejected_signatures": rejected_signatures[:5],
            "house_preference_conflicts": conflicts,
            "applied_penalties": penalties,
            "governance_tier": str(governance_tier or "").strip().lower() or None,
        }

    def _house_preference_conflicts(
        self,
        *,
        preferences: object,
        review_type: str,
        reserve_context: dict[str, Any],
    ) -> list[str]:
        if not isinstance(preferences, list):
            return []
        conflicts: list[str] = []
        for item in preferences:
            if isinstance(item, dict):
                pref_type = str(item.get("type", "")).strip().lower()
                if review_type == "drop_review" and pref_type == "max_drop_count":
                    limit = self._to_optional_float(item.get("value"))
                    if (
                        limit is not None
                        and float(reserve_context.get("drop_count", 0) or 0) > limit
                    ):
                        conflicts.append(
                            f"Candidate exceeds house preference max_drop_count={int(limit)}."
                        )
                if review_type == "tail_review" and pref_type == "prefer_stable_tail":
                    late_subunit = reserve_context.get("late_subunit_observed_ages", [])
                    gap_ratio = (
                        self._to_optional_float(
                            reserve_context.get("attachment_gap_ratio")
                        )
                        or 0.0
                    )
                    if (
                        isinstance(late_subunit, list) and late_subunit
                    ) or gap_ratio > 0.1:
                        conflicts.append(
                            "Candidate conflicts with house preference to prefer stable tail behavior."
                        )
                if (
                    review_type == "tail_review"
                    and pref_type == "max_attachment_gap_ratio"
                ):
                    limit = self._to_optional_float(item.get("value"))
                    gap_ratio = (
                        self._to_optional_float(
                            reserve_context.get("attachment_gap_ratio")
                        )
                        or 0.0
                    )
                    if limit is not None and gap_ratio > limit:
                        conflicts.append(
                            f"Candidate attachment gap ratio {gap_ratio:.2f} exceeds house limit {limit:.2f}."
                        )
                continue
            text = str(item).strip().lower()
            if not text:
                continue
            if review_type == "drop_review":
                match = re.search(r"avoid more than\s+(\d+)\s+dropped?", text)
                if match and int(match.group(1)) < int(
                    reserve_context.get("drop_count", 0) or 0
                ):
                    conflicts.append(
                        f"Candidate exceeds house preference to avoid more than {match.group(1)} drops."
                    )
            if review_type == "tail_review" and "prefer stable tail" in text:
                late_subunit = reserve_context.get("late_subunit_observed_ages", [])
                gap_ratio = (
                    self._to_optional_float(reserve_context.get("attachment_gap_ratio"))
                    or 0.0
                )
                if (isinstance(late_subunit, list) and late_subunit) or gap_ratio > 0.1:
                    conflicts.append(
                        "Candidate conflicts with house preference to prefer stable tail behavior."
                    )
        return conflicts

    def _bf_continuity_context(
        self,
        *,
        segment_memory: dict[str, Any] | None,
        overall_class: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        policy_trace = {
            "rejected_before": False,
            "rejected_signatures": [],
            "house_preference_conflicts": [],
            "applied_penalties": {},
            "governance_tier": None,
        }
        if not isinstance(segment_memory, dict):
            return notes, policy_trace
        last_selection = segment_memory.get("last_selection", {})
        method_by_uwy = (
            last_selection.get("method_by_uwy", {})
            if isinstance(last_selection, dict)
            else {}
        )
        if isinstance(method_by_uwy, dict):
            chosen = {
                str(value) for value in method_by_uwy.values() if str(value).strip()
            }
            if chosen == {"chainladder"} and overall_class == "bf_preferred":
                notes.append(
                    {
                        "code": "prior_selection_tension",
                        "severity": "medium",
                        "message": "Current BF suitability assessment conflicts with the prior all-CL segment selection.",
                        "context": {"prior_methods": list(chosen)},
                    }
                )
                policy_trace["house_preference_conflicts"].append(
                    "Current BF suitability differs from the prior all-CL selection pattern."
                )
        return notes, policy_trace

    def _build_candidate_recommendation(
        self,
        best: dict[str, Any] | None,
        ordered: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(best, dict):
            return {
                "recommendation_class": "avoid",
                "candidate_id": None,
                "summary": "No tested candidates were available.",
                "caveats": ["missing_candidates"],
                "alternatives": [],
            }
        recommendation_class = str(best.get("recommendation_class", "watch"))
        alternatives = [
            str(item.get("candidate_id"))
            for item in ordered[1:3]
            if isinstance(item, dict) and str(item.get("candidate_id", "")).strip()
        ]
        summary_by_class = {
            "recommend": "Tested evidence supports adopting the top-ranked candidate.",
            "reasonable_alternative": "The top-ranked candidate improves the review, but caveats remain material.",
            "watch": "The top-ranked candidate is a sensitivity worth watching, not a clear selection change.",
            "avoid": "The tested candidate does not clear governance and continuity hurdles for recommendation.",
        }
        caveats = [
            str(item.get("message"))
            for item in best.get("continuity_notes", [])
            if isinstance(item, dict)
        ]
        return {
            "recommendation_class": recommendation_class,
            "candidate_id": best.get("candidate_id"),
            "summary": summary_by_class.get(
                recommendation_class, summary_by_class["watch"]
            ),
            "caveats": caveats,
            "alternatives": alternatives,
        }

    @staticmethod
    def _diagnostic_classification(code: str) -> dict[str, Any] | None:
        if code == "DATA_QUALITY_GATE":
            return {
                "type": "data_quality",
                "reserve_relevance": "high",
                "next_diagnostic": "Resolve missing, non-monotonic, or non-positive premium cells before changing assumptions.",
                "pause_recommendation": True,
            }
        if code == "CALENDAR_YEAR_DRIFT" or "PORTFOLIO_SHIFT" in code:
            return {
                "type": "calendar_distortion",
                "reserve_relevance": "high",
                "next_diagnostic": "Compare calendar diagonals and confirm whether mix or handling changed.",
                "pause_recommendation": True,
            }
        if code == "NEGATIVE_DEVELOPMENT_TRIAGE":
            return {
                "type": "case_reserve_shift",
                "reserve_relevance": "high",
                "next_diagnostic": "Reconcile paid and incurred changes by AY and latest valuation movement.",
                "pause_recommendation": True,
            }
        if code.startswith("TAIL_SENSITIVITY"):
            return {
                "type": "sparse_maturity",
                "reserve_relevance": "medium",
                "next_diagnostic": "Retest tail attachment and fit windows with multiple curves.",
                "pause_recommendation": False,
            }
        if code.startswith("LATEST_DIAGONAL_DEVIATION"):
            return {
                "type": "case_reserve_shift",
                "reserve_relevance": "medium",
                "next_diagnostic": "Explain latest diagonal movement before changing development assumptions.",
                "pause_recommendation": False,
            }
        return None

    @staticmethod
    def _movement_classification(code: str) -> dict[str, Any] | None:
        if code.startswith("LARGE_LOSS_PROXY"):
            return {
                "type": "large_loss_contamination",
                "reserve_relevance": "high",
                "next_diagnostic": "Run a large-loss segmented view before acting on aggregate development outliers.",
                "pause_recommendation": False,
            }
        if code.startswith("PREMIUM_LATE_MOVEMENT"):
            return {
                "type": "segment_definition_change",
                "reserve_relevance": "medium",
                "next_diagnostic": "Confirm whether premium timing or segment composition changed.",
                "pause_recommendation": False,
            }
        if code.startswith("OUTSTANDING_CONCENTRATION"):
            return {
                "type": "case_reserve_shift",
                "reserve_relevance": "medium",
                "next_diagnostic": "Check case adequacy and settlement speed for the affected AY.",
                "pause_recommendation": False,
            }
        if code.startswith("INCURRED_SPIKE"):
            return {
                "type": "large_loss_contamination",
                "reserve_relevance": "medium",
                "next_diagnostic": "Inspect claim-level drivers for the spike period before dropping link ratios.",
                "pause_recommendation": False,
            }
        return None

    @staticmethod
    def _impact_sum(findings: list[dict[str, Any]]) -> float:
        total = 0.0
        for item in findings:
            if not isinstance(item, dict):
                continue
            total += abs(float(item.get("impact_estimate", 0.0) or 0.0))
        return total

    @staticmethod
    def _class_counts(rows: list[dict[str, Any]], *, key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key, "") or "").strip()
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        return counts

    @staticmethod
    def _overall_bf_class(rows: list[dict[str, Any]]) -> str:
        classes = [
            str(item.get("suitability_class", "inconclusive"))
            for item in rows
            if isinstance(item, dict)
        ]
        if not classes:
            return "inconclusive"
        unique = set(classes)
        if unique == {"bf_preferred"}:
            return "bf_preferred"
        if unique == {"cl_preferred"}:
            return "cl_preferred"
        if "bf_preferred" in unique and "cl_preferred" in unique:
            return "mixed"
        if "bf_preferred" in unique:
            return "mixed"
        if "cl_preferred" in unique:
            return "cl_preferred"
        return "inconclusive"

    def _bf_apriori_guidance(self, baseline_eval: Any) -> dict[str, Any]:
        for rec in getattr(baseline_eval, "recommendations", []):
            if str(getattr(rec, "code", "")) != "RECOMMEND_BF_APRIORI":
                continue
            return {
                "available": True,
                "message": getattr(rec, "message", ""),
                "rationale": getattr(rec, "rationale", ""),
                "bf_apriori": dict(
                    getattr(rec, "proposed_parameters", {}).get("bf_apriori", {})
                ),
            }
        return {"available": False, "bf_apriori": {}}

    def _diagnostics_flags_by_uwy(self, findings: list[Any]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for finding in findings:
            code = str(getattr(finding, "code", "") or "")
            match = re.search(r"_(\d{4})(?:_|$)", code)
            if not match:
                continue
            grouped.setdefault(match.group(1), []).append(code)
        return grouped

    @staticmethod
    def _percent_reported_support(percent_reported: float) -> float:
        if percent_reported <= 0 or percent_reported > 1.25:
            return -0.25
        if percent_reported < 0.45:
            return 0.15
        if percent_reported < 0.75:
            return 0.05
        return -0.1

    @staticmethod
    def _maturity_by_uwy(results_df: pd.DataFrame | None) -> dict[str, float]:
        if results_df is None or len(results_df) == 0:
            return {}
        maturity: dict[str, float] = {}
        for idx, row in results_df.iterrows():
            uwy = AssumptionReviewService._uwy_label(idx)
            incurred = float(row.get("incurred", 0.0) or 0.0)
            ultimate = float(row.get("ultimate", 0.0) or 0.0)
            if ultimate <= 0:
                maturity[uwy] = 0.0
                continue
            maturity[uwy] = round(min(max(incurred / ultimate, 0.0), 1.0), 4)
        return maturity

    def _anomaly_penalty(self, anomaly_triage: dict[str, Any]) -> float:
        triaged = (
            anomaly_triage.get("triaged_findings", [])
            if isinstance(anomaly_triage, dict)
            else []
        )
        if anomaly_triage.get("pause_recommendation"):
            return 0.35
        high_relevance = [
            item
            for item in triaged
            if isinstance(item, dict) and item.get("reserve_relevance") == "high"
        ]
        return round(min(len(high_relevance) * 0.05, 0.2), 4)

    @staticmethod
    def _governance_penalty(governance_tier: object) -> float:
        tier = str(governance_tier or "").strip().lower()
        if tier == "red":
            return 0.45
        if tier == "amber":
            return 0.15
        return 0.0

    @staticmethod
    def _run_metadata(evaluation: Any) -> dict[str, Any]:
        run_metadata = getattr(evaluation, "run_metadata", None)
        if hasattr(run_metadata, "model_dump"):
            return run_metadata.model_dump(mode="json")
        if isinstance(run_metadata, dict):
            return dict(run_metadata)
        return {}

    @staticmethod
    def _default_fit_period_candidates(reserving: Reserving) -> list[list[int]]:
        ages = AssumptionReviewService._available_ages(reserving)
        if len(ages) < 2:
            return []
        if len(ages) == 2:
            return [[ages[0], ages[-1]]]
        return [[ages[-2], ages[-1]], [ages[-3], ages[-1]]]

    @staticmethod
    def _default_attachment_candidates(reserving: Reserving) -> list[int]:
        ages = AssumptionReviewService._available_ages(reserving)
        return ages[-3:] if len(ages) >= 3 else ages

    @staticmethod
    def _available_ages(reserving: Reserving) -> list[int]:
        try:
            heatmap_data = reserving.get_triangle_heatmap_data()
            link_ratios = heatmap_data.get("link_ratios")
        except Exception:
            return []
        if not isinstance(link_ratios, pd.DataFrame) or link_ratios.empty:
            return []
        ages = [Reserving._parse_cdf_label_to_age(col) for col in link_ratios.columns]
        return [age for age in ages if age is not None]

    @staticmethod
    def _unique_ints(values: list[object]) -> list[int]:
        normalized: list[int] = []
        for value in values:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                continue
            if candidate not in normalized:
                normalized.append(candidate)
        return normalized

    @staticmethod
    def _normalize_fit_period(
        raw_fit_period: list[int],
    ) -> tuple[int, int | None] | None:
        if not raw_fit_period:
            return None
        normalized = sorted({int(value) for value in raw_fit_period})
        if len(normalized) == 1:
            return (normalized[0], None)
        return (normalized[0], normalized[-1])

    @staticmethod
    def _normalize_drop_pair(item: object) -> tuple[str, int] | None:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            return (str(item[0]), int(item[1]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clone_params(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "average": Reserving._normalize_average(params.get("average", "volume")),
            "drop": [list(item) for item in params.get("drop", [])],
            "drop_valuation": [list(item) for item in params.get("drop_valuation", [])],
            "tail": {
                "curve": str(params.get("tail", {}).get("curve", "weibull")),
                "attachment_age": params.get("tail", {}).get("attachment_age"),
                "projection_period": int(
                    params.get("tail", {}).get("projection_period", 0) or 0
                ),
                "fit_period": list(params.get("tail", {}).get("fit_period", [])),
            },
            "bf_apriori": dict(params.get("bf_apriori", {})),
            "final_ultimate": str(params.get("final_ultimate", "chainladder")),
            "selected_ultimate_by_uwy": dict(
                params.get("selected_ultimate_by_uwy", {})
            ),
        }

    @staticmethod
    def _to_optional_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _evidence_value(value: object) -> float:
        if hasattr(value, "value"):
            try:
                return float(getattr(value, "value"))
            except (TypeError, ValueError):
                return 0.0
        if isinstance(value, dict):
            try:
                return float(value.get("value", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _evidence_payload(value: object) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _uwy_label(origin: object) -> str:
        if hasattr(origin, "year"):
            return str(origin.year)
        text = str(origin)
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else text
