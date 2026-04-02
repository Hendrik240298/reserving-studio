from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import os
import threading
import time
from typing import Any, Literal, cast
import uuid

import numpy as np
import pandas as pd

from source.app import build_workflow_from_dataframes, load_config
from source.api.schemas import (
    AssumptionDetailRequest,
    AssumptionDetailResponse,
    AnomalyTriageRequest,
    AnomalyTriageResponse,
    BfSuitabilityRequest,
    BfSuitabilityResponse,
    DataCompareRequest,
    DataCompareResponse,
    DataViewRequest,
    DataViewResponse,
    DerivedDropScenarioRequest,
    DerivedDropScenarioResponse,
    DropReviewRequest,
    DropReviewResponse,
    DiagnosticFinding,
    DiagnosticRecommendation,
    DiagnosticEvidence,
    DiagnosticsIterateRequest,
    DiagnosticsIterateResponse,
    DiagnosticsRequest,
    DiagnosticsResponse,
    HighestA2ADropRequest,
    HighestA2ADropResponse,
    LateEmergenceRequest,
    LateEmergenceResponse,
    LinkRatioRankRequest,
    LinkRatioRankResponse,
    LdfConsistencyRequest,
    LdfConsistencyResponse,
    MovementDiagnosticsRequest,
    MovementDiagnosticsResponse,
    ParamsStore,
    QuarterClosePackRequest,
    QuarterClosePackResponse,
    QuarterCloseReviewRequest,
    QuarterCloseReviewResponse,
    RecalculateRequest,
    RecalculateResponse,
    ReserveChangeRequest,
    ReserveChangeResponse,
    TailEvaluationRequest,
    TailEvaluationResponse,
    ResultsResponse,
    RunMetadata,
    ScenarioEvaluation,
    ResultsStoreMeta,
    SessionSaveRequest,
    SessionSaveResponse,
    SessionStateResponse,
    TailReviewRequest,
    TailReviewResponse,
    WorkflowFromDataframesRequest,
    WorkflowInitializationResponse,
)
from source.config_manager import ConfigManager
from source.reserving import Reserving
from source.services.assumption_review_service import AssumptionReviewService
from source.services.data_view_service import (
    DataViewQuery as ServiceDataViewQuery,
    DataViewService,
    serialize_dataframe,
)
from source.services.diagnostics_service import DiagnosticsService
from source.services.movement_diagnostics_service import MovementDiagnosticsService
from source.services.scenario_evaluation_service import ScenarioEvaluationService
from source.services.scenario_scoring_service import ScenarioScoringService
from source.services.segment_memory_service import SegmentMemoryService
from source.services.quarter_close_service import QuarterCloseService
from source.services.uncertainty_service import UncertaintyService
from source.services.valuation_snapshot_service import ValuationSnapshotService


logger = logging.getLogger(__name__)


class SessionConflictError(ValueError):
    pass


@dataclass
class SessionContext:
    session_id: str
    segment: str
    reserving: Reserving
    sync_version: int
    params_store: ParamsStore
    results_store_meta: ResultsStoreMeta
    last_results_payload: dict
    source_claims_rows: list[dict[str, Any]] = field(default_factory=list)
    source_premium_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioCandidate:
    scenario_id: str
    params: dict[str, Any]
    summary: str
    parent_scenario_id: str | None
    transform: str
    rationale_evidence_ids: list[str]


class InMemoryReservingBackend:
    SCENARIO_GENERATOR_VERSION = "v1.2"
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

    def __init__(self) -> None:
        self._config: ConfigManager | None = self._load_config()
        self._lock = threading.RLock()
        self._sessions_by_id: dict[str, SessionContext] = {}
        self._sessions_by_segment: dict[str, str] = {}
        self._observability_enabled = os.environ.get(
            "RESERVING_OBSERVABILITY", "1"
        ).strip().lower() not in {"0", "false", "off"}
        self._diagnostics_service = DiagnosticsService()
        self._uncertainty_service = UncertaintyService()
        self._scenario_scoring_service = ScenarioScoringService()
        self._scenario_evaluation_service = ScenarioEvaluationService(
            diagnostics_service=self._diagnostics_service,
            uncertainty_service=self._uncertainty_service,
            scoring_service=self._scenario_scoring_service,
            scenario_generator_version=self.SCENARIO_GENERATOR_VERSION,
        )
        self._assumption_review_service = AssumptionReviewService(
            evaluation_service=self._scenario_evaluation_service,
            scoring_service=self._scenario_scoring_service,
            diagnostics_service=self._diagnostics_service,
        )
        self._valuation_snapshot_service = ValuationSnapshotService(
            evaluation_service=self._scenario_evaluation_service,
        )
        self._quarter_close_service = QuarterCloseService(
            evaluation_service=self._scenario_evaluation_service,
            assumption_review_service=self._assumption_review_service,
            valuation_snapshot_service=self._valuation_snapshot_service,
        )

    def create_workflow_from_dataframes(
        self,
        payload: WorkflowFromDataframesRequest,
    ) -> WorkflowInitializationResponse:
        started = time.perf_counter()
        claims_df = pd.DataFrame(payload.claims_rows)
        premium_df = pd.DataFrame(payload.premium_rows)
        if claims_df.empty:
            raise ValueError("claims_rows must contain at least one row")
        if premium_df.empty:
            raise ValueError("premium_rows must contain at least one row")

        reserving = build_workflow_from_dataframes(
            claims_df,
            premium_df,
            config=self._config,
        )

        with self._lock:
            session_id = self._new_session_id(payload.segment)
            params_store = self._default_params_store()
            results_payload = self._build_results_payload(reserving)
            context = SessionContext(
                session_id=session_id,
                segment=payload.segment,
                reserving=reserving,
                sync_version=0,
                params_store=params_store,
                results_store_meta=ResultsStoreMeta(
                    cache_key=results_payload.get("cache_key"),
                    model_cache_key=results_payload.get("model_cache_key"),
                    figure_version=results_payload.get("figure_version"),
                    sync_version=0,
                ),
                last_results_payload=results_payload,
                source_claims_rows=[
                    dict(item) for item in payload.claims_rows if isinstance(item, dict)
                ],
                source_premium_rows=[
                    dict(item)
                    for item in payload.premium_rows
                    if isinstance(item, dict)
                ],
            )
            self._sessions_by_id[session_id] = context
            self._sessions_by_segment[payload.segment] = session_id

        if self._observability_enabled:
            logger.info(
                "[OBS] workflow.create segment=%s session_id=%s claims_rows=%s premium_rows=%s duration_ms=%s",
                payload.segment,
                session_id,
                len(payload.claims_rows),
                len(payload.premium_rows),
                int((time.perf_counter() - started) * 1000),
            )

        return WorkflowInitializationResponse(
            session_id=session_id,
            segment=payload.segment,
            sync_version=0,
            initial_params=params_store,
            initial_results_summary={
                "row_count": len(results_payload.get("results_table_rows", [])),
                "last_updated": results_payload.get("last_updated", ""),
            },
        )

    def get_session(self, segment: str) -> SessionStateResponse | None:
        with self._lock:
            context = self._get_context_by_segment(segment)
            if context is None:
                return None
            return SessionStateResponse(
                session_id=context.session_id,
                segment=context.segment,
                params_store=context.params_store,
                results_store_meta=context.results_store_meta,
                valuation_context=self._build_valuation_context(context),
                sync_version=context.sync_version,
            )

    def save_session(
        self, segment: str, payload: SessionSaveRequest
    ) -> SessionSaveResponse:
        with self._lock:
            context = self._get_context_by_segment(segment)
            if context is None:
                raise LookupError(f"Segment session not found: {segment}")
            if payload.expected_sync_version != context.sync_version:
                raise SessionConflictError(
                    "Expected sync version does not match current session version"
                )

            next_version = context.sync_version + 1
            context.sync_version = next_version
            context.params_store = payload.params_store
            context.results_store_meta = payload.results_store_meta
            context.results_store_meta.sync_version = next_version

            self._persist_config_session(context)

            return SessionSaveResponse(
                segment=segment,
                sync_version=next_version,
                saved_at=datetime.now(timezone.utc),
            )

    def recalculate(self, payload: RecalculateRequest) -> RecalculateResponse:
        started = time.perf_counter()
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")

            drops = self._normalize_drop_pairs(payload.drop)
            drop_valuation = self._normalize_drop_valuation(payload.drop_valuation)
            fit_period = self._normalize_fit_period(payload.tail.fit_period)
            tail_projection_months = int(payload.tail.projection_period)
            months_per_dev = self._infer_months_per_dev(context.reserving)
            extrap_periods = tail_projection_months // months_per_dev
            projection_period = extrap_periods * months_per_dev
            normalized_average = Reserving._normalize_average(payload.average)

            context.reserving.set_development(
                average=normalized_average,
                drop=drops,
                drop_valuation=drop_valuation,
            )
            context.reserving.set_tail(
                curve=payload.tail.curve,
                attachment_age=payload.tail.attachment_age,
                extrap_periods=extrap_periods,
                projection_period=projection_period,
                fit_period=fit_period,
            )
            if payload.bf_apriori:
                context.reserving.set_bornhuetter_ferguson(apriori=payload.bf_apriori)
            else:
                context.reserving.set_bornhuetter_ferguson(apriori=0.6)
            context.reserving.reserve(
                final_ultimate=payload.final_ultimate,
                selected_ultimate_by_uwy=dict(payload.selected_ultimate_by_uwy),
            )

            context.params_store = ParamsStore(
                request_id=context.params_store.request_id + 1,
                source="api-recalculate",
                force_recalc=False,
                drop_store=payload.drop,
                tail_attachment_age=payload.tail.attachment_age,
                tail_projection_months=tail_projection_months,
                tail_fit_period_selection=payload.tail.fit_period,
                average=normalized_average,
                tail_curve=payload.tail.curve,
                bf_apriori_by_uwy=dict(payload.bf_apriori),
                selected_ultimate_by_uwy=dict(payload.selected_ultimate_by_uwy),
                sync_version=context.sync_version,
            )

            results_payload = self._build_results_payload(context.reserving)
            context.last_results_payload = results_payload
            context.results_store_meta = ResultsStoreMeta(
                cache_key=results_payload.get("cache_key"),
                model_cache_key=results_payload.get("model_cache_key"),
                figure_version=results_payload.get("figure_version"),
                sync_version=context.sync_version,
            )

            response = RecalculateResponse(
                session_id=context.session_id,
                results_table_rows=results_payload.get("results_table_rows", []),
                triangle_figure=results_payload.get("triangle_figure", {}),
                emergence_figure=results_payload.get("emergence_figure", {}),
                heatmap_payload=results_payload.get("heatmap_payload", {}),
                cache_key=results_payload.get("cache_key", ""),
                model_cache_key=results_payload.get("model_cache_key", ""),
                figure_version=results_payload.get("figure_version"),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            if self._observability_enabled:
                logger.info(
                    "[OBS] recalculate session_id=%s drop_count=%s tail_curve=%s duration_ms=%s",
                    context.session_id,
                    len(payload.drop),
                    payload.tail.curve,
                    response.duration_ms,
                )
            return response

    def run_diagnostics(self, payload: DiagnosticsRequest) -> DiagnosticsResponse:
        started = time.perf_counter()
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            results_df = context.reserving.get_results()
            heatmap_data = context.reserving.get_triangle_heatmap_data()
            diagnostics_service, calibration = self._calibrated_diagnostics_service(
                segment=context.segment,
                results_df=results_df,
                heatmap_data=heatmap_data,
            )
            run_result = diagnostics_service.run(
                results_df=results_df,
                heatmap_data=heatmap_data,
            )
            run_metadata = self._build_run_metadata(
                results_df=results_df,
                heatmap_data=heatmap_data,
            )

            mapped_findings = [
                self._map_finding(item, run_metadata=run_metadata)
                for item in run_result.findings
            ]
            mapped_recommendations = [
                self._map_recommendation(item, run_metadata=run_metadata)
                for item in run_result.recommendations
            ]
            if not payload.include_recommendations:
                mapped_recommendations = []

            severity_components = self._severity_components(mapped_findings)
            governance = self._governance_assessment(
                findings=mapped_findings,
                severity_components=severity_components,
            )
            metrics = cast(dict[str, Any], dict(run_result.metrics))
            metrics["severity_components"] = severity_components
            metrics["governance_tier"] = governance["tier"]
            metrics["governance_escalation_triggers"] = governance[
                "escalation_triggers"
            ]
            metrics["governance_requires_human_review"] = governance[
                "requires_human_review"
            ]
            metrics["threshold_calibration"] = calibration
            uncertainty = self._uncertainty_service.baseline_uncertainty(
                results_df=results_df,
                heatmap_data=heatmap_data,
            )
            metrics["uncertainty"] = uncertainty

            response = DiagnosticsResponse(
                session_id=context.session_id,
                findings=mapped_findings,
                recommendations=mapped_recommendations,
                metrics=metrics,
                governance=governance,
                calibration=calibration,
                uncertainty=uncertainty,
                run_metadata=run_metadata,
            )
            if self._observability_enabled:
                logger.info(
                    "[OBS] diagnostics.run session_id=%s findings=%s recommendations=%s severity_score=%s duration_ms=%s",
                    context.session_id,
                    len(response.findings),
                    len(response.recommendations),
                    response.metrics.get("severity_score"),
                    int((time.perf_counter() - started) * 1000),
                )
            return response

    def iterate_diagnostics(
        self,
        payload: DiagnosticsIterateRequest,
    ) -> DiagnosticsIterateResponse:
        started = time.perf_counter()
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")

            if self._observability_enabled:
                logger.info(
                    "[OBS] iterate.start session_id=%s segment=%s max_scenarios=%s",
                    context.session_id,
                    context.segment,
                    payload.max_scenarios,
                )

            baseline_params = self._params_from_store(context)
            baseline_eval = self._evaluate_scenario(
                context=context,
                scenario_id="baseline",
                params=baseline_params,
                summary="Current session configuration",
                parent_scenario_id=None,
                transform="baseline",
                rationale_evidence_ids=[],
            )

            scenario_candidates = self._build_scenario_candidates(
                context=context,
                baseline=baseline_params,
                baseline_eval=baseline_eval,
                max_scenarios=payload.max_scenarios,
            )

            evaluations: list[ScenarioEvaluation] = []
            for candidate in scenario_candidates:
                scenario_started = time.perf_counter()
                evaluations.append(
                    self._evaluate_scenario(
                        context=context,
                        scenario_id=candidate.scenario_id,
                        params=candidate.params,
                        summary=candidate.summary,
                        parent_scenario_id=candidate.parent_scenario_id,
                        transform=candidate.transform,
                        rationale_evidence_ids=candidate.rationale_evidence_ids,
                    )
                )
                if self._observability_enabled:
                    scenario_eval = evaluations[-1]
                    logger.info(
                        "[OBS] iterate.scenario session_id=%s scenario_id=%s score=%.4f findings=%s recommendations=%s duration_ms=%s",
                        context.session_id,
                        scenario_eval.scenario_id,
                        scenario_eval.score,
                        len(scenario_eval.findings),
                        len(scenario_eval.recommendations),
                        int((time.perf_counter() - scenario_started) * 1000),
                    )

            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )

            ordered = sorted(evaluations, key=lambda item: item.score)
            duration_ms = int((time.perf_counter() - started) * 1000)
            best = ordered[0] if ordered else None
            metrics = {
                "scenario_count": len(ordered),
                "duration_ms": duration_ms,
                "baseline_score": baseline_eval.score,
                "best_scenario_id": best.scenario_id if best is not None else None,
                "best_scenario_score": best.score if best is not None else None,
                "diagnostics_version": DiagnosticsService.DIAGNOSTICS_VERSION,
                "scenario_generator_version": self.SCENARIO_GENERATOR_VERSION,
                "best_governance_tier": (
                    best.governance.get("tier") if best is not None else None
                ),
            }

            bootstrap_uncertainty = (
                self._uncertainty_service.bootstrap_predictive_distribution(
                    results_df=context.reserving.get_results(),
                    heatmap_data=context.reserving.get_triangle_heatmap_data(),
                )
            )
            tail_assessment = self._uncertainty_service.tail_model_assessment(
                scenarios=[
                    {
                        "scenario_id": item.scenario_id,
                        "score": item.score,
                        "transform": item.lineage.get("transform", ""),
                        "parameters": item.parameters,
                    }
                    for item in ordered
                ]
            )
            aggregate_uncertainty = {
                "baseline": baseline_eval.uncertainty,
                "bootstrap": bootstrap_uncertainty,
                "tail_model": tail_assessment,
            }
            metrics["uncertainty"] = aggregate_uncertainty
            if self._observability_enabled:
                logger.info(
                    "[OBS] iterate.complete session_id=%s scenarios=%s duration_ms=%s best=%s best_score=%s",
                    context.session_id,
                    metrics["scenario_count"],
                    duration_ms,
                    metrics["best_scenario_id"],
                    metrics["best_scenario_score"],
                )
            return DiagnosticsIterateResponse(
                session_id=context.session_id,
                baseline=baseline_eval if payload.include_baseline else None,
                scenarios=ordered,
                iteration_metrics=metrics,
                governance=best.governance
                if best is not None
                else baseline_eval.governance,
                calibration=best.calibration
                if best is not None
                else baseline_eval.calibration,
                uncertainty=aggregate_uncertainty,
                run_metadata=baseline_eval.run_metadata,
            )

    def get_results(self, session_id: str) -> ResultsResponse | None:
        with self._lock:
            context = self._get_context_by_session_id(session_id)
            if context is None:
                return None
            return ResultsResponse(
                session_id=context.session_id, results=context.last_results_payload
            )

    def get_data_view(self, payload: DataViewRequest) -> DataViewResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            service = DataViewService(context.reserving)
            query = self._service_query(payload.query)
            frame = service.get_data_view(query)
            return DataViewResponse(
                session_id=context.session_id,
                query=payload.query.model_dump(mode="json"),
                data=serialize_dataframe(frame),
                summary=service.summarize_view(query)
                if payload.include_summary
                else {},
            )

    def compare_data_views(self, payload: DataCompareRequest) -> DataCompareResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            service = DataViewService(context.reserving)
            frame, summary = service.compare_views(
                self._service_query(payload.left),
                self._service_query(payload.right),
                comparison_mode=payload.comparison_mode,
            )
            return DataCompareResponse(
                session_id=context.session_id,
                comparison_mode=summary.get("comparison_mode", payload.comparison_mode),
                data=serialize_dataframe(frame),
                summary=summary,
            )

    def run_movement_diagnostics(
        self,
        payload: MovementDiagnosticsRequest,
    ) -> MovementDiagnosticsResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            response = MovementDiagnosticsService(context.reserving).run()
            return MovementDiagnosticsResponse(
                session_id=context.session_id,
                findings=response.get("findings", []),
                summary=response.get("summary", {}),
            )

    def run_ldf_consistency(
        self,
        payload: LdfConsistencyRequest,
    ) -> LdfConsistencyResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            response = MovementDiagnosticsService(
                context.reserving
            ).run_ldf_consistency()
            return LdfConsistencyResponse(
                session_id=context.session_id,
                findings=response.get("findings", []),
                summary=response.get("summary", {}),
            )

    def project_late_emergence(
        self,
        payload: LateEmergenceRequest,
    ) -> LateEmergenceResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            response = MovementDiagnosticsService(
                context.reserving
            ).run_late_emergence_benchmark(uwy=payload.uwy)
            return LateEmergenceResponse(
                session_id=context.session_id,
                rows=response.get("rows", []),
                summary=response.get("summary", {}),
            )

    def explain_reserve_change(
        self,
        payload: ReserveChangeRequest,
    ) -> ReserveChangeResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")

            baseline_params = self._params_from_store(context)
            candidate_params = self._clone_params(payload.model_dump(mode="json"))
            candidate_params.pop("session_id", None)

            baseline_eval = self._scenario_totals_for_params(
                context=context, params=baseline_params
            )
            step_rows: list[dict[str, Any]] = []

            current = self._clone_params(baseline_params)
            development_keys = ["average", "drop", "drop_valuation"]
            for key in development_keys:
                current[key] = candidate_params.get(key, current.get(key))
            dev_eval = self._scenario_totals_for_params(context=context, params=current)
            step_rows.append(
                self._build_attribution_row(
                    component="development",
                    previous=baseline_eval,
                    current=dev_eval,
                )
            )

            current["tail"] = dict(
                candidate_params.get("tail", current.get("tail", {}))
            )
            tail_eval = self._scenario_totals_for_params(
                context=context, params=current
            )
            step_rows.append(
                self._build_attribution_row(
                    component="tail",
                    previous=dev_eval,
                    current=tail_eval,
                )
            )

            current["bf_apriori"] = dict(candidate_params.get("bf_apriori", {}))
            bf_eval = self._scenario_totals_for_params(context=context, params=current)
            step_rows.append(
                self._build_attribution_row(
                    component="bf_apriori",
                    previous=tail_eval,
                    current=bf_eval,
                )
            )

            current["final_ultimate"] = candidate_params.get(
                "final_ultimate",
                current.get("final_ultimate", "chainladder"),
            )
            current["selected_ultimate_by_uwy"] = dict(
                candidate_params.get("selected_ultimate_by_uwy", {})
            )
            candidate_eval = self._scenario_totals_for_params(
                context=context, params=current
            )
            step_rows.append(
                self._build_attribution_row(
                    component="selection",
                    previous=bf_eval,
                    current=candidate_eval,
                )
            )

            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )

            return ReserveChangeResponse(
                session_id=context.session_id,
                baseline=baseline_eval,
                candidate=candidate_eval,
                attribution={
                    "baseline_vs_candidate_delta": round(
                        float(candidate_eval.get("total_ibnr", 0.0))
                        - float(baseline_eval.get("total_ibnr", 0.0)),
                        6,
                    ),
                    "steps": step_rows,
                },
                rows=step_rows,
            )

    def run_highest_a2a_drop_scenario(
        self,
        payload: HighestA2ADropRequest,
    ) -> HighestA2ADropResponse:
        derived = self.run_derived_drop_scenario(
            DerivedDropScenarioRequest(
                session_id=payload.session_id,
                rule={
                    "source": "link_ratios",
                    "selection_mode": "max",
                    "scope": "per_development_period",
                    "limit": 200,
                    "include_existing_drops": True,
                },
            )
        )
        return HighestA2ADropResponse(
            session_id=derived.session_id,
            drop=derived.drop,
            top_factors=derived.selected_rows,
            baseline=derived.baseline,
            candidate=derived.candidate,
            scenario=derived.scenario,
        )

    def rank_link_ratios(
        self,
        payload: LinkRatioRankRequest,
    ) -> LinkRatioRankResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            rows = self._rank_link_ratio_rows(
                context,
                selection_mode=payload.selection_mode,
                scope=payload.scope,
                limit=payload.limit,
                threshold_operator=payload.threshold_operator,
                threshold_value=payload.threshold_value,
            )
            return LinkRatioRankResponse(
                session_id=context.session_id,
                selection_mode=payload.selection_mode,
                scope=payload.scope,
                rows=rows,
                summary={
                    "row_count": len(rows),
                    "top_rows": rows[:5],
                },
            )

    def run_derived_drop_scenario(
        self,
        payload: DerivedDropScenarioRequest,
    ) -> DerivedDropScenarioResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")

            baseline_params = self._params_from_store(context)
            baseline_eval = self._evaluate_scenario(
                context=context,
                scenario_id="baseline",
                params=self._clone_params(baseline_params),
                summary="Current configuration",
                parent_scenario_id=None,
                transform="baseline",
                rationale_evidence_ids=[],
            )
            candidate_params = self._clone_params(baseline_params)
            rules = list(payload.rules or [])
            if not rules:
                rules = [payload.rule]
            rows: list[dict[str, Any]] = []
            drop_pairs: list[tuple[str, int]] = []
            seen_pairs: set[tuple[str, int]] = set()
            for rule in rules:
                rule_rows = self._selected_link_ratio_rows(
                    context,
                    selection_mode=rule.selection_mode,
                    scope=rule.scope,
                    limit=rule.limit,
                    threshold_operator=rule.threshold_operator,
                    threshold_value=rule.threshold_value,
                )
                for item in rule_rows:
                    pair = (str(item["origin"]), int(item["development_period"]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    rows.append(item)
                    drop_pairs.append(pair)
            existing_drop_pairs: set[tuple[str, int]] = set()
            include_existing_drops = any(rule.include_existing_drops for rule in rules)
            if include_existing_drops:
                existing_drop_pairs = {
                    (str(item[0]), int(item[1]))
                    for item in candidate_params.get("drop", [])
                }
            for origin, age in drop_pairs:
                existing_drop_pairs.add((origin, age))
            candidate_params["drop"] = [
                list(item) for item in sorted(existing_drop_pairs)
            ]

            candidate_eval = self._evaluate_scenario(
                context=context,
                scenario_id=self._scenario_id_from_rules(rules),
                params=candidate_params,
                summary=self._scenario_summary_from_rules(rules),
                parent_scenario_id="baseline",
                transform=self._scenario_transform_from_rules(rules),
                rationale_evidence_ids=[],
            )

            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )

            return DerivedDropScenarioResponse(
                session_id=context.session_id,
                rule={
                    "primary": payload.rule.model_dump(mode="json"),
                    "rules": [rule.model_dump(mode="json") for rule in rules],
                },
                drop=[list(item) for item in drop_pairs],
                selected_rows=rows,
                baseline=baseline_eval.model_dump(mode="json"),
                candidate=candidate_eval.model_dump(mode="json"),
                scenario={
                    "scenario_id": candidate_eval.scenario_id,
                    "summary": candidate_eval.summary,
                    "score_delta": round(candidate_eval.score - baseline_eval.score, 4),
                    "drop_count": len(candidate_params.get("drop", [])),
                    "parameters": candidate_params,
                },
            )

    def evaluate_tail_fit(
        self,
        payload: TailEvaluationRequest,
    ) -> TailEvaluationResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")

            params = self._clone_params(payload.model_dump(mode="json"))
            params.pop("session_id", None)
            baseline_params = self._params_from_store(context)

            self._apply_params_to_reserving(context, params)
            heatmap_data = context.reserving.get_triangle_heatmap_data()
            link_ratios_raw = heatmap_data.get("link_ratios")
            if not isinstance(link_ratios_raw, pd.DataFrame) or link_ratios_raw.empty:
                raise ValueError("Tail evaluation requires link ratio data")

            observed_row = link_ratios_raw.loc[
                link_ratios_raw.index.astype(str) == "LDF"
            ]
            fitted_row = link_ratios_raw.loc[
                link_ratios_raw.index.astype(str) == "Tail"
            ]
            if observed_row.empty or fitted_row.empty:
                raise ValueError("Tail evaluation requires both LDF and Tail rows")

            fit_period = self._normalize_fit_period(payload.tail.fit_period)
            residuals: list[dict[str, Any]] = []
            observed_points: list[dict[str, Any]] = []
            fitted_points: list[dict[str, Any]] = []
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
                observed_points.append({"age": age, "ldf": round(observed, 6)})
                fitted_points.append({"age": age, "ldf": round(fitted, 6)})
                residuals.append(
                    {
                        "age": age,
                        "observed_ldf": round(observed, 6),
                        "fitted_ldf": round(fitted, 6),
                        "error": round(fitted - observed, 6),
                    }
                )

            r2 = None
            rmse = None
            if observed_values:
                observed_series = np.array(observed_values, dtype=float)
                fitted_series = np.array(fitted_values, dtype=float)
                rmse = float(np.sqrt(np.mean((fitted_series - observed_series) ** 2)))
                centered = observed_series - observed_series.mean()
                denom = float(np.sum(centered**2))
                if denom > 0:
                    r2 = float(
                        1.0 - np.sum((observed_series - fitted_series) ** 2) / denom
                    )

            attachment_previous_age = None
            attachment_previous_ldf = None
            attachment_first_fitted_ldf = None
            attachment_gap_ratio = None
            late_subunit_observed_ages: list[int] = []
            attachment_age = payload.tail.attachment_age
            if attachment_age is not None:
                prior_ages = [age for age in observed_by_age if age < attachment_age]
                if prior_ages:
                    attachment_previous_age = max(prior_ages)
                    attachment_previous_ldf = observed_by_age.get(
                        attachment_previous_age
                    )
                attachment_first_fitted_ldf = fitted_by_age.get(attachment_age)
                if (
                    attachment_previous_ldf is not None
                    and attachment_previous_ldf > 0
                    and attachment_first_fitted_ldf is not None
                ):
                    attachment_gap_ratio = (
                        max(
                            attachment_previous_ldf - attachment_first_fitted_ldf,
                            0.0,
                        )
                        / attachment_previous_ldf
                    )
                late_subunit_observed_ages = [
                    age
                    for age, value in sorted(observed_by_age.items())
                    if age >= attachment_age and value < 1.0
                ]

            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )

            return TailEvaluationResponse(
                session_id=context.session_id,
                tail_curve=Reserving._normalize_tail_curve(payload.tail.curve),
                fit_period=list(payload.tail.fit_period),
                attachment_age=payload.tail.attachment_age,
                projection_period=payload.tail.projection_period,
                r2=round(r2, 6) if r2 is not None else None,
                rmse=round(rmse, 6) if rmse is not None else None,
                point_count=len(observed_points),
                residuals=residuals,
                observed_ldf=observed_points,
                fitted_tail_ldf=fitted_points,
                attachment_previous_age=attachment_previous_age,
                attachment_previous_ldf=round(attachment_previous_ldf, 6)
                if attachment_previous_ldf is not None
                else None,
                attachment_first_fitted_ldf=round(attachment_first_fitted_ldf, 6)
                if attachment_first_fitted_ldf is not None
                else None,
                attachment_gap_ratio=round(attachment_gap_ratio, 6)
                if attachment_gap_ratio is not None
                else None,
                late_subunit_observed_ages=late_subunit_observed_ages,
            )

    def get_assumption_context_detail(
        self,
        payload: AssumptionDetailRequest,
    ) -> AssumptionDetailResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            original_params = self._params_from_store(context)
            params, analysis_basis = self._resolve_assumption_detail_basis(
                context=context,
                payload=payload,
                original_params=original_params,
            )
            self._apply_params_to_reserving(context, params)
            try:
                heatmap_data = context.reserving.get_triangle_heatmap_data()
                link_ratios_raw = heatmap_data.get("link_ratios")
                if (
                    not isinstance(link_ratios_raw, pd.DataFrame)
                    or link_ratios_raw.empty
                ):
                    raise ValueError("Assumption detail requires link ratio data")

                observed_row = link_ratios_raw.loc[
                    link_ratios_raw.index.astype(str) == "LDF"
                ]
                fitted_row = link_ratios_raw.loc[
                    link_ratios_raw.index.astype(str) == "Tail"
                ]
                if observed_row.empty or fitted_row.empty:
                    raise ValueError(
                        "Assumption detail requires both LDF and Tail rows"
                    )

                triangle_only = link_ratios_raw.loc[
                    ~link_ratios_raw.index.astype(str).isin(["LDF", "Tail"])
                ]
                selected_ldf: list[dict[str, Any]] = []
                fitted_tail_ldf: list[dict[str, Any]] = []
                observed_a2a: list[dict[str, Any]] = []

                for col in link_ratios_raw.columns:
                    age = Reserving._parse_cdf_label_to_age(col)
                    if not self._age_in_window(
                        age,
                        start_age=payload.start_age,
                        end_age=payload.end_age,
                    ):
                        continue
                    label = str(col)
                    selected_value = self._to_optional_float(
                        observed_row.iloc[0].get(col)
                    )
                    fitted_value = self._to_optional_float(fitted_row.iloc[0].get(col))
                    if selected_value is not None:
                        selected_ldf.append(
                            {
                                "age": age,
                                "development_label": label,
                                "ldf": round(selected_value, 6),
                            }
                        )
                    if fitted_value is not None:
                        fitted_tail_ldf.append(
                            {
                                "age": age,
                                "development_label": label,
                                "ldf": round(fitted_value, 6),
                            }
                        )
                    if (
                        payload.development_period is None
                        or age != payload.development_period
                    ):
                        continue
                    for origin in triangle_only.index:
                        a2a_value = self._to_optional_float(
                            triangle_only.loc[origin, col]
                        )
                        if a2a_value is None:
                            continue
                        observed_a2a.append(
                            {
                                "origin": self._origin_label(origin),
                                "age": age,
                                "development_label": label,
                                "a2a": round(a2a_value, 6),
                            }
                        )

                return AssumptionDetailResponse(
                    session_id=context.session_id,
                    analysis_basis=analysis_basis,
                    parameters=params,
                    selected_ldf=selected_ldf,
                    fitted_tail_ldf=fitted_tail_ldf,
                    observed_a2a=observed_a2a,
                    bf_apriori_by_uwy=dict(params.get("bf_apriori", {})),
                    selected_ultimate_by_uwy=dict(
                        params.get("selected_ultimate_by_uwy", {})
                    ),
                )
            finally:
                self._apply_params_to_reserving(context, original_params)

    def _resolve_assumption_detail_basis(
        self,
        *,
        context: SessionContext,
        payload: AssumptionDetailRequest,
        original_params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        scenario_id = str(payload.scenario_id or "").strip() or None
        if isinstance(payload.parameters, dict) and payload.parameters:
            params = self._clone_params(payload.parameters)
            return params, self._build_analysis_basis_payload(
                session_id=context.session_id,
                basis_type=str(payload.basis_type or "scenario"),
                scenario_id=scenario_id,
                parameters=params,
                is_active_session=params == self._clone_params(original_params),
            )
        if scenario_id and scenario_id != "baseline":
            raise ValueError(
                "Scenario-bound assumption detail requires explicit parameters for this request"
            )
        params = self._clone_params(original_params)
        return params, self._build_analysis_basis_payload(
            session_id=context.session_id,
            basis_type=str(payload.basis_type or "baseline"),
            scenario_id=scenario_id or "baseline",
            parameters=params,
            is_active_session=True,
        )

    @staticmethod
    def _build_analysis_basis_payload(
        *,
        session_id: str,
        basis_type: str,
        scenario_id: str | None,
        parameters: dict[str, Any],
        is_active_session: bool,
    ) -> dict[str, Any]:
        canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        return {
            "basis_type": str(basis_type or "baseline"),
            "session_id": session_id,
            "scenario_id": scenario_id,
            "scenario_signature": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
                :16
            ],
            "is_active_session": bool(is_active_session),
            "parameters": InMemoryReservingBackend._clone_params(parameters),
        }

    def run_drop_review(self, payload: DropReviewRequest) -> DropReviewResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            baseline_params = self._params_from_store(context)
            review = self._get_assumption_review_service().review_drops(
                segment=context.segment,
                reserving=context.reserving,
                baseline_params=baseline_params,
                segment_memory=self._load_segment_memory(context.segment),
                candidate_limit=payload.candidate_limit,
            )
            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )
            return DropReviewResponse(
                session_id=context.session_id,
                baseline=review.get("baseline", {}),
                candidates=review.get("candidates", []),
                recommendation=review.get("recommendation", {}),
                continuity_notes=review.get("continuity_notes", []),
                policy_trace=review.get("policy_trace", {}),
                evidence_summary=review.get("evidence_summary", {}),
                run_metadata=review.get("run_metadata", {}),
            )

    def run_tail_review(self, payload: TailReviewRequest) -> TailReviewResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            baseline_params = self._params_from_store(context)
            review = self._get_assumption_review_service().review_tail(
                segment=context.segment,
                reserving=context.reserving,
                baseline_params=baseline_params,
                segment_memory=self._load_segment_memory(context.segment),
                candidate_limit=payload.candidate_limit,
            )
            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )
            return TailReviewResponse(
                session_id=context.session_id,
                baseline=review.get("baseline", {}),
                candidates=review.get("candidates", []),
                recommendation=review.get("recommendation", {}),
                continuity_notes=review.get("continuity_notes", []),
                policy_trace=review.get("policy_trace", {}),
                evidence_summary=review.get("evidence_summary", {}),
                run_metadata=review.get("run_metadata", {}),
            )

    def run_bf_suitability_review(
        self,
        payload: BfSuitabilityRequest,
    ) -> BfSuitabilityResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            baseline_params = self._params_from_store(context)
            review = self._get_assumption_review_service().review_bf_suitability(
                segment=context.segment,
                reserving=context.reserving,
                baseline_params=baseline_params,
                segment_memory=self._load_segment_memory(context.segment),
            )
            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )
            return BfSuitabilityResponse(
                session_id=context.session_id,
                rows=review.get("rows", []),
                overall_class=review.get("overall_class", "inconclusive"),
                summary=review.get("summary", {}),
                apriori_guidance=review.get("apriori_guidance", {}),
                continuity_notes=review.get("continuity_notes", []),
                policy_trace=review.get("policy_trace", {}),
                run_metadata=review.get("run_metadata", {}),
            )

    def run_anomaly_triage(
        self,
        payload: AnomalyTriageRequest,
    ) -> AnomalyTriageResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            baseline_params = self._params_from_store(context)
            review = self._get_assumption_review_service().triage_anomalies(
                segment=context.segment,
                reserving=context.reserving,
                baseline_params=baseline_params,
            )
            self._apply_params_to_reserving(context, baseline_params)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )
            return AnomalyTriageResponse(
                session_id=context.session_id,
                triaged_findings=review.get("triaged_findings", []),
                summary=review.get("summary", {}),
                pause_recommendation=bool(review.get("pause_recommendation", False)),
                run_metadata=review.get("run_metadata", {}),
            )

    def run_quarter_close_review(
        self,
        payload: QuarterCloseReviewRequest,
    ) -> QuarterCloseReviewResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            review = self._run_quarter_close_review_payload(context)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )
            return QuarterCloseReviewResponse(
                session_id=context.session_id,
                comparison=review.get("comparison", {}),
                diagnostics=review.get("diagnostics", {}),
                assumption_reviews=review.get("assumption_reviews", {}),
                scenario_summary=review.get("scenario_summary", {}),
                continuity=review.get("continuity", {}),
                recommendation=review.get("recommendation", {}),
                evidence_ids=review.get("evidence_ids", []),
                run_metadata=review.get("run_metadata", {}),
            )

    def build_quarter_close_pack(
        self,
        payload: QuarterClosePackRequest,
    ) -> QuarterClosePackResponse:
        with self._lock:
            context = self._get_context_by_session_id(payload.session_id)
            if context is None:
                raise LookupError(f"Session not found: {payload.session_id}")
            review = self._run_quarter_close_review_payload(context)
            pack = self._get_quarter_close_service().build_pack(review_result=review)
            context.last_results_payload = self._build_results_payload(
                context.reserving
            )
            return QuarterClosePackResponse(
                session_id=context.session_id,
                pack=pack,
                run_metadata=review.get("run_metadata", {}),
            )

    def _run_quarter_close_review_payload(
        self,
        context: SessionContext,
    ) -> dict[str, Any]:
        if not context.source_claims_rows or not context.source_premium_rows:
            raise ValueError(
                "Quarter-close review requires source claims_rows and premium_rows in the session context"
            )
        claims_df = pd.DataFrame(context.source_claims_rows)
        premium_df = pd.DataFrame(context.source_premium_rows)
        if claims_df.empty or premium_df.empty:
            raise ValueError(
                "Quarter-close review requires non-empty source claims and premium data"
            )
        return self._get_quarter_close_service().run_review(
            segment=context.segment,
            reserving=context.reserving,
            claims_df=claims_df,
            premium_df=premium_df,
            baseline_params=self._params_from_store(context),
            config=self._config,
            segment_memory=self._load_segment_memory(context.segment),
        )

    def _build_results_payload(self, reserving: Reserving) -> dict:
        results_df = reserving.get_results()
        try:
            emergence_df = reserving.get_emergence_pattern()
            emergence_payload = self._serialize_dataframe(emergence_df)
        except Exception:
            emergence_payload = {}
        try:
            heatmap = reserving.get_triangle_heatmap_data()
            heatmap_payload = {
                "link_ratios": self._serialize_dataframe(heatmap.get("link_ratios")),
                "incurred": self._serialize_dataframe(heatmap.get("incurred")),
                "paid": self._serialize_dataframe(heatmap.get("paid")),
                "premium": self._serialize_dataframe(heatmap.get("premium")),
            }
        except Exception:
            heatmap_payload = {}

        rows = self._build_results_table_rows(results_df)
        cache_key = str(uuid.uuid4())
        return {
            "results_table_rows": rows,
            "triangle_figure": {},
            "emergence_figure": emergence_payload,
            "heatmap_payload": heatmap_payload,
            "cache_key": cache_key,
            "model_cache_key": cache_key,
            "figure_version": 1,
            "last_updated": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }

    @staticmethod
    def _service_query(query) -> ServiceDataViewQuery:
        return ServiceDataViewQuery(
            metric=str(query.metric),
            view=str(query.view),
            denominator=query.denominator,
            denominator_view=query.denominator_view,
        )

    def _scenario_totals_for_params(
        self,
        *,
        context: SessionContext,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return self._get_scenario_evaluation_service().scenario_totals_for_params(
            reserving=context.reserving,
            params=self._clone_params(params),
        )

    @staticmethod
    def _build_attribution_row(
        *,
        component: str,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        delta = round(
            float(current.get("total_ibnr", 0.0))
            - float(previous.get("total_ibnr", 0.0)),
            6,
        )
        return {
            "component": component,
            "previous_total_ibnr": previous.get("total_ibnr"),
            "current_total_ibnr": current.get("total_ibnr"),
            "delta_ibnr": delta,
        }

    @staticmethod
    def _build_results_table_rows(
        results_df: pd.DataFrame | None,
    ) -> list[dict[str, str]]:
        if results_df is None or len(results_df) == 0:
            return []
        rows: list[dict[str, str]] = []
        for idx, row in results_df.iterrows():
            year_value = getattr(idx, "year", None)
            if year_value is not None:
                uwy = str(year_value)
            else:
                text = str(idx)
                uwy = text[:4] if len(text) >= 4 else text
            incurred = float(row.get("incurred", 0.0) or 0.0)
            premium = float(row.get("Premium", 0.0) or 0.0)
            cl_ultimate = float(row.get("cl_ultimate", 0.0) or 0.0)
            bf_ultimate = float(row.get("bf_ultimate", 0.0) or 0.0)
            selected_ultimate = float(row.get("ultimate", cl_ultimate) or cl_ultimate)
            ibnr = selected_ultimate - incurred

            rows.append(
                {
                    "uwy": uwy,
                    "incurred_display": f"{incurred:,.0f}",
                    "premium_display": f"{premium:,.0f}",
                    "cl_ultimate_display": f"{cl_ultimate:,.2f}",
                    "bf_ultimate_display": f"{bf_ultimate:,.2f}",
                    "ultimate_display": f"{selected_ultimate:,.2f}",
                    "ibnr_display": f"{ibnr:,.2f}",
                    "selected_method": str(row.get("selected_method", "chainladder")),
                }
            )
        return rows

    @staticmethod
    def _serialize_dataframe(dataframe: object) -> dict:
        if not isinstance(dataframe, pd.DataFrame):
            return {}
        serializable = dataframe.copy()
        if isinstance(serializable.columns, pd.MultiIndex):
            serializable.columns = [
                "|".join(str(part) for part in col if part is not None)
                for col in serializable.columns
            ]
        serializable = serializable.reset_index()
        records = serializable.to_dict(orient="records")
        normalized: list[dict[str, Any]] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    str(key): InMemoryReservingBackend._json_safe_value(value)
                    for key, value in row.items()
                }
            )
        return {"records": normalized}

    def _persist_config_session(self, context: SessionContext) -> None:
        if self._config is None:
            return
        payload = {
            "average": context.params_store.average,
            "tail_curve": context.params_store.tail_curve,
            "drops": context.params_store.drop_store,
            "tail_attachment_age": context.params_store.tail_attachment_age,
            "tail_projection_months": context.params_store.tail_projection_months,
            "tail_fit_period": context.params_store.tail_fit_period_selection,
            "bf_apriori_by_uwy": context.params_store.bf_apriori_by_uwy,
            "selected_ultimate_by_uwy": context.params_store.selected_ultimate_by_uwy,
        }
        self._config.save_session_with_version(payload)

    @staticmethod
    def _normalize_drop_pairs(
        raw_pairs: list[list[str | int]],
    ) -> list[tuple[str, int]] | None:
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
    def _normalize_drop_valuation(raw_pairs: list[list[str | int]]) -> list[str] | None:
        years: list[str] = []
        for pair in raw_pairs:
            if not isinstance(pair, list) or len(pair) < 1:
                continue
            years.append(str(pair[0]))
        return years or None

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
    def _infer_months_per_dev(reserving: Reserving) -> int:
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
    def _new_session_id(segment: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{segment}-{timestamp}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _default_params_store() -> ParamsStore:
        return ParamsStore(
            request_id=0,
            source="api-initialization",
            force_recalc=False,
            drop_store=[],
            tail_attachment_age=None,
            tail_projection_months=0,
            tail_fit_period_selection=[],
            average="volume",
            tail_curve="weibull",
            bf_apriori_by_uwy={},
            selected_ultimate_by_uwy={},
            sync_version=0,
        )

    def _get_context_by_segment(self, segment: str) -> SessionContext | None:
        session_id = self._sessions_by_segment.get(segment)
        if session_id is None:
            return None
        return self._sessions_by_id.get(session_id)

    def _get_context_by_session_id(self, session_id: str) -> SessionContext | None:
        return self._sessions_by_id.get(session_id)

    def _apply_params_to_reserving(self, context: SessionContext, params: dict) -> None:
        self._get_scenario_evaluation_service().apply_params_to_reserving(
            reserving=context.reserving,
            params=params,
        )

    def _filter_valid_drop_pairs(
        self,
        context: SessionContext,
        drops: list[tuple[str, int]] | None,
    ) -> list[tuple[str, int]] | None:
        if not drops:
            return None
        try:
            heatmap_data = context.reserving.get_triangle_heatmap_data()
            link_ratios_raw = heatmap_data.get("link_ratios")
        except Exception:
            return drops
        if not isinstance(link_ratios_raw, pd.DataFrame) or link_ratios_raw.empty:
            return drops

        valid_origins = {
            self._origin_label(origin)
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

    def _evaluate_scenario(
        self,
        *,
        context: SessionContext,
        scenario_id: str,
        params: dict,
        summary: str,
        parent_scenario_id: str | None,
        transform: str,
        rationale_evidence_ids: list[str],
    ) -> ScenarioEvaluation:
        return self._get_scenario_evaluation_service().evaluate_scenario(
            segment=context.segment,
            reserving=context.reserving,
            params=params,
            scenario_id=scenario_id,
            summary=summary,
            parent_scenario_id=parent_scenario_id,
            transform=transform,
            rationale_evidence_ids=rationale_evidence_ids,
        )

    def _highest_a2a_drop_candidates(
        self,
        context: SessionContext,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
        rows = self._selected_link_ratio_rows(
            context,
            selection_mode="max",
            scope="per_development_period",
            limit=200,
        )
        return rows, [
            (str(item["origin"]), int(item["development_period"])) for item in rows
        ]

    def _rank_link_ratio_rows(
        self,
        context: SessionContext,
        *,
        selection_mode: str,
        scope: str,
        limit: int,
        threshold_operator: str | None = None,
        threshold_value: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._selected_link_ratio_rows(
            context,
            selection_mode=selection_mode,
            scope=scope,
            limit=limit,
            threshold_operator=threshold_operator,
            threshold_value=threshold_value,
        )

    def _selected_link_ratio_rows(
        self,
        context: SessionContext,
        *,
        selection_mode: str,
        scope: str,
        limit: int,
        threshold_operator: str | None = None,
        threshold_value: float | None = None,
    ) -> list[dict[str, Any]]:
        heatmap_data = context.reserving.get_triangle_heatmap_data()
        link_ratios_raw = heatmap_data.get("link_ratios")
        if not isinstance(link_ratios_raw, pd.DataFrame) or link_ratios_raw.empty:
            return []
        link_ratios = link_ratios_raw.apply(pd.to_numeric, errors="coerce")
        triangle_only = link_ratios.loc[
            ~link_ratios.index.astype(str).isin(["LDF", "Tail"])
        ]
        if triangle_only.empty:
            return []

        mode = str(selection_mode).strip().lower()
        scope_value = str(scope).strip().lower()
        if mode not in {"max", "min"}:
            raise ValueError(f"Unsupported selection_mode '{selection_mode}'")
        if scope_value not in {"per_development_period", "global"}:
            raise ValueError(f"Unsupported scope '{scope}'")
        if threshold_operator is not None and threshold_operator not in {
            "lt",
            "lte",
            "gt",
            "gte",
        }:
            raise ValueError(f"Unsupported threshold_operator '{threshold_operator}'")

        rows: list[dict[str, Any]] = []
        for col in triangle_only.columns:
            column = triangle_only[col].dropna()
            if column.empty:
                continue
            development_period = Reserving._parse_cdf_label_to_age(col)
            if development_period is None:
                continue
            filtered = column
            if threshold_operator is not None and threshold_value is not None:
                filtered = self._apply_threshold_filter(
                    column,
                    operator=threshold_operator,
                    threshold_value=float(threshold_value),
                )
            if filtered.empty:
                continue
            ordered = filtered.sort_values(ascending=(mode == "min"))
            if scope_value == "per_development_period":
                ordered = ordered.iloc[:1]
            for origin, value in ordered.items():
                rows.append(
                    {
                        "development_period": development_period,
                        "origin": self._origin_label(origin),
                        "a2a": round(float(value), 6),
                    }
                )

        if scope_value == "global":
            rows.sort(key=lambda item: float(item["a2a"]), reverse=(mode == "max"))
            return rows[:limit]
        rows.sort(key=lambda item: int(item["development_period"]))
        return rows[:limit]

    @staticmethod
    def _scenario_id_from_rule(rule) -> str:
        return f"derived_drop_{rule.selection_mode}_{rule.scope}"

    @staticmethod
    def _scenario_id_from_rules(rules) -> str:
        if len(rules) == 1:
            return InMemoryReservingBackend._scenario_id_from_rule(rules[0])
        return "derived_drop_multi_rule"

    @staticmethod
    def _scenario_summary_from_rule(rule) -> str:
        direction = "highest" if rule.selection_mode == "max" else "lowest"
        threshold_text = ""
        if rule.threshold_operator and rule.threshold_value is not None:
            threshold_text = (
                f" among factors {rule.threshold_operator} {rule.threshold_value:g}"
            )
        scope = (
            "in each development period"
            if rule.scope == "per_development_period"
            else f"globally (top {rule.limit})"
        )
        return f"Drop the {direction} observed a2a factor {scope}{threshold_text}"

    @staticmethod
    def _scenario_summary_from_rules(rules) -> str:
        if len(rules) == 1:
            return InMemoryReservingBackend._scenario_summary_from_rule(rules[0])
        parts = [
            InMemoryReservingBackend._scenario_summary_from_rule(rule) for rule in rules
        ]
        return "; then combine rules: " + " | ".join(parts)

    @staticmethod
    def _scenario_transform_from_rule(rule) -> str:
        return f"drop_link_ratios_{rule.selection_mode}_{rule.scope}"

    @staticmethod
    def _scenario_transform_from_rules(rules) -> str:
        if len(rules) == 1:
            return InMemoryReservingBackend._scenario_transform_from_rule(rules[0])
        return "drop_link_ratios_multi_rule"

    @staticmethod
    def _apply_threshold_filter(
        column: pd.Series,
        *,
        operator: str,
        threshold_value: float,
    ) -> pd.Series:
        if operator == "lt":
            return column[column < threshold_value]
        if operator == "lte":
            return column[column <= threshold_value]
        if operator == "gt":
            return column[column > threshold_value]
        return column[column >= threshold_value]

    @staticmethod
    def _origin_label(origin: object) -> str:
        if hasattr(origin, "year"):
            return str(origin.year)
        text = str(origin)
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else text

    @staticmethod
    def _age_in_window(
        age: int | None,
        *,
        start_age: int | None,
        end_age: int | None,
    ) -> bool:
        if age is None:
            return False
        if start_age is not None and age < start_age:
            return False
        if end_age is not None and age > end_age:
            return False
        return True

    def _build_scenario_candidates(
        self,
        *,
        context: SessionContext,
        baseline: dict,
        baseline_eval: ScenarioEvaluation,
        max_scenarios: int,
    ) -> list[ScenarioCandidate]:
        scenarios: list[ScenarioCandidate] = []
        drop_recs = [
            rec
            for rec in baseline_eval.recommendations
            if rec.code.startswith("RECOMMEND_DROP_")
        ]
        tail_rec = next(
            (
                rec
                for rec in baseline_eval.recommendations
                if rec.code == "RECOMMEND_TAIL_FIT"
            ),
            None,
        )
        bf_rec = next(
            (
                rec
                for rec in baseline_eval.recommendations
                if rec.code == "RECOMMEND_BF_APRIORI"
            ),
            None,
        )

        for index, rec in enumerate(drop_recs[:4], start=1):
            params = self._clone_params(baseline)
            for pair in rec.proposed_parameters.get("drop", []):
                if pair not in params["drop"]:
                    params["drop"].append(pair)
            scenarios.append(
                ScenarioCandidate(
                    scenario_id=f"drop_{index}",
                    params=params,
                    summary=rec.message,
                    parent_scenario_id="baseline",
                    transform="apply_drop_recommendation",
                    rationale_evidence_ids=[str(rec.evidence.evidence_id)]
                    if rec.evidence.evidence_id
                    else [],
                )
            )

        if len(drop_recs) >= 2:
            params = self._clone_params(baseline)
            for rec in drop_recs[:2]:
                for pair in rec.proposed_parameters.get("drop", []):
                    if pair not in params["drop"]:
                        params["drop"].append(pair)
            rationale_ids = [
                str(rec.evidence.evidence_id)
                for rec in drop_recs[:2]
                if rec.evidence.evidence_id
            ]
            scenarios.append(
                ScenarioCandidate(
                    scenario_id="drop_combo_1",
                    params=params,
                    summary="Combine top two drop candidates",
                    parent_scenario_id="baseline",
                    transform="combine_drop_recommendations",
                    rationale_evidence_ids=rationale_ids,
                )
            )

        if tail_rec is not None:
            tail_params = tail_rec.proposed_parameters.get("tail", {})
            curves = tail_params.get("curve_candidates", [])
            fit_periods = tail_params.get("fit_period_candidates", [])
            attachment_age = tail_params.get("recommended_attachment_age")
            for curve in curves:
                for fit_period in fit_periods:
                    params = self._clone_params(baseline)
                    params["tail"]["curve"] = curve
                    params["tail"]["attachment_age"] = attachment_age
                    params["tail"]["fit_period"] = fit_period
                    scenarios.append(
                        ScenarioCandidate(
                            scenario_id=f"tail_{curve}_{fit_period[0]}_{fit_period[-1]}",
                            params=params,
                            summary=f"Tail sensitivity: curve={curve}, fit_period={fit_period}",
                            parent_scenario_id="baseline",
                            transform="tail_curve_fit_period_grid",
                            rationale_evidence_ids=[str(tail_rec.evidence.evidence_id)]
                            if tail_rec.evidence.evidence_id
                            else [],
                        )
                    )

        if bf_rec is not None:
            params = self._clone_params(baseline)
            params["bf_apriori"] = dict(
                bf_rec.proposed_parameters.get("bf_apriori", {})
            )
            scenarios.append(
                ScenarioCandidate(
                    scenario_id="bf_apriori_recommended",
                    params=params,
                    summary="Apply maturity-weighted BF apriori recommendations",
                    parent_scenario_id="baseline",
                    transform="apply_bf_apriori_recommendation",
                    rationale_evidence_ids=[str(bf_rec.evidence.evidence_id)]
                    if bf_rec.evidence.evidence_id
                    else [],
                )
            )

        if bf_rec is not None and tail_rec is not None:
            params = self._clone_params(baseline)
            params["bf_apriori"] = dict(
                bf_rec.proposed_parameters.get("bf_apriori", {})
            )
            tail_params = tail_rec.proposed_parameters.get("tail", {})
            attachment_age = tail_params.get("recommended_attachment_age")
            fit_periods = tail_params.get("fit_period_candidates", [])
            params["tail"]["attachment_age"] = attachment_age
            if fit_periods:
                params["tail"]["fit_period"] = fit_periods[0]
            rationale_ids: list[str] = []
            if bf_rec.evidence.evidence_id:
                rationale_ids.append(str(bf_rec.evidence.evidence_id))
            if tail_rec.evidence.evidence_id:
                rationale_ids.append(str(tail_rec.evidence.evidence_id))
            scenarios.append(
                ScenarioCandidate(
                    scenario_id="bf_plus_tail",
                    params=params,
                    summary="Combine BF apriori recommendation with tail-fit recommendation",
                    parent_scenario_id="baseline",
                    transform="combine_bf_and_tail_recommendations",
                    rationale_evidence_ids=rationale_ids,
                )
            )

        return scenarios[:max_scenarios]

    @staticmethod
    def _map_finding(item, *, run_metadata: RunMetadata) -> DiagnosticFinding:
        return ScenarioEvaluationService.map_finding(item, run_metadata=run_metadata)

    @staticmethod
    def _map_recommendation(
        item, *, run_metadata: RunMetadata
    ) -> DiagnosticRecommendation:
        return ScenarioEvaluationService.map_recommendation(
            item,
            run_metadata=run_metadata,
        )

    def _build_run_metadata(
        self,
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> RunMetadata:
        return self._get_scenario_evaluation_service().build_run_metadata(
            results_df=results_df,
            heatmap_data=heatmap_data,
        )

    @staticmethod
    def _data_fingerprint(
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> str:
        return ScenarioEvaluationService.data_fingerprint(
            results_df=results_df,
            heatmap_data=heatmap_data,
        )

    @staticmethod
    def _safe_df_records(dataframe: pd.DataFrame | None) -> list[dict]:
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
                    str(key): InMemoryReservingBackend._json_safe_value(value)
                    for key, value in row.items()
                }
            )
        return normalized

    @staticmethod
    def _json_safe_value(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [InMemoryReservingBackend._json_safe_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): InMemoryReservingBackend._json_safe_value(item)
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
    def _make_evidence_id(*, run_id: str, diagnostic_id: str, metric_id: str) -> str:
        raw = f"{run_id}|{diagnostic_id}|{metric_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _to_optional_float(value: object) -> float | None:
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
    def _to_optional_string(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _to_string_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _normalize_direction(value: object) -> Literal["good", "bad", "neutral"] | None:
        normalized = str(value).strip().lower()
        if normalized in {"good", "bad", "neutral"}:
            return cast(Literal["good", "bad", "neutral"], normalized)
        return None

    @staticmethod
    def _normalize_severity_level(
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
    def _normalize_review_level(
        value: object,
    ) -> Literal["green", "amber", "red"] | None:
        normalized = str(value).strip().lower()
        if normalized in {"green", "amber", "red"}:
            return cast(Literal["green", "amber", "red"], normalized)
        return None

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
    def _severity_components(findings: list[DiagnosticFinding]) -> dict[str, float]:
        return ScenarioEvaluationService.severity_components(findings)

    def _calibrated_diagnostics_service(
        self,
        *,
        segment: str,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> tuple[DiagnosticsService, dict[str, Any]]:
        return self._get_scenario_evaluation_service().calibrated_diagnostics_service(
            segment=segment,
            results_df=results_df,
            heatmap_data=heatmap_data,
        )

    def _calibrate_backtest_thresholds(
        self,
        *,
        segment: str,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> dict[str, Any]:
        return self._get_scenario_evaluation_service().calibrate_backtest_thresholds(
            segment=segment,
            results_df=results_df,
            heatmap_data=heatmap_data,
        )

    def _maturity_regime(self, results_df: pd.DataFrame | None) -> str:
        return self._get_scenario_evaluation_service().maturity_regime(results_df)

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        return ScenarioEvaluationService.quantile(values, q)

    @staticmethod
    def _governance_assessment(
        *,
        findings: list[DiagnosticFinding],
        severity_components: dict[str, float],
    ) -> dict[str, Any]:
        return ScenarioEvaluationService.governance_assessment(
            findings=findings,
            severity_components=severity_components,
        )

    @staticmethod
    def _governance_tier(findings: list[DiagnosticFinding]) -> str:
        severity_components = InMemoryReservingBackend._severity_components(findings)
        return str(
            InMemoryReservingBackend._governance_assessment(
                findings=findings,
                severity_components=severity_components,
            )["tier"]
        )

    @staticmethod
    def _params_from_store(context: SessionContext) -> dict:
        return {
            "average": Reserving._normalize_average(context.params_store.average),
            "drop": [list(item) for item in context.params_store.drop_store],
            "drop_valuation": [],
            "tail": {
                "curve": context.params_store.tail_curve,
                "attachment_age": context.params_store.tail_attachment_age,
                "projection_period": int(context.params_store.tail_projection_months),
                "fit_period": list(context.params_store.tail_fit_period_selection),
            },
            "bf_apriori": dict(context.params_store.bf_apriori_by_uwy),
            "final_ultimate": "chainladder",
            "selected_ultimate_by_uwy": dict(
                context.params_store.selected_ultimate_by_uwy
            ),
        }

    @staticmethod
    def _clone_params(params: dict) -> dict:
        cloned = {
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
        return cloned

    @staticmethod
    def _normalize_final_ultimate(
        value: object,
    ) -> Literal["chainladder", "bornhuetter_ferguson"]:
        return ScenarioEvaluationService.normalize_final_ultimate(value)

    def _autocomplete_bf_apriori(
        self,
        *,
        reserving: Reserving,
        bf_apriori: dict,
    ) -> dict[str, float]:
        return ScenarioEvaluationService.autocomplete_bf_apriori(
            reserving=reserving,
            bf_apriori=bf_apriori,
        )

    def _build_valuation_context(self, context: SessionContext) -> dict[str, Any]:
        if not context.source_claims_rows or not context.source_premium_rows:
            return {}
        claims_df = pd.DataFrame(context.source_claims_rows)
        premium_df = pd.DataFrame(context.source_premium_rows)
        if claims_df.empty or premium_df.empty:
            return {}
        service = self._get_valuation_snapshot_service()
        current = service.build_current_snapshot(
            reserving=context.reserving,
            claims_df=claims_df,
            premium_df=premium_df,
        )
        prior_proxy: dict[str, Any] = {}
        try:
            prior_proxy = service.build_prior_proxy_snapshot(
                claims_df=claims_df,
                premium_df=premium_df,
                params=self._params_from_store(context),
                config=self._config,
            )
        except Exception:
            prior_proxy = {}
        return {
            "current": current,
            "prior_proxy": prior_proxy,
        }

    def _get_scenario_evaluation_service(self) -> ScenarioEvaluationService:
        service = getattr(self, "_scenario_evaluation_service", None)
        if isinstance(service, ScenarioEvaluationService):
            return service
        scoring = getattr(self, "_scenario_scoring_service", None)
        if not isinstance(scoring, ScenarioScoringService):
            scoring = ScenarioScoringService()
            self._scenario_scoring_service = scoring
        service = ScenarioEvaluationService(
            diagnostics_service=getattr(self, "_diagnostics_service", None),
            uncertainty_service=getattr(self, "_uncertainty_service", None),
            scoring_service=scoring,
            scenario_generator_version=self.SCENARIO_GENERATOR_VERSION,
        )
        self._scenario_evaluation_service = service
        return service

    def _get_assumption_review_service(self) -> AssumptionReviewService:
        service = getattr(self, "_assumption_review_service", None)
        if isinstance(service, AssumptionReviewService):
            return service
        service = AssumptionReviewService(
            evaluation_service=self._get_scenario_evaluation_service(),
            scoring_service=getattr(self, "_scenario_scoring_service", None),
            diagnostics_service=getattr(self, "_diagnostics_service", None),
        )
        self._assumption_review_service = service
        return service

    def _load_segment_memory(self, segment: str) -> dict[str, Any]:
        if self._config is None:
            return SegmentMemoryService().load({}, segment=segment)
        raw_memory = self._config.load_ai_segment_memory(segment=segment)
        return SegmentMemoryService().load(raw_memory, segment=segment)

    def _get_valuation_snapshot_service(self) -> ValuationSnapshotService:
        service = getattr(self, "_valuation_snapshot_service", None)
        if isinstance(service, ValuationSnapshotService):
            return service
        service = ValuationSnapshotService(
            evaluation_service=self._get_scenario_evaluation_service(),
        )
        self._valuation_snapshot_service = service
        return service

    def _get_quarter_close_service(self) -> QuarterCloseService:
        service = getattr(self, "_quarter_close_service", None)
        if isinstance(service, QuarterCloseService):
            return service
        service = QuarterCloseService(
            evaluation_service=self._get_scenario_evaluation_service(),
            assumption_review_service=self._get_assumption_review_service(),
            valuation_snapshot_service=self._get_valuation_snapshot_service(),
        )
        self._quarter_close_service = service
        return service

    @staticmethod
    def _load_config() -> ConfigManager | None:
        explicit_path = os.environ.get("RESERVING_CONFIG")
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return ConfigManager.from_yaml(path)
        return load_config()
