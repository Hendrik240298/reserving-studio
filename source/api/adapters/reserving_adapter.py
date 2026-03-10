from __future__ import annotations

from dataclasses import dataclass
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

import pandas as pd

from source.app import build_workflow_from_dataframes, load_config
from source.api.schemas import (
    DiagnosticFinding,
    DiagnosticRecommendation,
    DiagnosticEvidence,
    DiagnosticsIterateRequest,
    DiagnosticsIterateResponse,
    DiagnosticsRequest,
    DiagnosticsResponse,
    ParamsStore,
    RecalculateRequest,
    RecalculateResponse,
    ResultsResponse,
    RunMetadata,
    ScenarioEvaluation,
    ResultsStoreMeta,
    SessionSaveRequest,
    SessionSaveResponse,
    SessionStateResponse,
    WorkflowFromDataframesRequest,
    WorkflowInitializationResponse,
)
from source.config_manager import ConfigManager
from source.reserving import Reserving
from source.services.diagnostics_service import DiagnosticsService
from source.services.uncertainty_service import UncertaintyService


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

            context.reserving.set_development(
                average=payload.average,
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
                average=payload.average,
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
        drops = self._normalize_drop_pairs(params.get("drop", []))
        drop_valuation = self._normalize_drop_valuation(
            params.get("drop_valuation", [])
        )
        tail_config = params.get("tail", {})
        tail_fit_period = self._normalize_fit_period(tail_config.get("fit_period", []))
        tail_projection_months = int(tail_config.get("projection_period", 0) or 0)
        months_per_dev = self._infer_months_per_dev(context.reserving)
        extrap_periods = tail_projection_months // months_per_dev
        projection_period = extrap_periods * months_per_dev

        context.reserving.set_development(
            average=str(params.get("average", "volume")),
            drop=drops,
            drop_valuation=drop_valuation,
        )
        context.reserving.set_tail(
            curve=str(tail_config.get("curve", "weibull")),
            attachment_age=tail_config.get("attachment_age"),
            extrap_periods=extrap_periods,
            projection_period=projection_period,
            fit_period=tail_fit_period,
        )

        bf_apriori = params.get("bf_apriori", {})
        if isinstance(bf_apriori, dict) and bf_apriori:
            completed_apriori = self._autocomplete_bf_apriori(
                reserving=context.reserving,
                bf_apriori=bf_apriori,
            )
            context.reserving.set_bornhuetter_ferguson(apriori=completed_apriori)
        else:
            context.reserving.set_bornhuetter_ferguson(apriori=0.6)

        context.reserving.reserve(
            final_ultimate=self._normalize_final_ultimate(
                params.get("final_ultimate", "chainladder")
            ),
            selected_ultimate_by_uwy=dict(params.get("selected_ultimate_by_uwy", {})),
        )

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
        self._apply_params_to_reserving(context, params)
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
        findings = [
            self._map_finding(item, run_metadata=run_metadata)
            for item in run_result.findings
        ]
        recommendations = [
            self._map_recommendation(item, run_metadata=run_metadata)
            for item in run_result.recommendations
        ]
        drop_count = len(params.get("drop", []))
        score = self._diagnostics_service.compute_severity_score(run_result.findings)
        score += 0.2 * float(drop_count)
        severity_components = self._severity_components(findings)
        governance = self._governance_assessment(
            findings=findings,
            severity_components=severity_components,
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

        return ScenarioEvaluation(
            scenario_id=scenario_id,
            score=round(score, 4),
            summary=summary,
            parameters=params,
            findings=findings,
            recommendations=recommendations,
            metrics=scenario_metrics,
            lineage={
                "parent_scenario_id": parent_scenario_id,
                "transform": transform,
                "rationale_evidence_ids": rationale_evidence_ids,
            },
            governance=governance,
            calibration=calibration,
            uncertainty=uncertainty,
            run_metadata=run_metadata,
        )

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
            for curve in curves:
                for fit_period in fit_periods:
                    params = self._clone_params(baseline)
                    params["tail"]["curve"] = curve
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
            fit_periods = tail_params.get("fit_period_candidates", [])
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
                InMemoryReservingBackend._make_evidence_id(
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
        applicability_conditions = InMemoryReservingBackend._to_string_list(
            evidence.get("applicability_conditions")
        )
        alternative_hypotheses = InMemoryReservingBackend._to_string_list(
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
                unit=InMemoryReservingBackend._to_optional_string(evidence.get("unit")),
                direction=InMemoryReservingBackend._normalize_direction(
                    evidence.get("direction")
                ),
                p_value_or_score=InMemoryReservingBackend._to_optional_float(
                    evidence.get("p_value_or_score")
                ),
                severity_band=cast(
                    Literal["low", "medium", "high", "critical"] | None,
                    InMemoryReservingBackend._normalize_severity_level(
                        evidence.get("severity_band"),
                        fallback=severity,
                    ),
                ),
                applicability_conditions=applicability_conditions,
                alternative_hypotheses=alternative_hypotheses,
                confidence=InMemoryReservingBackend._to_optional_float(
                    evidence.get("confidence")
                ),
                required_review_level=InMemoryReservingBackend._normalize_review_level(
                    evidence.get("required_review_level")
                ),
            ),
            suggested_actions=list(item.suggested_actions),
        )

    @staticmethod
    def _map_recommendation(
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
                InMemoryReservingBackend._make_evidence_id(
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
                unit=InMemoryReservingBackend._to_optional_string(evidence.get("unit")),
                direction=InMemoryReservingBackend._normalize_direction(
                    evidence.get("direction")
                ),
                p_value_or_score=InMemoryReservingBackend._to_optional_float(
                    evidence.get("p_value_or_score")
                ),
                severity_band=cast(
                    Literal["low", "medium", "high", "critical"] | None,
                    InMemoryReservingBackend._normalize_severity_level(
                        evidence.get("severity_band"),
                        fallback=priority,
                    ),
                ),
                applicability_conditions=InMemoryReservingBackend._to_string_list(
                    evidence.get("applicability_conditions")
                ),
                alternative_hypotheses=InMemoryReservingBackend._to_string_list(
                    evidence.get("alternative_hypotheses")
                ),
                confidence=InMemoryReservingBackend._to_optional_float(
                    evidence.get("confidence")
                ),
                required_review_level=InMemoryReservingBackend._normalize_review_level(
                    evidence.get("required_review_level")
                ),
            ),
            proposed_parameters=dict(item.proposed_parameters),
        )

    def _build_run_metadata(
        self,
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> RunMetadata:
        return RunMetadata(
            run_id=str(uuid.uuid4()),
            generated_at=datetime.now(timezone.utc),
            data_fingerprint=self._data_fingerprint(
                results_df=results_df,
                heatmap_data=heatmap_data,
            ),
            diagnostics_version=DiagnosticsService.DIAGNOSTICS_VERSION,
            scenario_generator_version=self.SCENARIO_GENERATOR_VERSION,
        )

    @staticmethod
    def _data_fingerprint(
        *,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> str:
        payload: dict[str, object] = {
            "results": InMemoryReservingBackend._safe_df_records(results_df),
            "heatmap": {},
        }
        if isinstance(heatmap_data, dict):
            serialized: dict[str, object] = {}
            for key in ["incurred", "paid", "premium", "link_ratios"]:
                serialized[key] = InMemoryReservingBackend._safe_df_records(
                    InMemoryReservingBackend._to_dataframe(heatmap_data.get(key))
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

    def _calibrated_diagnostics_service(
        self,
        *,
        segment: str,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> tuple[DiagnosticsService, dict[str, Any]]:
        calibration = self._calibrate_backtest_thresholds(
            segment=segment,
            results_df=results_df,
            heatmap_data=heatmap_data,
        )
        diagnostics_service = DiagnosticsService(
            backtest_bias_threshold=float(calibration["backtest_bias_threshold"]),
            backtest_mae_threshold=float(calibration["backtest_mae_threshold"]),
        )
        return diagnostics_service, calibration

    def _calibrate_backtest_thresholds(
        self,
        *,
        segment: str,
        results_df: pd.DataFrame | None,
        heatmap_data: dict | None,
    ) -> dict[str, Any]:
        maturity_regime = self._maturity_regime(results_df)
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

        empirical_bias = self._quantile(abs_residuals, 0.55)
        empirical_mae = self._quantile(abs_residuals, 0.8)
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

    def _maturity_regime(self, results_df: pd.DataFrame | None) -> str:
        maturity_map = self._diagnostics_service._build_maturity_map(results_df)
        if not maturity_map:
            return "mixed"
        average_maturity = sum(maturity_map.values()) / len(maturity_map)
        if average_maturity < 0.4:
            return "immature"
        if average_maturity >= 0.75:
            return "mature"
        return "mixed"

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
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
    def _governance_assessment(
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
            "average": context.params_store.average,
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
            "average": str(params.get("average", "volume")),
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
        normalized = str(value).strip().lower()
        if normalized == "bornhuetter_ferguson":
            return "bornhuetter_ferguson"
        return "chainladder"

    def _autocomplete_bf_apriori(
        self,
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

        missing: list[str] = []
        for origin in origins:
            year_value = getattr(origin, "year", None)
            key = str(year_value) if year_value is not None else str(origin)
            if key in completed:
                continue
            completed[key] = fallback
            missing.append(key)

        if self._observability_enabled and missing:
            logger.info(
                "[OBS] bf_apriori.autocomplete missing_count=%s fallback=%s",
                len(missing),
                fallback,
            )
        return completed

    @staticmethod
    def _load_config() -> ConfigManager | None:
        explicit_path = os.environ.get("RESERVING_CONFIG")
        if explicit_path:
            path = Path(explicit_path)
            if path.exists():
                return ConfigManager.from_yaml(path)
        return load_config()
