from __future__ import annotations

from typing import Any

from ai.execution_records import attach_execution_metadata
from ai.request_validation import (
    build_passthrough_request_validation,
    validate_recalculate_like_arguments,
)
from source.api.schemas import (
    AssumptionDetailRequest,
    AnomalyTriageRequest,
    BfSuitabilityRequest,
    DataCompareRequest,
    DataViewRequest,
    DerivedDropScenarioRequest,
    DiagnosticsIterateRequest,
    DiagnosticsRequest,
    DropReviewRequest,
    HighestA2ADropRequest,
    LateEmergenceRequest,
    LinkRatioRankRequest,
    LdfConsistencyRequest,
    MovementDiagnosticsRequest,
    QuarterClosePackRequest,
    QuarterCloseReviewRequest,
    RecalculateRequest,
    ReserveChangeRequest,
    ResultsRequest,
    TailEvaluationRequest,
    TailReviewRequest,
    WorkflowFromDataframesRequest,
)

from ai.tool_payloads import (
    build_tool_specs,
    extract_last_derived_drop_detail,
    extract_finding_detail,
    extract_result_row_detail,
    extract_scenario_detail,
    summarize_data_compare_payload,
    summarize_data_view_payload,
    summarize_diagnostics_payload,
    summarize_derived_drop_payload,
    summarize_drop_review_payload,
    summarize_iteration_payload,
    summarize_bf_suitability_payload,
    summarize_link_ratio_rank_payload,
    summarize_late_emergence_payload,
    summarize_ldf_consistency_payload,
    summarize_movement_diagnostics_payload,
    summarize_anomaly_triage_review_payload,
    summarize_quarter_close_pack_payload,
    summarize_quarter_close_review_payload,
    summarize_recalculate_payload,
    summarize_highest_a2a_drop_payload,
    summarize_reserve_change_payload,
    summarize_tail_evaluation_payload,
    summarize_tail_review_payload,
    summarize_results_payload,
    summarize_session_payload,
)


class BackendReservingTools:
    def __init__(
        self, *, backend: Any, tool_specs: list[dict[str, Any]] | None = None
    ) -> None:
        self._backend = backend
        self._tool_specs = tool_specs or build_tool_specs()
        self._raw_cache: dict[str, dict[str, Any]] = {
            "session": {},
            "diagnostics": {},
            "iteration": {},
            "results": {},
            "recalculate": {},
            "assumption_detail": {},
            "data_view": {},
            "movement": {},
            "ldf_consistency": {},
            "late_emergence": {},
            "reserve_change": {},
            "highest_a2a_drop": {},
            "link_ratio_rank": {},
            "derived_drop": {},
            "tail_evaluation": {},
            "drop_review": {},
            "tail_review": {},
            "bf_suitability": {},
            "anomaly_triage": {},
            "quarter_close_review": {},
            "quarter_close_pack": {},
        }

    @property
    def tool_specs(self) -> list[dict[str, Any]]:
        return list(self._tool_specs)

    def create_workflow(
        self,
        *,
        segment: str,
        claims_rows: list[dict[str, Any]],
        premium_rows: list[dict[str, Any]],
        granularity: str | None = None,
    ) -> dict[str, Any]:
        response = self._backend.create_workflow_from_dataframes(
            WorkflowFromDataframesRequest(
                segment=segment,
                claims_rows=claims_rows,
                premium_rows=premium_rows,
                granularity=granularity,
            )
        )
        return response.model_dump(mode="json")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "tool_get_session_summary":
            segment = str(arguments["segment"])
            response = self._backend.get_session(segment)
            if response is None:
                raise LookupError(f"Segment session not found: {segment}")
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            self._raw_cache["session"][segment] = payload
            if session_id:
                self._raw_cache["session"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_session_payload(payload),
            )
        if name == "tool_evaluate_tail_fit":
            validation = validate_recalculate_like_arguments(arguments)
            if validation.rejection_reason:
                return self._finalize_summary(
                    tool_name=name,
                    args=arguments,
                    summary=_rejected_tool_summary(arguments, validation.rejection_reason),
                    validation=validation,
                )
            sanitized_arguments = dict(validation.effective_inputs)
            response = self._backend.evaluate_tail_fit(
                TailEvaluationRequest(**sanitized_arguments)
            )
            payload = response.model_dump(mode="json")
            if validation.input_adjustments:
                payload["input_adjustments"] = validation.input_adjustments
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["tail_evaluation"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_tail_evaluation_payload(payload),
                validation=validation,
            )
        if name == "tool_run_diagnostics_summary":
            response = self._backend.run_diagnostics(DiagnosticsRequest(**arguments))
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["diagnostics"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_diagnostics_payload(payload),
            )
        if name == "tool_run_drop_review":
            response = self._backend.run_drop_review(DropReviewRequest(**arguments))
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["drop_review"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_drop_review_payload(payload),
            )
        if name == "tool_run_tail_review":
            response = self._backend.run_tail_review(TailReviewRequest(**arguments))
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["tail_review"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_tail_review_payload(payload),
            )
        if name == "tool_run_bf_suitability_review":
            response = self._backend.run_bf_suitability_review(
                BfSuitabilityRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["bf_suitability"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_bf_suitability_payload(payload),
            )
        if name == "tool_run_anomaly_triage":
            response = self._backend.run_anomaly_triage(
                AnomalyTriageRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["anomaly_triage"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_anomaly_triage_review_payload(payload),
            )
        if name == "tool_run_quarter_close_review":
            response = self._backend.run_quarter_close_review(
                QuarterCloseReviewRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["quarter_close_review"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_quarter_close_review_payload(payload),
            )
        if name == "tool_get_quarter_close_pack":
            response = self._backend.build_quarter_close_pack(
                QuarterClosePackRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["quarter_close_pack"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_quarter_close_pack_payload(payload),
            )
        if name == "tool_iterate_diagnostics_summary":
            response = self._backend.iterate_diagnostics(
                DiagnosticsIterateRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["iteration"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_iteration_payload(payload),
            )
        if name == "tool_get_results_summary":
            session_id = str(arguments["session_id"])
            response = self._backend.get_results(
                ResultsRequest(
                    session_id=session_id,
                    basis_type=arguments.get("basis_type"),
                    basis_key=arguments.get("basis_key"),
                    scenario_id=arguments.get("scenario_id"),
                    parameters=arguments.get("parameters", {}),
                )
            )
            if response is None:
                raise LookupError(f"Session results not found: {session_id}")
            payload = response.model_dump(mode="json")
            self._raw_cache["results"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_results_payload(payload),
            )
        if name in {"tool_get_data_view_summary", "tool_get_data_view"}:
            session_id = str(arguments["session_id"])
            response = self._backend.get_data_view(
                DataViewRequest(
                    session_id=session_id,
                    query={
                        "metric": arguments.get("metric", "incurred"),
                        "view": arguments.get("view", "cumulative"),
                        "denominator": arguments.get("denominator"),
                        "denominator_view": arguments.get("denominator_view"),
                    },
                    include_summary=True,
                    basis_type=arguments.get("basis_type"),
                    basis_key=arguments.get("basis_key"),
                    scenario_id=arguments.get("scenario_id"),
                    parameters=arguments.get("parameters", {}),
                )
            )
            payload = response.model_dump(mode="json")
            self._raw_cache["data_view"][session_id] = payload
            if name == "tool_get_data_view":
                return self._finalize_summary(
                    tool_name=name,
                    args=arguments,
                    summary=payload,
                )
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_data_view_payload(payload),
            )
        if name == "tool_get_assumption_context_detail":
            session_id = str(arguments["session_id"])
            response = self._backend.get_assumption_context_detail(
                AssumptionDetailRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            self._raw_cache["assumption_detail"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=payload,
            )
        if name == "tool_compare_data_views":
            session_id = str(arguments["session_id"])
            response = self._backend.compare_data_views(
                DataCompareRequest(
                    session_id=session_id,
                    left={
                        "metric": arguments.get("left_metric", "incurred"),
                        "view": arguments.get("left_view", "cumulative"),
                        "denominator": arguments.get("left_denominator"),
                        "denominator_view": arguments.get("left_denominator_view"),
                    },
                    right={
                        "metric": arguments.get("right_metric", "incurred"),
                        "view": arguments.get("right_view", "cumulative"),
                        "denominator": arguments.get("right_denominator"),
                        "denominator_view": arguments.get("right_denominator_view"),
                    },
                    comparison_mode=str(arguments.get("comparison_mode", "difference")),
                )
            )
            payload = response.model_dump(mode="json")
            self._raw_cache["data_view"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_data_compare_payload(payload),
            )
        if name == "tool_run_movement_diagnostics":
            response = self._backend.run_movement_diagnostics(
                MovementDiagnosticsRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["movement"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_movement_diagnostics_payload(payload),
            )
        if name == "tool_run_ldf_consistency_diagnostics":
            response = self._backend.run_ldf_consistency(
                LdfConsistencyRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["ldf_consistency"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_ldf_consistency_payload(payload),
            )
        if name == "tool_project_late_emergence_benchmark":
            response = self._backend.project_late_emergence(
                LateEmergenceRequest(**arguments)
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["late_emergence"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_late_emergence_payload(payload),
            )
        if name == "tool_explain_reserve_change":
            validation = validate_recalculate_like_arguments(arguments)
            if validation.rejection_reason:
                return self._finalize_summary(
                    tool_name=name,
                    args=arguments,
                    summary=_rejected_tool_summary(arguments, validation.rejection_reason),
                    validation=validation,
                )
            sanitized_arguments = dict(validation.effective_inputs)
            response = self._backend.explain_reserve_change(
                ReserveChangeRequest(**sanitized_arguments)
            )
            payload = response.model_dump(mode="json")
            if validation.input_adjustments:
                payload["input_adjustments"] = validation.input_adjustments
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["reserve_change"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_reserve_change_payload(payload),
                validation=validation,
            )
        if name == "tool_run_highest_a2a_drop_scenario":
            response = self._backend.run_highest_a2a_drop_scenario(
                HighestA2ADropRequest(
                    session_id=str(arguments["session_id"]),
                    basis_type=arguments.get("basis_type"),
                    basis_key=arguments.get("basis_key"),
                    scenario_id=arguments.get("scenario_id"),
                    parameters=arguments.get("parameters", {}),
                )
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["highest_a2a_drop"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_highest_a2a_drop_payload(payload),
            )
        if name == "tool_rank_link_ratios":
            response = self._backend.rank_link_ratios(
                LinkRatioRankRequest(
                    session_id=str(arguments["session_id"]),
                    selection_mode=arguments.get("selection_mode", "max"),
                    scope=arguments.get("scope", "per_development_period"),
                    limit=arguments.get("limit", 5),
                    threshold_operator=arguments.get("threshold_operator"),
                    threshold_value=arguments.get("threshold_value"),
                    basis_type=arguments.get("basis_type"),
                    basis_key=arguments.get("basis_key"),
                    scenario_id=arguments.get("scenario_id"),
                    parameters=arguments.get("parameters", {}),
                )
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["link_ratio_rank"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_link_ratio_rank_payload(payload),
            )
        if name == "tool_run_derived_drop_scenario":
            response = self._backend.run_derived_drop_scenario(
                DerivedDropScenarioRequest(
                    session_id=str(arguments["session_id"]),
                    basis_type=arguments.get("basis_type"),
                    basis_key=arguments.get("basis_key"),
                    scenario_id=arguments.get("scenario_id"),
                    parameters=arguments.get("parameters", {}),
                    rule={
                        "source": arguments.get("source", "link_ratios"),
                        "selection_mode": arguments.get("selection_mode", "max"),
                        "scope": arguments.get("scope", "per_development_period"),
                        "limit": arguments.get("limit", 5),
                        "include_existing_drops": arguments.get(
                            "include_existing_drops", True
                        ),
                        "threshold_operator": arguments.get("threshold_operator"),
                        "threshold_value": arguments.get("threshold_value"),
                    },
                    rules=arguments.get("rules") or [],
                )
            )
            payload = response.model_dump(mode="json")
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["derived_drop"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_derived_drop_payload(payload),
            )
        if name == "tool_get_finding_detail":
            session_id = str(arguments["session_id"])
            return extract_finding_detail(
                self._raw_cache["diagnostics"].get(session_id),
                self._raw_cache["iteration"].get(session_id),
                code=_optional_str(arguments.get("code")),
                evidence_id=_optional_str(arguments.get("evidence_id")),
                scenario_id=_optional_str(arguments.get("scenario_id")),
            )
        if name == "tool_get_last_derived_drop_detail":
            session_id = str(arguments["session_id"])
            return extract_last_derived_drop_detail(
                self._raw_cache["derived_drop"].get(session_id),
                self._raw_cache["highest_a2a_drop"].get(session_id),
            )
        if name == "tool_get_scenario_detail":
            session_id = str(arguments["session_id"])
            return extract_scenario_detail(
                self._raw_cache["iteration"].get(session_id),
                scenario_id=str(arguments["scenario_id"]),
            )
        if name == "tool_get_result_for_uwy":
            session_id = str(arguments["session_id"])
            return extract_result_row_detail(
                self._raw_cache["results"].get(session_id),
                uwy=str(arguments["uwy"]),
            )
        if name == "tool_recalculate":
            validation = validate_recalculate_like_arguments(arguments)
            if validation.rejection_reason:
                return self._finalize_summary(
                    tool_name=name,
                    args=arguments,
                    summary=_rejected_tool_summary(arguments, validation.rejection_reason),
                    validation=validation,
                )
            sanitized_arguments = dict(validation.effective_inputs)
            sanitized_arguments["persist_to_session"] = False
            response = self._backend.recalculate(
                RecalculateRequest(**sanitized_arguments)
            )
            payload = response.model_dump(mode="json")
            if validation.input_adjustments:
                payload["input_adjustments"] = validation.input_adjustments
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["recalculate"][session_id] = payload
                self._raw_cache["results"][session_id] = payload
            return self._finalize_summary(
                tool_name=name,
                args=arguments,
                summary=summarize_recalculate_payload(payload),
                validation=validation,
            )
        raise ValueError(f"Unsupported tool: {name}")

    @staticmethod
    def _finalize_summary(
        *,
        tool_name: str,
        args: dict[str, Any],
        summary: dict[str, Any],
        validation: Any | None = None,
    ) -> dict[str, Any]:
        validation_result = (
            validation
            if validation is not None
            else build_passthrough_request_validation(args)
        )
        session_id = str(summary.get("session_id") or args.get("session_id") or "").strip()
        return attach_execution_metadata(
            summary,
            tool_name=tool_name,
            validation=validation_result,
            session_id=session_id or None,
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _rejected_tool_summary(arguments: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "session_id": str(arguments.get("session_id") or "").strip() or None,
        "rejected": True,
        "rejection_reason": reason,
    }
