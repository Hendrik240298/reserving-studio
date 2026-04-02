from __future__ import annotations

from datetime import date, datetime
import json
import logging
import os
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError

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


class ReservingApiTools:
    def __init__(self, *, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._observability_enabled = os.environ.get(
            "AI_OBSERVABILITY", "1"
        ).strip().lower() not in {"0", "false", "off"}
        self._logger = logging.getLogger(__name__)
        self._tool_specs = build_tool_specs()
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

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "tool_get_session_summary":
            segment = str(arguments["segment"])
            payload = self.request_json("GET", f"/v1/sessions/{segment}")
            session_id = str(payload.get("session_id", ""))
            self._raw_cache["session"][segment] = payload
            if session_id:
                self._raw_cache["session"][session_id] = payload
            return summarize_session_payload(payload)
        if name == "tool_evaluate_tail_fit":
            sanitized_arguments, input_adjustments = (
                _sanitize_recalculate_like_arguments(arguments)
            )
            payload = self.request_json(
                "POST",
                "/v1/tail/evaluate",
                sanitized_arguments,
            )
            if input_adjustments:
                payload["input_adjustments"] = input_adjustments
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["tail_evaluation"][session_id] = payload
            return summarize_tail_evaluation_payload(payload)
        if name == "tool_run_diagnostics_summary":
            payload = self.request_json(
                "POST",
                "/v1/diagnostics/run",
                {
                    "session_id": arguments["session_id"],
                    "diagnostic_profile": arguments.get("diagnostic_profile"),
                    "include_recommendations": bool(
                        arguments.get("include_recommendations", True)
                    ),
                },
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["diagnostics"][session_id] = payload
            return summarize_diagnostics_payload(payload)
        if name == "tool_run_drop_review":
            payload = self.request_json(
                "POST",
                "/v1/reviews/drop",
                {
                    "session_id": arguments["session_id"],
                    "candidate_limit": int(arguments.get("candidate_limit", 5)),
                },
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["drop_review"][session_id] = payload
            return summarize_drop_review_payload(payload)
        if name == "tool_run_tail_review":
            payload = self.request_json(
                "POST",
                "/v1/reviews/tail",
                {
                    "session_id": arguments["session_id"],
                    "candidate_limit": int(arguments.get("candidate_limit", 12)),
                },
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["tail_review"][session_id] = payload
            return summarize_tail_review_payload(payload)
        if name == "tool_run_bf_suitability_review":
            payload = self.request_json(
                "POST",
                "/v1/reviews/bf-suitability",
                {"session_id": arguments["session_id"]},
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["bf_suitability"][session_id] = payload
            return summarize_bf_suitability_payload(payload)
        if name == "tool_run_anomaly_triage":
            payload = self.request_json(
                "POST",
                "/v1/reviews/anomaly-triage",
                {"session_id": arguments["session_id"]},
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["anomaly_triage"][session_id] = payload
            return summarize_anomaly_triage_review_payload(payload)
        if name == "tool_run_quarter_close_review":
            payload = self.request_json(
                "POST",
                "/v1/reviews/quarter-close",
                {"session_id": arguments["session_id"]},
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["quarter_close_review"][session_id] = payload
            return summarize_quarter_close_review_payload(payload)
        if name == "tool_get_quarter_close_pack":
            payload = self.request_json(
                "POST",
                "/v1/reviews/quarter-close/pack",
                {"session_id": arguments["session_id"]},
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["quarter_close_pack"][session_id] = payload
            return summarize_quarter_close_pack_payload(payload)
        if name == "tool_iterate_diagnostics_summary":
            payload = self.request_json(
                "POST",
                "/v1/diagnostics/iterate",
                {
                    "session_id": arguments["session_id"],
                    "max_scenarios": int(arguments.get("max_scenarios", 24)),
                    "include_baseline": bool(arguments.get("include_baseline", True)),
                },
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["iteration"][session_id] = payload
            return summarize_iteration_payload(payload)
        if name == "tool_get_results_summary":
            session_id = str(arguments["session_id"])
            payload = self.request_json("GET", f"/v1/results/{session_id}")
            self._raw_cache["results"][session_id] = payload
            return summarize_results_payload(payload)
        if name in {"tool_get_data_view_summary", "tool_get_data_view"}:
            session_id = str(arguments["session_id"])
            payload = self.request_json(
                "POST",
                "/v1/data/view",
                {
                    "session_id": session_id,
                    "query": {
                        "metric": arguments.get("metric", "incurred"),
                        "view": arguments.get("view", "cumulative"),
                        "denominator": arguments.get("denominator"),
                        "denominator_view": arguments.get("denominator_view"),
                    },
                    "include_summary": True,
                },
            )
            self._raw_cache["data_view"][session_id] = payload
            if name == "tool_get_data_view":
                return payload
            return summarize_data_view_payload(payload)
        if name == "tool_get_assumption_context_detail":
            session_id = str(arguments["session_id"])
            payload = self.request_json(
                "POST",
                "/v1/reserving/assumption-detail",
                {
                    "session_id": session_id,
                    "start_age": arguments.get("start_age"),
                    "end_age": arguments.get("end_age"),
                    "development_period": arguments.get("development_period"),
                    "basis_type": arguments.get("basis_type"),
                    "scenario_id": arguments.get("scenario_id"),
                    "parameters": arguments.get("parameters", {}),
                },
            )
            self._raw_cache["assumption_detail"][session_id] = payload
            return payload
        if name == "tool_compare_data_views":
            session_id = str(arguments["session_id"])
            payload = self.request_json(
                "POST",
                "/v1/data/compare",
                {
                    "session_id": session_id,
                    "left": {
                        "metric": arguments.get("left_metric", "incurred"),
                        "view": arguments.get("left_view", "cumulative"),
                        "denominator": arguments.get("left_denominator"),
                        "denominator_view": arguments.get("left_denominator_view"),
                    },
                    "right": {
                        "metric": arguments.get("right_metric", "incurred"),
                        "view": arguments.get("right_view", "cumulative"),
                        "denominator": arguments.get("right_denominator"),
                        "denominator_view": arguments.get("right_denominator_view"),
                    },
                    "comparison_mode": arguments.get("comparison_mode", "difference"),
                },
            )
            self._raw_cache["data_view"][session_id] = payload
            return summarize_data_compare_payload(payload)
        if name == "tool_run_movement_diagnostics":
            payload = self.request_json(
                "POST",
                "/v1/diagnostics/movement",
                {"session_id": arguments["session_id"]},
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["movement"][session_id] = payload
            return summarize_movement_diagnostics_payload(payload)
        if name == "tool_run_ldf_consistency_diagnostics":
            payload = self.request_json(
                "POST",
                "/v1/diagnostics/ldf-consistency",
                {"session_id": arguments["session_id"]},
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["ldf_consistency"][session_id] = payload
            return summarize_ldf_consistency_payload(payload)
        if name == "tool_project_late_emergence_benchmark":
            payload = self.request_json(
                "POST",
                "/v1/diagnostics/late-emergence",
                {
                    "session_id": arguments["session_id"],
                    "uwy": arguments.get("uwy"),
                },
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["late_emergence"][session_id] = payload
            return summarize_late_emergence_payload(payload)
        if name == "tool_explain_reserve_change":
            sanitized_arguments, input_adjustments = (
                _sanitize_recalculate_like_arguments(arguments)
            )
            payload = self.request_json(
                "POST",
                "/v1/reserving/explain-change",
                sanitized_arguments,
            )
            if input_adjustments:
                payload["input_adjustments"] = input_adjustments
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["reserve_change"][session_id] = payload
            return summarize_reserve_change_payload(payload)
        if name == "tool_run_highest_a2a_drop_scenario":
            payload = self.request_json(
                "POST",
                "/v1/reserving/highest-a2a-drop",
                arguments,
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["highest_a2a_drop"][session_id] = payload
            return summarize_highest_a2a_drop_payload(payload)
        if name == "tool_rank_link_ratios":
            payload = self.request_json(
                "POST",
                "/v1/link-ratios/rank",
                {
                    "session_id": arguments["session_id"],
                    "selection_mode": arguments.get("selection_mode", "max"),
                    "scope": arguments.get("scope", "per_development_period"),
                    "limit": arguments.get("limit", 5),
                    "threshold_operator": arguments.get("threshold_operator"),
                    "threshold_value": arguments.get("threshold_value"),
                },
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["link_ratio_rank"][session_id] = payload
            return summarize_link_ratio_rank_payload(payload)
        if name == "tool_run_derived_drop_scenario":
            payload = self.request_json(
                "POST",
                "/v1/reserving/derived-drop-scenario",
                {
                    "session_id": arguments["session_id"],
                    "rule": {
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
                    "rules": arguments.get("rules") or [],
                },
            )
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["derived_drop"][session_id] = payload
            return summarize_derived_drop_payload(payload)
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
            sanitized_arguments, input_adjustments = (
                _sanitize_recalculate_like_arguments(arguments)
            )
            payload = self.request_json(
                "POST",
                "/v1/reserving/recalculate",
                sanitized_arguments,
            )
            if input_adjustments:
                payload["input_adjustments"] = input_adjustments
            session_id = str(payload.get("session_id", ""))
            if session_id:
                self._raw_cache["recalculate"][session_id] = payload
            return summarize_recalculate_payload(payload)
        raise ValueError(f"Unsupported tool: {name}")

    def create_workflow(
        self,
        *,
        segment: str,
        claims_rows: list[dict[str, Any]],
        premium_rows: list[dict[str, Any]],
        granularity: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "segment": segment,
            "claims_rows": claims_rows,
            "premium_rows": premium_rows,
        }
        if granularity:
            payload["granularity"] = granularity
        return self.request_json("POST", "/v1/workflows/from-dataframes", payload)

    def request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        encoded_body: bytes | None = None
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if body is not None:
            encoded_body = json.dumps(body, default=_json_default).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}{path}",
            method=method,
            data=encoded_body,
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if self._observability_enabled:
                    self._logger.info(
                        "[OBS] api.request method=%s path=%s status=%s duration_ms=%s",
                        method,
                        path,
                        response.status,
                        int((time.perf_counter() - started) * 1000),
                    )
                return payload
        except HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            if self._observability_enabled:
                self._logger.error(
                    "[OBS] api.request_failed method=%s path=%s status=%s duration_ms=%s",
                    method,
                    path,
                    error.code,
                    int((time.perf_counter() - started) * 1000),
                )
            raise RuntimeError(
                f"API request failed ({error.code}) {path}: {response_body}"
            ) from error


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sanitize_recalculate_like_arguments(
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    sanitized = dict(arguments)
    adjustments: list[str] = []
    average = sanitized.get("average")
    if average is not None:
        normalized_average = _normalize_average_or_volume(average)
        if normalized_average != str(average).strip().lower():
            adjustments.append(
                f"Normalized average '{average}' to '{normalized_average}'."
            )
        sanitized["average"] = normalized_average
    tail = sanitized.get("tail")
    if isinstance(tail, dict):
        tail_copy = dict(tail)
        curve = tail_copy.get("curve")
        if curve is not None:
            normalized_curve = _normalize_tail_curve_or_default(curve)
            if normalized_curve != str(curve).strip().lower():
                adjustments.append(
                    f"Normalized tail curve '{curve}' to '{normalized_curve}'."
                )
            tail_copy["curve"] = normalized_curve
        fit_period = tail_copy.get("fit_period")
        if isinstance(fit_period, list) and len(fit_period) > 2:
            normalized = sorted({int(value) for value in fit_period})
            tail_copy["fit_period"] = [normalized[0], normalized[-1]]
            adjustments.append(
                f"Collapsed tail.fit_period to [{normalized[0]}, {normalized[-1]}]."
            )
        sanitized["tail"] = tail_copy
    selected = sanitized.get("selected_ultimate_by_uwy")
    if isinstance(selected, dict):
        valid_selected: dict[str, str] = {}
        dropped = 0
        for key, value in selected.items():
            normalized_method = _normalize_selected_method(value)
            if normalized_method is None:
                dropped += 1
                continue
            valid_selected[str(key)] = normalized_method
        if dropped:
            adjustments.append(
                f"Dropped {dropped} invalid selected_ultimate_by_uwy override(s) and kept only method values."
            )
        sanitized["selected_ultimate_by_uwy"] = valid_selected
    for field_name in ("drop", "drop_valuation"):
        field_value = sanitized.get(field_name)
        if isinstance(field_value, list):
            sanitized_pairs, dropped = _sanitize_drop_like_pairs(field_value)
            if dropped:
                adjustments.append(
                    f"Dropped {dropped} invalid {field_name} entr{'y' if dropped == 1 else 'ies'}."
                )
            sanitized[field_name] = sanitized_pairs
    return sanitized, adjustments


def _normalize_average_or_volume(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "volume": "volume",
        "weighted": "volume",
        "weighted_average": "volume",
        "weighted_average_all": "volume",
        "volume_weighted": "volume",
        "volume_weighted_average": "volume",
        "volume_weighted_all": "volume",
        "weighted_average_3_year": "volume",
        "simple": "simple",
        "simple_average": "simple",
        "arithmetic": "simple",
    }
    return aliases.get(normalized, "volume")


def _normalize_tail_curve_or_default(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "exponential": "exponential",
        "exp": "exponential",
        "inverse_power": "inverse_power",
        "inversepower": "inverse_power",
        "power": "inverse_power",
        "power_curve": "inverse_power",
        "powercurve": "inverse_power",
        "inverse_power_curve": "inverse_power",
        "weibull": "weibull",
    }
    return aliases.get(normalized, "weibull")


def _normalize_selected_method(value: object) -> str | None:
    normalized = str(value).strip().lower()
    if normalized in {"chainladder", "bornhuetter_ferguson"}:
        return normalized
    return None


def _sanitize_drop_like_pairs(value: list[Any]) -> tuple[list[list[str | int]], int]:
    valid_pairs: list[list[str | int]] = []
    dropped = 0
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            dropped += 1
            continue
        origin, development = pair
        if origin is None or development is None:
            dropped += 1
            continue
        if not isinstance(development, int) or isinstance(development, bool):
            dropped += 1
            continue
        valid_pairs.append([str(origin), development])
    return valid_pairs, dropped


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
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
    return str(value)
