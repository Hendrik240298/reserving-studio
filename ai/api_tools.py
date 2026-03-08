from __future__ import annotations

from datetime import date, datetime
import json
import logging
import os
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError


class ReservingApiTools:
    def __init__(self, *, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._observability_enabled = os.environ.get(
            "AI_OBSERVABILITY", "1"
        ).strip().lower() not in {"0", "false", "off"}
        self._logger = logging.getLogger(__name__)

    @property
    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "tool_get_session",
                    "description": "Get current session state for a segment",
                    "parameters": {
                        "type": "object",
                        "properties": {"segment": {"type": "string"}},
                        "required": ["segment"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_recalculate",
                    "description": "Recalculate reserving results for a session",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "average": {"type": "string"},
                            "drop": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": ["string", "integer"]},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                            "drop_valuation": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": ["string", "integer"]},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                            "tail": {
                                "type": "object",
                                "properties": {
                                    "curve": {"type": "string"},
                                    "attachment_age": {
                                        "type": ["integer", "null"],
                                    },
                                    "projection_period": {"type": "integer"},
                                    "fit_period": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                },
                                "required": [
                                    "curve",
                                    "projection_period",
                                    "fit_period",
                                ],
                            },
                            "bf_apriori": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                            "final_ultimate": {
                                "type": "string",
                                "enum": ["chainladder", "bornhuetter_ferguson"],
                            },
                            "selected_ultimate_by_uwy": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string",
                                    "enum": ["chainladder", "bornhuetter_ferguson"],
                                },
                            },
                        },
                        "required": [
                            "session_id",
                            "average",
                            "drop",
                            "drop_valuation",
                            "tail",
                            "bf_apriori",
                            "final_ultimate",
                            "selected_ultimate_by_uwy",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_run_diagnostics",
                    "description": "Run deterministic diagnostics for a session",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "diagnostic_profile": {"type": ["string", "null"]},
                            "include_recommendations": {"type": "boolean"},
                        },
                        "required": ["session_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_iterate_diagnostics",
                    "description": (
                        "Run iterative scenario search across drop, tail, and BF apriori candidates"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "max_scenarios": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "include_baseline": {"type": "boolean"},
                        },
                        "required": ["session_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_get_results",
                    "description": "Get latest reserving results payload for a session",
                    "parameters": {
                        "type": "object",
                        "properties": {"session_id": {"type": "string"}},
                        "required": ["session_id"],
                    },
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "tool_get_session":
            segment = str(arguments["segment"])
            return self._request_json("GET", f"/v1/sessions/{segment}")
        if name == "tool_recalculate":
            return self._request_json("POST", "/v1/reserving/recalculate", arguments)
        if name == "tool_run_diagnostics":
            payload = {
                "session_id": arguments["session_id"],
                "diagnostic_profile": arguments.get("diagnostic_profile"),
                "include_recommendations": bool(
                    arguments.get("include_recommendations", True)
                ),
            }
            return self._request_json("POST", "/v1/diagnostics/run", payload)
        if name == "tool_iterate_diagnostics":
            payload = {
                "session_id": arguments["session_id"],
                "max_scenarios": int(arguments.get("max_scenarios", 24)),
                "include_baseline": bool(arguments.get("include_baseline", True)),
            }
            return self._request_json("POST", "/v1/diagnostics/iterate", payload)
        if name == "tool_get_results":
            session_id = str(arguments["session_id"])
            return self._request_json("GET", f"/v1/results/{session_id}")
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
        return self._request_json("POST", "/v1/workflows/from-dataframes", payload)

    def _request_json(
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
