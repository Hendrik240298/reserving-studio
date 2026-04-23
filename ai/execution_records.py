from __future__ import annotations

from typing import Any
import uuid

from ai.control_plane_types import normalize_execution_record, utc_now_iso
from ai.request_validation import RequestValidationResult


def attach_execution_metadata(
    summary: dict[str, Any],
    *,
    tool_name: str,
    validation: RequestValidationResult,
    session_id: str | None,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    summary_payload = dict(summary)
    record = build_execution_record(
        tool_name=tool_name,
        workflow_name=workflow_name,
        validation=validation,
        result_summary=summary_payload,
        session_id=session_id,
    )
    summary_payload["execution_status"] = record["execution_status"]
    summary_payload["warnings"] = record["warnings"]
    summary_payload["material_adjustments"] = record["material_adjustments"]
    summary_payload["execution_record"] = record
    return summary_payload


def build_execution_record(
    *,
    tool_name: str,
    validation: RequestValidationResult,
    result_summary: dict[str, Any],
    session_id: str | None,
    workflow_name: str | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    return normalize_execution_record(
        {
            "execution_id": f"exec-{uuid.uuid4().hex[:12]}",
            "workflow_name": workflow_name,
            "tool_name": tool_name,
            "requested_inputs": dict(validation.requested_inputs),
            "effective_inputs": dict(validation.effective_inputs),
            "execution_status": validation.execution_status,
            "warnings": list(validation.warnings),
            "material_adjustments": list(validation.material_adjustments),
            "result_summary": dict(result_summary),
            "session_id": session_id,
            "chat_id": chat_id,
            "evidence_ids": collect_evidence_ids(result_summary),
            "created_at": utc_now_iso(),
        }
    )


def collect_evidence_ids(payload: object) -> list[str]:
    collected: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            maybe_id = node.get("evidence_id")
            if isinstance(maybe_id, str) and maybe_id.strip():
                collected.append(maybe_id.strip())
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in collected:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def execution_status_is_successful(status: str | None) -> bool:
    normalized = str(status or "").strip()
    return normalized in {
        "executed_exactly",
        "executed_with_non_material_normalization",
    }


def latest_execution_record(
    execution_records: object,
) -> dict[str, Any]:
    if not isinstance(execution_records, list):
        return {}
    for item in reversed(execution_records):
        if isinstance(item, dict):
            return dict(item)
    return {}
