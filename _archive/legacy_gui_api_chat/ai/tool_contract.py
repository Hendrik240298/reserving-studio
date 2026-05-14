from __future__ import annotations

from typing import Any

from ai.execution_records import collect_evidence_ids


def normalize_tool_result(
    *,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    segment: str | None,
    evidence_key: str,
) -> dict[str, Any]:
    payload = dict(result)
    execution_status = str(payload.get("execution_status") or "ok")
    evidence_ids = collect_evidence_ids(payload)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    governance = (
        payload.get("governance") if isinstance(payload.get("governance"), dict) else {}
    )
    uncertainty = (
        payload.get("uncertainty")
        if isinstance(payload.get("uncertainty"), dict)
        else {}
    )
    return {
        "tool_name": tool_name,
        "evidence_key": evidence_key,
        "summary": payload,
        "metrics": metrics,
        "provenance": {
            "session_id": payload.get("session_id") or args.get("session_id"),
            "segment": segment,
            "run_id": _extract_run_id(payload),
            "evidence_ids": evidence_ids,
        },
        "governance": governance,
        "uncertainty": uncertainty,
        "status": execution_status,
        "execution": {
            "status": execution_status,
            "warnings": payload.get("warnings", []),
            "material_adjustments": payload.get("material_adjustments", []),
        },
    }


def _extract_run_id(payload: dict[str, Any]) -> str | None:
    run_metadata = payload.get("run_metadata")
    if isinstance(run_metadata, dict):
        value = run_metadata.get("run_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
