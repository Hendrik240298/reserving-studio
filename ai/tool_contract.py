from __future__ import annotations

from typing import Any


def normalize_tool_result(
    *,
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    segment: str | None,
    evidence_key: str,
) -> dict[str, Any]:
    payload = dict(result)
    evidence_ids = _collect_evidence_ids(payload)
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
        "status": "ok",
    }


def _extract_run_id(payload: dict[str, Any]) -> str | None:
    run_metadata = payload.get("run_metadata")
    if isinstance(run_metadata, dict):
        value = run_metadata.get("run_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _collect_evidence_ids(payload: object) -> list[str]:
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
