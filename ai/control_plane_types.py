from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Literal, TypedDict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class BasisSource(TypedDict, total=False):
    tool_name: str | None
    workflow_name: str | None
    session_id: str | None


class AcceptedAnalysisBasis(TypedDict, total=False):
    basis_key: str | None
    basis_type: str
    scenario_label: str | None
    parameters: dict[str, Any]
    source: BasisSource
    accepted_at: str | None
    accepted_from_execution_id: str | None
    matches_active_session: bool
    session_id: str | None
    scenario_id: str | None
    candidate_id: str | None
    scenario_signature: str | None
    source_tool: str | None
    source_review_type: str | None
    is_active_session: bool


class ProposalBasis(TypedDict, total=False):
    proposal_id: str
    basis_key: str | None
    basis_type: str
    scenario_label: str | None
    parameters: dict[str, Any]
    origin_execution_id: str | None
    origin_workflow: str | None
    recommendation_strength: str | None
    caveats: list[str]
    status: Literal["pending", "accepted", "rejected", "superseded"]
    created_at: str | None
    session_id: str | None
    scenario_id: str | None
    candidate_id: str | None
    source_tool: str | None
    source_review_type: str | None
    matches_active_session: bool
    presented_in_message_id: str | None
    superseded_by_proposal_id: str | None


class PreviewBasis(TypedDict, total=False):
    basis_key: str | None
    parameters: dict[str, Any]
    origin_execution_id: str | None
    expires_after_turn: bool
    session_id: str | None
    scenario_id: str | None
    basis_type: str


class ExecutionRecord(TypedDict, total=False):
    execution_id: str
    workflow_name: str | None
    tool_name: str | None
    requested_inputs: dict[str, Any]
    effective_inputs: dict[str, Any]
    execution_status: str
    warnings: list[str]
    material_adjustments: list[str]
    result_summary: dict[str, Any]
    session_id: str | None
    chat_id: str | None
    evidence_ids: list[str]
    created_at: str | None


class BasisTransitionRecord(TypedDict, total=False):
    transition_id: str
    chat_id: str | None
    from_basis_key: str | None
    to_basis_key: str | None
    transition_type: str
    origin_proposal_id: str | None
    created_at: str | None


def normalize_basis_parameters(parameters: object) -> dict[str, Any]:
    if not isinstance(parameters, dict) or not parameters:
        return {}
    try:
        return json.loads(json.dumps(parameters, sort_keys=True))
    except (TypeError, ValueError):
        return dict(parameters)


def basis_key_from_parameters(parameters: object) -> str | None:
    normalized = normalize_basis_parameters(parameters)
    if not normalized:
        return None
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def scenario_label_from_basis_payload(basis: object) -> str | None:
    if not isinstance(basis, dict):
        return None
    for key in ("scenario_label", "candidate_id", "scenario_id"):
        value = str(basis.get(key) or "").strip()
        if value:
            return value
    basis_type = str(basis.get("basis_type") or "").strip().lower()
    if basis_type == "baseline":
        return "active baseline session"
    if basis_type == "bespoke":
        return "custom parameter basis"
    return None


def normalize_accepted_analysis_basis(
    basis: object,
    *,
    accepted_at: str | None = None,
    accepted_from_execution_id: str | None = None,
) -> AcceptedAnalysisBasis:
    if not isinstance(basis, dict) or not basis:
        return {}
    parameters = normalize_basis_parameters(basis.get("parameters"))
    basis_key = str(basis.get("basis_key") or "").strip() or basis_key_from_parameters(
        parameters
    )
    scenario_signature = str(basis.get("scenario_signature") or "").strip() or basis_key
    scenario_id = str(basis.get("scenario_id") or "").strip() or None
    candidate_id = str(basis.get("candidate_id") or "").strip() or None
    session_id = str(basis.get("session_id") or "").strip() or None
    source_tool = str(basis.get("source_tool") or "").strip() or None
    source_review_type = str(basis.get("source_review_type") or "").strip() or None
    matches_active_session = bool(
        basis.get("matches_active_session", basis.get("is_active_session", False))
    )
    normalized: AcceptedAnalysisBasis = {
        "basis_key": basis_key or None,
        "basis_type": str(basis.get("basis_type") or "baseline"),
        "scenario_label": scenario_label_from_basis_payload(basis),
        "parameters": parameters,
        "source": {
            "tool_name": source_tool,
            "workflow_name": source_review_type,
            "session_id": session_id,
        },
        "accepted_at": accepted_at
        if accepted_at is not None
        else (str(basis.get("accepted_at") or "").strip() or None),
        "accepted_from_execution_id": accepted_from_execution_id
        if accepted_from_execution_id is not None
        else (str(basis.get("accepted_from_execution_id") or "").strip() or None),
        "matches_active_session": matches_active_session,
        "session_id": session_id,
        "scenario_id": scenario_id,
        "candidate_id": candidate_id,
        "scenario_signature": scenario_signature or None,
        "source_tool": source_tool,
        "source_review_type": source_review_type,
        "is_active_session": matches_active_session,
    }
    return normalized


def normalize_proposal_basis(proposal: object) -> ProposalBasis:
    if not isinstance(proposal, dict) or not proposal:
        return {}
    parameters = normalize_basis_parameters(proposal.get("parameters"))
    basis_key = str(proposal.get("basis_key") or "").strip() or basis_key_from_parameters(
        parameters
    )
    normalized: ProposalBasis = {
        "proposal_id": str(proposal.get("proposal_id") or "").strip(),
        "basis_key": basis_key or None,
        "basis_type": str(proposal.get("basis_type") or "baseline"),
        "scenario_label": scenario_label_from_basis_payload(proposal),
        "parameters": parameters,
        "origin_execution_id": str(proposal.get("origin_execution_id") or "").strip()
        or None,
        "origin_workflow": str(proposal.get("origin_workflow") or "").strip()
        or None,
        "recommendation_strength": str(
            proposal.get("recommendation_strength") or ""
        ).strip()
        or None,
        "caveats": [
            str(item)
            for item in proposal.get("caveats", [])
            if str(item).strip()
        ]
        if isinstance(proposal.get("caveats"), list)
        else [],
        "status": str(proposal.get("status") or "pending"),
        "created_at": str(proposal.get("created_at") or "").strip() or None,
        "session_id": str(proposal.get("session_id") or "").strip() or None,
        "scenario_id": str(proposal.get("scenario_id") or "").strip() or None,
        "candidate_id": str(proposal.get("candidate_id") or "").strip() or None,
        "source_tool": str(proposal.get("source_tool") or "").strip() or None,
        "source_review_type": str(proposal.get("source_review_type") or "").strip()
        or None,
        "matches_active_session": bool(
            proposal.get(
                "matches_active_session", proposal.get("is_active_session", False)
            )
        ),
        "presented_in_message_id": str(
            proposal.get("presented_in_message_id") or ""
        ).strip()
        or None,
        "superseded_by_proposal_id": str(
            proposal.get("superseded_by_proposal_id") or ""
        ).strip()
        or None,
    }
    if not normalized["proposal_id"]:
        normalized["proposal_id"] = f"proposal-{(basis_key or 'unknown')[:12]}"
    return normalized


def normalize_preview_basis(preview: object) -> PreviewBasis:
    if not isinstance(preview, dict) or not preview:
        return {}
    parameters = normalize_basis_parameters(preview.get("parameters"))
    basis_key = str(preview.get("basis_key") or "").strip() or basis_key_from_parameters(
        parameters
    )
    return {
        "basis_key": basis_key or None,
        "parameters": parameters,
        "origin_execution_id": str(preview.get("origin_execution_id") or "").strip()
        or None,
        "expires_after_turn": bool(preview.get("expires_after_turn", True)),
        "session_id": str(preview.get("session_id") or "").strip() or None,
        "scenario_id": str(preview.get("scenario_id") or "").strip() or None,
        "basis_type": str(preview.get("basis_type") or "bespoke"),
    }


def normalize_execution_record(record: object) -> ExecutionRecord:
    if not isinstance(record, dict) or not record:
        return {}
    return {
        "execution_id": str(record.get("execution_id") or "").strip(),
        "workflow_name": str(record.get("workflow_name") or "").strip() or None,
        "tool_name": str(record.get("tool_name") or "").strip() or None,
        "requested_inputs": dict(record.get("requested_inputs") or {}),
        "effective_inputs": dict(record.get("effective_inputs") or {}),
        "execution_status": str(record.get("execution_status") or "").strip(),
        "warnings": [
            str(item) for item in record.get("warnings", []) if str(item).strip()
        ]
        if isinstance(record.get("warnings"), list)
        else [],
        "material_adjustments": [
            str(item)
            for item in record.get("material_adjustments", [])
            if str(item).strip()
        ]
        if isinstance(record.get("material_adjustments"), list)
        else [],
        "result_summary": dict(record.get("result_summary") or {}),
        "session_id": str(record.get("session_id") or "").strip() or None,
        "chat_id": str(record.get("chat_id") or "").strip() or None,
        "evidence_ids": [
            str(item) for item in record.get("evidence_ids", []) if str(item).strip()
        ]
        if isinstance(record.get("evidence_ids"), list)
        else [],
        "created_at": str(record.get("created_at") or "").strip() or None,
    }


def normalize_basis_transition_record(record: object) -> BasisTransitionRecord:
    if not isinstance(record, dict) or not record:
        return {}
    return {
        "transition_id": str(record.get("transition_id") or "").strip(),
        "chat_id": str(record.get("chat_id") or "").strip() or None,
        "from_basis_key": str(record.get("from_basis_key") or "").strip() or None,
        "to_basis_key": str(record.get("to_basis_key") or "").strip() or None,
        "transition_type": str(record.get("transition_type") or "").strip(),
        "origin_proposal_id": str(record.get("origin_proposal_id") or "").strip()
        or None,
        "created_at": str(record.get("created_at") or "").strip() or None,
    }
