from __future__ import annotations

from typing import Any
import uuid

from ai.basis_manager import BasisManager
from ai.control_plane_types import (
    AcceptedAnalysisBasis,
    BasisTransitionRecord,
    ProposalBasis,
    basis_key_from_parameters,
    normalize_accepted_analysis_basis,
    normalize_basis_transition_record,
    normalize_proposal_basis,
    utc_now_iso,
)


MAX_COMBINED_DROP_PROPOSAL_COUNT = 5


class ProposalManager:
    @staticmethod
    def build_from_deterministic_packet(
        *,
        deterministic_packet: dict[str, Any],
        accepted_analysis_basis: dict[str, Any] | None,
        basis_cache: dict[str, Any] | None,
    ) -> ProposalBasis:
        if not isinstance(deterministic_packet, dict) or not deterministic_packet:
            return {}
        recommendation = (
            deterministic_packet.get("recommendation")
            if isinstance(deterministic_packet.get("recommendation"), dict)
            else {}
        )
        recommendation_status = str(recommendation.get("status") or "").strip().lower()
        if recommendation_status not in {"recommended", "reasonable_alternative"}:
            return {}

        plan = deterministic_packet.get("plan") if isinstance(deterministic_packet.get("plan"), dict) else {}
        review = deterministic_packet.get("review") if isinstance(deterministic_packet.get("review"), dict) else {}
        basis_behavior = plan.get("basis_behavior") if isinstance(plan.get("basis_behavior"), list) else []
        if "proposal_possible" not in {str(item) for item in basis_behavior}:
            return {}

        composite_review = (
            deterministic_packet.get("composite_review")
            if isinstance(deterministic_packet.get("composite_review"), dict)
            else {}
        )
        provenance = composite_review.get("provenance") if isinstance(composite_review.get("provenance"), dict) else {}

        accepted_basis = BasisManager.accepted_basis(
            accepted_analysis_basis=accepted_analysis_basis
        )
        combined_drop_proposal = _build_combined_drop_review_proposal(
            composite_review=composite_review,
            accepted_basis=accepted_basis,
            plan=plan,
            review=review,
            recommendation=recommendation,
            recommendation_status=recommendation_status,
            provenance=provenance,
        )
        if combined_drop_proposal:
            return normalize_proposal_basis(combined_drop_proposal)

        recommended_basis_key = str(recommendation.get("recommended_basis_key") or "").strip()
        if not recommended_basis_key:
            return {}

        resolved_basis = BasisManager.lookup_basis_by_key(
            basis_key=recommended_basis_key,
            current_basis=accepted_basis,
            basis_cache=basis_cache if isinstance(basis_cache, dict) else {},
        )
        if not resolved_basis:
            return {}
        if not str(resolved_basis.get("basis_key") or "").strip():
            return {}

        if (
            accepted_basis
            and str(accepted_basis.get("basis_key") or "").strip()
            and str(accepted_basis.get("basis_key") or "").strip()
            == str(resolved_basis.get("basis_key") or "").strip()
        ):
            return {}

        proposal = {
            "proposal_id": f"proposal-{uuid.uuid4().hex[:12]}",
            "basis_key": resolved_basis.get("basis_key"),
            "basis_type": resolved_basis.get("basis_type"),
            "scenario_label": resolved_basis.get("scenario_label"),
            "parameters": resolved_basis.get("parameters", {}),
            "origin_execution_id": provenance.get("run_id"),
            "origin_workflow": str(plan.get("playbook") or "").strip() or None,
            "recommendation_strength": recommendation_status or None,
            "caveats": [
                str(item)
                for item in [
                    *(review.get("caveats") if isinstance(review.get("caveats"), list) else []),
                    *(recommendation.get("rationale") if isinstance(recommendation.get("rationale"), list) else []),
                ]
                if str(item).strip()
            ],
            "status": "pending",
            "created_at": utc_now_iso(),
            "session_id": resolved_basis.get("session_id"),
            "scenario_id": resolved_basis.get("scenario_id"),
            "candidate_id": resolved_basis.get("candidate_id"),
            "source_tool": resolved_basis.get("source_tool"),
            "source_review_type": resolved_basis.get("source_review_type"),
            "matches_active_session": bool(resolved_basis.get("matches_active_session", False)),
        }
        return normalize_proposal_basis(proposal)

    @staticmethod
    def attach_to_message(
        proposal: dict[str, Any], *, message_id: str | None
    ) -> ProposalBasis:
        normalized = normalize_proposal_basis(proposal)
        if not normalized:
            return {}
        if isinstance(message_id, str) and message_id.strip():
            normalized["presented_in_message_id"] = message_id.strip()
        return normalized

    @staticmethod
    def mark_superseded(
        proposal: dict[str, Any], *, superseded_by_proposal_id: str | None
    ) -> ProposalBasis:
        normalized = normalize_proposal_basis(proposal)
        if not normalized:
            return {}
        normalized["status"] = "superseded"
        normalized["superseded_by_proposal_id"] = (
            str(superseded_by_proposal_id or "").strip() or None
        )
        return normalized

    @staticmethod
    def accept(
        proposal: dict[str, Any],
        *,
        chat_id: str,
        current_accepted_basis: dict[str, Any] | None,
    ) -> tuple[AcceptedAnalysisBasis, ProposalBasis, BasisTransitionRecord]:
        normalized = normalize_proposal_basis(proposal)
        if not normalized:
            return {}, {}, {}
        accepted_basis = normalize_accepted_analysis_basis(
            {
                "basis_key": normalized.get("basis_key"),
                "basis_type": normalized.get("basis_type"),
                "scenario_label": normalized.get("scenario_label"),
                "parameters": normalized.get("parameters", {}),
                "session_id": normalized.get("session_id"),
                "scenario_id": normalized.get("scenario_id"),
                "candidate_id": normalized.get("candidate_id"),
                "source_tool": normalized.get("source_tool"),
                "source_review_type": normalized.get("source_review_type"),
                "matches_active_session": normalized.get("matches_active_session", False),
                "is_active_session": normalized.get("matches_active_session", False),
            },
            accepted_at=utc_now_iso(),
            accepted_from_execution_id=normalized.get("origin_execution_id"),
        )
        updated_proposal = normalize_proposal_basis({
            **normalized,
            "status": "accepted",
        })
        current_basis = BasisManager.accepted_basis(
            accepted_analysis_basis=current_accepted_basis
        )
        transition = normalize_basis_transition_record(
            {
                "transition_id": f"transition-{uuid.uuid4().hex[:12]}",
                "chat_id": chat_id,
                "from_basis_key": current_basis.get("basis_key"),
                "to_basis_key": accepted_basis.get("basis_key"),
                "transition_type": "proposal_accepted",
                "origin_proposal_id": normalized.get("proposal_id"),
                "created_at": utc_now_iso(),
            }
        )
        return accepted_basis, updated_proposal, transition

    @staticmethod
    def reject(
        proposal: dict[str, Any], *, chat_id: str
    ) -> tuple[ProposalBasis, BasisTransitionRecord]:
        normalized = normalize_proposal_basis(proposal)
        if not normalized:
            return {}, {}
        updated_proposal = normalize_proposal_basis({
            **normalized,
            "status": "rejected",
        })
        transition = normalize_basis_transition_record(
            {
                "transition_id": f"transition-{uuid.uuid4().hex[:12]}",
                "chat_id": chat_id,
                "from_basis_key": normalized.get("basis_key"),
                "to_basis_key": normalized.get("basis_key"),
                "transition_type": "proposal_rejected",
                "origin_proposal_id": normalized.get("proposal_id"),
                "created_at": utc_now_iso(),
            }
        )
        return updated_proposal, transition


def _build_combined_drop_review_proposal(
    *,
    composite_review: dict[str, Any],
    accepted_basis: dict[str, Any],
    plan: dict[str, Any],
    review: dict[str, Any],
    recommendation: dict[str, Any],
    recommendation_status: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    summary = (
        composite_review.get("summary")
        if isinstance(composite_review.get("summary"), dict)
        else {}
    )
    if str(summary.get("review_type") or composite_review.get("evidence_key") or "").strip() != "drop_review":
        return {}
    candidates = summary.get("top_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return {}

    accepted_parameters = (
        accepted_basis.get("parameters")
        if isinstance(accepted_basis.get("parameters"), dict)
        else {}
    )
    analysis_basis = (
        summary.get("analysis_basis")
        if isinstance(summary.get("analysis_basis"), dict)
        else {}
    )
    analysis_parameters = (
        analysis_basis.get("parameters")
        if isinstance(analysis_basis.get("parameters"), dict)
        else {}
    )
    parameters = _basis_parameters_with_defaults(accepted_parameters or analysis_parameters)
    drops = [
        normalized
        for item in parameters.get("drop", [])
        if (normalized := _normalize_drop_pair(item)) is not None
    ]
    seen = {tuple(item) for item in drops}
    selected_candidate_ids: list[str] = []

    for candidate in candidates:
        if len(drops) >= MAX_COMBINED_DROP_PROPOSAL_COUNT:
            break
        if not isinstance(candidate, dict) or not _drop_candidate_is_eligible(candidate):
            continue
        candidate_parameters = (
            candidate.get("parameters")
            if isinstance(candidate.get("parameters"), dict)
            else {}
        )
        candidate_drops = candidate_parameters.get("drop")
        if not isinstance(candidate_drops, list):
            continue
        added = False
        for item in candidate_drops:
            if len(drops) >= MAX_COMBINED_DROP_PROPOSAL_COUNT:
                break
            normalized = _normalize_drop_pair(item)
            if normalized is None:
                continue
            identity = tuple(normalized)
            if identity in seen:
                continue
            seen.add(identity)
            drops.append(normalized)
            added = True
        if added:
            candidate_id = str(candidate.get("candidate_id") or candidate.get("scenario_id") or "").strip()
            if candidate_id:
                selected_candidate_ids.append(candidate_id)

    if len(selected_candidate_ids) < 2:
        return {}
    parameters["drop"] = drops
    basis_key = basis_key_from_parameters(parameters)
    if not basis_key:
        return {}
    candidate_id = f"combined_drop_review_top_{len(selected_candidate_ids)}"
    return {
        "proposal_id": f"proposal-{uuid.uuid4().hex[:12]}",
        "basis_key": basis_key,
        "basis_type": "review_candidate",
        "scenario_label": candidate_id,
        "parameters": parameters,
        "origin_execution_id": provenance.get("run_id"),
        "origin_workflow": str(plan.get("playbook") or "drop_review").strip() or "drop_review",
        "recommendation_strength": recommendation_status or None,
        "caveats": [
            str(item)
            for item in [
                *(review.get("caveats") if isinstance(review.get("caveats"), list) else []),
                *(recommendation.get("rationale") if isinstance(recommendation.get("rationale"), list) else []),
                f"combined_drop_candidates={','.join(selected_candidate_ids)}",
            ]
            if str(item).strip()
        ],
        "status": "pending",
        "created_at": utc_now_iso(),
        "session_id": accepted_basis.get("session_id") or summary.get("session_id"),
        "scenario_id": f"review_{candidate_id}_{basis_key}",
        "candidate_id": candidate_id,
        "source_tool": "tool_run_drop_review",
        "source_review_type": "drop_review",
        "matches_active_session": False,
    }


def _basis_parameters_with_defaults(parameters: dict[str, Any]) -> dict[str, Any]:
    source = dict(parameters) if isinstance(parameters, dict) else {}
    tail = source.get("tail") if isinstance(source.get("tail"), dict) else {}
    return {
        "average": source.get("average", "volume"),
        "drop": [list(item) for item in source.get("drop", []) if isinstance(item, list | tuple)],
        "drop_valuation": source.get("drop_valuation", []),
        "tail": {
            "curve": tail.get("curve", "weibull"),
            "attachment_age": tail.get("attachment_age"),
            "projection_period": tail.get("projection_period", 0),
            "fit_period": tail.get("fit_period", []),
        },
        "bf_apriori": source.get("bf_apriori", {}),
        "final_ultimate": source.get("final_ultimate", "chainladder"),
        "selected_ultimate_by_uwy": source.get("selected_ultimate_by_uwy", {}),
    }


def _drop_candidate_is_eligible(candidate: dict[str, Any]) -> bool:
    recommendation_class = str(candidate.get("recommendation_class") or "").strip().lower()
    if recommendation_class not in {"recommend", "recommended", "reasonable_alternative"}:
        return False
    policy_trace = candidate.get("policy_trace") if isinstance(candidate.get("policy_trace"), dict) else {}
    if bool(policy_trace.get("rejected_before", False)):
        return False
    if str(policy_trace.get("governance_tier") or "").strip().lower() == "red":
        return False
    conflicts = policy_trace.get("house_preference_conflicts")
    if isinstance(conflicts, list) and conflicts:
        return False
    return True


def _normalize_drop_pair(value: object) -> list[str | int] | None:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return None
    origin, development = value[0], value[1]
    if origin is None or development is None:
        return None
    try:
        development_age = int(development)
    except (TypeError, ValueError):
        return None
    return [str(origin), development_age]
