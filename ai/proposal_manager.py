from __future__ import annotations

from typing import Any
import uuid

from ai.basis_manager import BasisManager
from ai.control_plane_types import (
    AcceptedAnalysisBasis,
    BasisTransitionRecord,
    ProposalBasis,
    normalize_accepted_analysis_basis,
    normalize_basis_transition_record,
    normalize_proposal_basis,
    utc_now_iso,
)


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

        recommended_basis_key = str(recommendation.get("recommended_basis_key") or "").strip()
        if not recommended_basis_key:
            return {}

        accepted_basis = BasisManager.accepted_basis(
            accepted_analysis_basis=accepted_analysis_basis
        )
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
