from __future__ import annotations

from typing import Any

from ai.plan_models import ExecutionPlan, ReviewOutcome


class ReviewerGate:
    def review(
        self,
        *,
        plan: ExecutionPlan,
        evidence_packets: list[dict[str, Any]],
    ) -> ReviewOutcome:
        collected = [
            str(item.get("evidence_key", "")).strip()
            for item in evidence_packets
            if str(item.get("evidence_key", "")).strip()
        ]
        missing = [key for key in plan.required_evidence if key not in set(collected)]
        issues: list[str] = []
        caveats: list[str] = []

        if len(set(collected)) < int(plan.minimum_evidence_count):
            issues.append("minimum_evidence_not_met")

        if self._has_unsupported_method(evidence_packets):
            issues.append("unsupported_method_or_parameter")

        if self._has_recommendation_without_evidence(evidence_packets):
            issues.append("recommendation_missing_evidence_id")

        if self._has_red_governance(evidence_packets):
            issues.append("red_governance_or_data_quality")

        if self._has_missing_provenance(evidence_packets):
            issues.append("numeric_claim_missing_provenance")

        if self._requires_continuity(plan) and not self._has_continuity_coverage(
            evidence_packets
        ):
            issues.append("continuity_memory_not_evaluated")

        if self._has_amber_governance(evidence_packets):
            caveats.append("human_review_recommended")
        if self._has_rejected_before_flag(evidence_packets):
            caveats.append("rejected_before")
        if self._has_house_preference_conflict(evidence_packets):
            caveats.append("house_preference_conflict")
        if self._has_pause_recommendation(evidence_packets):
            caveats.append("pause_recommendation")

        if issues:
            status = "hard_fail"
        elif caveats or missing:
            status = "pass_with_caveats"
        else:
            status = "pass"

        return ReviewOutcome(
            status=status,
            collected_evidence=sorted(set(collected)),
            missing_evidence=missing,
            issues=issues,
            caveats=caveats,
        )

    @staticmethod
    def validate_memory_update_proposals(
        proposals: list[dict[str, Any]] | None,
        *,
        evidence_packets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(proposals, list):
            return []
        available_evidence = ReviewerGate._collect_available_evidence_ids(
            evidence_packets
        )
        allowed_fields = {
            "segment_overview",
            "known_issues",
            "recent_quarter_notes",
            "open_items",
        }
        validated: list[dict[str, Any]] = []
        for item in proposals:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field", "")).strip()
            operation = str(item.get("operation", "")).strip().lower()
            rationale = str(item.get("rationale", "")).strip()
            if field not in allowed_fields or operation not in {"append", "replace"}:
                continue
            if not rationale:
                continue
            evidence_ids = item.get("evidence_ids")
            if isinstance(evidence_ids, list):
                normalized_ids = [
                    str(entry).strip() for entry in evidence_ids if str(entry).strip()
                ]
                if normalized_ids and not set(normalized_ids).issubset(
                    available_evidence
                ):
                    continue
            validated.append(dict(item))
        return validated

    @staticmethod
    def _has_unsupported_method(evidence_packets: list[dict[str, Any]]) -> bool:
        supported = {"chainladder", "bornhuetter_ferguson"}
        supported_tail_curves = {"weibull", "exponential", "inverse_power"}
        for packet in evidence_packets:
            summary = packet.get("summary") if isinstance(packet, dict) else None
            if not isinstance(summary, dict):
                continue
            recommendations = summary.get("top_recommendations")
            if not isinstance(recommendations, list):
                continue
            for item in recommendations:
                if not isinstance(item, dict):
                    continue
                params = item.get("proposed_parameters")
                if not isinstance(params, dict):
                    continue
                final_ultimate = params.get("final_ultimate")
                if final_ultimate is not None and str(final_ultimate) not in supported:
                    return True
                tail = params.get("tail")
                if isinstance(tail, dict):
                    curve = tail.get("curve")
                    if curve is not None and str(curve) not in supported_tail_curves:
                        return True
        return False

    @staticmethod
    def _has_recommendation_without_evidence(
        evidence_packets: list[dict[str, Any]],
    ) -> bool:
        for packet in evidence_packets:
            summary = packet.get("summary") if isinstance(packet, dict) else None
            if not isinstance(summary, dict):
                continue
            recommendations = summary.get("top_recommendations")
            if not isinstance(recommendations, list):
                continue
            for item in recommendations:
                if not isinstance(item, dict):
                    continue
                proposed = item.get("proposed_parameters")
                if not isinstance(proposed, dict) or not proposed:
                    continue
                evidence_id = str(item.get("evidence_id", "")).strip()
                if not evidence_id:
                    return True
        return False

    @staticmethod
    def _has_red_governance(evidence_packets: list[dict[str, Any]]) -> bool:
        for packet in evidence_packets:
            governance = packet.get("governance") if isinstance(packet, dict) else None
            if (
                isinstance(governance, dict)
                and str(governance.get("tier", "")).lower() == "red"
            ):
                return True
            summary = packet.get("summary") if isinstance(packet, dict) else None
            if not isinstance(summary, dict):
                continue
            top_findings = summary.get("top_findings")
            if not isinstance(top_findings, list):
                continue
            for item in top_findings:
                if not isinstance(item, dict):
                    continue
                severity = str(item.get("severity", "")).lower()
                code = str(item.get("code", ""))
                if severity == "critical" and "DATA_QUALITY" in code:
                    return True
        return False

    @staticmethod
    def _has_amber_governance(evidence_packets: list[dict[str, Any]]) -> bool:
        for packet in evidence_packets:
            governance = packet.get("governance") if isinstance(packet, dict) else None
            if (
                isinstance(governance, dict)
                and str(governance.get("tier", "")).lower() == "amber"
            ):
                return True
        return False

    @staticmethod
    def _has_missing_provenance(evidence_packets: list[dict[str, Any]]) -> bool:
        for packet in evidence_packets:
            provenance = packet.get("provenance") if isinstance(packet, dict) else None
            if not isinstance(provenance, dict):
                return True
            if not provenance.get("session_id"):
                return True
        return False

    @staticmethod
    def _requires_continuity(plan: ExecutionPlan) -> bool:
        return bool(plan.requires_continuity)

    @staticmethod
    def _has_continuity_coverage(evidence_packets: list[dict[str, Any]]) -> bool:
        for packet in evidence_packets:
            summary = packet.get("summary") if isinstance(packet, dict) else None
            if not isinstance(summary, dict):
                continue
            continuity_notes = summary.get("continuity_notes")
            continuity = summary.get("continuity")
            if isinstance(continuity_notes, list):
                return True
            if isinstance(continuity, dict) and (
                isinstance(continuity.get("continuity_notes"), list)
                or continuity.get("memory_schema_version") is not None
            ):
                return True
        return False

    @staticmethod
    def _has_rejected_before_flag(evidence_packets: list[dict[str, Any]]) -> bool:
        for packet in evidence_packets:
            summary = packet.get("summary") if isinstance(packet, dict) else None
            if not isinstance(summary, dict):
                continue
            if ReviewerGate._policy_trace_has_flag(summary, "rejected_before"):
                return True
        return False

    @staticmethod
    def _has_house_preference_conflict(evidence_packets: list[dict[str, Any]]) -> bool:
        for packet in evidence_packets:
            summary = packet.get("summary") if isinstance(packet, dict) else None
            if not isinstance(summary, dict):
                continue
            policy_trace = (
                summary.get("policy_trace")
                if isinstance(summary.get("policy_trace"), dict)
                else {}
            )
            conflicts = policy_trace.get("house_preference_conflicts")
            if isinstance(conflicts, list) and conflicts:
                return True
            recommendation = (
                summary.get("recommendation")
                if isinstance(summary.get("recommendation"), dict)
                else {}
            )
            recommendation_trace = (
                recommendation.get("policy_trace")
                if isinstance(recommendation.get("policy_trace"), dict)
                else {}
            )
            recommendation_conflicts = recommendation_trace.get(
                "house_preference_conflicts"
            )
            if isinstance(recommendation_conflicts, list) and recommendation_conflicts:
                return True
            for item in list(summary.get("top_candidates", [])) + list(
                summary.get("top_ranked", [])
            ):
                if not isinstance(item, dict):
                    continue
                candidate_trace = (
                    item.get("policy_trace")
                    if isinstance(item.get("policy_trace"), dict)
                    else {}
                )
                candidate_conflicts = candidate_trace.get("house_preference_conflicts")
                if isinstance(candidate_conflicts, list) and candidate_conflicts:
                    return True
        return False

    @staticmethod
    def _has_pause_recommendation(evidence_packets: list[dict[str, Any]]) -> bool:
        for packet in evidence_packets:
            summary = packet.get("summary") if isinstance(packet, dict) else None
            if not isinstance(summary, dict):
                continue
            if bool(summary.get("pause_recommendation", False)):
                return True
            recommendation = (
                summary.get("recommendation")
                if isinstance(summary.get("recommendation"), dict)
                else {}
            )
            if (
                str(recommendation.get("status", "")).strip().lower()
                == "hold_for_review"
            ):
                return True
        return False

    @staticmethod
    def _policy_trace_has_flag(summary: dict[str, Any], flag_name: str) -> bool:
        policy_trace = (
            summary.get("policy_trace")
            if isinstance(summary.get("policy_trace"), dict)
            else {}
        )
        if bool(policy_trace.get(flag_name)):
            return True
        recommendation = (
            summary.get("recommendation")
            if isinstance(summary.get("recommendation"), dict)
            else {}
        )
        recommendation_trace = (
            recommendation.get("policy_trace")
            if isinstance(recommendation.get("policy_trace"), dict)
            else {}
        )
        if bool(recommendation_trace.get(flag_name)):
            return True
        for item in list(summary.get("top_candidates", [])) + list(
            summary.get("top_ranked", [])
        ):
            if not isinstance(item, dict):
                continue
            candidate_trace = (
                item.get("policy_trace")
                if isinstance(item.get("policy_trace"), dict)
                else {}
            )
            if bool(candidate_trace.get(flag_name)):
                return True
        return False

    @staticmethod
    def _collect_available_evidence_ids(
        evidence_packets: list[dict[str, Any]],
    ) -> set[str]:
        available: set[str] = set()
        for packet in evidence_packets:
            provenance = packet.get("provenance") if isinstance(packet, dict) else None
            if not isinstance(provenance, dict):
                continue
            evidence_ids = provenance.get("evidence_ids")
            if not isinstance(evidence_ids, list):
                continue
            for item in evidence_ids:
                value = str(item).strip()
                if value:
                    available.add(value)
        return available
