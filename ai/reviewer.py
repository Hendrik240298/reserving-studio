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

        if self._has_amber_governance(evidence_packets):
            caveats.append("human_review_recommended")

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
