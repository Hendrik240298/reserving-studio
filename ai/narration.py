from __future__ import annotations

from typing import Any

from ai.basis_manager import BasisManager
from ai.execution_records import latest_execution_record


ANSWER_CONTRACTS: dict[str, dict[str, list[str]]] = {
    "recommendation_with_proposal": {
        "required_sections": [
            "basis_used",
            "what_was_reviewed",
            "recommendation",
            "caveats",
            "proposal_status",
        ],
        "blocked_claims": [
            "basis_changed_without_acceptance",
            "applied_pending_proposal",
        ],
    },
    "review_summary_only": {
        "required_sections": [
            "basis_used",
            "what_was_reviewed",
            "findings",
            "caveats",
        ],
        "blocked_claims": [
            "basis_changed_without_acceptance",
            "proposal_created_when_disallowed",
        ],
    },
    "observational_explanation": {
        "required_sections": [
            "basis_or_data_scope",
            "what_was_inspected",
            "main_explanation",
            "limitations",
        ],
        "blocked_claims": [
            "basis_changed_without_acceptance",
            "unsupported_recommendation",
        ],
    },
    "data_summary": {
        "required_sections": [
            "data_scope",
            "what_was_shown",
            "key_values_or_summary",
            "limitations",
        ],
        "blocked_claims": [
            "basis_changed_without_acceptance",
            "unsupported_recommendation",
        ],
    },
}


class NarrationAssembler:
    @staticmethod
    def build(
        *,
        deterministic_packet: dict[str, Any],
        accepted_analysis_basis: dict[str, Any] | None,
        proposal_basis: dict[str, Any] | None,
        execution_records: list[dict[str, Any]] | None,
        guardrail_state: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        plan = _dict(deterministic_packet.get("plan"))
        review = _dict(deterministic_packet.get("review"))
        recommendation = _dict(deterministic_packet.get("recommendation"))
        evidence_packets = _list_of_dicts(deterministic_packet.get("evidence_packets"))
        presentation = _dict(deterministic_packet.get("presentation"))
        workflow_name = str(
            plan.get("workflow_name") or plan.get("playbook") or "unclassified"
        ).strip()
        answer_contract = str(plan.get("answer_contract") or "review_summary_only").strip()
        contract = ANSWER_CONTRACTS.get(answer_contract, ANSWER_CONTRACTS["review_summary_only"])
        accepted_basis = _dict(accepted_analysis_basis)
        proposal = _dict(proposal_basis)
        latest_record = latest_execution_record(execution_records)
        execution = _execution_payload(latest_record, evidence_packets)
        caveats = _dedupe(
            [
                *_string_list(review.get("caveats")),
                *_string_list(review.get("issues")),
                *[f"Missing evidence: {item}" for item in _string_list(review.get("missing_evidence"))],
                *[str(item.get("message")) for item in _list_of_dicts(deterministic_packet.get("continuity_notes")) if str(item.get("message") or "").strip()],
                str(presentation.get("key_caveat") or ""),
                *_string_list(execution.get("warnings")),
                *_string_list(execution.get("material_adjustments")),
            ]
        )
        proposal_exists = bool(proposal) and str(proposal.get("status") or "") == "pending"
        basis_changed = False
        blocked_claims = list(contract["blocked_claims"])
        if proposal_exists:
            blocked_claims.extend(["proposal_is_accepted", "proposal_changed_basis"])
        else:
            blocked_claims.extend(["pending_proposal_when_none", "natural_language_acceptance"])
        if str(execution.get("status") or "") not in {
            "",
            "executed_exactly",
            "executed_with_non_material_normalization",
        }:
            blocked_claims.append("success_style_execution_claim")

        return {
            "schema_version": 1,
            "answer_contract": answer_contract,
            "workflow_name": workflow_name,
            "basis": {
                "current_basis_label": BasisManager.label(accepted_basis),
                "accepted_basis": accepted_basis,
                "proposal_basis": proposal,
                "basis_changed": basis_changed,
            },
            "reviewed": {
                "scope": _review_scope(workflow_name, plan, evidence_packets),
                "tools_run": _tools_run(plan, evidence_packets),
                "evidence_count": len(evidence_packets),
                "evidence_ids": _evidence_ids(presentation, evidence_packets),
                "review_status": review.get("status"),
            },
            "recommendation": {
                "status": recommendation.get("status"),
                "summary": recommendation.get("summary"),
                "recommended_label": _recommended_label(recommendation, proposal),
                "recommended_basis_key": proposal.get("basis_key")
                or recommendation.get("recommended_basis_key"),
                "rationale": _string_list(recommendation.get("rationale")),
            },
            "supporting_evidence": _supporting_evidence(deterministic_packet),
            "caveats": caveats,
            "execution": execution,
            "proposal": {
                "exists": proposal_exists,
                "status": proposal.get("status"),
                "proposal_id": proposal.get("proposal_id"),
                "label": proposal.get("scenario_label")
                or proposal.get("candidate_id")
                or proposal.get("scenario_id"),
                "instruction": _proposal_instruction(proposal_exists),
            },
            "guardrails": dict(guardrail_state or {}),
            "required_answer_sections": list(contract["required_sections"]),
            "blocked_claims": _dedupe(blocked_claims),
        }


def build_narration_prompt(narration_packet: dict[str, Any]) -> str:
    return (
        "Use this deterministic narration packet as the answer scaffold. "
        "Include every required_answer_sections item. Do not make any blocked_claims. "
        "Do not say the Analysis Basis changed unless basis.basis_changed is true. "
        "If proposal.exists is true, say the recommendation is pending explicit Yes/No acceptance. "
        "If proposal.exists is false, do not mention a pending proposal and do not ask the user to accept anything. "
        "Acceptance and rejection happen only through the proposal card buttons, never through natural-language replies. "
        "Write concise user-facing prose grounded only in this packet.\n"
        + _json_dumps(narration_packet)
    )


def render_narration_fallback(narration_packet: dict[str, Any]) -> str:
    basis = _dict(narration_packet.get("basis"))
    reviewed = _dict(narration_packet.get("reviewed"))
    recommendation = _dict(narration_packet.get("recommendation"))
    proposal = _dict(narration_packet.get("proposal"))
    execution = _dict(narration_packet.get("execution"))
    supporting = _dict(narration_packet.get("supporting_evidence"))
    caveats = _string_list(narration_packet.get("caveats"))
    sections = [str(basis.get("current_basis_label") or "Basis used: current baseline session.")]
    sections.append(
        "### What was reviewed\n"
        + str(reviewed.get("scope") or "A deterministic reserving workflow was run.").rstrip(".")
        + "."
    )
    status = str(recommendation.get("status") or "watch")
    summary = str(recommendation.get("summary") or "No stronger recommendation was produced.")
    sections.append(f"### Conclusion\n**{status}**. {summary}")
    review_lines = _fallback_review_lines(supporting)
    if review_lines:
        sections.append("### Evidence Highlights\n" + "\n".join(f"- {line}." for line in review_lines))
    if caveats:
        sections.append("### Caveats\n" + "\n".join(f"- {item}." for item in caveats[:5]))
    else:
        sections.append("### Caveats\n- None material from the deterministic packet.")
    if proposal.get("exists"):
        label = str(proposal.get("label") or "proposed basis")
        sections.append(
            "### Proposal Status\n"
            f"{label} is pending explicit Yes/No acceptance. The Analysis Basis is unchanged until accepted."
        )
    execution_status = str(execution.get("status") or "")
    if execution_status and execution_status not in {
        "executed_exactly",
        "executed_with_non_material_normalization",
    }:
        sections.append(
            "### Execution Note\nLatest execution status is "
            f"{execution_status}; use effective inputs and warnings, not success-style wording."
        )
    return "\n\n".join(section for section in sections if section.strip())


def _fallback_review_lines(supporting: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for review in _list_of_dicts(supporting.get("reviews")):
        label = str(review.get("review_type") or review.get("evidence_key") or "review").strip()
        finding_count = review.get("finding_count")
        if finding_count is not None:
            lines.append(f"{label}: {finding_count} finding(s) identified")
        if review.get("pause_recommendation") is not None:
            pause_label = "pause recommendations" if bool(review.get("pause_recommendation")) else "recommendations may continue"
            lines.append(f"{label}: {pause_label}")
        recommendation = _dict(review.get("recommendation"))
        candidate = str(
            recommendation.get("candidate_id")
            or recommendation.get("scenario_id")
            or recommendation.get("summary")
            or "no candidate"
        ).strip()
        recommendation_class = str(
            recommendation.get("recommendation_class")
            or recommendation.get("status")
            or "watch"
        ).strip()
        if label and candidate:
            lines.append(f"{label}: {candidate} ({recommendation_class})")
        candidates = _list_of_dicts(review.get("top_candidates"))
        if candidates:
            ranked = []
            for item in candidates[:3]:
                item_label = str(item.get("candidate_id") or item.get("summary") or "candidate").strip()
                score = item.get("score")
                if score is None:
                    ranked.append(item_label)
                else:
                    ranked.append(f"{item_label}, score {score}")
            if ranked:
                lines.append(f"top ranked {label}: " + ", ".join(ranked))
        findings = _list_of_dicts(review.get("top_findings"))
        for item in findings[:3]:
            message = str(item.get("message") or item.get("code") or "finding").strip()
            severity = str(item.get("severity") or "").strip()
            prefix = f"[{severity}] " if severity else ""
            if message:
                lines.append(f"{label}: {prefix}{message}")
    return lines


def _execution_payload(
    latest_record: dict[str, Any],
    evidence_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    if latest_record:
        return {
            "status": latest_record.get("execution_status"),
            "warnings": _string_list(latest_record.get("warnings")),
            "material_adjustments": _string_list(latest_record.get("material_adjustments")),
            "execution_id": latest_record.get("execution_id"),
            "tool_name": latest_record.get("tool_name"),
        }
    for packet in reversed(evidence_packets):
        execution = _dict(packet.get("execution"))
        if execution:
            return {
                "status": execution.get("status"),
                "warnings": _string_list(execution.get("warnings")),
                "material_adjustments": _string_list(execution.get("material_adjustments")),
                "execution_id": None,
                "tool_name": packet.get("tool_name"),
            }
    return {
        "status": "",
        "warnings": [],
        "material_adjustments": [],
        "execution_id": None,
        "tool_name": None,
    }


def _review_scope(
    workflow_name: str,
    plan: dict[str, Any],
    evidence_packets: list[dict[str, Any]],
) -> str:
    goal = str(plan.get("goal") or "").strip()
    if goal:
        return goal
    labels = {
        "quarter_close_review": "Quarter-close review",
        "data_anomaly_triage": "Data anomaly triage",
        "multi_review": "Multi-workflow review",
        "drop_review": "Drop review",
        "movement_review": "Movement review",
        "late_emergence_review": "Late emergence review",
        "reserve_change_explanation": "Reserve change explanation",
        "tail_selection": "Tail selection review",
        "method_suitability_review": "Method suitability review",
        "scenario_recommendation": "Scenario recommendation review",
        "data_exploration": "Data exploration",
    }
    if workflow_name in labels:
        return labels[workflow_name]
    keys = [str(item.get("evidence_key")) for item in evidence_packets if item.get("evidence_key")]
    return ", ".join(keys) or "Deterministic workflow"


def _tools_run(plan: dict[str, Any], evidence_packets: list[dict[str, Any]]) -> list[str]:
    steps = _list_of_dicts(plan.get("steps"))
    tools = [str(item.get("tool_name")) for item in steps if str(item.get("tool_name") or "").strip()]
    if tools:
        return _dedupe(tools)
    return _dedupe(
        [str(item.get("tool_name")) for item in evidence_packets if str(item.get("tool_name") or "").strip()]
    )


def _evidence_ids(
    presentation: dict[str, Any], evidence_packets: list[dict[str, Any]]
) -> list[str]:
    ids = _string_list(presentation.get("evidence_used"))
    if ids:
        return _dedupe(ids)
    collected: list[str] = []
    for packet in evidence_packets:
        provenance = _dict(packet.get("provenance"))
        collected.extend(_string_list(provenance.get("evidence_ids")))
    return _dedupe(collected)


def _recommended_label(
    recommendation: dict[str, Any], proposal: dict[str, Any]
) -> str | None:
    for value in (
        proposal.get("scenario_label"),
        proposal.get("candidate_id"),
        proposal.get("scenario_id"),
        recommendation.get("recommended_basis_id"),
        recommendation.get("recommended_scenario_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _supporting_evidence(packet: dict[str, Any]) -> dict[str, Any]:
    composite_review = _dict(packet.get("composite_review"))
    composite_summary = _dict(composite_review.get("summary"))
    composite_reviews = _list_of_dicts(packet.get("composite_reviews"))
    if not composite_reviews and composite_review:
        composite_reviews = [composite_review]
    return {
        "reviews": [_compact_supporting_review(item) for item in composite_reviews[:4]],
        "top_candidates": _list_of_dicts(composite_summary.get("top_candidates"))[:3],
        "top_ranked": _list_of_dicts(composite_summary.get("top_ranked"))[:3],
        "recommended_changes": _list_of_dicts(packet.get("recommended_changes"))[:3],
        "score_breakdown": _dict(packet.get("score_breakdown")),
        "policy_trace": _dict(packet.get("policy_trace")),
        "continuity_notes": _list_of_dicts(packet.get("continuity_notes"))[:3],
    }


def _compact_supporting_review(review: dict[str, Any]) -> dict[str, Any]:
    summary = _dict(review.get("summary"))
    recommendation = _dict(summary.get("recommendation"))
    return {
        "evidence_key": review.get("evidence_key"),
        "review_type": summary.get("review_type"),
        "recommendation": recommendation,
        "top_candidates": _list_of_dicts(summary.get("top_candidates"))[:3],
        "top_ranked": _list_of_dicts(summary.get("top_ranked"))[:3],
        "top_findings": _list_of_dicts(summary.get("top_findings"))[:3],
        "finding_count": summary.get("finding_count"),
        "pause_recommendation": summary.get("pause_recommendation"),
        "continuity_notes": _list_of_dicts(summary.get("continuity_notes"))[:3],
        "policy_trace": _dict(summary.get("policy_trace")),
    }


def _proposal_instruction(proposal_exists: bool) -> str:
    if proposal_exists:
        return "This recommendation is pending acceptance through the proposal card buttons. Analysis Basis is unchanged."
    return "No pending basis proposal is attached to this answer."


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True)
