from __future__ import annotations

from typing import Any


def build_deterministic_packet(
    *,
    plan: dict[str, Any],
    review: dict[str, Any],
    recommendation: dict[str, Any],
    evidence_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_ids = _collect_evidence_ids(evidence_packets)
    caveats = review.get("caveats") if isinstance(review.get("caveats"), list) else []
    alternatives = (
        recommendation.get("alternative_scenario_ids")
        if isinstance(recommendation.get("alternative_scenario_ids"), list)
        else []
    )
    composite_review = _extract_composite_review(evidence_packets)
    continuity_notes = _extract_continuity_notes(composite_review)
    score_breakdown = _extract_score_breakdown(composite_review)
    policy_trace = _extract_policy_trace(composite_review)
    recommended_changes = _extract_recommended_changes(composite_review)
    if not alternatives:
        alternatives = _extract_alternatives(composite_review)
    packet = {
        "plan": dict(plan),
        "review": dict(review),
        "recommendation": dict(recommendation),
        "evidence_packets": [dict(item) for item in evidence_packets],
        "composite_review": composite_review,
        "continuity_notes": continuity_notes,
        "score_breakdown": score_breakdown,
        "policy_trace": policy_trace,
        "recommended_changes": recommended_changes,
        "presentation": {
            "conclusion": recommendation.get("status"),
            "evidence_used": evidence_ids,
            "alternative_considered": alternatives,
            "key_caveat": _key_caveat(caveats, continuity_notes),
            "next_best_question": _next_best_question(review, recommendation),
        },
    }
    return packet


def _collect_evidence_ids(evidence_packets: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for packet in evidence_packets:
        provenance = packet.get("provenance") if isinstance(packet, dict) else None
        if not isinstance(provenance, dict):
            continue
        evidence_ids = provenance.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            continue
        for item in evidence_ids:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
    return ordered


def _next_best_question(
    review: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    issues = review.get("issues") if isinstance(review.get("issues"), list) else []
    if issues:
        return "Which unresolved review issue must be cleared before sign-off?"
    status = str(recommendation.get("status", "")).strip().lower()
    if status in {"recommended", "reasonable_alternative"}:
        return "What human judgment or segment-specific caveat could overturn this recommendation?"
    return "What additional evidence would justify a stronger recommendation?"


def _extract_composite_review(evidence_packets: list[dict[str, Any]]) -> dict[str, Any]:
    preferred_keys = [
        "quarter_close_review",
        "drop_review",
        "tail_review",
        "bf_suitability_review",
        "anomaly_triage",
    ]
    for key in preferred_keys:
        for packet in evidence_packets:
            if str(packet.get("evidence_key", "")).strip() != key:
                continue
            summary = (
                packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
            )
            return {
                "evidence_key": key,
                "summary": summary,
                "provenance": packet.get("provenance", {}),
                "governance": packet.get("governance", {}),
            }
    return {}


def _extract_continuity_notes(composite_review: dict[str, Any]) -> list[dict[str, Any]]:
    summary = (
        composite_review.get("summary")
        if isinstance(composite_review.get("summary"), dict)
        else {}
    )
    continuity = (
        summary.get("continuity") if isinstance(summary.get("continuity"), dict) else {}
    )
    if isinstance(continuity.get("continuity_notes"), list):
        return [
            dict(item)
            for item in continuity.get("continuity_notes", [])
            if isinstance(item, dict)
        ]
    notes = summary.get("continuity_notes")
    if isinstance(notes, list):
        return [dict(item) for item in notes if isinstance(item, dict)]
    return []


def _extract_score_breakdown(composite_review: dict[str, Any]) -> dict[str, Any]:
    summary = (
        composite_review.get("summary")
        if isinstance(composite_review.get("summary"), dict)
        else {}
    )
    candidates = (
        summary.get("top_candidates")
        if isinstance(summary.get("top_candidates"), list)
        else []
    )
    if candidates and isinstance(candidates[0], dict):
        breakdown = candidates[0].get("score_breakdown")
        if isinstance(breakdown, dict):
            return breakdown
    top_ranked = (
        summary.get("top_ranked") if isinstance(summary.get("top_ranked"), list) else []
    )
    if top_ranked and isinstance(top_ranked[0], dict):
        breakdown = top_ranked[0].get("score_breakdown")
        if isinstance(breakdown, dict):
            return breakdown
    return {}


def _extract_policy_trace(composite_review: dict[str, Any]) -> dict[str, Any]:
    summary = (
        composite_review.get("summary")
        if isinstance(composite_review.get("summary"), dict)
        else {}
    )
    recommendation = (
        summary.get("recommendation")
        if isinstance(summary.get("recommendation"), dict)
        else {}
    )
    if isinstance(recommendation.get("policy_trace"), dict):
        return dict(recommendation.get("policy_trace", {}))
    policy_trace = summary.get("policy_trace")
    if isinstance(policy_trace, dict):
        return dict(policy_trace)
    return {}


def _extract_recommended_changes(
    composite_review: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = (
        composite_review.get("summary")
        if isinstance(composite_review.get("summary"), dict)
        else {}
    )
    recommendation = (
        summary.get("recommendation")
        if isinstance(summary.get("recommendation"), dict)
        else {}
    )
    changes = recommendation.get("recommended_changes")
    if isinstance(changes, list):
        return [dict(item) for item in changes if isinstance(item, dict)]
    candidate_id = recommendation.get("candidate_id")
    top_candidates = (
        summary.get("top_candidates")
        if isinstance(summary.get("top_candidates"), list)
        else []
    )
    for item in top_candidates:
        if not isinstance(item, dict):
            continue
        if str(item.get("candidate_id", "")).strip() == str(candidate_id or "").strip():
            return [dict(item)]
    return []


def _extract_alternatives(composite_review: dict[str, Any]) -> list[str]:
    summary = (
        composite_review.get("summary")
        if isinstance(composite_review.get("summary"), dict)
        else {}
    )
    recommendation = (
        summary.get("recommendation")
        if isinstance(summary.get("recommendation"), dict)
        else {}
    )
    if isinstance(recommendation.get("alternatives"), list):
        return [
            str(item)
            for item in recommendation.get("alternatives", [])
            if str(item).strip()
        ]
    top_ranked = (
        summary.get("top_ranked") if isinstance(summary.get("top_ranked"), list) else []
    )
    return [
        str(item.get("candidate_id"))
        for item in top_ranked[1:3]
        if isinstance(item, dict) and str(item.get("candidate_id", "")).strip()
    ]


def _key_caveat(caveats: list[str], continuity_notes: list[dict[str, Any]]) -> str:
    if caveats:
        return str(caveats[0])
    for item in continuity_notes:
        message = str(item.get("message", "")).strip()
        if message:
            return message
    return ""
