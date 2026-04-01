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
    packet = {
        "plan": dict(plan),
        "review": dict(review),
        "recommendation": dict(recommendation),
        "evidence_packets": [dict(item) for item in evidence_packets],
        "presentation": {
            "conclusion": recommendation.get("status"),
            "evidence_used": evidence_ids,
            "alternative_considered": alternatives,
            "key_caveat": str(caveats[0]) if caveats else "",
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
