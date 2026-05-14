from __future__ import annotations

from typing import Any

from ai.plan_models import RecommendationDecision, ReviewOutcome


class RecommendationPolicy:
    def decide(
        self,
        *,
        review: ReviewOutcome,
        evidence_packets: list[dict[str, Any]],
    ) -> RecommendationDecision:
        if review.status == "hard_fail":
            return RecommendationDecision(
                status="hold_for_review",
                summary="Pause recommendation until reviewer issues are resolved.",
                rationale=list(review.issues),
            )

        quarter_close_packet = _find_packet(evidence_packets, "quarter_close_review")
        if isinstance(quarter_close_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_quarter_close(quarter_close_packet),
                review,
            )

        drop_review_packet = _find_packet(evidence_packets, "drop_review")
        tail_review_packet = _find_packet(evidence_packets, "tail_review")
        if isinstance(drop_review_packet, dict) and isinstance(tail_review_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_multiple_candidate_reviews(
                    [drop_review_packet, tail_review_packet]
                ),
                review,
            )

        if isinstance(drop_review_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_composite_candidate_review(
                    drop_review_packet,
                    review_type="drop review",
                ),
                review,
            )

        if isinstance(tail_review_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_composite_candidate_review(
                    tail_review_packet,
                    review_type="tail review",
                ),
                review,
            )

        bf_packet = _find_packet(evidence_packets, "bf_suitability_review")
        if isinstance(bf_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_bf_review(bf_packet),
                review,
            )

        anomaly_packet = _find_packet(evidence_packets, "anomaly_triage")
        if isinstance(anomaly_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_anomaly_triage(anomaly_packet),
                review,
            )

        derived_drop_packet = _find_packet(evidence_packets, "derived_drop_scenario")
        if isinstance(derived_drop_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_derived_drop(derived_drop_packet),
                review,
            )

        combined_drop_packet = _find_packet(evidence_packets, "combined_drop_recalculation")
        if isinstance(combined_drop_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_combined_drop_recalculation(combined_drop_packet),
                review,
            )

        bf_recalculation_packet = _find_packet(evidence_packets, "bf_recalculation")
        if isinstance(bf_recalculation_packet, dict):
            return _apply_review_caveats_to_decision(
                _decision_from_bf_recalculation(bf_recalculation_packet),
                review,
            )

        iteration_packet = next(
            (
                item
                for item in evidence_packets
                if str(item.get("evidence_key", "")) == "scenario_comparison"
            ),
            None,
        )
        if not isinstance(iteration_packet, dict):
            status = (
                "watch" if review.status == "pass_with_caveats" else "not_recommended"
            )
            return RecommendationDecision(
                status=status,
                summary="No tested scenario comparison was available for a stronger recommendation.",
                rationale=list(review.caveats or review.missing_evidence),
            )

        summary = iteration_packet.get("summary")
        if not isinstance(summary, dict):
            return RecommendationDecision(
                status="watch",
                summary="Scenario comparison payload was incomplete.",
            )

        baseline = (
            summary.get("baseline") if isinstance(summary.get("baseline"), dict) else {}
        )
        top_scenarios = (
            summary.get("top_scenarios")
            if isinstance(summary.get("top_scenarios"), list)
            else []
        )
        best = (
            top_scenarios[0]
            if top_scenarios and isinstance(top_scenarios[0], dict)
            else {}
        )
        baseline_score = _to_float(baseline.get("score"))
        best_score = _to_float(best.get("score"))
        improvement = baseline_score - best_score
        governance_tier = str(
            best.get("governance_tier") or baseline.get("governance_tier") or ""
        ).lower()
        best_scenario_id = _to_optional_str(best.get("scenario_id"))
        best_basis_key = _to_optional_str(best.get("basis_key"))
        if not best_scenario_id or not best_basis_key:
            return RecommendationDecision(
                status="watch",
                summary="Scenario comparison did not identify a stable tested best basis.",
                rationale=["missing_best_scenario_id_or_basis_key"],
            )
        alternatives = [
            str(item.get("scenario_id"))
            for item in top_scenarios[1:3]
            if isinstance(item, dict) and str(item.get("scenario_id", "")).strip()
        ]
        alternative_basis_keys = [
            str(item.get("basis_key"))
            for item in top_scenarios[1:3]
            if isinstance(item, dict) and str(item.get("basis_key", "")).strip()
        ]

        if improvement > 0.5 and governance_tier == "green" and review.status == "pass":
            return RecommendationDecision(
                status="recommended",
                summary="A tested scenario improved diagnostics materially without triggering governance concerns.",
                rationale=[f"score_improvement={improvement:.3f}", "governance=green"],
                recommended_basis_key=best_basis_key,
                recommended_scenario_id=best_scenario_id,
                alternative_basis_keys=alternative_basis_keys,
                alternative_scenario_ids=alternatives,
            )
        if improvement > 0.1:
            return RecommendationDecision(
                status="reasonable_alternative",
                summary="A tested scenario improved diagnostics, but review caveats remain.",
                rationale=[
                    f"score_improvement={improvement:.3f}",
                    f"governance={governance_tier or 'unknown'}",
                    *review.caveats,
                ],
                recommended_basis_key=best_basis_key,
                recommended_scenario_id=best_scenario_id,
                alternative_basis_keys=alternative_basis_keys,
                alternative_scenario_ids=alternatives,
            )
        return RecommendationDecision(
            status="watch",
            summary="Tested alternatives did not improve enough to support a stronger recommendation.",
            rationale=[f"score_improvement={improvement:.3f}"],
            recommended_basis_key=best_basis_key,
            recommended_scenario_id=best_scenario_id,
            alternative_basis_keys=alternative_basis_keys,
            alternative_scenario_ids=alternatives,
        )


def _to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _to_optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _basis_key_for_candidate(
    *,
    summary: dict[str, Any],
    candidate_id: str | None,
    scenario_id: str | None,
) -> str | None:
    labels = {str(item).strip() for item in (candidate_id, scenario_id) if str(item or "").strip()}
    if not labels:
        return None
    for key in ("top_candidates", "top_ranked"):
        items = summary.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_labels = {
                str(item.get("candidate_id") or "").strip(),
                str(item.get("scenario_id") or "").strip(),
            }
            if labels.intersection(item_labels):
                return _to_optional_str(item.get("basis_key"))
    return None


def _basis_keys_for_labels(
    *,
    summary: dict[str, Any],
    labels: list[str],
) -> list[str]:
    wanted = {str(item).strip() for item in labels if str(item or "").strip()}
    if not wanted:
        return []
    keys: list[str] = []
    for key in ("top_candidates", "top_ranked"):
        items = summary.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_labels = {
                str(item.get("candidate_id") or "").strip(),
                str(item.get("scenario_id") or "").strip(),
            }
            basis_key = _to_optional_str(item.get("basis_key"))
            if basis_key and wanted.intersection(item_labels) and basis_key not in keys:
                keys.append(basis_key)
    return keys


def _find_packet(
    evidence_packets: list[dict[str, Any]],
    evidence_key: str,
) -> dict[str, Any] | None:
    for item in evidence_packets:
        if str(item.get("evidence_key", "")).strip() == evidence_key:
            return item
    return None


def _decision_from_quarter_close(packet: dict[str, Any]) -> RecommendationDecision:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    recommendation = (
        summary.get("recommendation")
        if isinstance(summary.get("recommendation"), dict)
        else {}
    )
    status = str(recommendation.get("status", "watch")).strip() or "watch"
    changes = (
        recommendation.get("recommended_changes")
        if isinstance(recommendation.get("recommended_changes"), list)
        else []
    )
    recommended_id = None
    recommended_basis_key = None
    recommended_basis_id = None
    if changes and isinstance(changes[0], dict):
        recommended_id = _to_optional_str(changes[0].get("candidate_id"))
        recommended_basis_key = _to_optional_str(changes[0].get("basis_key"))
        recommended_basis_id = _to_optional_str(changes[0].get("scenario_id"))
    alternatives = [
        str(item.get("candidate_id"))
        for item in changes[1:3]
        if isinstance(item, dict) and str(item.get("candidate_id", "")).strip()
    ]
    alternative_basis_ids = [
        str(item.get("scenario_id"))
        for item in changes[1:3]
        if isinstance(item, dict) and str(item.get("scenario_id", "")).strip()
    ]
    alternative_basis_keys = [
        str(item.get("basis_key"))
        for item in changes[1:3]
        if isinstance(item, dict) and str(item.get("basis_key", "")).strip()
    ]
    rationale = [
        *(
            recommendation.get("why_reasonable")
            if isinstance(recommendation.get("why_reasonable"), list)
            else []
        ),
        *(
            recommendation.get("caveats")
            if isinstance(recommendation.get("caveats"), list)
            else []
        ),
    ]
    return RecommendationDecision(
        status=status,
        summary=str(recommendation.get("summary", "")).strip()
        or "Quarter-close review completed.",
        rationale=[str(item) for item in rationale if str(item).strip()],
        recommended_basis_key=recommended_basis_key,
        recommended_scenario_id=recommended_id,
        recommended_basis_id=recommended_basis_id,
        alternative_basis_keys=alternative_basis_keys,
        alternative_scenario_ids=alternatives,
        alternative_basis_ids=alternative_basis_ids,
    )


def _decision_from_composite_candidate_review(
    packet: dict[str, Any],
    *,
    review_type: str,
) -> RecommendationDecision:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    recommendation = (
        summary.get("recommendation")
        if isinstance(summary.get("recommendation"), dict)
        else {}
    )
    recommendation_class = (
        str(recommendation.get("recommendation_class", "watch")).strip().lower()
    )
    status_map = {
        "recommend": "recommended",
        "reasonable_alternative": "reasonable_alternative",
        "watch": "watch",
        "avoid": "not_recommended",
    }
    status = status_map.get(recommendation_class, "watch")
    candidate_id = _to_optional_str(recommendation.get("candidate_id"))
    scenario_id = _to_optional_str(recommendation.get("scenario_id"))
    recommended_basis_key = _basis_key_for_candidate(
        summary=summary,
        candidate_id=candidate_id,
        scenario_id=scenario_id,
    )
    alternatives = [
        str(item)
        for item in recommendation.get("alternatives", [])
        if str(item).strip()
    ]
    alternative_basis_ids = [
        str(item)
        for item in recommendation.get("alternative_scenario_ids", [])
        if str(item).strip()
    ]
    alternative_basis_keys = _basis_keys_for_labels(
        summary=summary,
        labels=[*alternatives, *alternative_basis_ids],
    )
    rationale = [
        recommendation_class,
        *(
            recommendation.get("caveats")
            if isinstance(recommendation.get("caveats"), list)
            else []
        ),
    ]
    return RecommendationDecision(
        status=status,
        summary=str(recommendation.get("summary", "")).strip()
        or f"Composite {review_type} completed.",
        rationale=[str(item) for item in rationale if str(item).strip()],
        recommended_basis_key=recommended_basis_key,
        recommended_scenario_id=candidate_id,
        recommended_basis_id=scenario_id,
        alternative_basis_keys=alternative_basis_keys,
        alternative_scenario_ids=alternatives,
        alternative_basis_ids=alternative_basis_ids,
    )


def _decision_from_multiple_candidate_reviews(
    packets: list[dict[str, Any]],
) -> RecommendationDecision:
    labels: list[str] = []
    rationale: list[str] = []
    for packet in packets:
        summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
        review_type = str(
            summary.get("review_type") or packet.get("evidence_key") or "review"
        ).strip()
        recommendation = (
            summary.get("recommendation")
            if isinstance(summary.get("recommendation"), dict)
            else {}
        )
        recommendation_class = str(
            recommendation.get("recommendation_class")
            or recommendation.get("status")
            or "watch"
        ).strip()
        label = str(
            recommendation.get("candidate_id")
            or recommendation.get("scenario_id")
            or recommendation.get("summary")
            or "no candidate"
        ).strip()
        labels.append(f"{review_type}: {label}")
        rationale.append(f"{review_type}={recommendation_class or 'watch'}")
    return RecommendationDecision(
        status="reviewed",
        summary="Drop and tail reviews completed; use the separate per-review recommendations rather than a single combined basis proposal.",
        rationale=[*labels, *rationale],
    )


def _decision_from_bf_review(packet: dict[str, Any]) -> RecommendationDecision:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    overall_class = str(summary.get("overall_class", "inconclusive")).strip().lower()
    if overall_class == "bf_preferred":
        return RecommendationDecision(
            status="reasonable_alternative",
            summary="BF suitability review indicates BF is preferred for at least part of the portfolio.",
            rationale=["bf_preferred"],
        )
    if overall_class == "mixed":
        return RecommendationDecision(
            status="reasonable_alternative",
            summary="BF suitability review is mixed by UWY and requires actuarial judgment by maturity.",
            rationale=["mixed"],
        )
    if overall_class == "cl_preferred":
        return RecommendationDecision(
            status="watch",
            summary="BF suitability review supports staying with Chainladder for the current selection context.",
            rationale=["cl_preferred"],
        )
    return RecommendationDecision(
        status="watch",
        summary="BF suitability review is inconclusive without stronger apriori support or further review.",
        rationale=[overall_class or "inconclusive"],
    )


def _decision_from_anomaly_triage(packet: dict[str, Any]) -> RecommendationDecision:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    if bool(summary.get("pause_recommendation", False)):
        return RecommendationDecision(
            status="hold_for_review",
            summary="Anomaly triage indicates recommendations should pause until the flagged issues are reviewed.",
            rationale=["pause_recommendation"],
        )
    return RecommendationDecision(
        status="watch",
        summary="Anomaly triage completed without a deterministic pause recommendation.",
        rationale=["triage_complete"],
    )


def _decision_from_derived_drop(packet: dict[str, Any]) -> RecommendationDecision:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    basis_key = _to_optional_str(summary.get("basis_key"))
    scenario_id = _to_optional_str(summary.get("scenario_id"))
    if not basis_key:
        return RecommendationDecision(
            status="watch",
            summary="Derived drop scenario did not produce a stable basis key.",
            rationale=["missing_derived_drop_basis_key"],
        )
    score_delta = _to_float(summary.get("score_delta"))
    direction = "improved" if score_delta < 0 else "changed"
    return RecommendationDecision(
        status="reasonable_alternative",
        summary=f"A derived drop scenario was tested and {direction} the diagnostic score by {abs(score_delta):.3f}.",
        rationale=[
            f"score_delta={score_delta:.3f}",
            f"drop_count={summary.get('drop_count')}",
        ],
        recommended_basis_key=basis_key,
        recommended_scenario_id=scenario_id,
    )


def _decision_from_combined_drop_recalculation(
    packet: dict[str, Any],
) -> RecommendationDecision:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    analysis_basis = (
        summary.get("analysis_basis")
        if isinstance(summary.get("analysis_basis"), dict)
        else {}
    )
    recommendation = (
        summary.get("recommendation")
        if isinstance(summary.get("recommendation"), dict)
        else {}
    )
    basis_key = _to_optional_str(recommendation.get("basis_key")) or _to_optional_str(
        analysis_basis.get("basis_key")
    )
    scenario_id = _to_optional_str(recommendation.get("scenario_id")) or _to_optional_str(
        analysis_basis.get("scenario_id")
    )
    if not basis_key:
        return RecommendationDecision(
            status="watch",
            summary="Combined drop recalculation did not produce a stable basis key.",
            rationale=["missing_combined_drop_basis_key"],
        )
    return RecommendationDecision(
        status="reasonable_alternative",
        summary=str(recommendation.get("summary") or "").strip()
        or "Combined prior drop candidates were recalculated as one tested basis.",
        rationale=["combined_prior_drop_candidates"],
        recommended_basis_key=basis_key,
        recommended_scenario_id=_to_optional_str(recommendation.get("candidate_id")),
        recommended_basis_id=scenario_id,
    )


def _decision_from_bf_recalculation(
    packet: dict[str, Any],
) -> RecommendationDecision:
    summary = packet.get("summary") if isinstance(packet.get("summary"), dict) else {}
    analysis_basis = (
        summary.get("analysis_basis")
        if isinstance(summary.get("analysis_basis"), dict)
        else {}
    )
    basis_key = _to_optional_str(analysis_basis.get("basis_key"))
    if not basis_key:
        return RecommendationDecision(
            status="watch",
            summary="BF recalculation did not produce a stable basis key.",
            rationale=["missing_bf_basis_key"],
        )
    parameters = (
        analysis_basis.get("parameters")
        if isinstance(analysis_basis.get("parameters"), dict)
        else {}
    )
    selected = parameters.get("selected_ultimate_by_uwy")
    selected_count = len(selected) if isinstance(selected, dict) else 0
    return RecommendationDecision(
        status="reasonable_alternative",
        summary=f"BF was applied incrementally to {selected_count} selected underwriting year(s), preserving the existing drop and tail settings.",
        rationale=["bf_incremental_recalculation"],
        recommended_basis_key=basis_key,
        recommended_scenario_id="bf_incremental_recalculation",
    )


def _apply_review_caveats_to_decision(
    decision: RecommendationDecision,
    review: ReviewOutcome,
) -> RecommendationDecision:
    status = decision.status
    rationale = list(decision.rationale)
    if "pause_recommendation" in review.caveats:
        status = "hold_for_review"
        rationale.append("pause_recommendation")
    elif status == "recommended" and review.status == "pass_with_caveats":
        status = "reasonable_alternative"
    if "rejected_before" in review.caveats and status == "recommended":
        status = "reasonable_alternative"
        rationale.append("rejected_before")
    if "house_preference_conflict" in review.caveats and status == "recommended":
        status = "reasonable_alternative"
        rationale.append("house_preference_conflict")
    if review.status == "pass_with_caveats":
        rationale.extend(review.caveats)
    return RecommendationDecision(
        status=status,
        summary=decision.summary,
        rationale=rationale,
        recommended_basis_key=decision.recommended_basis_key,
        recommended_scenario_id=decision.recommended_scenario_id,
        recommended_basis_id=decision.recommended_basis_id,
        alternative_basis_keys=list(decision.alternative_basis_keys),
        alternative_scenario_ids=list(decision.alternative_scenario_ids),
        alternative_basis_ids=list(decision.alternative_basis_ids),
    )
