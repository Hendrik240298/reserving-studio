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
        if not best_scenario_id:
            return RecommendationDecision(
                status="watch",
                summary="Scenario comparison did not identify a stable tested best scenario.",
                rationale=["missing_best_scenario_id"],
            )
        alternatives = [
            str(item.get("scenario_id"))
            for item in top_scenarios[1:3]
            if isinstance(item, dict) and str(item.get("scenario_id", "")).strip()
        ]

        if improvement > 0.5 and governance_tier == "green" and review.status == "pass":
            return RecommendationDecision(
                status="recommended",
                summary="A tested scenario improved diagnostics materially without triggering governance concerns.",
                rationale=[f"score_improvement={improvement:.3f}", "governance=green"],
                recommended_scenario_id=best_scenario_id,
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
                recommended_scenario_id=best_scenario_id,
                alternative_scenario_ids=alternatives,
            )
        return RecommendationDecision(
            status="watch",
            summary="Tested alternatives did not improve enough to support a stronger recommendation.",
            rationale=[f"score_improvement={improvement:.3f}"],
            recommended_scenario_id=best_scenario_id,
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
