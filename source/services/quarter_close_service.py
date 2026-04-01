from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

import pandas as pd

from source.config_manager import ConfigManager
from source.reserving import Reserving
from source.services.assumption_review_service import (
    AssumptionReviewService,
    BaselineContext,
)
from source.services.scenario_evaluation_service import ScenarioEvaluationService
from source.services.segment_memory_service import SegmentMemoryService
from source.services.valuation_snapshot_service import ValuationSnapshotService


class QuarterCloseService:
    def __init__(
        self,
        *,
        evaluation_service: ScenarioEvaluationService | None = None,
        assumption_review_service: AssumptionReviewService | None = None,
        valuation_snapshot_service: ValuationSnapshotService | None = None,
        segment_memory_service: SegmentMemoryService | None = None,
    ) -> None:
        self._evaluation_service = evaluation_service or ScenarioEvaluationService()
        self._assumption_review_service = (
            assumption_review_service or AssumptionReviewService()
        )
        self._valuation_snapshot_service = (
            valuation_snapshot_service or ValuationSnapshotService()
        )
        self._segment_memory_service = segment_memory_service or SegmentMemoryService()

    def run_review(
        self,
        *,
        segment: str,
        reserving: Reserving,
        claims_df: pd.DataFrame,
        premium_df: pd.DataFrame,
        baseline_params: dict[str, Any],
        config: ConfigManager | None = None,
        segment_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_params = self._clone_params(baseline_params)
        normalized_memory = self._segment_memory_service.load(
            segment_memory,
            segment=segment,
        )
        workflow_run_id = str(uuid.uuid4())
        self._evaluation_service.apply_params_to_reserving(
            reserving=reserving,
            params=normalized_params,
        )
        try:
            current_snapshot = self._valuation_snapshot_service.build_current_snapshot(
                reserving=reserving,
                claims_df=claims_df,
                premium_df=premium_df,
            )
            prior_proxy_snapshot: dict[str, Any] = {}
            comparison_limitations: list[str] = []
            try:
                prior_proxy_snapshot = (
                    self._valuation_snapshot_service.build_prior_proxy_snapshot(
                        claims_df=claims_df,
                        premium_df=premium_df,
                        params=normalized_params,
                        config=config,
                    )
                )
            except Exception as exc:
                comparison_limitations.append(
                    f"prior_proxy_unavailable: {str(exc).strip()}"
                )
            delta_summary = self._valuation_snapshot_service.build_delta_summary(
                current_snapshot=current_snapshot,
                prior_snapshot=prior_proxy_snapshot,
            )
            baseline_context = self._assumption_review_service.build_baseline_context(
                segment=segment,
                reserving=reserving,
                baseline_params=normalized_params,
            )
            drop_review = self._assumption_review_service.review_drops(
                segment=segment,
                reserving=reserving,
                baseline_params=normalized_params,
                segment_memory=normalized_memory,
                baseline_context=baseline_context,
            )
            tail_review = self._assumption_review_service.review_tail(
                segment=segment,
                reserving=reserving,
                baseline_params=normalized_params,
                segment_memory=normalized_memory,
                baseline_context=baseline_context,
            )
            bf_review = self._assumption_review_service.review_bf_suitability(
                segment=segment,
                reserving=reserving,
                baseline_params=normalized_params,
                segment_memory=normalized_memory,
                baseline_context=baseline_context,
            )
            continuity = self._build_continuity_summary(
                segment_memory=normalized_memory,
                drop_review=drop_review,
                tail_review=tail_review,
                bf_review=bf_review,
            )
            scenario_summary = self._build_scenario_summary(
                drop_review=drop_review,
                tail_review=tail_review,
                bf_review=bf_review,
            )
            evidence_ids = self._collect_evidence_ids(baseline_context)
            recommendation = self._build_recommendation(
                comparison_limitations=comparison_limitations,
                anomaly_triage=baseline_context.anomaly_triage,
                drop_review=drop_review,
                tail_review=tail_review,
                bf_review=bf_review,
                continuity=continuity,
            )
            return {
                "review_type": "quarter_close",
                "comparison": {
                    "current_snapshot": current_snapshot,
                    "prior_proxy_snapshot": prior_proxy_snapshot,
                    "delta_summary": delta_summary,
                    "limitations": comparison_limitations,
                },
                "diagnostics": {
                    "anomaly_triage": baseline_context.anomaly_triage,
                    "movement": baseline_context.movement_diagnostics,
                    "ldf_consistency": baseline_context.ldf_consistency,
                    "late_emergence": baseline_context.late_emergence,
                },
                "assumption_reviews": {
                    "drop_review": drop_review,
                    "tail_review": tail_review,
                    "bf_suitability": bf_review,
                },
                "scenario_summary": scenario_summary,
                "continuity": continuity,
                "recommendation": recommendation,
                "evidence_ids": evidence_ids,
                "run_metadata": {
                    "workflow_run_id": workflow_run_id,
                    "generated_at": self._utc_now(),
                    "comparison_basis": delta_summary.get("comparison_basis"),
                    "current_data_fingerprint": current_snapshot.get(
                        "data_fingerprint"
                    ),
                    "prior_proxy_data_fingerprint": prior_proxy_snapshot.get(
                        "data_fingerprint"
                    ),
                    "memory_schema_version": continuity.get("memory_schema_version"),
                    "component_run_ids": {
                        "baseline": self._run_id_from_payload(
                            baseline_context.evaluation.run_metadata
                        ),
                        "drop_review": self._run_id_from_payload(
                            drop_review.get("run_metadata")
                        ),
                        "tail_review": self._run_id_from_payload(
                            tail_review.get("run_metadata")
                        ),
                        "bf_suitability": self._run_id_from_payload(
                            bf_review.get("run_metadata")
                        ),
                        "anomaly_triage": self._run_id_from_payload(
                            baseline_context.anomaly_triage.get("run_metadata")
                        ),
                    },
                },
            }
        finally:
            self._evaluation_service.apply_params_to_reserving(
                reserving=reserving,
                params=normalized_params,
            )

    def build_pack(self, *, review_result: dict[str, Any]) -> dict[str, Any]:
        comparison = (
            review_result.get("comparison")
            if isinstance(review_result.get("comparison"), dict)
            else {}
        )
        diagnostics = (
            review_result.get("diagnostics")
            if isinstance(review_result.get("diagnostics"), dict)
            else {}
        )
        assumption_reviews = (
            review_result.get("assumption_reviews")
            if isinstance(review_result.get("assumption_reviews"), dict)
            else {}
        )
        scenario_summary = (
            review_result.get("scenario_summary")
            if isinstance(review_result.get("scenario_summary"), dict)
            else {}
        )
        continuity = (
            review_result.get("continuity")
            if isinstance(review_result.get("continuity"), dict)
            else {}
        )
        recommendation = (
            review_result.get("recommendation")
            if isinstance(review_result.get("recommendation"), dict)
            else {}
        )
        run_metadata = (
            review_result.get("run_metadata")
            if isinstance(review_result.get("run_metadata"), dict)
            else {}
        )
        return {
            "review_type": "quarter_close_pack",
            "pack_metadata": {
                "generated_at": self._utc_now(),
                "workflow_run_id": run_metadata.get("workflow_run_id"),
                "comparison_basis": run_metadata.get("comparison_basis"),
                "memory_schema_version": run_metadata.get("memory_schema_version"),
            },
            "sections": {
                "data_changes": {
                    "delta_summary": comparison.get("delta_summary", {}),
                    "current_snapshot": comparison.get("current_snapshot", {}).get(
                        "summary", {}
                    )
                    if isinstance(comparison.get("current_snapshot"), dict)
                    else {},
                    "prior_proxy_snapshot": comparison.get(
                        "prior_proxy_snapshot", {}
                    ).get("summary", {})
                    if isinstance(comparison.get("prior_proxy_snapshot"), dict)
                    else {},
                    "movement_summary": diagnostics.get("movement", {}).get(
                        "summary", {}
                    )
                    if isinstance(diagnostics.get("movement"), dict)
                    else {},
                    "limitations": comparison.get("limitations", []),
                },
                "assumptions_retested": {
                    "drop_review": self._review_packet_summary(
                        assumption_reviews.get("drop_review")
                    ),
                    "tail_review": self._review_packet_summary(
                        assumption_reviews.get("tail_review")
                    ),
                    "bf_suitability": {
                        "overall_class": assumption_reviews.get(
                            "bf_suitability", {}
                        ).get("overall_class")
                        if isinstance(assumption_reviews.get("bf_suitability"), dict)
                        else None,
                        "summary": assumption_reviews.get("bf_suitability", {}).get(
                            "summary", {}
                        )
                        if isinstance(assumption_reviews.get("bf_suitability"), dict)
                        else {},
                    },
                },
                "scenarios_considered": {
                    "top_ranked": scenario_summary.get("top_ranked", []),
                    "considered": scenario_summary.get("considered", []),
                    "evidence_ids": list(review_result.get("evidence_ids", [])),
                },
                "recommended_changes": recommendation.get("recommended_changes", []),
                "why_reasonable": recommendation.get("why_reasonable", []),
                "caveats": recommendation.get("caveats", []),
                "judgment_items": recommendation.get("judgment_items", []),
                "signoff_questions": recommendation.get("signoff_questions", []),
                "policy_trace": recommendation.get("policy_trace", {}),
                "continuity_notes": continuity.get("continuity_notes", []),
            },
        }

    def _build_continuity_summary(
        self,
        *,
        segment_memory: dict[str, Any],
        drop_review: dict[str, Any],
        tail_review: dict[str, Any],
        bf_review: dict[str, Any],
    ) -> dict[str, Any]:
        summary = self._segment_memory_service.continuity_summary(segment_memory)
        summary["continuity_notes"] = self._dedupe_notes(
            [
                *self._note_list(drop_review.get("continuity_notes")),
                *self._note_list(tail_review.get("continuity_notes")),
                *self._note_list(bf_review.get("continuity_notes")),
            ]
        )
        return summary

    def _build_scenario_summary(
        self,
        *,
        drop_review: dict[str, Any],
        tail_review: dict[str, Any],
        bf_review: dict[str, Any],
    ) -> dict[str, Any]:
        considered: list[dict[str, Any]] = []
        for review_type, review in [
            ("drop_review", drop_review),
            ("tail_review", tail_review),
        ]:
            for item in review.get("candidates", [])[:5]:
                if not isinstance(item, dict):
                    continue
                considered.append(
                    {
                        "review_type": review_type,
                        "candidate_id": item.get("candidate_id"),
                        "summary": item.get("summary"),
                        "rank": item.get("rank"),
                        "score": item.get("score"),
                        "recommendation_class": item.get("recommendation_class"),
                        "parameters": item.get("parameters", {}),
                        "score_breakdown": item.get("score_breakdown", {}),
                        "policy_trace": item.get("policy_trace", {}),
                        "continuity_notes": item.get("continuity_notes", []),
                        "governance_tier": item.get("metrics", {}).get(
                            "governance_tier"
                        )
                        if isinstance(item.get("metrics"), dict)
                        else None,
                    }
                )
        considered.sort(
            key=lambda item: (
                self._recommendation_strength(item.get("recommendation_class")),
                float(item.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return {
            "considered": considered,
            "top_ranked": considered[:5],
            "bf_suitability": {
                "overall_class": bf_review.get("overall_class"),
                "summary": bf_review.get("summary", {}),
            },
        }

    def _build_recommendation(
        self,
        *,
        comparison_limitations: list[str],
        anomaly_triage: dict[str, Any],
        drop_review: dict[str, Any],
        tail_review: dict[str, Any],
        bf_review: dict[str, Any],
        continuity: dict[str, Any],
    ) -> dict[str, Any]:
        anomaly_pause = bool(anomaly_triage.get("pause_recommendation", False))
        recommended_changes = self._recommended_changes(
            drop_review=drop_review,
            tail_review=tail_review,
        )
        caveats = self._unique_strings(
            [
                *comparison_limitations,
                *self._top_triage_messages(anomaly_triage),
                *[
                    str(item.get("message", "")).strip()
                    for item in continuity.get("continuity_notes", [])
                    if isinstance(item, dict)
                ],
            ]
        )
        judgment_items = self._build_judgment_items(
            anomaly_pause=anomaly_pause,
            bf_review=bf_review,
            continuity=continuity,
        )
        signoff_questions = self._build_signoff_questions(
            anomaly_pause=anomaly_pause,
            recommended_changes=recommended_changes,
            bf_review=bf_review,
            caveats=caveats,
        )
        if anomaly_pause:
            status = "hold_for_review"
            summary = (
                "Quarter-close review should pause assumption changes until the "
                "highest-priority anomalies are resolved."
            )
        elif any(
            str(item.get("recommendation_class")) == "recommend"
            for item in recommended_changes
        ):
            status = "recommended"
            summary = (
                "Quarter-close review identified tested assumption changes that clear "
                "the current governance and continuity checks."
            )
        elif recommended_changes:
            status = "reasonable_alternative"
            summary = (
                "Quarter-close review found useful alternative assumption changes, "
                "but caveats remain material."
            )
        else:
            status = "watch"
            summary = (
                "Quarter-close review did not identify a strong enough tested change "
                "to support a selection update."
            )
        return {
            "status": status,
            "summary": summary,
            "recommended_changes": recommended_changes,
            "why_reasonable": self._build_why_reasonable(
                status=status,
                recommended_changes=recommended_changes,
                bf_review=bf_review,
            ),
            "caveats": caveats,
            "judgment_items": judgment_items,
            "signoff_questions": signoff_questions,
            "policy_trace": {
                "anomaly_pause": anomaly_pause,
                "selected_candidate_ids": [
                    item.get("candidate_id") for item in recommended_changes
                ],
                "bf_overall_class": bf_review.get("overall_class"),
                "continuity_note_count": len(continuity.get("continuity_notes", [])),
            },
        }

    def _recommended_changes(
        self,
        *,
        drop_review: dict[str, Any],
        tail_review: dict[str, Any],
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for review_type, review in [
            ("drop_review", drop_review),
            ("tail_review", tail_review),
        ]:
            recommendation = (
                review.get("recommendation")
                if isinstance(review.get("recommendation"), dict)
                else {}
            )
            recommendation_class = str(
                recommendation.get("recommendation_class", "watch")
            )
            if recommendation_class not in {"recommend", "reasonable_alternative"}:
                continue
            candidate = self._candidate_by_id(
                review,
                recommendation.get("candidate_id"),
            )
            if not isinstance(candidate, dict):
                continue
            changes.append(
                {
                    "review_type": review_type,
                    "candidate_id": candidate.get("candidate_id"),
                    "recommendation_class": recommendation_class,
                    "summary": candidate.get("summary")
                    or recommendation.get("summary"),
                    "parameters": candidate.get("parameters", {}),
                    "score": candidate.get("score"),
                    "score_breakdown": candidate.get("score_breakdown", {}),
                    "policy_trace": candidate.get("policy_trace", {}),
                    "continuity_notes": candidate.get("continuity_notes", []),
                }
            )
        return changes

    def _build_judgment_items(
        self,
        *,
        anomaly_pause: bool,
        bf_review: dict[str, Any],
        continuity: dict[str, Any],
    ) -> list[str]:
        items: list[str] = []
        if anomaly_pause:
            items.append(
                "Resolve high-relevance anomaly findings before changing selected assumptions."
            )
        bf_class = str(bf_review.get("overall_class", "inconclusive")).strip()
        if bf_class == "mixed":
            items.append(
                "BF suitability is mixed across UWYs and should be reviewed at UWY level before any method change."
            )
        elif bf_class == "bf_preferred":
            items.append(
                "BF appears more suitable than all-CL for the current quarter and needs actuarial review before adoption."
            )
        for note in continuity.get("continuity_notes", []):
            if not isinstance(note, dict):
                continue
            message = str(note.get("message", "")).strip()
            if message:
                items.append(message)
        return self._unique_strings(items)

    def _build_signoff_questions(
        self,
        *,
        anomaly_pause: bool,
        recommended_changes: list[dict[str, Any]],
        bf_review: dict[str, Any],
        caveats: list[str],
    ) -> list[str]:
        questions: list[str] = []
        if anomaly_pause:
            questions.append(
                "Have the highest-priority anomaly findings been reconciled well enough to resume assumption selection?"
            )
        for item in recommended_changes:
            review_type = str(item.get("review_type", "assumption review")).replace(
                "_", " "
            )
            questions.append(
                f"Is the proposed {review_type} change supported by both diagnostics improvement and segment continuity?"
            )
        bf_class = str(bf_review.get("overall_class", "inconclusive")).strip()
        if bf_class in {"bf_preferred", "mixed"}:
            questions.append(
                "Should BF suitability conclusions change the selected method for any immature UWYs this quarter?"
            )
        if caveats:
            questions.append(
                "Do the documented caveats require peer review or explicit management discussion before sign-off?"
            )
        return self._unique_strings(questions)

    @staticmethod
    def _build_why_reasonable(
        *,
        status: str,
        recommended_changes: list[dict[str, Any]],
        bf_review: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        if status == "hold_for_review":
            reasons.append(
                "Recommendation strength is capped because anomaly triage identified findings that should be resolved first."
            )
        for item in recommended_changes:
            review_type = str(item.get("review_type", "")).replace("_", " ")
            reasons.append(
                f"Top-ranked {review_type} candidate cleared the current review threshold with a structured score breakdown."
            )
        bf_class = str(bf_review.get("overall_class", "inconclusive")).strip()
        if bf_class and bf_class != "inconclusive":
            reasons.append(
                f"BF suitability result for the quarter is {bf_class}, which informs method judgment even when no method change is proposed."
            )
        return QuarterCloseService._unique_strings(reasons)

    @staticmethod
    def _candidate_by_id(
        review: dict[str, Any],
        candidate_id: object,
    ) -> dict[str, Any] | None:
        target = str(candidate_id or "").strip()
        if not target:
            return None
        for item in review.get("candidates", []):
            if (
                isinstance(item, dict)
                and str(item.get("candidate_id", "")).strip() == target
            ):
                return item
        return None

    @staticmethod
    def _note_list(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @staticmethod
    def _dedupe_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for item in notes:
            key = (
                f"{str(item.get('code', '')).strip()}|"
                f"{str(item.get('message', '')).strip()}"
            )
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return ordered

    @staticmethod
    def _top_triage_messages(anomaly_triage: dict[str, Any]) -> list[str]:
        triaged = (
            anomaly_triage.get("triaged_findings", [])
            if isinstance(anomaly_triage, dict)
            else []
        )
        messages: list[str] = []
        for item in triaged[:5]:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message", "")).strip()
            if message:
                messages.append(message)
        return messages

    def _collect_evidence_ids(self, baseline_context: BaselineContext) -> list[str]:
        evidence_ids: list[str] = []
        for finding in getattr(baseline_context.evaluation, "findings", []):
            evidence = getattr(finding, "evidence", None)
            evidence_id = self._extract_evidence_id(evidence)
            if evidence_id:
                evidence_ids.append(evidence_id)
        for recommendation in getattr(
            baseline_context.evaluation, "recommendations", []
        ):
            evidence = getattr(recommendation, "evidence", None)
            evidence_id = self._extract_evidence_id(evidence)
            if evidence_id:
                evidence_ids.append(evidence_id)
        return self._unique_strings(evidence_ids)

    @staticmethod
    def _extract_evidence_id(evidence: object) -> str:
        if hasattr(evidence, "evidence_id"):
            return str(getattr(evidence, "evidence_id") or "").strip()
        if isinstance(evidence, dict):
            return str(evidence.get("evidence_id", "") or "").strip()
        return ""

    @staticmethod
    def _run_id_from_payload(payload: object) -> str | None:
        if hasattr(payload, "run_id"):
            run_id = str(getattr(payload, "run_id") or "").strip()
            return run_id or None
        if isinstance(payload, dict):
            run_id = str(payload.get("run_id", "") or "").strip()
            return run_id or None
        return None

    @staticmethod
    def _review_packet_summary(review: object) -> dict[str, Any]:
        if not isinstance(review, dict):
            return {}
        return {
            "recommendation": review.get("recommendation", {}),
            "candidate_count": len(review.get("candidates", [])),
            "continuity_notes": review.get("continuity_notes", []),
            "policy_trace": review.get("policy_trace", {}),
        }

    @staticmethod
    def _recommendation_strength(value: object) -> int:
        levels = {
            "recommend": 4,
            "reasonable_alternative": 3,
            "watch": 2,
            "avoid": 1,
        }
        return levels.get(str(value or "").strip().lower(), 0)

    @staticmethod
    def _unique_strings(values: list[object]) -> list[str]:
        output: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in output:
                output.append(text)
        return output

    @staticmethod
    def _clone_params(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "average": Reserving._normalize_average(params.get("average", "volume")),
            "drop": [list(item) for item in params.get("drop", [])],
            "drop_valuation": [list(item) for item in params.get("drop_valuation", [])],
            "tail": {
                "curve": str(params.get("tail", {}).get("curve", "weibull")),
                "attachment_age": params.get("tail", {}).get("attachment_age"),
                "projection_period": int(
                    params.get("tail", {}).get("projection_period", 0) or 0
                ),
                "fit_period": list(params.get("tail", {}).get("fit_period", [])),
            },
            "bf_apriori": dict(params.get("bf_apriori", {})),
            "final_ultimate": str(params.get("final_ultimate", "chainladder")),
            "selected_ultimate_by_uwy": dict(
                params.get("selected_ultimate_by_uwy", {})
            ),
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
