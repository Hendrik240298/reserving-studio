from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from source.config_manager import ConfigManager
from source.reserving import Reserving


class AIReviewService:
    def __init__(
        self,
        reserving: Reserving,
        *,
        config: ConfigManager | None = None,
    ) -> None:
        self._reserving = reserving
        self._config = config

    def build_review_payload(self, *, max_scenarios: int = 16) -> dict[str, Any]:
        from source.api.adapters.reserving_adapter import (
            InMemoryReservingBackend,
            SessionContext,
        )
        from source.api.schemas import ParamsStore, ResultsStoreMeta
        from source.services.diagnostics_service import DiagnosticsService
        from source.services.uncertainty_service import UncertaintyService

        session = self._config.load_session() if self._config is not None else {}
        params_store = ParamsStore(
            request_id=0,
            source="ai-review",
            force_recalc=False,
            drop_store=self._normalize_drop_store(session.get("drops")),
            tail_attachment_age=self._normalize_int(session.get("tail_attachment_age")),
            tail_projection_months=self._normalize_int(
                session.get("tail_projection_months", 0)
            )
            or 0,
            tail_fit_period_selection=self._normalize_int_list(
                session.get("tail_fit_period")
            ),
            average=str(session.get("average", "volume")),
            tail_curve=str(session.get("tail_curve", "weibull")),
            bf_apriori_by_uwy=self._normalize_float_mapping(
                session.get("bf_apriori_by_uwy")
            ),
            selected_ultimate_by_uwy=self._normalize_str_mapping(
                session.get("selected_ultimate_by_uwy")
            ),
            sync_version=0,
        )

        backend = InMemoryReservingBackend.__new__(InMemoryReservingBackend)
        backend._diagnostics_service = DiagnosticsService()
        backend._uncertainty_service = UncertaintyService()
        context = SessionContext(
            session_id=f"{self._segment_key()}-ai-review",
            segment=self._segment_key(),
            reserving=self._reserving,
            sync_version=0,
            params_store=params_store,
            results_store_meta=ResultsStoreMeta(),
            last_results_payload={},
        )

        baseline_params = backend._params_from_store(context)
        baseline_eval = backend._evaluate_scenario(
            context=context,
            scenario_id="baseline",
            params=baseline_params,
            summary="Current configuration",
            parent_scenario_id=None,
            transform="baseline",
            rationale_evidence_ids=[],
        )
        scenario_candidates = backend._build_scenario_candidates(
            context=context,
            baseline=baseline_params,
            baseline_eval=baseline_eval,
            max_scenarios=max_scenarios,
        )
        scenario_evals = [
            backend._evaluate_scenario(
                context=context,
                scenario_id=candidate.scenario_id,
                params=candidate.params,
                summary=candidate.summary,
                parent_scenario_id=candidate.parent_scenario_id,
                transform=candidate.transform,
                rationale_evidence_ids=candidate.rationale_evidence_ids,
            )
            for candidate in scenario_candidates
        ]
        ordered = sorted(scenario_evals, key=lambda item: item.score)
        best = ordered[0] if ordered else baseline_eval

        bootstrap = backend._uncertainty_service.bootstrap_predictive_distribution(
            results_df=self._reserving.get_results(),
            heatmap_data=self._reserving.get_triangle_heatmap_data(),
        )
        tail_model = backend._uncertainty_service.tail_model_assessment(
            scenarios=[
                {
                    "scenario_id": item.scenario_id,
                    "score": item.score,
                    "transform": item.lineage.get("transform", ""),
                    "parameters": item.parameters,
                }
                for item in ordered
            ]
        )
        uncertainty = {
            "baseline": baseline_eval.uncertainty,
            "bootstrap": bootstrap,
            "tail_model": tail_model,
        }

        scenario_matrix = [
            {
                "scenario_id": item.scenario_id,
                "score": item.score,
                "governance_tier": item.governance.get("tier", "green"),
                "summary": item.summary,
                "transform": item.lineage.get("transform", "baseline"),
                "evidence_refs": ", ".join(
                    item.lineage.get("rationale_evidence_ids", [])
                ),
            }
            for item in ([baseline_eval] + ordered)
        ]
        evidence_trace = self._build_evidence_trace(baseline_eval)
        commentary = self.build_commentary(
            baseline=baseline_eval,
            best=best,
            uncertainty=uncertainty,
        )

        backend._apply_params_to_reserving(context, baseline_params)

        return {
            "generated_at": self._utc_now(),
            "segment": self._segment_key(),
            "governance": best.governance,
            "uncertainty": uncertainty,
            "scenario_matrix": scenario_matrix,
            "evidence_trace": evidence_trace,
            "ai_findings": [item.model_dump() for item in baseline_eval.findings],
            "ai_commentary": commentary,
            "ai_evidence_refs": [
                str(row.get("evidence_id", "")).strip()
                for row in evidence_trace
                if str(row.get("evidence_id", "")).strip()
            ],
            "ai_model_meta": {
                "engine": "deterministic-ai-review",
                "generated_at": self._utc_now(),
            },
            "ai_override": {},
        }

    @staticmethod
    def build_commentary(
        *, baseline: Any, best: Any, uncertainty: dict[str, Any]
    ) -> str:
        baseline_score = float(getattr(baseline, "score", 0.0) or 0.0)
        best_score = float(getattr(best, "score", baseline_score) or baseline_score)
        improvement = baseline_score - best_score
        governance_tier = str(best.governance.get("tier", "green")).upper()
        baseline_uncertainty = uncertainty.get("baseline", {})
        tail_uncertainty = uncertainty.get("tail_model", {})
        process_cv = baseline_uncertainty.get("total_process_cv")
        p90 = uncertainty.get("bootstrap", {}).get("p90")
        instability = tail_uncertainty.get("instability_flag")
        return (
            f"Executive view: best scenario improves the diagnostic score by {improvement:.2f} points versus baseline. "
            f"Governance tier is {governance_tier}; process CV is {process_cv}; bootstrap p90 is {p90}; "
            f"tail instability flag is {instability}. Review the evidence trace before sign-off."
        )

    @staticmethod
    def render_chat_transcript(history: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for item in history:
            label = (
                "You" if str(item.get("role", "")).lower() == "user" else "Assistant"
            )
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"**{label}:** {content}")
        return "\n\n".join(lines)

    @staticmethod
    def answer_chat_prompt(prompt: str, review_data: dict[str, Any] | None) -> str:
        if not isinstance(review_data, dict):
            return "Run the review first so I can answer from deterministic evidence."
        text = str(prompt).strip().lower()
        governance = review_data.get("governance", {})
        uncertainty = review_data.get("uncertainty", {})
        scenario_matrix = review_data.get("scenario_matrix", [])
        evidence_trace = review_data.get("evidence_trace", [])

        if "governance" in text or "escalation" in text:
            tier = str(governance.get("tier", "unknown")).upper()
            triggers = governance.get("escalation_triggers", [])
            joined = ", ".join(str(item) for item in triggers) or "none"
            return f"Governance tier is {tier}. Escalation triggers: {joined}."

        if "uncertainty" in text or "bootstrap" in text or "msep" in text:
            baseline = uncertainty.get("baseline", {})
            bootstrap = uncertainty.get("bootstrap", {})
            return (
                f"Process CV is {baseline.get('total_process_cv')}; bootstrap p50 is {bootstrap.get('p50')}; "
                f"bootstrap p90 is {bootstrap.get('p90')}."
            )

        if "scenario" in text or "best" in text:
            if scenario_matrix:
                top = scenario_matrix[0]
                return (
                    f"Top scenario is {top.get('scenario_id')} with score {top.get('score')} "
                    f"and governance tier {top.get('governance_tier')}."
                )
            return "No scenario matrix available."

        if "evidence" in text or "why" in text:
            if evidence_trace:
                top = evidence_trace[0]
                return (
                    f"Primary evidence is {top.get('code')} with evidence ID "
                    f"{top.get('evidence_id')}."
                )
            return "No evidence trace available."

        return "Ask about governance, uncertainty, scenarios, or evidence trace and I will answer from the review payload."

    @staticmethod
    def summarize_governance(
        governance: dict[str, Any], uncertainty: dict[str, Any]
    ) -> str:
        tier = str(governance.get("tier", "unknown")).upper()
        triggers = governance.get("escalation_triggers", [])
        trigger_text = (
            ", ".join(str(item) for item in triggers) or "no escalation triggers"
        )
        cv = uncertainty.get("baseline", {}).get("total_process_cv")
        tail_flag = uncertainty.get("tail_model", {}).get("instability_flag")
        return f"{tier} governance with {trigger_text}. Process CV={cv}. Tail instability={tail_flag}."

    @staticmethod
    def build_decision_packet(review_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "generated_at": AIReviewService._utc_now(),
            "ai_model_meta": review_data.get("ai_model_meta", {}),
            "ai_commentary": review_data.get("ai_commentary", ""),
            "governance": review_data.get("governance", {}),
            "uncertainty": review_data.get("uncertainty", {}),
            "scenario_matrix": review_data.get("scenario_matrix", []),
            "evidence_trace": review_data.get("evidence_trace", []),
            "ai_evidence_refs": review_data.get("ai_evidence_refs", []),
            "ai_override": review_data.get("ai_override", {}),
        }

    def persist_review(self, review_data: dict[str, Any]) -> None:
        if self._config is None:
            return
        session_payload = self._config.load_session()
        for key in [
            "ai_findings",
            "ai_commentary",
            "ai_evidence_refs",
            "ai_model_meta",
            "governance",
            "uncertainty",
            "ai_override",
        ]:
            session_payload[key] = review_data.get(key, {})
        session_payload["ai_decision_packet"] = self.build_decision_packet(review_data)
        self._config.save_session_with_version(session_payload)

    @staticmethod
    def _build_evidence_trace(baseline_eval: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for finding in baseline_eval.findings:
            rows.append(
                {
                    "kind": "finding",
                    "evidence_id": finding.evidence.evidence_id or "",
                    "code": finding.code,
                    "severity": finding.severity,
                    "message": finding.message,
                    "metric": finding.evidence.metric_id,
                    "value": finding.evidence.value,
                    "threshold": finding.evidence.threshold,
                }
            )
        for recommendation in baseline_eval.recommendations:
            rows.append(
                {
                    "kind": "recommendation",
                    "evidence_id": recommendation.evidence.evidence_id or "",
                    "code": recommendation.code,
                    "severity": recommendation.priority,
                    "message": recommendation.message,
                    "metric": recommendation.evidence.metric_id,
                    "value": recommendation.evidence.value,
                    "threshold": recommendation.evidence.threshold,
                }
            )
        return rows

    def _segment_key(self) -> str:
        if self._config is not None:
            return str(self._config.get_segment())
        return "segment"

    @staticmethod
    def _normalize_drop_store(raw: object) -> list[list[str | int]]:
        if not isinstance(raw, list):
            return []
        output: list[list[str | int]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    output.append([str(item[0]), int(item[1])])
                except (TypeError, ValueError):
                    continue
        return output

    @staticmethod
    def _normalize_int(raw: object) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_int_list(raw: object) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, (int, float, str)):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        output: list[int] = []
        for item in raw:
            try:
                output.append(int(item))
            except (TypeError, ValueError):
                continue
        return output

    @staticmethod
    def _normalize_float_mapping(raw: object) -> dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        output: dict[str, float] = {}
        for key, value in raw.items():
            try:
                output[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return output

    @staticmethod
    def _normalize_str_mapping(raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        return {str(key): str(value) for key, value in raw.items()}

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
