from __future__ import annotations

from typing import Any


class ScenarioScoringService:
    VERSION = "v1"
    REVIEW_VERSION = "v2"

    _SEVERITY_WEIGHTS = {
        "low": 0.5,
        "medium": 2.0,
        "high": 5.0,
        "critical": 8.0,
    }

    def score(
        self,
        *,
        findings: list[Any],
        drop_count: int,
        continuity_penalty: float = 0.0,
        governance_penalty: float = 0.0,
        extra_penalties: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        diagnostics_severity = self._severity_score(findings)
        drop_count_penalty = max(int(drop_count), 0) * 0.2
        penalties = {
            "drop_count": round(float(drop_count_penalty), 4),
            "continuity": round(max(float(continuity_penalty), 0.0), 4),
            "governance": round(max(float(governance_penalty), 0.0), 4),
        }
        for key, value in (extra_penalties or {}).items():
            penalties[str(key)] = round(max(float(value), 0.0), 4)

        components = {
            "diagnostics_severity": round(float(diagnostics_severity), 4),
        }
        total_score = round(
            sum(float(value) for value in components.values())
            + sum(float(value) for value in penalties.values()),
            4,
        )
        return {
            "score": total_score,
            "components": components,
            "penalties": penalties,
            "formula_version": self.VERSION,
        }

    def score_drop_candidate(
        self,
        *,
        outlier_support: float,
        consistency_improvement: float,
        late_emergence_support: float,
        reserve_impact_penalty: float,
        fragility_penalty: float,
        governance_penalty: float,
        continuity_penalty: float,
    ) -> dict[str, Any]:
        return self._review_score(
            components={
                "outlier_support": outlier_support,
                "consistency_improvement": consistency_improvement,
                "late_emergence_support": late_emergence_support,
            },
            penalties={
                "reserve_impact": reserve_impact_penalty,
                "fragility": fragility_penalty,
                "governance": governance_penalty,
                "continuity": continuity_penalty,
            },
        )

    def score_tail_candidate(
        self,
        *,
        fit_quality: float,
        continuity_score: float,
        stability_score: float,
        reserve_reasonableness: float,
        continuity_gap_penalty: float,
        subunit_penalty: float,
        instability_penalty: float,
        governance_penalty: float,
        continuity_penalty: float,
    ) -> dict[str, Any]:
        return self._review_score(
            components={
                "fit_quality": fit_quality,
                "continuity_score": continuity_score,
                "stability_score": stability_score,
                "reserve_reasonableness": reserve_reasonableness,
            },
            penalties={
                "continuity_gap": continuity_gap_penalty,
                "subunit": subunit_penalty,
                "instability": instability_penalty,
                "governance": governance_penalty,
                "continuity": continuity_penalty,
            },
        )

    def score_bf_suitability(
        self,
        *,
        maturity_support: float,
        volatility_support: float,
        cl_sensitivity_support: float,
        percent_reported_support: float,
        apriori_readiness_support: float,
    ) -> dict[str, Any]:
        components = {
            "maturity_support": self._round_score(maturity_support),
            "volatility_support": self._round_score(volatility_support),
            "cl_sensitivity_support": self._round_score(cl_sensitivity_support),
            "percent_reported_support": self._round_score(percent_reported_support),
            "apriori_readiness_support": self._round_score(apriori_readiness_support),
        }
        total_score = round(sum(float(value) for value in components.values()), 4)
        return {
            "score": total_score,
            "components": components,
            "penalties": {},
            "formula_version": self.REVIEW_VERSION,
        }

    def classify_recommendation(
        self,
        *,
        total_score: float,
        governance_tier: str | None,
        rejected_before: bool = False,
    ) -> str:
        tier = str(governance_tier or "").strip().lower()
        if rejected_before or tier == "red" or total_score < 0.2:
            return "avoid"
        if total_score >= 0.85 and tier == "green":
            return "recommend"
        if total_score >= 0.45:
            return "reasonable_alternative"
        return "watch"

    def classify_bf_suitability(self, *, total_score: float) -> str:
        if total_score >= 0.45:
            return "bf_preferred"
        if total_score <= -0.35:
            return "cl_preferred"
        return "inconclusive"

    def _review_score(
        self,
        *,
        components: dict[str, float],
        penalties: dict[str, float],
    ) -> dict[str, Any]:
        normalized_components = {
            str(key): self._round_score(value) for key, value in components.items()
        }
        normalized_penalties = {
            str(key): self._round_score(max(float(value), 0.0))
            for key, value in penalties.items()
        }
        total_score = round(
            sum(float(value) for value in normalized_components.values())
            - sum(float(value) for value in normalized_penalties.values()),
            4,
        )
        return {
            "score": total_score,
            "components": normalized_components,
            "penalties": normalized_penalties,
            "formula_version": self.REVIEW_VERSION,
        }

    def _severity_score(self, findings: list[Any]) -> float:
        score = 0.0
        for finding in findings:
            severity = self._severity_from_item(finding)
            score += float(self._SEVERITY_WEIGHTS.get(severity, 1.0))
        return float(score)

    @staticmethod
    def _severity_from_item(item: object) -> str:
        if isinstance(item, dict):
            return str(item.get("severity", "")).strip().lower()
        severity = getattr(item, "severity", "")
        return str(severity).strip().lower()

    @staticmethod
    def _round_score(value: float) -> float:
        return round(float(value), 4)
