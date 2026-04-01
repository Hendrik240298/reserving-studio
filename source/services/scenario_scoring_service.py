from __future__ import annotations

from typing import Any


class ScenarioScoringService:
    VERSION = "v1"

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
