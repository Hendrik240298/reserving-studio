from __future__ import annotations

from typing import Any

from ai.plan_models import ExecutionPlan
from ai.playbook_registry import build_execution_plan


class PlaybookPlanner:
    def select_playbook(self, user_prompt: str) -> str:
        prompt = str(user_prompt or "").strip().lower()
        if not prompt:
            return ""
        if any(
            keyword in prompt
            for keyword in {
                "quarter close",
                "quarter-close",
                "close pack",
                "close review",
                "quarterly review pack",
            }
        ):
            return "quarter_close_review"
        if any(
            keyword in prompt
            for keyword in {
                "data quality",
                "anomaly",
                "triage",
                "missing diagonal",
                "impossible link ratio",
                "calendar year distortion",
                "large loss contamination",
            }
        ):
            return "data_anomaly_triage"
        if any(
            keyword in prompt
            for keyword in {
                "drop review",
                "drop any ratios",
                "which ratios should be dropped",
                "which ratio should be dropped",
                "should be dropped",
                "should we drop",
                "should i drop",
                "drop ratios",
            }
        ):
            return "drop_review"
        if any(
            keyword in prompt
            for keyword in {
                "movement",
                "movements",
                "unexpected",
                "unusual",
                "this quarter",
                "current quarter",
                "latest diagonal",
                "what happened",
            }
        ) and any(
            keyword in prompt
            for keyword in {"claims", "incurred", "paid", "outstanding", "premium"}
        ):
            return "movement_review"
        if any(
            keyword in prompt
            for keyword in {
                "how much more",
                "still emerge",
                "late emergence",
                "still come",
                "still develop",
            }
        ):
            return "late_emergence_review"
        if any(
            keyword in prompt
            for keyword in {
                "why did reserve",
                "why does reserve",
                "explain reserve change",
                "driver of reserve",
                "reserve change",
                "impact on reserve",
            }
        ):
            return "reserve_change_explanation"
        if any(
            keyword in prompt
            for keyword in {
                "tail",
                "weibull",
                "inverse power",
                "inverse_power",
                "exponential",
                "fit period",
                "tail fit",
            }
        ):
            return "tail_selection"
        if any(
            keyword in prompt
            for keyword in {
                "cl vs bf",
                "chainladder vs bf",
                "bornhuetter",
                "method suitable",
                "bf better",
                "chainladder better",
            }
        ):
            return "method_suitability_review"
        if any(
            keyword in prompt
            for keyword in {
                "recommend",
                "scenario",
                "drop",
                "change",
                "adjust",
                "what should",
                "which should",
                "best",
                "compare",
                "trade-off",
                "tradeoff",
            }
        ):
            return "scenario_recommendation"
        if any(
            keyword in prompt
            for keyword in {
                "show me",
                "compare data",
                "triangle",
                "view",
                "ratio",
                "table",
            }
        ):
            return "data_exploration"
        return ""

    def plan(
        self,
        *,
        user_prompt: str,
        session_context: dict[str, Any] | None,
        segment_memory: dict[str, Any] | None = None,
    ) -> ExecutionPlan | None:
        playbook = self.select_playbook(user_prompt)
        if not playbook:
            return None
        session_id = None
        segment = None
        if isinstance(session_context, dict):
            raw_session_id = session_context.get("session_id")
            raw_segment = session_context.get("segment")
            if isinstance(raw_session_id, str) and raw_session_id.strip():
                session_id = raw_session_id.strip()
            if isinstance(raw_segment, str) and raw_segment.strip():
                segment = raw_segment.strip()
        return build_execution_plan(
            playbook=playbook,
            session_id=session_id,
            segment=segment,
            segment_memory=segment_memory,
        )
