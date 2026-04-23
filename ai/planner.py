from __future__ import annotations

from typing import Any

from ai.plan_models import ExecutionPlan
from ai.playbook_registry import build_execution_plan
from ai.workflow_definitions import select_workflow_definition


class PlaybookPlanner:
    def select_playbook(self, user_prompt: str) -> str:
        definition = select_workflow_definition(user_prompt)
        return definition.workflow_name if definition is not None else ""

    def plan(
        self,
        *,
        user_prompt: str,
        session_context: dict[str, Any] | None,
        segment_memory: dict[str, Any] | None = None,
        analysis_basis: dict[str, Any] | None = None,
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
            analysis_basis=analysis_basis,
        )
