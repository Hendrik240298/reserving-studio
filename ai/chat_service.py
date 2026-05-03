from __future__ import annotations

import threading
from typing import Any, Callable

from ai.assistant_service import AssistantService
from ai.control_plane_types import (
    normalize_accepted_analysis_basis,
    normalize_basis_transition_record,
    normalize_preview_basis,
    normalize_proposal_basis,
)
from ai.chat_store import ChatSession, InMemoryChatStore
from ai.proposal_manager import ProposalManager


class AIChatService:
    def __init__(
        self,
        *,
        assistant_factory: Callable[[], AssistantService],
        store: InMemoryChatStore | None = None,
    ) -> None:
        self._assistant_factory = assistant_factory
        self._store = store or InMemoryChatStore()

    def create_chat(
        self,
        *,
        segment: str | None = None,
        reserving_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        return self._store.create_chat(
            segment=segment,
            reserving_session_id=reserving_session_id,
            metadata=metadata,
        )

    def get_chat(self, chat_id: str) -> ChatSession | None:
        return self._store.get_chat(chat_id)

    def send_message(
        self,
        chat_id: str,
        content: str,
        *,
        display_content: str | None = None,
    ) -> dict[str, Any]:
        return self.start_message(chat_id, content, display_content=display_content)

    def start_message(
        self,
        chat_id: str,
        content: str,
        *,
        display_content: str | None = None,
    ) -> dict[str, Any]:
        prompt = str(content or "").strip()
        if not prompt:
            raise ValueError("Message content must not be empty")
        visible_prompt = str(
            display_content if display_content is not None else content
        ).strip()
        if not visible_prompt:
            visible_prompt = prompt

        session = self._store.get_chat(chat_id)
        if session is None:
            raise LookupError(f"Chat session not found: {chat_id}")
        if bool(session.metadata.get("streaming")):
            raise ValueError("Another assistant response is still in progress")

        self._store.append_message(chat_id, {"role": "user", "content": visible_prompt})
        updated_session, message_id = self._store.start_assistant_message(chat_id)
        worker = threading.Thread(
            target=self._run_message,
            kwargs={
                "chat_id": chat_id,
                "prompt": prompt,
                "session": session,
                "message_id": message_id,
            },
            daemon=True,
        )
        worker.start()
        return self._build_response(updated_session)

    def _run_message(
        self,
        *,
        chat_id: str,
        prompt: str,
        session: ChatSession,
        message_id: str,
    ) -> None:
        try:
            assistant = self._assistant_factory()
            result = assistant.run_turn(
                user_prompt=prompt,
                conversation_history=self._conversation_history(session),
                session_context={
                    "segment": session.segment,
                    "session_id": session.reserving_session_id,
                },
                accepted_analysis_basis=session.accepted_analysis_basis,
                proposal_basis=session.proposal_basis,
                preview_basis=session.preview_basis,
                execution_records=session.execution_records,
                basis_transition_history=session.basis_transition_history,
                working_memory={
                    **dict(session.working_memory),
                    "scenario_ledger": [dict(item) for item in session.scenario_ledger],
                },
                event_callback=lambda event_type, payload: self._handle_event(
                    chat_id,
                    message_id,
                    event_type,
                    payload,
                ),
            )

            if isinstance(result.get("session_id"), str):
                self._store.update_context(
                    chat_id,
                    reserving_session_id=str(result["session_id"]),
                )
            memory_snapshot = result.get("memory_snapshot")
            if isinstance(memory_snapshot, dict):
                accepted_analysis_basis = memory_snapshot.get("accepted_analysis_basis")
                if not isinstance(accepted_analysis_basis, dict):
                    accepted_analysis_basis = memory_snapshot.get("analysis_basis")
                proposal_basis = normalize_proposal_basis(
                    memory_snapshot.get("proposal_basis")
                )
                prior_pending_proposal = normalize_proposal_basis(session.proposal_basis)
                if (
                    prior_pending_proposal
                    and str(prior_pending_proposal.get("status") or "").strip()
                    == "pending"
                    and proposal_basis
                    and str(proposal_basis.get("proposal_id") or "").strip()
                    != str(prior_pending_proposal.get("proposal_id") or "").strip()
                ):
                    superseded = ProposalManager.mark_superseded(
                        prior_pending_proposal,
                        superseded_by_proposal_id=proposal_basis.get("proposal_id"),
                    )
                    prior_message_id = str(
                        prior_pending_proposal.get("presented_in_message_id") or ""
                    ).strip()
                    if prior_message_id:
                        self._store.update_message_fields(
                            chat_id,
                            message_id=prior_message_id,
                            fields={"proposal_basis": superseded},
                        )
                if proposal_basis and str(proposal_basis.get("status") or "").strip() == "pending":
                    same_as_prior = bool(
                        prior_pending_proposal
                        and str(prior_pending_proposal.get("proposal_id") or "").strip()
                        == str(proposal_basis.get("proposal_id") or "").strip()
                    )
                    if same_as_prior and prior_pending_proposal.get("presented_in_message_id"):
                        proposal_basis = dict(prior_pending_proposal)
                    else:
                        proposal_basis = ProposalManager.attach_to_message(
                            proposal_basis,
                            message_id=message_id,
                        )
                stripped_working_memory = {
                    key: value
                    for key, value in memory_snapshot.items()
                    if key
                    not in {
                        "analysis_basis",
                        "accepted_analysis_basis",
                        "proposal_basis",
                        "preview_basis",
                        "execution_records",
                        "basis_transition_history",
                    }
                }
                self._store.update_memory(
                    chat_id,
                    working_memory=stripped_working_memory,
                    accepted_analysis_basis=accepted_analysis_basis,
                    proposal_basis=proposal_basis,
                    preview_basis=memory_snapshot.get("preview_basis"),
                    scenario_ledger=memory_snapshot.get("scenario_ledger"),
                    execution_records=memory_snapshot.get("execution_records"),
                    basis_transition_history=memory_snapshot.get(
                        "basis_transition_history"
                    ),
                    deterministic_packet=memory_snapshot.get("deterministic_packet"),
                )
                if proposal_basis and str(proposal_basis.get("status") or "").strip() == "pending":
                    presented_in_message_id = str(
                        proposal_basis.get("presented_in_message_id") or ""
                    ).strip()
                    if presented_in_message_id:
                        self._store.update_message_fields(
                            chat_id,
                            message_id=presented_in_message_id,
                            fields={"proposal_basis": proposal_basis},
                        )
            self._store.update_assistant_message(
                chat_id,
                message_id=message_id,
                replace_content=str(result.get("content", "")),
                streaming=False,
                fallback_used=bool(result.get("fallback_used", False)),
                fallback_reason=str(result.get("fallback_reason") or ""),
                fallback_detail=str(result.get("fallback_detail") or ""),
                status="Done",
            )
        except Exception as error:
            self._store.update_assistant_message(
                chat_id,
                message_id=message_id,
                replace_content=f"AI assistant error: {error}",
                streaming=False,
                fallback_used=False,
                status="Failed",
            )

    def _handle_event(
        self,
        chat_id: str,
        message_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type == "content_chunk":
            self._store.update_assistant_message(
                chat_id,
                message_id=message_id,
                append_content=str(payload.get("text", "")),
                status="Writing answer...",
            )
            return
        if event_type == "status":
            self._store.update_assistant_message(
                chat_id,
                message_id=message_id,
                status=str(payload.get("message", "Thinking...")),
            )
            return
        if event_type == "tool_event":
            self._store.extend_tool_events(chat_id, [payload])
            return

    def build_chat_response(self, chat_id: str) -> dict[str, Any]:
        session = self._store.get_chat(chat_id)
        if session is None:
            raise LookupError(f"Chat session not found: {chat_id}")
        return self._build_response(session)

    def accept_proposal(self, chat_id: str, proposal_id: str) -> dict[str, Any]:
        session = self._store.get_chat(chat_id)
        if session is None:
            raise LookupError(f"Chat session not found: {chat_id}")
        proposal = normalize_proposal_basis(session.proposal_basis)
        if not proposal:
            raise ValueError("No proposal is pending for this chat")
        if str(proposal.get("status") or "").strip() != "pending":
            raise ValueError("Current proposal is not pending")
        if str(proposal.get("proposal_id") or "").strip() != str(proposal_id or "").strip():
            raise ValueError("Proposal ID does not match the pending proposal")
        accepted_basis, updated_proposal, transition = ProposalManager.accept(
            proposal,
            chat_id=chat_id,
            current_accepted_basis=session.accepted_analysis_basis,
        )
        deterministic_packet = dict(session.deterministic_packet)
        deterministic_packet["accepted_analysis_basis"] = dict(accepted_basis)
        deterministic_packet["proposal_basis"] = dict(updated_proposal)
        updated = self._store.update_memory(
            chat_id,
            working_memory=session.working_memory,
            accepted_analysis_basis=accepted_basis,
            proposal_basis=updated_proposal,
            preview_basis=session.preview_basis,
            scenario_ledger=session.scenario_ledger,
            execution_records=session.execution_records,
            basis_transition_history=[
                *session.basis_transition_history,
                normalize_basis_transition_record(transition),
            ],
            deterministic_packet=deterministic_packet,
        )
        presented_in_message_id = str(
            updated_proposal.get("presented_in_message_id") or ""
        ).strip()
        if presented_in_message_id:
            updated = self._store.update_message_fields(
                chat_id,
                message_id=presented_in_message_id,
                fields={"proposal_basis": updated_proposal},
            )
        return self._build_response(updated)

    def reject_proposal(self, chat_id: str, proposal_id: str) -> dict[str, Any]:
        session = self._store.get_chat(chat_id)
        if session is None:
            raise LookupError(f"Chat session not found: {chat_id}")
        proposal = normalize_proposal_basis(session.proposal_basis)
        if not proposal:
            raise ValueError("No proposal is pending for this chat")
        if str(proposal.get("status") or "").strip() != "pending":
            raise ValueError("Current proposal is not pending")
        if str(proposal.get("proposal_id") or "").strip() != str(proposal_id or "").strip():
            raise ValueError("Proposal ID does not match the pending proposal")
        updated_proposal, transition = ProposalManager.reject(proposal, chat_id=chat_id)
        deterministic_packet = dict(session.deterministic_packet)
        deterministic_packet["proposal_basis"] = dict(updated_proposal)
        updated = self._store.update_memory(
            chat_id,
            working_memory=session.working_memory,
            accepted_analysis_basis=session.accepted_analysis_basis,
            proposal_basis=updated_proposal,
            preview_basis=session.preview_basis,
            scenario_ledger=session.scenario_ledger,
            execution_records=session.execution_records,
            basis_transition_history=[
                *session.basis_transition_history,
                normalize_basis_transition_record(transition),
            ],
            deterministic_packet=deterministic_packet,
        )
        presented_in_message_id = str(
            updated_proposal.get("presented_in_message_id") or ""
        ).strip()
        if presented_in_message_id:
            updated = self._store.update_message_fields(
                chat_id,
                message_id=presented_in_message_id,
                fields={"proposal_basis": updated_proposal},
            )
        return self._build_response(updated)

    def update_working_memory_fields(
        self,
        chat_id: str,
        *,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        session = self._store.get_chat(chat_id)
        if session is None:
            raise LookupError(f"Chat session not found: {chat_id}")
        next_memory = dict(session.working_memory)
        next_memory.update(dict(fields or {}))
        accepted_analysis_basis = session.accepted_analysis_basis
        proposal_basis = session.proposal_basis
        preview_basis = session.preview_basis
        if "analysis_basis" in next_memory:
            accepted_analysis_basis = normalize_accepted_analysis_basis(
                next_memory.pop("analysis_basis")
            )
        if "accepted_analysis_basis" in next_memory:
            accepted_analysis_basis = normalize_accepted_analysis_basis(
                next_memory.pop("accepted_analysis_basis")
            )
        if "proposal_basis" in next_memory:
            proposal_basis = normalize_proposal_basis(next_memory.pop("proposal_basis"))
        if "preview_basis" in next_memory:
            preview_basis = normalize_preview_basis(next_memory.pop("preview_basis"))
        updated = self._store.update_memory(
            chat_id,
            working_memory=next_memory,
            accepted_analysis_basis=accepted_analysis_basis,
            proposal_basis=proposal_basis,
            preview_basis=preview_basis,
            scenario_ledger=session.scenario_ledger,
            execution_records=session.execution_records,
            basis_transition_history=session.basis_transition_history,
            deterministic_packet=session.deterministic_packet,
        )
        return self._build_response(updated)

    def _build_response(self, refreshed: ChatSession) -> dict[str, Any]:
        accepted_analysis_basis = normalize_accepted_analysis_basis(
            refreshed.accepted_analysis_basis
        )
        proposal_basis = normalize_proposal_basis(refreshed.proposal_basis)
        return {
            "chat_id": refreshed.chat_id,
            "segment": refreshed.segment,
            "reserving_session_id": refreshed.reserving_session_id,
            "assistant_message": self._latest_assistant_message(refreshed),
            "fallback_used": self._latest_fallback_used(refreshed),
            "fallback_reason": self._latest_fallback_field(
                refreshed,
                "fallback_reason",
            ),
            "fallback_detail": self._latest_fallback_field(
                refreshed,
                "fallback_detail",
            ),
            "tool_events": [dict(item) for item in refreshed.tool_events],
            "messages": [dict(item) for item in refreshed.messages],
            "working_memory": dict(refreshed.working_memory),
            "accepted_analysis_basis": dict(accepted_analysis_basis),
            "analysis_basis": dict(accepted_analysis_basis),
            "proposal_basis": dict(proposal_basis),
            "preview_basis": dict(refreshed.preview_basis),
            "scenario_ledger": [dict(item) for item in refreshed.scenario_ledger],
            "execution_records": [dict(item) for item in refreshed.execution_records],
            "basis_transition_history": [
                dict(item) for item in refreshed.basis_transition_history
            ],
            "deterministic_packet": dict(refreshed.deterministic_packet),
            "narration_packet": self._narration_packet(refreshed),
            "memory_update_proposals": [
                dict(item)
                for item in refreshed.working_memory.get("memory_update_proposals", [])
                if isinstance(item, dict)
            ]
            if isinstance(refreshed.working_memory.get("memory_update_proposals"), list)
            else [],
            "streaming": bool(refreshed.metadata.get("streaming")),
            "stream_status": str(refreshed.metadata.get("stream_status", "")),
            "updated_at": refreshed.updated_at,
        }

    @staticmethod
    def _latest_assistant_message(session: ChatSession) -> str:
        for item in reversed(session.messages):
            if str(item.get("role", "")).lower() == "assistant":
                return str(item.get("content", ""))
        return ""

    @staticmethod
    def _latest_fallback_used(session: ChatSession) -> bool:
        for item in reversed(session.messages):
            if str(item.get("role", "")).lower() == "assistant":
                return bool(item.get("fallback_used", False))
        return False

    @staticmethod
    def _latest_fallback_field(session: ChatSession, field: str) -> str:
        for item in reversed(session.messages):
            if str(item.get("role", "")).lower() == "assistant":
                return str(item.get(field) or "")
        return ""

    @staticmethod
    def _narration_packet(session: ChatSession) -> dict[str, Any]:
        packet = session.working_memory.get("narration_packet")
        if isinstance(packet, dict) and packet:
            return dict(packet)
        deterministic_packet = session.deterministic_packet
        nested = deterministic_packet.get("narration_packet")
        return dict(nested) if isinstance(nested, dict) else {}

    @staticmethod
    def _conversation_history(session: ChatSession) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for item in session.messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            history.append({"role": role, "content": content})
        return history
