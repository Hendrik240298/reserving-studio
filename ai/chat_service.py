from __future__ import annotations

import threading
from typing import Any, Callable

from ai.assistant_service import AssistantService
from ai.chat_store import ChatSession, InMemoryChatStore


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

    def send_message(self, chat_id: str, content: str) -> dict[str, Any]:
        return self.start_message(chat_id, content)

    def start_message(self, chat_id: str, content: str) -> dict[str, Any]:
        prompt = str(content or "").strip()
        if not prompt:
            raise ValueError("Message content must not be empty")

        session = self._store.get_chat(chat_id)
        if session is None:
            raise LookupError(f"Chat session not found: {chat_id}")
        if bool(session.metadata.get("streaming")):
            raise ValueError("Another assistant response is still in progress")

        self._store.append_message(chat_id, {"role": "user", "content": prompt})
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
                self._store.update_memory(
                    chat_id,
                    working_memory=memory_snapshot,
                    scenario_ledger=memory_snapshot.get("scenario_ledger"),
                )
            self._store.update_assistant_message(
                chat_id,
                message_id=message_id,
                replace_content=str(result.get("content", "")),
                streaming=False,
                fallback_used=bool(result.get("fallback_used", False)),
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

    def _build_response(self, refreshed: ChatSession) -> dict[str, Any]:
        return {
            "chat_id": refreshed.chat_id,
            "segment": refreshed.segment,
            "reserving_session_id": refreshed.reserving_session_id,
            "assistant_message": self._latest_assistant_message(refreshed),
            "fallback_used": self._latest_fallback_used(refreshed),
            "tool_events": [dict(item) for item in refreshed.tool_events],
            "messages": [dict(item) for item in refreshed.messages],
            "working_memory": dict(refreshed.working_memory),
            "scenario_ledger": [dict(item) for item in refreshed.scenario_ledger],
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
    def _conversation_history(session: ChatSession) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for item in session.messages:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            history.append({"role": role, "content": content})
        return history
