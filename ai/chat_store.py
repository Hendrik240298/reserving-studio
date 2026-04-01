from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ChatSession:
    chat_id: str
    segment: str | None = None
    reserving_session_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    scenario_ledger: list[dict[str, Any]] = field(default_factory=list)
    deterministic_packet: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)


class InMemoryChatStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, ChatSession] = {}

    def create_chat(
        self,
        *,
        segment: str | None = None,
        reserving_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        with self._lock:
            chat_id = f"chat-{uuid.uuid4().hex[:12]}"
            session = ChatSession(
                chat_id=chat_id,
                segment=segment,
                reserving_session_id=reserving_session_id,
                metadata=dict(metadata or {}),
            )
            self._sessions[chat_id] = session
            return self._copy_session(session)

    def get_chat(self, chat_id: str) -> ChatSession | None:
        with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                return None
            return self._copy_session(session)

    def append_message(self, chat_id: str, message: dict[str, Any]) -> ChatSession:
        with self._lock:
            session = self._require(chat_id)
            session.messages.append(dict(message))
            session.updated_at = _utc_now()
            return self._copy_session(session)

    def start_assistant_message(
        self,
        chat_id: str,
        *,
        initial_content: str = "",
        status: str = "Thinking...",
    ) -> tuple[ChatSession, str]:
        with self._lock:
            session = self._require(chat_id)
            message_id = f"msg-{uuid.uuid4().hex[:12]}"
            session.messages.append(
                {
                    "message_id": message_id,
                    "role": "assistant",
                    "content": initial_content,
                    "streaming": True,
                }
            )
            session.metadata["streaming"] = True
            session.metadata["stream_status"] = status
            session.metadata["active_message_id"] = message_id
            session.updated_at = _utc_now()
            return self._copy_session(session), message_id

    def update_assistant_message(
        self,
        chat_id: str,
        *,
        message_id: str,
        append_content: str | None = None,
        replace_content: str | None = None,
        streaming: bool | None = None,
        fallback_used: bool | None = None,
        status: str | None = None,
    ) -> ChatSession:
        with self._lock:
            session = self._require(chat_id)
            for item in reversed(session.messages):
                if str(item.get("message_id", "")) != message_id:
                    continue
                if replace_content is not None:
                    item["content"] = replace_content
                elif append_content:
                    item["content"] = str(item.get("content", "")) + append_content
                if streaming is not None:
                    item["streaming"] = bool(streaming)
                if fallback_used is not None:
                    item["fallback_used"] = bool(fallback_used)
                break
            if streaming is not None:
                session.metadata["streaming"] = bool(streaming)
                if not streaming:
                    session.metadata["active_message_id"] = None
            if status is not None:
                session.metadata["stream_status"] = status
            session.updated_at = _utc_now()
            return self._copy_session(session)

    def extend_tool_events(
        self,
        chat_id: str,
        events: list[dict[str, Any]],
    ) -> ChatSession:
        with self._lock:
            session = self._require(chat_id)
            session.tool_events.extend(dict(item) for item in events)
            session.updated_at = _utc_now()
            return self._copy_session(session)

    def update_context(
        self,
        chat_id: str,
        *,
        segment: str | None = None,
        reserving_session_id: str | None = None,
    ) -> ChatSession:
        with self._lock:
            session = self._require(chat_id)
            if segment is not None:
                session.segment = segment
            if reserving_session_id is not None:
                session.reserving_session_id = reserving_session_id
            session.updated_at = _utc_now()
            return self._copy_session(session)

    def update_memory(
        self,
        chat_id: str,
        *,
        working_memory: dict[str, Any] | None = None,
        scenario_ledger: list[dict[str, Any]] | None = None,
        deterministic_packet: dict[str, Any] | None = None,
    ) -> ChatSession:
        with self._lock:
            session = self._require(chat_id)
            if isinstance(working_memory, dict):
                session.working_memory = dict(working_memory)
            if isinstance(scenario_ledger, list):
                session.scenario_ledger = [
                    dict(item) for item in scenario_ledger if isinstance(item, dict)
                ]
            if isinstance(deterministic_packet, dict):
                session.deterministic_packet = dict(deterministic_packet)
            session.updated_at = _utc_now()
            return self._copy_session(session)

    def _require(self, chat_id: str) -> ChatSession:
        session = self._sessions.get(chat_id)
        if session is None:
            raise LookupError(f"Chat session not found: {chat_id}")
        return session

    @staticmethod
    def _copy_session(session: ChatSession) -> ChatSession:
        return ChatSession(
            chat_id=session.chat_id,
            segment=session.segment,
            reserving_session_id=session.reserving_session_id,
            messages=[dict(item) for item in session.messages],
            tool_events=[dict(item) for item in session.tool_events],
            working_memory=dict(session.working_memory),
            scenario_ledger=[dict(item) for item in session.scenario_ledger],
            deterministic_packet=dict(session.deterministic_packet),
            metadata=dict(session.metadata),
            created_at=session.created_at,
            updated_at=session.updated_at,
        )


class FileChatStore(InMemoryChatStore):
    def __init__(self, directory: str | Path) -> None:
        super().__init__()
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def create_chat(
        self,
        *,
        segment: str | None = None,
        reserving_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        with self._lock:
            session = super().create_chat(
                segment=segment,
                reserving_session_id=reserving_session_id,
                metadata=metadata,
            )
            self._persist_session_unlocked(session)
            return session

    def get_chat(self, chat_id: str) -> ChatSession | None:
        with self._lock:
            session = self._sessions.get(chat_id)
            if session is None:
                loaded = self._load_session_from_disk(chat_id)
                if loaded is not None:
                    self._sessions[chat_id] = loaded
                    session = loaded
            if session is None:
                return None
            return self._copy_session(session)

    def append_message(self, chat_id: str, message: dict[str, Any]) -> ChatSession:
        with self._lock:
            session = super().append_message(chat_id, message)
            self._persist_session_unlocked(session)
            return session

    def start_assistant_message(
        self,
        chat_id: str,
        *,
        initial_content: str = "",
        status: str = "Thinking...",
    ) -> tuple[ChatSession, str]:
        with self._lock:
            session, message_id = super().start_assistant_message(
                chat_id,
                initial_content=initial_content,
                status=status,
            )
            self._persist_session_unlocked(session)
            return session, message_id

    def update_assistant_message(
        self,
        chat_id: str,
        *,
        message_id: str,
        append_content: str | None = None,
        replace_content: str | None = None,
        streaming: bool | None = None,
        fallback_used: bool | None = None,
        status: str | None = None,
    ) -> ChatSession:
        with self._lock:
            session = super().update_assistant_message(
                chat_id,
                message_id=message_id,
                append_content=append_content,
                replace_content=replace_content,
                streaming=streaming,
                fallback_used=fallback_used,
                status=status,
            )
            if self._should_persist_assistant_update(
                append_content=append_content,
                replace_content=replace_content,
                streaming=streaming,
                fallback_used=fallback_used,
            ):
                self._persist_session_unlocked(session)
            return session

    def extend_tool_events(
        self,
        chat_id: str,
        events: list[dict[str, Any]],
    ) -> ChatSession:
        with self._lock:
            session = super().extend_tool_events(chat_id, events)
            self._persist_session_unlocked(session)
            return session

    def update_context(
        self,
        chat_id: str,
        *,
        segment: str | None = None,
        reserving_session_id: str | None = None,
    ) -> ChatSession:
        with self._lock:
            session = super().update_context(
                chat_id,
                segment=segment,
                reserving_session_id=reserving_session_id,
            )
            self._persist_session_unlocked(session)
            return session

    def update_memory(
        self,
        chat_id: str,
        *,
        working_memory: dict[str, Any] | None = None,
        scenario_ledger: list[dict[str, Any]] | None = None,
        deterministic_packet: dict[str, Any] | None = None,
    ) -> ChatSession:
        with self._lock:
            session = super().update_memory(
                chat_id,
                working_memory=working_memory,
                scenario_ledger=scenario_ledger,
                deterministic_packet=deterministic_packet,
            )
            self._persist_session_unlocked(session)
            return session

    @staticmethod
    def _should_persist_assistant_update(
        *,
        append_content: str | None,
        replace_content: str | None,
        streaming: bool | None,
        fallback_used: bool | None,
    ) -> bool:
        if replace_content is not None:
            return True
        if streaming is False:
            return True
        if fallback_used is not None:
            return True
        if append_content:
            return False
        return False

    def _session_path(self, chat_id: str) -> Path:
        return self._directory / f"{chat_id}.json"

    def _persist_session_unlocked(self, session: ChatSession) -> None:
        session_path = self._session_path(session.chat_id)
        payload = self._session_payload(session)
        tmp_path = session_path.with_suffix(session_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(session_path)

    def _load_session_from_disk(self, chat_id: str) -> ChatSession | None:
        session_path = self._session_path(chat_id)
        if not session_path.exists():
            return None
        try:
            with session_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        return self._session_from_payload(chat_id=chat_id, payload=payload)

    @staticmethod
    def _session_payload(session: ChatSession) -> dict[str, Any]:
        return {
            "chat_id": session.chat_id,
            "segment": session.segment,
            "reserving_session_id": session.reserving_session_id,
            "messages": [dict(item) for item in session.messages],
            "tool_events": [dict(item) for item in session.tool_events],
            "working_memory": dict(session.working_memory),
            "scenario_ledger": [dict(item) for item in session.scenario_ledger],
            "deterministic_packet": dict(session.deterministic_packet),
            "metadata": dict(session.metadata),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    @staticmethod
    def _session_from_payload(
        *,
        chat_id: str,
        payload: dict[str, Any],
    ) -> ChatSession | None:
        stored_chat_id = str(payload.get("chat_id", "")).strip()
        resolved_chat_id = stored_chat_id or chat_id
        if resolved_chat_id != chat_id:
            return None
        return ChatSession(
            chat_id=resolved_chat_id,
            segment=payload.get("segment"),
            reserving_session_id=payload.get("reserving_session_id"),
            messages=[
                dict(item)
                for item in payload.get("messages", [])
                if isinstance(item, dict)
            ],
            tool_events=[
                dict(item)
                for item in payload.get("tool_events", [])
                if isinstance(item, dict)
            ],
            working_memory=(
                dict(payload.get("working_memory"))
                if isinstance(payload.get("working_memory"), dict)
                else {}
            ),
            scenario_ledger=[
                dict(item)
                for item in payload.get("scenario_ledger", [])
                if isinstance(item, dict)
            ],
            deterministic_packet=(
                dict(payload.get("deterministic_packet"))
                if isinstance(payload.get("deterministic_packet"), dict)
                else {}
            ),
            metadata=(
                dict(payload.get("metadata"))
                if isinstance(payload.get("metadata"), dict)
                else {}
            ),
            created_at=str(payload.get("created_at") or _utc_now()),
            updated_at=str(payload.get("updated_at") or _utc_now()),
        )
