from __future__ import annotations

import json
from typing import Any

from ai.api_tools import ReservingApiTools


class AIChatApiClient:
    def __init__(self, *, base_url: str) -> None:
        self._api = ReservingApiTools(base_url=base_url)

    def create_chat(
        self,
        *,
        segment: str | None = None,
        reserving_session_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if segment is not None:
            payload["segment"] = segment
        if reserving_session_id is not None:
            payload["reserving_session_id"] = reserving_session_id
        return self._api.request_json("POST", "/v1/ai/chats", payload)

    def send_message(self, *, chat_id: str, content: str) -> dict[str, Any]:
        return self._api.request_json(
            "POST",
            f"/v1/ai/chats/{chat_id}/messages",
            {"content": content},
        )

    def get_chat(self, *, chat_id: str) -> dict[str, Any]:
        return self._api.request_json("GET", f"/v1/ai/chats/{chat_id}")
