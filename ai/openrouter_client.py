from __future__ import annotations

import json
import os
from typing import Any
from urllib import request
from urllib.error import HTTPError


class OpenRouterClient:
    def __init__(self) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self._api_key = api_key
        self._base_url = os.environ.get(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).rstrip("/")
        self._model = os.environ.get("AI_MODEL", "minimax/minimax-m2.5")
        self._http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        self._title = os.environ.get("OPENROUTER_APP_TITLE", "reserving-studio")

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        raw_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._title:
            headers["X-Title"] = self._title

        req = request.Request(
            url=f"{self._base_url}/chat/completions",
            method="POST",
            data=raw_body,
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter request failed ({error.code}): {body}"
            ) from error
