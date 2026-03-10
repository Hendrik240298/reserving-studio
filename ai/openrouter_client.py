from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


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
        self._fallback_model = os.environ.get("AI_FALLBACK_MODEL")
        self._max_retries = int(os.environ.get("AI_PROVIDER_MAX_RETRIES", "2"))
        self._retry_base_delay_seconds = float(
            os.environ.get("AI_PROVIDER_RETRY_BASE_DELAY_SECONDS", "0.7")
        )
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
        models = [self._model]
        if self._fallback_model and self._fallback_model != self._model:
            models.append(self._fallback_model)

        last_error: RuntimeError | None = None
        for model in models:
            try:
                return self._chat_completion_with_model(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except RuntimeError as error:
                last_error = error
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenRouter request failed with unknown error")

    def _chat_completion_with_model(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
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

        attempt = 0
        last_error: RuntimeError | None = None
        while attempt <= self._max_retries:
            try:
                with request.urlopen(req, timeout=120) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                should_retry = error.code in {429, 502, 503, 504}
                last_error = RuntimeError(
                    f"OpenRouter request failed ({error.code}) model={model}: {body}"
                )
                if not should_retry or attempt >= self._max_retries:
                    raise last_error from error
            except URLError as error:
                last_error = RuntimeError(
                    f"OpenRouter request failed (network) model={model}: {error}"
                )
                if attempt >= self._max_retries:
                    raise last_error from error

            attempt += 1
            delay = self._retry_base_delay_seconds * attempt
            time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"OpenRouter request failed for model={model}")
