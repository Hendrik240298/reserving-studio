from __future__ import annotations

from typing import Any

from source.app import load_config
from source.config_manager import ConfigManager


class SegmentMemoryStore:
    def __init__(self, *, config: ConfigManager | None = None) -> None:
        self._config = config if config is not None else load_config()

    def load(self, *, segment: str | None) -> dict[str, Any]:
        if self._config is None or not isinstance(segment, str) or not segment.strip():
            return {}
        payload = self._config.load_ai_segment_memory(segment=segment.strip())
        return payload if isinstance(payload, dict) else {}

    def save(
        self,
        *,
        segment: str | None,
        memory: dict[str, Any],
    ) -> None:
        if self._config is None or not isinstance(segment, str) or not segment.strip():
            return
        payload = dict(memory)
        payload.setdefault("segment_id", segment.strip())
        self._config.save_ai_segment_memory(payload, segment=segment.strip())
