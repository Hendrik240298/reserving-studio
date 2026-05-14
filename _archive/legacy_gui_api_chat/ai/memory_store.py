from __future__ import annotations

from typing import Any

from source.app import load_config
from source.config_manager import ConfigManager
from source.services.segment_memory_service import SegmentMemoryService


class SegmentMemoryStore:
    def __init__(self, *, config: ConfigManager | None = None) -> None:
        self._config = config if config is not None else load_config()
        self._service = SegmentMemoryService()

    def load(self, *, segment: str | None) -> dict[str, Any]:
        if self._config is None or not isinstance(segment, str) or not segment.strip():
            return self._service.load({}, segment=segment)
        payload = self._config.load_ai_segment_memory(segment=segment.strip())
        return self._service.load(
            payload if isinstance(payload, dict) else {},
            segment=segment.strip(),
        )

    def save(
        self,
        *,
        segment: str | None,
        memory: dict[str, Any],
    ) -> None:
        if self._config is None or not isinstance(segment, str) or not segment.strip():
            return
        raw_existing = self._config.load_ai_segment_memory(segment=segment.strip())
        payload = self._service.merge(
            existing_memory=raw_existing if isinstance(raw_existing, dict) else {},
            incoming_memory=memory,
            segment=segment.strip(),
        )
        self._config.save_ai_segment_memory(payload, segment=segment.strip())
