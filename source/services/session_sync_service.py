from __future__ import annotations

import logging
import time

from source.config_manager import ConfigManager
from source.services.types import ParamsState, ResultsPayload


class SessionSyncService:
    def __init__(
        self,
        *,
        config: ConfigManager | None,
        segment_key_provider,
        live_results_store: dict[str, dict],
        live_session_store: dict[str, dict] | None = None,
        persist_debounce_seconds: float = 0.25,
    ) -> None:
        self._config = config
        self._segment_key_provider = segment_key_provider
        self._live_results_store = live_results_store
        self._live_session_store = (
            live_session_store if live_session_store is not None else {}
        )
        self._persist_debounce_seconds = max(float(persist_debounce_seconds), 0.0)

    def _build_session_payload(self, params: ParamsState) -> dict:
        return {
            "average": params.get("average"),
            "tail_curve": params.get("tail_curve"),
            "drops": params.get("drop_store"),
            "tail_attachment_age": params.get("tail_attachment_age"),
            "tail_projection_months": params.get("tail_projection_months"),
            "tail_fit_period": params.get("tail_fit_period_selection"),
            "bf_apriori_by_uwy": params.get("bf_apriori_by_uwy"),
            "selected_ultimate_by_uwy": params.get("selected_ultimate_by_uwy"),
        }

    @staticmethod
    def _normalize_session_payload(payload: dict) -> dict:
        normalized = payload.copy()
        normalized.pop("sync_version", None)
        normalized.pop("updated_at", None)
        return normalized

    def _get_normalized_payload(self, payload: dict) -> dict:
        if self._config is not None:
            return self._config.normalize_session_payload(payload)
        return self._normalize_session_payload(payload)

    def _get_or_create_live_session_state(self, segment_key: str) -> dict:
        state = self._live_session_store.get(segment_key)
        if isinstance(state, dict):
            return state

        persisted_payload: dict = {}
        persisted_sync_version = 0
        if self._config is not None:
            persisted_payload = self._config.load_session()
            persisted_sync_version = self._config.get_sync_version()

        state = {
            "session_payload": persisted_payload,
            "normalized_payload": self._get_normalized_payload(persisted_payload),
            "sync_version": persisted_sync_version,
            "persisted_sync_version": persisted_sync_version,
        }
        self._live_session_store[segment_key] = state
        return state

    def apply_sync_source_payload(
        self,
        *,
        results_payload: ResultsPayload,
        params: ParamsState,
    ) -> tuple[ResultsPayload, dict | None, dict | None]:
        incoming_sync = params.get("sync_version")
        try:
            sync_version = int(incoming_sync)
        except (TypeError, ValueError):
            sync_version = 0

        segment_key = self._segment_key_provider()
        session_payload = self._build_session_payload(params)
        normalized_payload = self._get_normalized_payload(session_payload)
        state = self._get_or_create_live_session_state(segment_key)
        state["session_payload"] = session_payload
        state["normalized_payload"] = normalized_payload
        state["sync_version"] = sync_version

        results_payload["sync_version"] = sync_version
        self._live_results_store[segment_key] = results_payload
        return results_payload, None, None

    def apply_local_source_payload(
        self,
        *,
        results_payload: ResultsPayload,
        params: ParamsState,
        current_payload: dict | None,
        sync_ready: bool,
    ) -> tuple[ResultsPayload, dict | None, dict | None]:
        segment_key = self._segment_key_provider()
        state = self._get_or_create_live_session_state(segment_key)
        previous_sync_version = 0
        try:
            previous_sync_version = int(state.get("sync_version", 0))
        except (TypeError, ValueError):
            previous_sync_version = 0

        if previous_sync_version <= 0 and isinstance(current_payload, dict):
            try:
                previous_sync_version = int(current_payload.get("sync_version", 0))
            except (TypeError, ValueError):
                previous_sync_version = 0

        session_payload = self._build_session_payload(params)
        normalized_payload = self._get_normalized_payload(session_payload)
        if normalized_payload == state.get("normalized_payload"):
            results_payload["sync_version"] = previous_sync_version
            self._live_results_store[segment_key] = results_payload
            logging.info(
                "Session unchanged in memory for segment '%s'; skipping save scheduling",
                segment_key,
            )
            return results_payload, None, None

        sync_version = previous_sync_version + 1
        state["session_payload"] = session_payload
        state["normalized_payload"] = normalized_payload
        state["sync_version"] = sync_version

        results_payload["sync_version"] = sync_version
        self._live_results_store[segment_key] = results_payload

        save_request = {
            "segment_key": segment_key,
            "session_payload": session_payload,
            "sync_version": sync_version,
            "scheduled_at": time.time(),
            "due_at": time.time() + self._persist_debounce_seconds,
        }
        logging.info(
            "Session save scheduled for segment '%s' version=%s debounce_ms=%.0f",
            segment_key,
            sync_version,
            self._persist_debounce_seconds * 1000,
        )

        if not bool(sync_ready):
            return results_payload, None, save_request
        if sync_version <= previous_sync_version:
            return results_payload, None, save_request

        publish_message = {
            "sync_version": sync_version,
            "updated_at": results_payload.get("last_updated"),
            "session_payload": session_payload,
        }
        return results_payload, publish_message, save_request

    def flush_pending_session_save(self, save_request: dict | None) -> int | None:
        if not isinstance(save_request, dict):
            return None
        if self._config is None:
            return None

        segment_key = str(
            save_request.get("segment_key") or self._segment_key_provider()
        )
        sync_version_raw = save_request.get("sync_version", 0)
        try:
            sync_version = int(sync_version_raw)
        except (TypeError, ValueError):
            return None

        state = self._get_or_create_live_session_state(segment_key)
        try:
            persisted_sync_version = int(state.get("persisted_sync_version", 0))
        except (TypeError, ValueError):
            persisted_sync_version = 0
        if sync_version <= persisted_sync_version:
            logging.info(
                "Session save already persisted for segment '%s' version=%s",
                segment_key,
                persisted_sync_version,
            )
            return persisted_sync_version

        session_payload = save_request.get("session_payload")
        if not isinstance(session_payload, dict):
            return None

        started = time.perf_counter()
        persisted_version = self._config.persist_session_snapshot(
            session_payload,
            sync_version,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        state["persisted_sync_version"] = persisted_version
        logging.info(
            "Session save flushed for segment '%s' version=%s total_ms=%.0f",
            segment_key,
            persisted_version,
            elapsed_ms,
        )
        return persisted_version
