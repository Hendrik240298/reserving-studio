from __future__ import annotations

from source.config_manager import ConfigManager
from source.services.types import ParamsState, ResultsPayload


class SessionSyncService:
    def __init__(
        self,
        *,
        config: ConfigManager | None,
        segment_key_provider,
        live_results_store: dict[str, dict],
    ) -> None:
        self._config = config
        self._segment_key_provider = segment_key_provider
        self._live_results_store = live_results_store

    def apply_sync_source_payload(
        self,
        *,
        results_payload: ResultsPayload,
        params: ParamsState,
    ) -> tuple[ResultsPayload, dict | None]:
        incoming_sync = params.get("sync_version")
        try:
            sync_version = int(incoming_sync)
        except (TypeError, ValueError):
            sync_version = 0
        results_payload["sync_version"] = sync_version
        segment_key = self._segment_key_provider()
        self._live_results_store[segment_key] = results_payload
        return results_payload, None

    def apply_local_source_payload(
        self,
        *,
        results_payload: ResultsPayload,
        params: ParamsState,
        current_payload: dict | None,
        sync_ready: bool,
    ) -> tuple[ResultsPayload, dict | None]:
        average = params.get("average")
        tail_curve = params.get("tail_curve")
        drop_store = params.get("drop_store")
        tail_attachment_age = params.get("tail_attachment_age")
        tail_fit_period_selection = params.get("tail_fit_period_selection")
        bf_apriori_by_uwy = params.get("bf_apriori_by_uwy")

        sync_version = 0
        if self._config is not None:
            sync_version = self._config.save_session_with_version(
                {
                    "average": average,
                    "tail_curve": tail_curve,
                    "drops": drop_store,
                    "tail_attachment_age": tail_attachment_age,
                    "tail_fit_period": tail_fit_period_selection,
                    "bf_apriori_by_uwy": bf_apriori_by_uwy,
                }
            )
        elif isinstance(current_payload, dict):
            try:
                sync_version = int(current_payload.get("sync_version", 0)) + 1
            except (TypeError, ValueError):
                sync_version = 1
        else:
            sync_version = 1

        results_payload["sync_version"] = sync_version
        segment_key = self._segment_key_provider()
        self._live_results_store[segment_key] = results_payload

        previous_sync_version = 0
        if isinstance(current_payload, dict):
            try:
                previous_sync_version = int(current_payload.get("sync_version", 0))
            except (TypeError, ValueError):
                previous_sync_version = 0

        if not bool(sync_ready):
            return results_payload, None
        if sync_version <= previous_sync_version:
            return results_payload, None

        publish_message = {
            "sync_version": sync_version,
            "updated_at": results_payload.get("last_updated"),
        }
        return results_payload, publish_message
