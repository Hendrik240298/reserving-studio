from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.services.session_sync_service import SessionSyncService


class _FakeConfig:
    def __init__(self) -> None:
        self.saved_payload: dict | None = None
        self.persisted_version: int | None = None

    def load_session(self) -> dict:
        return {}

    def get_sync_version(self) -> int:
        return 0

    def normalize_session_payload(self, payload: dict) -> dict:
        return payload.copy()

    def persist_session_snapshot(self, data: dict, sync_version: int) -> int:
        self.saved_payload = data
        self.persisted_version = sync_version
        return sync_version


def test_local_sync_persists_tail_projection_months() -> None:
    fake_config = _FakeConfig()
    live_results_store: dict[str, dict] = {}
    service = SessionSyncService(
        config=fake_config,  # type: ignore[arg-type]
        segment_key_provider=lambda: "quarterly",
        live_results_store=live_results_store,
    )

    results_payload = {"last_updated": "2026-02-20T00:00:00Z"}
    params = {
        "average": "volume",
        "tail_curve": "weibull",
        "drop_store": [],
        "tail_attachment_age": 96,
        "tail_projection_months": 24,
        "tail_fit_period_selection": [3, 93],
        "bf_apriori_by_uwy": {"2001": 0.6},
        "selected_ultimate_by_uwy": {"2001": "chainladder"},
    }

    payload, publish_message, save_request = service.apply_local_source_payload(
        results_payload=results_payload,
        params=params,  # type: ignore[arg-type]
        current_payload=None,
        sync_ready=False,
    )

    assert fake_config.saved_payload is None
    assert save_request is not None
    persisted_version = service.flush_pending_session_save(save_request)

    assert fake_config.saved_payload is not None
    assert fake_config.saved_payload["tail_projection_months"] == 24
    assert persisted_version == 1
    assert payload["sync_version"] == 1
    assert publish_message is None
