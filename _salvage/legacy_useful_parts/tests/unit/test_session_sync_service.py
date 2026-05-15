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

    def save_session_with_version(self, data: dict) -> int:
        self.saved_payload = data
        return 7


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

    payload, publish_message = service.apply_local_source_payload(
        results_payload=results_payload,
        params=params,  # type: ignore[arg-type]
        current_payload=None,
        sync_ready=False,
    )

    assert fake_config.saved_payload is not None
    assert fake_config.saved_payload["tail_projection_months"] == 24
    assert payload["sync_version"] == 7
    assert publish_message is None
