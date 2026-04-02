from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.chat_service import AIChatService
from ai.chat_store import FileChatStore
from source.api.main import create_app


def test_file_chat_store_persists_completed_chat_turn(tmp_path: Path) -> None:
    class _AssistantStub:
        def run_turn(self, **_kwargs):
            return {
                "content": "Recommendation ready.",
                "fallback_used": False,
                "session_id": "s-1",
                "memory_snapshot": {
                    "analysis_basis": {
                        "basis_type": "review_candidate",
                        "scenario_id": "drop_1",
                        "is_active_session": False,
                        "parameters": {"drop": [["2022", 12]]},
                    },
                    "scenario_ledger": [{"scenario_id": "drop_1", "score": 1.0}],
                    "deterministic_packet": {
                        "plan": {"playbook": "scenario_recommendation"}
                    },
                },
            }

    store = FileChatStore(tmp_path / "chats")
    service = AIChatService(
        assistant_factory=lambda: _AssistantStub(),
        store=store,
    )
    session = service.create_chat(segment="industrial", reserving_session_id="s-1")

    service.start_message(session.chat_id, "What scenario do you recommend?")
    for _ in range(50):
        current = service.build_chat_response(session.chat_id)
        if not current.get("streaming"):
            break

    persisted_path = tmp_path / "chats" / f"{session.chat_id}.json"
    assert persisted_path.exists()

    payload = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert payload["chat_id"] == session.chat_id
    assert payload["messages"][-1]["content"] == "Recommendation ready."
    assert payload["scenario_ledger"][0]["scenario_id"] == "drop_1"
    assert payload["working_memory"]["analysis_basis"]["scenario_id"] == "drop_1"

    reloaded_store = FileChatStore(tmp_path / "chats")
    reloaded_session = reloaded_store.get_chat(session.chat_id)
    assert reloaded_session is not None
    assert reloaded_session.messages[-1]["content"] == "Recommendation ready."
    assert reloaded_session.working_memory["analysis_basis"]["scenario_id"] == "drop_1"


def test_create_app_uses_file_chat_store_when_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        """
paths:
  results: results/
  plots: plots/
  data: data/
  sessions: sessions/
last date: "March 2026"
segment: industrial
ai:
  chat_logging:
    enabled: true
    path: chats
""".strip(),
        encoding="utf-8",
    )

    app = create_app(config_path=config_path)

    assert isinstance(app.state.chat_store, FileChatStore)
    assert app.state.chat_store._directory == tmp_path / "chats"
