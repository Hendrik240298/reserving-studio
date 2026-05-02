from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.chat_service import AIChatService
from ai.chat_store import FileChatStore, InMemoryChatStore
from source.api.main import create_app


def test_file_chat_store_persists_completed_chat_turn(tmp_path: Path) -> None:
    class _AssistantStub:
        def run_turn(self, **_kwargs):
            return {
                "content": "Recommendation ready.",
                "fallback_used": False,
                "session_id": "s-1",
                "memory_snapshot": {
                    "accepted_analysis_basis": {
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
    assert payload["accepted_analysis_basis"]["scenario_id"] == "drop_1"
    assert "analysis_basis" not in payload["working_memory"]

    reloaded_store = FileChatStore(tmp_path / "chats")
    reloaded_session = reloaded_store.get_chat(session.chat_id)
    assert reloaded_session is not None
    assert reloaded_session.messages[-1]["content"] == "Recommendation ready."
    assert reloaded_session.accepted_analysis_basis["scenario_id"] == "drop_1"


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


def test_chat_service_stores_display_content_but_sends_internal_prompt() -> None:
    captured: dict[str, str] = {}

    class _AssistantStub:
        def run_turn(self, **kwargs):
            captured["user_prompt"] = kwargs["user_prompt"]
            return {
                "content": "Clarification handled.",
                "fallback_used": False,
                "session_id": "s-1",
                "memory_snapshot": {
                    "accepted_analysis_basis": {},
                    "proposal_basis": {},
                    "scenario_ledger": [],
                    "basis_transition_history": [],
                    "deterministic_packet": {},
                },
            }

    service = AIChatService(
        assistant_factory=lambda: _AssistantStub(),
        store=InMemoryChatStore(),
    )
    session = service.create_chat(segment="industrial", reserving_session_id="s-1")

    service.send_message(
        session.chat_id,
        "INTERNAL CONTROL WRAPPER",
        display_content="give me first a comparison",
    )
    for _ in range(50):
        current = service.build_chat_response(session.chat_id)
        if not current.get("streaming"):
            break
        time.sleep(0.01)

    final_response = service.build_chat_response(session.chat_id)
    assert final_response["messages"][0]["content"] == "give me first a comparison"
    assert captured["user_prompt"] == "INTERNAL CONTROL WRAPPER"


def test_chat_service_accepts_pending_proposal_and_updates_message_state() -> None:
    service = AIChatService(
        assistant_factory=lambda: None,
        store=InMemoryChatStore(),
    )
    session = service.create_chat(segment="industrial", reserving_session_id="s-1")
    service._store.append_message(
        session.chat_id,
        {
            "message_id": "msg-1",
            "role": "assistant",
            "content": "Accept this tested scenario?",
            "proposal_basis": {
                "proposal_id": "proposal-1",
                "status": "pending",
                "scenario_id": "drop_1",
                "scenario_label": "drop_1",
                "basis_type": "review_candidate",
                "parameters": {"drop": [["2022", 24]]},
                "presented_in_message_id": "msg-1",
                "origin_execution_id": "run-1",
            },
        },
    )
    service._store.update_memory(
        session.chat_id,
        working_memory={},
        accepted_analysis_basis={},
        proposal_basis={
            "proposal_id": "proposal-1",
            "status": "pending",
            "scenario_id": "drop_1",
            "scenario_label": "drop_1",
            "basis_type": "review_candidate",
            "parameters": {"drop": [["2022", 24]]},
            "presented_in_message_id": "msg-1",
            "origin_execution_id": "run-1",
        },
        scenario_ledger=[],
        execution_records=[],
        basis_transition_history=[],
        deterministic_packet={},
    )

    response = service.accept_proposal(session.chat_id, "proposal-1")

    assert response["accepted_analysis_basis"]["scenario_id"] == "drop_1"
    assert response["accepted_analysis_basis"]["accepted_from_execution_id"] == "run-1"
    assert response["proposal_basis"]["status"] == "accepted"
    assert response["basis_transition_history"][0]["transition_type"] == "proposal_accepted"
    reloaded = service.get_chat(session.chat_id)
    assert reloaded is not None
    assert reloaded.messages[-1]["proposal_basis"]["status"] == "accepted"


def test_chat_service_rejects_pending_proposal_without_changing_basis() -> None:
    service = AIChatService(
        assistant_factory=lambda: None,
        store=InMemoryChatStore(),
    )
    session = service.create_chat(segment="industrial", reserving_session_id="s-1")
    service._store.append_message(
        session.chat_id,
        {
            "message_id": "msg-1",
            "role": "assistant",
            "content": "Reject this tested scenario?",
            "proposal_basis": {
                "proposal_id": "proposal-1",
                "status": "pending",
                "scenario_id": "drop_1",
                "scenario_label": "drop_1",
                "basis_type": "review_candidate",
                "parameters": {"drop": [["2022", 24]]},
                "presented_in_message_id": "msg-1",
            },
        },
    )
    service._store.update_memory(
        session.chat_id,
        working_memory={},
        accepted_analysis_basis={
            "basis_type": "baseline",
            "scenario_id": "baseline",
            "parameters": {},
            "is_active_session": True,
        },
        proposal_basis={
            "proposal_id": "proposal-1",
            "status": "pending",
            "scenario_id": "drop_1",
            "scenario_label": "drop_1",
            "basis_type": "review_candidate",
            "parameters": {"drop": [["2022", 24]]},
            "presented_in_message_id": "msg-1",
        },
        scenario_ledger=[],
        execution_records=[],
        basis_transition_history=[],
        deterministic_packet={},
    )

    response = service.reject_proposal(session.chat_id, "proposal-1")

    assert response["accepted_analysis_basis"]["scenario_id"] == "baseline"
    assert response["proposal_basis"]["status"] == "rejected"
    assert response["basis_transition_history"][0]["transition_type"] == "proposal_rejected"
    reloaded = service.get_chat(session.chat_id)
    assert reloaded is not None
    assert reloaded.messages[-1]["proposal_basis"]["status"] == "rejected"


def test_chat_service_supersedes_older_pending_proposal_on_new_recommendation() -> None:
    class _AssistantStub:
        def run_turn(self, **_kwargs):
            return {
                "content": "A newer recommendation is ready.",
                "fallback_used": False,
                "session_id": "s-1",
                "memory_snapshot": {
                    "accepted_analysis_basis": {},
                    "proposal_basis": {
                        "proposal_id": "proposal-2",
                        "status": "pending",
                        "scenario_id": "drop_2",
                        "scenario_label": "drop_2",
                        "basis_type": "review_candidate",
                        "parameters": {"drop": [["2021", 24]]},
                    },
                    "scenario_ledger": [],
                    "basis_transition_history": [],
                    "deterministic_packet": {},
                },
            }

    store = InMemoryChatStore()
    service = AIChatService(
        assistant_factory=lambda: _AssistantStub(),
        store=store,
    )
    session = service.create_chat(segment="industrial", reserving_session_id="s-1")
    store.append_message(
        session.chat_id,
        {
            "message_id": "msg-old",
            "role": "assistant",
            "content": "Original proposal.",
            "proposal_basis": {
                "proposal_id": "proposal-1",
                "status": "pending",
                "scenario_id": "drop_1",
                "scenario_label": "drop_1",
                "basis_type": "review_candidate",
                "parameters": {"drop": [["2022", 24]]},
                "presented_in_message_id": "msg-old",
            },
        },
    )
    store.update_memory(
        session.chat_id,
        working_memory={},
        accepted_analysis_basis={},
        proposal_basis={
            "proposal_id": "proposal-1",
            "status": "pending",
            "scenario_id": "drop_1",
            "scenario_label": "drop_1",
            "basis_type": "review_candidate",
            "parameters": {"drop": [["2022", 24]]},
            "presented_in_message_id": "msg-old",
        },
        scenario_ledger=[],
        execution_records=[],
        basis_transition_history=[],
        deterministic_packet={},
    )

    service.start_message(session.chat_id, "What do you recommend now?")
    for _ in range(50):
        current = service.build_chat_response(session.chat_id)
        if not current.get("streaming"):
            break
        time.sleep(0.01)

    final_response = service.build_chat_response(session.chat_id)
    assert final_response["proposal_basis"]["proposal_id"] == "proposal-2"
    reloaded = service.get_chat(session.chat_id)
    assert reloaded is not None
    old_message = next(
        item for item in reloaded.messages if item.get("message_id") == "msg-old"
    )
    assert old_message["proposal_basis"]["status"] == "superseded"
    assert old_message["proposal_basis"]["superseded_by_proposal_id"] == "proposal-2"
    new_message = next(
        item
        for item in reloaded.messages
        if item.get("role") == "assistant" and item.get("message_id") != "msg-old"
    )
    assert new_message["proposal_basis"]["proposal_id"] == "proposal-2"
    assert (
        new_message["proposal_basis"]["presented_in_message_id"]
        == new_message["message_id"]
    )


def test_chat_service_preserves_basis_transition_history_after_follow_up_turn() -> None:
    class _AssistantStub:
        def run_turn(self, **kwargs):
            return {
                "content": "Comparison ready.",
                "fallback_used": False,
                "session_id": "s-1",
                "memory_snapshot": {
                    "accepted_analysis_basis": kwargs["accepted_analysis_basis"],
                    "proposal_basis": kwargs["proposal_basis"],
                    "scenario_ledger": [],
                    "execution_records": kwargs["execution_records"],
                    "basis_transition_history": kwargs["basis_transition_history"],
                    "deterministic_packet": {},
                },
            }

    store = InMemoryChatStore()
    service = AIChatService(
        assistant_factory=lambda: _AssistantStub(),
        store=store,
    )
    session = service.create_chat(segment="industrial", reserving_session_id="s-1")
    store.update_memory(
        session.chat_id,
        working_memory={},
        accepted_analysis_basis={
            "basis_type": "review_candidate",
            "scenario_id": "drop_1",
            "parameters": {"drop": [["2022", 24]]},
        },
        proposal_basis={},
        scenario_ledger=[],
        execution_records=[],
        basis_transition_history=[
            {
                "transition_id": "transition-1",
                "chat_id": session.chat_id,
                "from_basis_key": None,
                "to_basis_key": "basis-1",
                "transition_type": "proposal_accepted",
                "origin_proposal_id": "proposal-1",
                "created_at": "2026-05-02T00:00:00Z",
            }
        ],
        deterministic_packet={},
    )

    service.start_message(session.chat_id, "Compare this basis to baseline")
    for _ in range(50):
        current = service.build_chat_response(session.chat_id)
        if not current.get("streaming"):
            break
        time.sleep(0.01)

    final_response = service.build_chat_response(session.chat_id)
    assert len(final_response["basis_transition_history"]) == 1
    assert (
        final_response["basis_transition_history"][0]["transition_type"]
        == "proposal_accepted"
    )
