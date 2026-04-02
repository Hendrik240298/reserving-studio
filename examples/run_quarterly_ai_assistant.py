from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.env_loader import load_dotenv
from ai.assistant_service import AssistantService
from ai.chat_service import AIChatService
from ai.chat_store import FileChatStore
from source.api.adapters.reserving_adapter import InMemoryReservingBackend
from source.api.schemas import WorkflowFromDataframesRequest
from source.ai_dashboard import launch_ai_dashboard
from source.app import build_workflow_from_dataframes
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config


def _load_config() -> ConfigManager:
    config_path = REPO_ROOT / "examples" / "config_quarterly.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Quarterly config not found at {config_path}")
    return ConfigManager.from_yaml(config_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the standalone quarterly AI Dash workspace."
    )
    parser.add_argument(
        "--print-cli-commentary",
        action="store_true",
        help="Also run the API-backed CLI assistant first and print commentary before opening Dash.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8052,
        help="Dash UI port (default: 8052).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env.local")

    config = _load_config()
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)
    backend = InMemoryReservingBackend()
    workflow = backend.create_workflow_from_dataframes(
        WorkflowFromDataframesRequest(
            segment=config.get_segment(),
            claims_rows=claims_df.to_dict(orient="records"),
            premium_rows=premium_df.to_dict(orient="records"),
            granularity=config.get_granularity(),
        )
    )
    chat_store = (
        FileChatStore(config.get_ai_chat_logging_path())
        if config.is_ai_chat_logging_enabled()
        else None
    )
    chat_service = AIChatService(
        assistant_factory=lambda: AssistantService.from_backend(backend=backend),
        store=chat_store,
    )
    chat = chat_service.create_chat(
        segment=config.get_segment(),
        reserving_session_id=workflow.session_id,
    )

    if args.print_cli_commentary:
        response = chat_service.send_message(
            chat.chat_id,
            (
                "Run diagnostics, iterate meaningful reserving scenarios, and provide detailed commentary "
                "with sections: Executive summary, key findings, governance and escalation, scenario trade-offs, "
                "and recommended actions. Include evidence references where available and explain uncertainty."
            ),
        )
        print("Workflow initialized:")
        print(workflow.model_dump(mode="json"))
        print("\nAssistant commentary:\n")
        print(response.get("assistant_message", ""))

    print(f"\nStarting standalone AI Dash UI on http://127.0.0.1:{args.port}\n")
    reserving = build_workflow_from_dataframes(
        claims_df,
        premium_df,
        config=config,
    )
    launch_ai_dashboard(
        reserving,
        config=config,
        chat_service=chat_service,
        chat_id=chat.chat_id,
        port=args.port,
    )


if __name__ == "__main__":
    main()
