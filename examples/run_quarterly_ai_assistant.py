from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.env_loader import load_dotenv
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
        default=8050,
        help="Dash UI port (default: 8050).",
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

    if args.print_cli_commentary:
        from ai.assistant_service import AssistantService

        assistant = AssistantService(api_base_url="http://127.0.0.1:8000")
        workflow = assistant.bootstrap_workflow(
            segment=config.get_segment(),
            claims_rows=claims_df.to_dict(orient="records"),
            premium_rows=premium_df.to_dict(orient="records"),
            granularity=config.get_granularity(),
        )
        session_id = workflow["session_id"]
        prompt = (
            f"Session initialized with session_id={session_id}, segment={config.get_segment()}. "
            "Run tool_get_session, tool_run_diagnostics, tool_iterate_diagnostics "
            "(include_baseline=true, max_scenarios=20), and tool_get_results before finalizing. "
            "Produce detailed reserving commentary with sections: Executive summary, key findings, "
            "governance and escalation, scenario trade-offs, and recommended actions. "
            "Include explicit evidence IDs for every material claim and include uncertainty statements "
            "where confidence is reduced."
        )
        answer = assistant.answer(user_prompt=prompt)
        print("Workflow initialized:")
        print(workflow)
        print("\nAssistant commentary:\n")
        print(answer)

    print(f"\nStarting standalone AI Dash UI on http://127.0.0.1:{args.port}\n")
    reserving = build_workflow_from_dataframes(
        claims_df,
        premium_df,
        config=config,
    )
    launch_ai_dashboard(reserving, config=config, port=args.port)


if __name__ == "__main__":
    main()
