from __future__ import annotations

import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.assistant_service import AssistantService
from ai.env_loader import load_dotenv
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config


def _load_config() -> ConfigManager:
    config_path = REPO_ROOT / "examples" / "config_quarterly.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Quarterly config not found at {config_path}")
    return ConfigManager.from_yaml(config_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env.local")

    config = _load_config()
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)

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
        "Run diagnostics and produce concise reserving commentary with explicit evidence IDs. "
        "Highlight method divergence, AY anomalies, and development instability."
    )
    answer = assistant.answer(user_prompt=prompt)

    print("Workflow initialized:")
    print(workflow)
    print("\nAssistant commentary:\n")
    print(answer)


if __name__ == "__main__":
    main()
