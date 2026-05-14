from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.app import (
    build_workflow_from_dataframes,
    create_interactive_session_controller,
    run_interactive_session,
)
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config


def _load_config() -> ConfigManager:
    config_path = REPO_ROOT / "examples" / "config_sql_template.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"SQL template config not found at {config_path}")
    return ConfigManager.from_yaml(config_path)


def main() -> None:
    config = _load_config()
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)

    reserving = build_workflow_from_dataframes(
        claims_df=claims_df,
        premium_df=premium_df,
        config=config,
    )
    controller = create_interactive_session_controller()

    print("Dashboard starting at http://127.0.0.1:8050")
    print("Use the UI, then click 'Finalize & Continue' in the Results tab.")

    finalized = run_interactive_session(
        reserving,
        config=config,
        controller=controller,
        port=8050,
        timeout_seconds=None,
    )

    print("\nInteractive session finalized.")
    print(f"Segment: {finalized.segment}")
    print(f"Finalized at: {finalized.finalized_at_utc.isoformat()}")
    print("Selected methods by UWY:")
    for uwy, method in finalized.params_store.selected_ultimate_by_uwy.items():
        print(f"  {uwy}: {method}")

    print("\nTop 5 rows of finalized numeric results:")
    print(finalized.results_df.head())


if __name__ == "__main__":
    main()
