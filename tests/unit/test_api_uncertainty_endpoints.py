from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source.api.main import create_app
from source.config_manager import ConfigManager
from source.input_loader import load_inputs_from_config


def _json_safe_rows(frame) -> list[dict]:
    records = frame.to_dict(orient="records")
    normalized: list[dict] = []
    for row in records:
        safe_row: dict = {}
        for key, value in row.items():
            isoformat = getattr(value, "isoformat", None)
            if callable(isoformat):
                try:
                    safe_row[str(key)] = isoformat()
                    continue
                except Exception:
                    pass
            safe_row[str(key)] = value
        normalized.append(safe_row)
    return normalized


def _load_quarterly_inputs() -> tuple[str, list[dict], list[dict], str]:
    config = ConfigManager.from_yaml(REPO_ROOT / "examples" / "config_quarterly.yml")
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)
    return (
        config.get_segment(),
        _json_safe_rows(claims_df),
        _json_safe_rows(premium_df),
        config.get_granularity(),
    )


def test_diagnostics_endpoints_include_uncertainty_blocks() -> None:
    app = create_app()
    client = TestClient(app)
    segment, claims_rows, premium_rows, granularity = _load_quarterly_inputs()

    workflow_response = client.post(
        "/v1/workflows/from-dataframes",
        json={
            "segment": segment,
            "claims_rows": claims_rows,
            "premium_rows": premium_rows,
            "granularity": granularity,
        },
    )
    assert workflow_response.status_code == 200
    session_id = workflow_response.json()["session_id"]

    diagnostics_response = client.post(
        "/v1/diagnostics/run",
        json={"session_id": session_id},
    )
    assert diagnostics_response.status_code == 200
    diagnostics_json = diagnostics_response.json()
    assert "uncertainty" in diagnostics_json
    assert diagnostics_json["uncertainty"]["version"] == "v1.1"
    assert "total_process_cv" in diagnostics_json["uncertainty"]

    iterate_response = client.post(
        "/v1/diagnostics/iterate",
        json={"session_id": session_id, "max_scenarios": 12, "include_baseline": True},
    )
    assert iterate_response.status_code == 200
    iterate_json = iterate_response.json()
    assert "uncertainty" in iterate_json
    assert "baseline" in iterate_json["uncertainty"]
    assert "bootstrap" in iterate_json["uncertainty"]
    assert "tail_model" in iterate_json["uncertainty"]
