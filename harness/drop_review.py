from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Literal

from source.config_manager import ConfigManager
from source.drop_analysis import (
    DEFAULT_METHOD,
    build_options_from_config,
    build_triangle,
    run_native_drop_analysis,
)
from source.input_loader import load_inputs_from_config

from harness.markdown import render_drop_review_packet


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG_PATH = Path("examples/config_quarterly.yml")
DEFAULT_ARTIFACT_DIR = Path("harness/artifacts")


@dataclass(frozen=True)
class DropReviewPacketResult:
    output_path: Path
    review_type: str
    recommendation_class: str | None
    candidate_id: str | None
    candidate_count: int
    warnings: list[str] = field(default_factory=list)


def run_drop_review_packet(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
    candidate_limit: int = 5,
    method: Literal["chainladder", "bornhuetter_ferguson"] = DEFAULT_METHOD,
    use_tail: bool = False,
    enforce_monotone_tail: bool = False,
) -> DropReviewPacketResult:
    resolved_config_path = _resolve_repo_path(config_path)
    if output_path is None:
        output_path = _default_output_path()
    resolved_output_path = _resolve_repo_path(output_path)

    requested_inputs = {
        "config_path": _display_path(resolved_config_path),
        "candidate_limit": candidate_limit,
        "method": method,
        "use_tail": use_tail,
        "enforce_monotone_tail": enforce_monotone_tail,
    }

    config = ConfigManager.from_yaml(resolved_config_path)
    claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)
    triangle = build_triangle(claims_df, premium_df, config=config)
    options = build_options_from_config(
        config,
        method=method,
        use_tail=use_tail,
        enforce_monotone_tail=enforce_monotone_tail,
    )
    review_payload = run_native_drop_analysis(
        triangle,
        options=options,
        candidate_limit=candidate_limit,
    )
    summary_payload = _summarize_drop_review_payload(review_payload)

    effective_inputs = {
        "segment": config.get_segment(),
        "granularity": config.get_granularity(),
        "dataset": config.get_workflow_dataset(),
        "quarterly_premium_csv": config.get_workflow_quarterly_premium_csv(),
        "candidate_limit": candidate_limit,
        "method": method,
        "use_tail": use_tail,
        "enforce_monotone_tail": enforce_monotone_tail,
        "average": options.average,
        "base_drop_count": len(options.base_drops),
    }
    command = _command_for_packet(
        config_path=config_path,
        output_path=output_path,
        candidate_limit=candidate_limit,
        method=method,
        use_tail=use_tail,
        enforce_monotone_tail=enforce_monotone_tail,
    )

    packet = render_drop_review_packet(
        review_payload=review_payload,
        summary_payload=summary_payload,
        requested_inputs=requested_inputs,
        effective_inputs=effective_inputs,
        command=command,
        timestamp=datetime.now(timezone.utc).isoformat(),
        code_version=_git_revision(),
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(packet, encoding="utf-8")

    recommendation = summary_payload.get("recommendation") or {}
    candidates = summary_payload.get("top_candidates") or []
    return DropReviewPacketResult(
        output_path=resolved_output_path,
        review_type=str(summary_payload.get("review_type") or "native_drop_analysis"),
        recommendation_class=_optional_str(recommendation.get("recommendation_class")),
        candidate_id=_optional_str(recommendation.get("candidate_id")),
        candidate_count=int(summary_payload.get("candidate_count") or len(candidates)),
        warnings=_collect_warnings(review_payload),
    )


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists() or candidate.parent.exists():
        return candidate
    return REPO_ROOT / path


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_ARTIFACT_DIR / f"drop_review_quarterly_{stamp}.md"


def _command_for_packet(
    *,
    config_path: Path,
    output_path: Path,
    candidate_limit: int,
    method: Literal["chainladder", "bornhuetter_ferguson"],
    use_tail: bool,
    enforce_monotone_tail: bool,
) -> str:
    parts = [
        "uv run python -m harness.cli drop-review",
        f"--config {config_path}",
        f"--candidate-limit {candidate_limit}",
        f"--method {method}",
        f"--output {output_path}",
    ]
    if use_tail:
        parts.append("--use-tail")
    if enforce_monotone_tail:
        parts.append("--enforce-monotone-tail")
    return " ".join(parts)


def _summarize_drop_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = _as_list(payload.get("candidates"))
    recommendation = _as_dict(payload.get("recommendation"))
    return {
        "review_type": payload.get("review_type"),
        "candidate_count": len(candidates),
        "top_candidates": [_compact_review_candidate(item) for item in candidates[:5]],
        "recommendation": _compact_review_recommendation(recommendation),
        "continuity_notes": [],
        "policy_trace": _as_dict(payload.get("policy_trace")),
        "evidence_summary": _as_dict(payload.get("evidence_summary")),
        "run_metadata": _as_dict(payload.get("run_metadata")),
    }


def _compact_review_candidate(item: Any) -> dict[str, Any]:
    candidate = _as_dict(item)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "recommendation_class": candidate.get("recommendation_class"),
        "summary": candidate.get("summary"),
        "score": candidate.get("score"),
        "metrics": candidate.get("metrics", {}),
    }


def _compact_review_recommendation(recommendation: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_class": recommendation.get("recommendation_class"),
        "candidate_id": recommendation.get("candidate_id"),
        "summary": recommendation.get("summary"),
        "caveats": _as_list(recommendation.get("caveats")),
        "alternatives": _as_list(recommendation.get("alternatives")),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    revision = result.stdout.strip()
    return revision or "unknown"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _collect_warnings(review_payload: dict) -> list[str]:
    warnings: list[str] = []
    recommendation = review_payload.get("recommendation")
    if isinstance(recommendation, dict):
        warnings.extend(str(item) for item in recommendation.get("caveats", []) if item)
    if not review_payload.get("candidates"):
        warnings.append(
            "Native drop analysis returned no deterministic drop candidates for the selected mode."
        )
    return warnings
