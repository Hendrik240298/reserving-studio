from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).parents[1]
DEFAULT_DROP_REVIEW_TEMPLATE = REPO_ROOT / "harness" / "templates" / "drop_review_packet.md"


def render_drop_review_packet(
    *,
    review_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    requested_inputs: dict[str, Any],
    effective_inputs: dict[str, Any],
    command: str,
    timestamp: str,
    code_version: str,
    template_path: Path = DEFAULT_DROP_REVIEW_TEMPLATE,
) -> str:
    """Render a human-readable markdown packet for a drop review run."""

    recommendation = _as_dict(summary_payload.get("recommendation"))
    candidates = _review_candidates(review_payload, summary_payload)
    continuity_notes = _as_list(summary_payload.get("continuity_notes"))
    policy_trace = _as_dict(summary_payload.get("policy_trace"))
    evidence_summary = _as_dict(summary_payload.get("evidence_summary"))
    run_metadata = _as_dict(summary_payload.get("run_metadata"))
    values = {
        "executive_summary": _summary_sentence(recommendation, candidates),
        "recommendation_result": _join_lines(
            [
                f"- Recommendation class: {_value(recommendation.get('recommendation_class'))}",
                f"- Candidate id: {_value(recommendation.get('candidate_id'))}",
                f"- Summary: {_value(recommendation.get('summary'))}",
            ]
        ),
        "key_evidence": _join_lines(_evidence_summary_lines(evidence_summary)),
        "candidate_ranking": _join_lines(_candidate_lines(candidates)),
        "actuarial_interpretation": _interpretation_sentence(recommendation),
        "caveats": _join_lines(_caveat_lines(recommendation, continuity_notes)),
        "command": command,
        "requested_inputs": _join_lines(_dict_bullets(requested_inputs)),
        "effective_inputs": _join_lines(_dict_bullets(effective_inputs)),
        "tool_calls": _join_lines(
            [
                "- `ConfigManager.from_yaml`: loaded reserving config.",
                "- `load_inputs_from_config`: loaded configured claims and premium data.",
                "- `build_workflow_from_dataframes`: built deterministic `Reserving` workflow.",
                "- `AssumptionReviewService.review_drops`: ran deterministic drop review.",
            ]
        ),
        "data_lineage": _join_lines(
            [
                f"- Config path: {_value(requested_inputs.get('config_path'))}",
                f"- Dataset: {_value(effective_inputs.get('dataset'))}",
                f"- Premium CSV: {_value(effective_inputs.get('quarterly_premium_csv'))}",
                f"- Segment: {_value(effective_inputs.get('segment'))}",
                f"- Granularity: {_value(effective_inputs.get('granularity'))}",
            ]
        ),
        "evidence_references": _join_lines(
            _evidence_reference_lines(evidence_summary, run_metadata)
        ),
        "warnings": _join_lines(_warning_lines(recommendation, review_payload)),
        "reproducibility": _join_lines(
            [
                f"- Timestamp: {timestamp}",
                f"- Code version: {code_version}",
                f"- Review type: {_value(summary_payload.get('review_type'))}",
                f"- Candidate count: {_value(summary_payload.get('candidate_count'))}",
            ]
        ),
    }
    return _fill_template(template_path.read_text(encoding="utf-8"), values)


def _summary_sentence(recommendation: dict[str, Any], candidates: list[Any]) -> str:
    candidate_id = recommendation.get("candidate_id")
    if candidate_id:
        return (
            f"Drop review returned {len(candidates)} candidates. "
            f"The recommended candidate is `{candidate_id}` with class "
            f"`{_value(recommendation.get('recommendation_class'))}`."
        )
    return (
        f"Drop review returned {len(candidates)} candidates, "
        "but no single candidate was selected as a recommendation."
    )


def _interpretation_sentence(recommendation: dict[str, Any]) -> str:
    summary = str(recommendation.get("summary") or "").strip()
    if summary:
        return summary
    return (
        "Use this packet as review evidence only. Any actuarial selection or "
        "booking action still requires human review and acceptance."
    )


def _candidate_lines(candidates: list[Any]) -> list[str]:
    if not candidates:
        return ["- No candidates returned."]
    lines: list[str] = []
    for index, item in enumerate(candidates, start=1):
        candidate = _as_dict(item)
        label = candidate.get("candidate_id") or f"candidate_{index}"
        lines.append(f"- {index}. `{label}`")
        for key in ("recommendation_class", "summary"):
            if candidate.get(key) not in (None, "", [], {}):
                lines.append(f"    - {key}: {_value(candidate.get(key))}")
        metrics = candidate.get("metrics")
        if metrics:
            lines.append(f"    - metrics: {_value(metrics)}")
    return lines


def _review_candidates(
    review_payload: dict[str, Any], summary_payload: dict[str, Any]
) -> list[Any]:
    candidates = _as_list(review_payload.get("candidates"))
    if candidates:
        return candidates
    return _as_list(summary_payload.get("top_candidates"))


def _caveat_lines(
    recommendation: dict[str, Any], continuity_notes: list[Any]
) -> list[str]:
    lines = [str(item) for item in recommendation.get("caveats", []) if item]
    for note in continuity_notes:
        note_dict = _as_dict(note)
        message = note_dict.get("message") or note_dict.get("code")
        if message:
            lines.append(str(message))
    if not lines:
        return ["- No caveats returned by the deterministic review."]
    return [f"- {line}" for line in lines]


def _warning_lines(
    recommendation: dict[str, Any], review_payload: dict[str, Any]
) -> list[str]:
    warnings = [str(item) for item in recommendation.get("caveats", []) if item]
    if not review_payload.get("candidates"):
        warnings.append("Drop review returned no candidates.")
    if not warnings:
        return ["- No warnings captured."]
    return [f"- {warning}" for warning in warnings]


def _evidence_reference_lines(
    evidence_summary: dict[str, Any], run_metadata: dict[str, Any]
) -> list[str]:
    lines = []
    if evidence_summary:
        lines.append(
            "- Evidence summary included in Key Evidence section "
            f"({len(evidence_summary)} groups)."
        )
    if run_metadata:
        lines.append(f"- Run metadata: {_value(run_metadata)}")
    if not lines:
        return ["- No explicit evidence references returned."]
    return lines


def _evidence_summary_lines(evidence_summary: dict[str, Any]) -> list[str]:
    if not evidence_summary:
        return ["- No evidence summary returned."]
    lines: list[str] = []
    for key, value in evidence_summary.items():
        if key == "baseline_drop_recommendations" and isinstance(value, dict):
            labels = [str(item) for item in list(value.keys())[:5]]
            lines.append(
                f"- Baseline drop recommendations: {len(value)} found"
                f"; first items: {', '.join(labels) if labels else 'none'}."
            )
            continue
        if key == "baseline_movement_summary" and isinstance(value, dict):
            findings = _as_list(value.get("top_findings"))
            lines.append(
                "- Baseline movement diagnostics: "
                f"{_value(value.get('finding_count'))} findings; "
                f"top finding codes: {_join_codes(findings)}."
            )
            continue
        if key == "baseline_late_emergence_rows" and isinstance(value, list):
            origins = [str(_as_dict(item).get("origin")) for item in value[:5]]
            lines.append(
                f"- Baseline late-emergence rows: {len(value)} rows; "
                f"first origins: {', '.join(origin for origin in origins if origin)}."
            )
            continue
        if key == "baseline_ldf_findings" and isinstance(value, list):
            messages = [str(_as_dict(item).get("message")) for item in value[:2]]
            lines.append(
                f"- Baseline LDF findings: {len(value)} findings; "
                f"examples: {' | '.join(message for message in messages if message)}."
            )
            continue
        lines.append(f"- {key}: {_value(value)}")
    return lines


def _join_codes(items: list[Any]) -> str:
    codes = [str(_as_dict(item).get("code")) for item in items[:5]]
    return ", ".join(code for code in codes if code) or "none"


def _dict_bullets(values: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in values.items():
        if value in (None, "", [], {}):
            continue
        lines.append(f"- {key}: {_value(value)}")
    return lines


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def _fill_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def _value(value: Any, *, max_length: int = 500) -> str:
    if value is None or value == "":
        return "not specified"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, default=str)
    else:
        text = str(value)
    if len(text) > max_length:
        text = f"{text[:max_length].rstrip()}... [truncated]"
    if isinstance(value, (dict, list)):
        return f"`{text}`"
    return text


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
