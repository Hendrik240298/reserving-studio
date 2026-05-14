from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.deterministic_packet import build_deterministic_packet
from ai.planner import PlaybookPlanner
from ai.recommendation_policy import RecommendationPolicy
from ai.reviewer import ReviewerGate
from ai.tool_contract import normalize_tool_result


FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "phase2"


def _load_cases() -> list[dict]:
    cases: list[dict] = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            cases.append(json.load(handle))
    return cases


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["name"])
def test_phase2_benchmark_cases(case: dict) -> None:
    planner = PlaybookPlanner()
    session_context = case.get("session_context", {})
    plan = planner.plan(
        user_prompt=case["prompt"],
        session_context=session_context,
        segment_memory=case.get("segment_memory", {}),
    )

    assert plan is not None, f"Planner returned no playbook for {case['name']}"
    actual_sequence = [step.tool_name for step in plan.steps]
    evidence_packets = []
    for step in plan.steps:
        payload = case.get("tool_outputs", {}).get(step.tool_name)
        assert payload is not None, f"Missing tool output fixture for {step.tool_name}"
        evidence_packets.append(
            normalize_tool_result(
                tool_name=step.tool_name,
                args=step.args,
                result=payload,
                segment=session_context.get("segment"),
                evidence_key=step.evidence_key,
            )
        )

    review = ReviewerGate().review(plan=plan, evidence_packets=evidence_packets)
    recommendation = RecommendationPolicy().decide(
        review=review,
        evidence_packets=evidence_packets,
    )
    packet = build_deterministic_packet(
        plan=plan.to_dict(),
        review=review.to_dict(),
        recommendation=recommendation.to_dict(),
        evidence_packets=evidence_packets,
    )

    expected = case.get("expected", {})
    scorecard = {
        "playbook_correct": plan.playbook == expected.get("playbook"),
        "tool_sequence_correct": actual_sequence == expected.get("tool_sequence", []),
        "review_status_correct": review.status == expected.get("review_status"),
        "recommendation_status_correct": recommendation.status
        == expected.get("recommendation_status"),
        "continuity_used": _contains_all(
            _continuity_codes(packet),
            expected.get("continuity_contains", []),
        ),
        "policy_trace_used": _has_policy_trace_flags(
            packet.get("policy_trace", {}),
            expected.get("policy_trace_flags", []),
        ),
        "evidence_grounded": _contains_all(
            packet.get("presentation", {}).get("evidence_used", []),
            expected.get("evidence_used_contains", []),
        ),
        "review_caveats_correct": _contains_all(
            review.caveats,
            expected.get("review_caveats_contains", []),
        ),
        "review_issues_correct": _contains_all(
            review.issues,
            expected.get("review_issues_contains", []),
        ),
        "recommended_id_correct": _recommended_id_matches(
            recommendation.recommended_scenario_id,
            expected.get("recommended_id"),
        ),
    }

    failures = [name for name, passed in scorecard.items() if not passed]
    assert not failures, (
        f"{case['name']} benchmark failed: {failures} scorecard={scorecard}"
    )


def _contains_all(actual: object, expected: object) -> bool:
    expected_values = (
        [str(item) for item in expected if str(item).strip()]
        if isinstance(expected, list)
        else []
    )
    if not expected_values:
        return True
    actual_values = (
        {str(item) for item in actual if str(item).strip()}
        if isinstance(actual, list)
        else set()
    )
    return all(item in actual_values for item in expected_values)


def _continuity_codes(packet: dict) -> list[str]:
    notes = (
        packet.get("continuity_notes")
        if isinstance(packet.get("continuity_notes"), list)
        else []
    )
    return [
        str(item.get("code", "")).strip() for item in notes if isinstance(item, dict)
    ]


def _has_policy_trace_flags(policy_trace: object, flags: object) -> bool:
    expected_flags = (
        [str(item) for item in flags if str(item).strip()]
        if isinstance(flags, list)
        else []
    )
    if not expected_flags:
        return True
    if not isinstance(policy_trace, dict):
        return False
    for flag in expected_flags:
        value = policy_trace.get(flag)
        if isinstance(value, list) and value:
            continue
        if value:
            continue
        return False
    return True


def _recommended_id_matches(actual: object, expected: object) -> bool:
    if expected is None:
        return True
    return str(actual or "").strip() == str(expected or "").strip()
