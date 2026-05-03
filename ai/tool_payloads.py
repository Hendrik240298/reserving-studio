from __future__ import annotations

import hashlib
import json
from typing import Any

from ai.control_plane_types import basis_key_from_parameters, scenario_label_from_basis_payload


def build_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "tool_evaluate_tail_fit",
                "description": "Run a tested tail-fit evaluation for a specified tail configuration and scenario parameters. Use this before recommending or comparing tail settings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "average": {"type": "string"},
                        "drop": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "drop_valuation": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "tail": {
                            "type": "object",
                            "properties": {
                                "curve": {"type": "string"},
                                "attachment_age": {"type": ["integer", "null"]},
                                "projection_period": {"type": "integer"},
                                "fit_period": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                            },
                            "required": ["curve", "projection_period", "fit_period"],
                        },
                        "bf_apriori": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "final_ultimate": {
                            "type": "string",
                            "enum": ["chainladder", "bornhuetter_ferguson"],
                        },
                        "selected_ultimate_by_uwy": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string",
                                "enum": ["chainladder", "bornhuetter_ferguson"],
                            },
                        },
                    },
                    "required": [
                        "session_id",
                        "average",
                        "drop",
                        "drop_valuation",
                        "tail",
                        "bf_apriori",
                        "final_ultimate",
                        "selected_ultimate_by_uwy",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_data_view_summary",
                "description": "Get a compact summary of a data-tab style view. Allowed metric names: incurred, paid, outstanding, premium. Allowed view names: cumulative, incremental. If the user says incurred claims, use metric=incurred.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "metric": {"type": "string"},
                        "view": {"type": "string"},
                        "denominator": {"type": ["string", "null"]},
                        "denominator_view": {"type": ["string", "null"]},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id", "metric", "view"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_data_view",
                "description": "Get a detailed data-tab style view. Prefer the summary tool first and use this when the extra detail is useful. Allowed metric names: incurred, paid, outstanding, premium. Allowed view names: cumulative, incremental.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "metric": {"type": "string"},
                        "view": {"type": "string"},
                        "denominator": {"type": ["string", "null"]},
                        "denominator_view": {"type": ["string", "null"]},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id", "metric", "view"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_assumption_context_detail",
                "description": "Get exact reserving assumption detail for the active session, a recommended scenario, or a bespoke parameter basis, including selected LDFs, fitted tail LDFs, BF apriori by UWY, selected methods by UWY, and the observed a2a vector for one development period when requested. Use this for exact numeric follow-up questions instead of answering from memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "start_age": {"type": ["integer", "null"]},
                        "end_age": {"type": ["integer", "null"]},
                        "development_period": {"type": ["integer", "null"]},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_compare_data_views",
                "description": "Compare two data views such as cumulative versus incremental or incurred versus paid and return a compact summary. Allowed metric names: incurred, paid, outstanding, premium. Allowed view names: cumulative, incremental.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "left_metric": {"type": "string"},
                        "left_view": {"type": "string"},
                        "left_denominator": {"type": ["string", "null"]},
                        "left_denominator_view": {"type": ["string", "null"]},
                        "right_metric": {"type": "string"},
                        "right_view": {"type": "string"},
                        "right_denominator": {"type": ["string", "null"]},
                        "right_denominator_view": {"type": ["string", "null"]},
                        "comparison_mode": {"type": "string"},
                    },
                    "required": [
                        "session_id",
                        "left_metric",
                        "left_view",
                        "right_metric",
                        "right_view",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_session_summary",
                "description": "Get a compact summary of the current reserving session for a segment. Use this first to orient yourself.",
                "parameters": {
                    "type": "object",
                    "properties": {"segment": {"type": "string"}},
                    "required": ["segment"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_diagnostics_summary",
                "description": "Run deterministic diagnostics and return a compact summary with top findings, recommendations, governance, and uncertainty.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "diagnostic_profile": {"type": ["string", "null"]},
                        "include_recommendations": {"type": "boolean"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_drop_review",
                "description": "Run the composite deterministic drop review and return ranked tested drop candidates with continuity and policy context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "candidate_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_tail_review",
                "description": "Run the composite deterministic tail review and return ranked tested tail candidates with continuity and policy context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "candidate_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                        },
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_bf_suitability_review",
                "description": "Run the composite BF suitability review and return UWY-level and overall CL versus BF suitability conclusions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_anomaly_triage",
                "description": "Run the composite anomaly triage review and return structured anomaly classes, reserve relevance, and pause guidance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_quarter_close_review",
                "description": "Run the composite quarter-close review and return the deterministic plan-test-conclude packet for the current session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_quarter_close_pack",
                "description": "Build the structured quarter-close pack for export-style review after the quarter-close review is available.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_iterate_diagnostics_summary",
                "description": "Run scenario search and return a compact leaderboard. Use this before recommending drops, tail assumptions, or BF apriori changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "max_scenarios": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                        },
                        "include_baseline": {"type": "boolean"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_movement_diagnostics",
                "description": "Run premium, incurred, outstanding, and aggregated large-loss proxy movement diagnostics.",
                "parameters": {
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_ldf_consistency_diagnostics",
                "description": "Check whether observed age-to-age factors are inconsistent with selected LDF assumptions and estimate local reserve impact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_project_late_emergence_benchmark",
                "description": "Benchmark how much later emergence comparable older underwriting years experienced after the current age.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "uwy": {"type": ["string", "null"]},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_explain_reserve_change",
                "description": "Compare a bespoke scenario to the current conversation basis and attribute the IBNR change across development, tail, BF, and final selection steps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "basis_parameters": {"type": ["object", "null"]},
                        "average": {"type": "string"},
                        "drop": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "drop_valuation": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "tail": {
                            "type": "object",
                            "properties": {
                                "curve": {"type": "string"},
                                "attachment_age": {"type": ["integer", "null"]},
                                "projection_period": {"type": "integer"},
                                "fit_period": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                            },
                            "required": [
                                "curve",
                                "projection_period",
                                "fit_period",
                            ],
                        },
                        "bf_apriori": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "final_ultimate": {
                            "type": "string",
                            "enum": ["chainladder", "bornhuetter_ferguson"],
                        },
                        "selected_ultimate_by_uwy": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string",
                                "enum": ["chainladder", "bornhuetter_ferguson"],
                            },
                        },
                    },
                    "required": [
                        "session_id",
                        "average",
                        "drop",
                        "drop_valuation",
                        "tail",
                        "bf_apriori",
                        "final_ultimate",
                        "selected_ultimate_by_uwy",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_rank_link_ratios",
                "description": "Rank observed age-to-age factors from the current conversation basis using a generic rule such as highest or lowest, either per development period or globally.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                        "selection_mode": {"type": "string", "enum": ["max", "min"]},
                        "scope": {
                            "type": "string",
                            "enum": ["per_development_period", "global"],
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        "threshold_operator": {
                            "type": ["string", "null"],
                            "enum": ["lt", "lte", "gt", "gte", None],
                        },
                        "threshold_value": {"type": ["number", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_derived_drop_scenario",
                "description": "Build a drop list from a generic link-ratio rule over the current conversation basis, then run that bespoke drop scenario against that basis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                        "source": {"type": "string", "enum": ["link_ratios"]},
                        "selection_mode": {"type": "string", "enum": ["max", "min"]},
                        "scope": {
                            "type": "string",
                            "enum": ["per_development_period", "global"],
                        },
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        "include_existing_drops": {"type": "boolean"},
                        "threshold_operator": {
                            "type": ["string", "null"],
                            "enum": ["lt", "lte", "gt", "gte", None],
                        },
                        "threshold_value": {"type": ["number", "null"]},
                        "rules": {
                            "type": ["array", "null"],
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {
                                        "type": "string",
                                        "enum": ["link_ratios"],
                                    },
                                    "selection_mode": {
                                        "type": "string",
                                        "enum": ["max", "min"],
                                    },
                                    "scope": {
                                        "type": "string",
                                        "enum": ["per_development_period", "global"],
                                    },
                                    "limit": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 200,
                                    },
                                    "include_existing_drops": {"type": "boolean"},
                                    "threshold_operator": {
                                        "type": ["string", "null"],
                                        "enum": ["lt", "lte", "gt", "gte", None],
                                    },
                                    "threshold_value": {"type": ["number", "null"]},
                                },
                            },
                        },
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_run_highest_a2a_drop_scenario",
                "description": "Build a drop list by taking the highest observed age-to-age factor in each development period from the current conversation basis, then run that bespoke drop scenario against that basis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_last_derived_drop_detail",
                "description": "Get the full cached detail for the last derived-drop scenario or highest-a2a-drop scenario, including the exact drop list and scenario parameters. Use this before follow-up reserve impact questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_results_summary",
                "description": "Get a compact summary of reserving results and key underwriting year rows for the current conversation basis, the active baseline session, or an explicit scenario basis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "basis_type": {"type": ["string", "null"]},
                        "basis_key": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                        "parameters": {"type": ["object", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_finding_detail",
                "description": "Get targeted detail for one diagnostic finding or recommendation using a code or evidence_id after reviewing a summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "code": {"type": ["string", "null"]},
                        "evidence_id": {"type": ["string", "null"]},
                        "scenario_id": {"type": ["string", "null"]},
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_scenario_detail",
                "description": "Get targeted detail for one scenario from the last scenario search, including parameters, score, top findings, and uncertainty.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "scenario_id": {"type": "string"},
                    },
                    "required": ["session_id", "scenario_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_get_result_for_uwy",
                "description": "Get one underwriting year result row from the latest reserving results for targeted comparison.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "uwy": {"type": "string"},
                    },
                    "required": ["session_id", "uwy"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_recalculate",
                "description": "Run a non-persisting targeted recalculation with explicit parameters when you need to test a bespoke scenario not already covered by scenario search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "average": {"type": "string"},
                        "drop": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "drop_valuation": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "tail": {
                            "type": "object",
                            "properties": {
                                "curve": {"type": "string"},
                                "attachment_age": {"type": ["integer", "null"]},
                                "projection_period": {"type": "integer"},
                                "fit_period": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                            },
                            "required": [
                                "curve",
                                "projection_period",
                                "fit_period",
                            ],
                        },
                        "bf_apriori": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "final_ultimate": {
                            "type": "string",
                            "enum": ["chainladder", "bornhuetter_ferguson"],
                        },
                        "selected_ultimate_by_uwy": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string",
                                "enum": ["chainladder", "bornhuetter_ferguson"],
                            },
                        },
                    },
                    "required": [
                        "session_id",
                        "average",
                        "drop",
                        "drop_valuation",
                        "tail",
                        "bf_apriori",
                        "final_ultimate",
                        "selected_ultimate_by_uwy",
                    ],
                },
            },
        },
    ]


def compact_tool_result_for_model(*, tool_name: str, result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if tool_name == "tool_get_assumption_context_detail":
        return summarize_assumption_detail_payload(result)
    if tool_name == "tool_get_data_view":
        return summarize_detailed_data_view_payload(result)
    return result


def summarize_session_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params_store")
    result_meta = payload.get("results_store_meta")
    valuation_context = payload.get("valuation_context")
    summary = {
        "session_id": payload.get("session_id"),
        "segment": payload.get("segment"),
        "sync_version": payload.get("sync_version"),
        "params": {},
        "results_meta": {},
        "valuation_context": {},
    }
    if isinstance(params, dict):
        summary["params"] = {
            "average": params.get("average"),
            "tail_curve": params.get("tail_curve"),
            "tail_attachment_age": params.get("tail_attachment_age"),
            "tail_projection_months": params.get("tail_projection_months"),
            "tail_fit_period_selection": params.get("tail_fit_period_selection", []),
            "drop_store": params.get("drop_store", []),
            "drop_count": len(params.get("drop_store", []) or []),
            "bf_apriori_by_uwy": params.get("bf_apriori_by_uwy", {}),
            "bf_apriori_year_count": len(params.get("bf_apriori_by_uwy", {}) or {}),
            "selected_ultimate_by_uwy": params.get("selected_ultimate_by_uwy", {}),
            "selected_ultimate_overrides": len(
                params.get("selected_ultimate_by_uwy", {}) or {}
            ),
        }
    if isinstance(result_meta, dict):
        summary["results_meta"] = {
            "figure_version": result_meta.get("figure_version"),
            "sync_version": result_meta.get("sync_version"),
            "cache_key_present": bool(result_meta.get("cache_key")),
        }
    if isinstance(valuation_context, dict):
        summary["valuation_context"] = {
            "current": valuation_context.get("current", {}),
            "prior_proxy": valuation_context.get("prior_proxy", {}),
        }
    return summary


def summarize_diagnostics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    findings = payload.get("findings")
    recommendations = payload.get("recommendations")
    governance = payload.get("governance")
    metrics = payload.get("metrics")
    uncertainty = payload.get("uncertainty")
    summary: dict[str, Any] = {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "finding_count": len(findings) if isinstance(findings, list) else 0,
        "recommendation_count": (
            len(recommendations) if isinstance(recommendations, list) else 0
        ),
        "governance": {},
        "top_findings": _top_findings(findings),
        "top_recommendations": _top_recommendations(recommendations),
        "uncertainty": _summarize_uncertainty(uncertainty),
    }
    if isinstance(governance, dict):
        summary["governance"] = {
            "tier": governance.get("tier"),
            "requires_human_review": governance.get("requires_human_review"),
            "escalation_triggers": governance.get("escalation_triggers", [])[:5],
        }
    if isinstance(metrics, dict):
        summary["metrics"] = {
            "assessment_confidence": metrics.get("assessment_confidence"),
            "governance_tier": metrics.get("governance_tier"),
        }
    return summary


def summarize_iteration_payload(payload: dict[str, Any]) -> dict[str, Any]:
    iteration_metrics = payload.get("iteration_metrics")
    baseline = payload.get("baseline")
    scenarios = payload.get("scenarios")
    summary: dict[str, Any] = {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "baseline": _compact_scenario(baseline),
        "scenario_count": len(scenarios) if isinstance(scenarios, list) else 0,
        "top_scenarios": [],
        "iteration_metrics": {},
        "uncertainty": _summarize_uncertainty(payload.get("uncertainty")),
    }
    if isinstance(scenarios, list):
        ordered = [item for item in scenarios if isinstance(item, dict)]
        ordered = sorted(ordered, key=lambda item: float(item.get("score", 0.0) or 0.0))
        summary["top_scenarios"] = [_compact_scenario(item) for item in ordered[:5]]
    if isinstance(iteration_metrics, dict):
        summary["iteration_metrics"] = {
            "duration_ms": iteration_metrics.get("duration_ms"),
            "best_scenario_id": iteration_metrics.get("best_scenario_id"),
        }
    return summary


def summarize_results_payload(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, dict):
        return {
            "session_id": payload.get("session_id"),
            "analysis_basis": payload.get("analysis_basis", {}),
            "result_row_count": 0,
            "top_rows": [],
        }
    rows = results.get("results_table_rows")
    compact_rows = []
    if isinstance(rows, list):
        compact_rows = [
            {
                "uwy": item.get("uwy"),
                "ultimate_display": item.get("ultimate_display"),
                "ibnr_display": item.get("ibnr_display"),
                "selected_method": item.get("selected_method"),
            }
            for item in rows[:5]
            if isinstance(item, dict)
        ]
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "result_row_count": len(rows) if isinstance(rows, list) else 0,
        "top_rows": compact_rows,
        "latest_rows": [
            {
                "uwy": item.get("uwy"),
                "ultimate_display": item.get("ultimate_display"),
                "ibnr_display": item.get("ibnr_display"),
                "selected_method": item.get("selected_method"),
            }
            for item in (rows[-5:] if isinstance(rows, list) else [])
            if isinstance(item, dict)
        ],
        "last_updated": results.get("last_updated"),
    }


def summarize_recalculate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results_table_rows")
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "duration_ms": payload.get("duration_ms"),
        "result_row_count": len(rows) if isinstance(rows, list) else 0,
        "top_rows": [
            {
                "uwy": item.get("uwy"),
                "ultimate_display": item.get("ultimate_display"),
                "ibnr_display": item.get("ibnr_display"),
                "selected_method": item.get("selected_method"),
            }
            for item in (rows[:5] if isinstance(rows, list) else [])
            if isinstance(item, dict)
        ],
        "latest_rows": [
            {
                "uwy": item.get("uwy"),
                "ultimate_display": item.get("ultimate_display"),
                "ibnr_display": item.get("ibnr_display"),
                "selected_method": item.get("selected_method"),
            }
            for item in (rows[-5:] if isinstance(rows, list) else [])
            if isinstance(item, dict)
        ],
        "input_adjustments": payload.get("input_adjustments", []),
    }


def summarize_data_view_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": _compact_analysis_basis(payload.get("analysis_basis")),
        "query": payload.get("query", {}),
        "shape": summary.get("shape", {}),
        "latest_age": summary.get("latest_age"),
        "top_latest_rows": summary.get("top_latest_rows", []),
        "latest_diagonal_total": summary.get("latest_diagonal_total"),
        "top_latest_diagonal_rows": summary.get("top_latest_diagonal_rows", []),
        "late_movement_candidates": summary.get("late_movement_candidates", []),
    }


def summarize_detailed_data_view_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary_payload = summarize_data_view_payload(payload)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    records = data.get("records") if isinstance(data.get("records"), list) else []
    return {
        **summary_payload,
        "record_count": len(records),
        "sample_rows": [dict(item) for item in records[:5] if isinstance(item, dict)],
    }


def summarize_assumption_detail_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": payload.get("session_id"),
        "metric": payload.get("metric"),
        "analysis_basis": _compact_analysis_basis(payload.get("analysis_basis")),
        "parameter_summary": _compact_parameter_summary(payload.get("parameters")),
        "selected_ldf": _dict_list(payload.get("selected_ldf")),
        "fitted_tail_ldf": _dict_list(payload.get("fitted_tail_ldf")),
        "tail_active": payload.get("tail_active"),
        "tail_mode": payload.get("tail_mode"),
        "tail_applies_from_age": payload.get("tail_applies_from_age"),
        "observed_a2a": _dict_list(payload.get("observed_a2a")),
        "bf_apriori_by_uwy": _string_key_dict(payload.get("bf_apriori_by_uwy")),
        "selected_ultimate_by_uwy": _string_key_dict(
            payload.get("selected_ultimate_by_uwy")
        ),
    }


def summarize_data_compare_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "session_id": payload.get("session_id"),
        "comparison_mode": payload.get("comparison_mode"),
        "latest_age": summary.get("latest_age"),
        "top_latest_differences": summary.get("top_latest_differences", []),
    }


def summarize_movement_diagnostics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    findings = (
        payload.get("findings") if isinstance(payload.get("findings"), list) else []
    )
    return {
        "session_id": payload.get("session_id"),
        "finding_count": summary.get("finding_count", len(findings)),
        "top_findings": summary.get("top_findings", findings[:5]),
    }


def summarize_ldf_consistency_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    findings = (
        payload.get("findings") if isinstance(payload.get("findings"), list) else []
    )
    return {
        "session_id": payload.get("session_id"),
        "finding_count": summary.get("finding_count", len(findings)),
        "top_findings": summary.get("top_findings", findings[:5]),
    }


def summarize_late_emergence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return {
        "session_id": payload.get("session_id"),
        "row_count": summary.get("row_count", len(rows)),
        "top_rows": summary.get("top_rows", rows[:5]),
    }


def summarize_reserve_change_payload(payload: dict[str, Any]) -> dict[str, Any]:
    attribution = (
        payload.get("attribution")
        if isinstance(payload.get("attribution"), dict)
        else {}
    )
    return {
        "session_id": payload.get("session_id"),
        "baseline_total_ibnr": payload.get("baseline", {}).get("total_ibnr")
        if isinstance(payload.get("baseline"), dict)
        else None,
        "candidate_total_ibnr": payload.get("candidate", {}).get("total_ibnr")
        if isinstance(payload.get("candidate"), dict)
        else None,
        "delta_ibnr": attribution.get("baseline_vs_candidate_delta"),
        "steps": attribution.get("steps", []),
        "input_adjustments": payload.get("input_adjustments", []),
    }


def summarize_highest_a2a_drop_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = (
        payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
    )
    baseline = (
        payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    )
    candidate = (
        payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    )
    return {
        "session_id": payload.get("session_id"),
        "drop_count": len(payload.get("drop", []) or []),
        "top_factors": (payload.get("top_factors") or [])[:5],
        "baseline_score": baseline.get("score"),
        "candidate_score": candidate.get("score"),
        "score_delta": scenario.get("score_delta"),
        "scenario_id": scenario.get("scenario_id"),
        "summary": scenario.get("summary"),
    }


def summarize_link_ratio_rank_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return {
        "session_id": payload.get("session_id"),
        "selection_mode": payload.get("selection_mode"),
        "scope": payload.get("scope"),
        "row_count": summary.get("row_count", len(rows)),
        "top_rows": summary.get("top_rows", rows[:5]),
    }


def summarize_derived_drop_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = (
        payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
    )
    baseline = (
        payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    )
    candidate = (
        payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    )
    parameters = (
        scenario.get("parameters") if isinstance(scenario.get("parameters"), dict) else {}
    )
    return {
        "session_id": payload.get("session_id"),
        "rule": payload.get("rule", {}),
        "drop_count": len(payload.get("drop", []) or []),
        "selected_rows": (payload.get("selected_rows") or [])[:5],
        "baseline_score": baseline.get("score"),
        "candidate_score": candidate.get("score"),
        "score_delta": scenario.get("score_delta"),
        "basis_key": basis_key_from_parameters(parameters),
        "scenario_id": scenario.get("scenario_id"),
        "scenario_label": scenario.get("scenario_id"),
        "summary": scenario.get("summary"),
        "parameters": parameters,
    }


def summarize_tail_evaluation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": payload.get("session_id"),
        "tail_curve": payload.get("tail_curve"),
        "fit_period": payload.get("fit_period", []),
        "attachment_age": payload.get("attachment_age"),
        "projection_period": payload.get("projection_period"),
        "tail_active": payload.get("tail_active"),
        "tail_mode": payload.get("tail_mode"),
        "tail_applies_from_age": payload.get("tail_applies_from_age"),
        "r2": payload.get("r2"),
        "rmse": payload.get("rmse"),
        "point_count": payload.get("point_count"),
        "input_adjustments": payload.get("input_adjustments", []),
        "top_residuals": (payload.get("residuals") or [])[:5],
    }


def summarize_drop_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = (
        payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    )
    recommendation = (
        payload.get("recommendation")
        if isinstance(payload.get("recommendation"), dict)
        else {}
    )
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "review_type": payload.get("review_type"),
        "candidate_count": len(candidates),
        "top_candidates": [_compact_review_candidate(item) for item in candidates[:5]],
        "recommendation": _compact_review_recommendation(recommendation),
        "continuity_notes": _top_continuity_notes(payload.get("continuity_notes")),
        "policy_trace": payload.get("policy_trace", {}),
        "evidence_summary": payload.get("evidence_summary", {}),
        "run_metadata": payload.get("run_metadata", {}),
    }


def summarize_tail_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = (
        payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    )
    recommendation = (
        payload.get("recommendation")
        if isinstance(payload.get("recommendation"), dict)
        else {}
    )
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "review_type": payload.get("review_type"),
        "candidate_count": len(candidates),
        "top_candidates": [_compact_review_candidate(item) for item in candidates[:5]],
        "recommendation": _compact_review_recommendation(recommendation),
        "continuity_notes": _top_continuity_notes(payload.get("continuity_notes")),
        "policy_trace": payload.get("policy_trace", {}),
        "evidence_summary": payload.get("evidence_summary", {}),
        "run_metadata": payload.get("run_metadata", {}),
    }


def summarize_bf_suitability_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "review_type": payload.get("review_type"),
        "overall_class": payload.get("overall_class"),
        "row_count": len(rows),
        "top_rows": rows[:5],
        "apriori_guidance": payload.get("apriori_guidance", {}),
        "continuity_notes": _top_continuity_notes(payload.get("continuity_notes")),
        "policy_trace": payload.get("policy_trace", {}),
        "run_metadata": payload.get("run_metadata", {}),
    }


def summarize_anomaly_triage_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    findings = (
        payload.get("triaged_findings")
        if isinstance(payload.get("triaged_findings"), list)
        else []
    )
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "review_type": payload.get("review_type"),
        "finding_count": len(findings),
        "pause_recommendation": bool(payload.get("pause_recommendation", False)),
        "top_findings": findings[:5],
        "summary": payload.get("summary", {}),
        "run_metadata": payload.get("run_metadata", {}),
    }


def summarize_quarter_close_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    recommendation = (
        payload.get("recommendation")
        if isinstance(payload.get("recommendation"), dict)
        else {}
    )
    scenario_summary = (
        payload.get("scenario_summary")
        if isinstance(payload.get("scenario_summary"), dict)
        else {}
    )
    continuity = (
        payload.get("continuity") if isinstance(payload.get("continuity"), dict) else {}
    )
    comparison = (
        payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    )
    return {
        "session_id": payload.get("session_id"),
        "analysis_basis": payload.get("analysis_basis", {}),
        "review_type": payload.get("review_type"),
        "comparison": {
            "delta_summary": comparison.get("delta_summary", {}),
            "limitations": comparison.get("limitations", []),
            "current_valuation_date": comparison.get("current_snapshot", {}).get(
                "valuation_date"
            )
            if isinstance(comparison.get("current_snapshot"), dict)
            else None,
            "prior_valuation_date": comparison.get("prior_proxy_snapshot", {}).get(
                "valuation_date"
            )
            if isinstance(comparison.get("prior_proxy_snapshot"), dict)
            else None,
        },
        "top_ranked": scenario_summary.get("top_ranked", [])[:5],
        "recommendation": recommendation,
        "continuity": {
            "segment_id": continuity.get("segment_id"),
            "memory_schema_version": continuity.get("memory_schema_version"),
            "continuity_notes": _top_continuity_notes(
                continuity.get("continuity_notes")
            ),
            "recent_rejected_signatures": continuity.get(
                "recent_rejected_signatures", []
            ),
            "house_preferences": continuity.get("house_preferences", []),
        },
        "evidence_ids": payload.get("evidence_ids", []),
        "run_metadata": payload.get("run_metadata", {}),
    }


def summarize_quarter_close_pack_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pack = payload.get("pack") if isinstance(payload.get("pack"), dict) else {}
    metadata = (
        pack.get("pack_metadata") if isinstance(pack.get("pack_metadata"), dict) else {}
    )
    sections = pack.get("sections") if isinstance(pack.get("sections"), dict) else {}
    return {
        "session_id": payload.get("session_id"),
        "review_type": payload.get("review_type"),
        "pack_metadata": metadata,
        "recommended_changes": sections.get("recommended_changes", [])[:5],
        "signoff_questions": sections.get("signoff_questions", [])[:5],
        "policy_trace": sections.get("policy_trace", {}),
        "continuity_notes": sections.get("continuity_notes", [])[:5],
        "run_metadata": payload.get("run_metadata", {}),
    }


def extract_last_derived_drop_detail(
    derived_payload: dict[str, Any] | None,
    highest_a2a_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = (
        derived_payload
        if isinstance(derived_payload, dict) and derived_payload
        else None
    )
    if (
        payload is None
        and isinstance(highest_a2a_payload, dict)
        and highest_a2a_payload
    ):
        payload = highest_a2a_payload
    if payload is None:
        return {"message": "No cached derived drop scenario found for this session."}
    candidate = (
        payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    )
    return {
        "session_id": payload.get("session_id"),
        "drop": payload.get("drop", []),
        "selected_rows": payload.get("selected_rows") or payload.get("top_factors", []),
        "rule": payload.get("rule", {}),
        "scenario": payload.get("scenario", {}),
        "candidate_parameters": candidate.get("parameters", {}),
        "candidate_score": candidate.get("score"),
        "baseline": payload.get("baseline", {}),
        "candidate": candidate,
        "drop_details": _extract_drop_details(
            candidate,
            drop=payload.get("drop", []),
            selected_rows=payload.get("selected_rows")
            or payload.get("top_factors", []),
            rule=payload.get("rule", {}),
        ),
    }


def extract_finding_detail(
    diagnostics_payload: dict[str, Any] | None,
    iteration_payload: dict[str, Any] | None,
    *,
    code: str | None,
    evidence_id: str | None,
    scenario_id: str | None,
) -> dict[str, Any]:
    search_code = str(code or "").strip()
    search_evidence = str(evidence_id or "").strip()
    if scenario_id:
        scenario = _find_scenario(iteration_payload, str(scenario_id))
        if not isinstance(scenario, dict):
            return {"error": f"Scenario not found: {scenario_id}"}
        matches = _match_findings(
            scenario.get("findings"),
            scenario.get("recommendations"),
            search_code,
            search_evidence,
        )
        return {
            "session_id": iteration_payload.get("session_id")
            if isinstance(iteration_payload, dict)
            else None,
            "scenario_id": scenario_id,
            "matches": matches,
        }
    matches = _match_findings(
        diagnostics_payload.get("findings")
        if isinstance(diagnostics_payload, dict)
        else None,
        diagnostics_payload.get("recommendations")
        if isinstance(diagnostics_payload, dict)
        else None,
        search_code,
        search_evidence,
    )
    return {
        "session_id": diagnostics_payload.get("session_id")
        if isinstance(diagnostics_payload, dict)
        else None,
        "matches": matches,
    }


def extract_scenario_detail(
    iteration_payload: dict[str, Any] | None,
    *,
    scenario_id: str,
) -> dict[str, Any]:
    scenario = _find_scenario(iteration_payload, scenario_id)
    if not isinstance(scenario, dict):
        return {"error": f"Scenario not found: {scenario_id}"}
    return {
        "scenario_id": scenario.get("scenario_id"),
        "score": scenario.get("score"),
        "summary": scenario.get("summary"),
        "parameters": scenario.get("parameters", {}),
        "governance": {
            "tier": scenario.get("governance", {}).get("tier")
            if isinstance(scenario.get("governance"), dict)
            else None,
        },
        "top_findings": _top_findings(scenario.get("findings")),
        "top_recommendations": _top_recommendations(scenario.get("recommendations")),
        "drop_details": _extract_drop_details(
            scenario,
            drop=(
                scenario.get("parameters", {}).get("drop", [])
                if isinstance(scenario.get("parameters"), dict)
                else []
            ),
        ),
        "uncertainty": _summarize_uncertainty(scenario.get("uncertainty")),
        "lineage": scenario.get("lineage", {}),
    }


def _extract_drop_details(
    payload: dict[str, Any] | None,
    *,
    drop: object,
    selected_rows: object = None,
    rule: object = None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    details: list[dict[str, Any]] = []
    selected_map = _selected_row_map(selected_rows)
    recommendations = payload.get("recommendations")
    findings = payload.get("findings")
    rule_list = _rule_list(rule)
    for item in drop or []:
        pair = _normalize_drop_pair(item)
        if pair is None:
            continue
        origin, age = pair
        matched_rec = _match_drop_recommendation(
            recommendations, origin=origin, age=age
        )
        matched_finding = _match_drop_finding(findings, origin=origin, age=age)
        observed = selected_map.get((origin, age), {})
        detail = {
            "origin": origin,
            "development_period": age,
            "observed_a2a": observed.get("a2a"),
            "support_status": "unsupported",
            "reason_label": None,
            "message": None,
            "rationale": None,
            "evidence_id": None,
            "code": None,
        }
        if matched_rec is not None:
            evidence = (
                matched_rec.get("evidence")
                if isinstance(matched_rec.get("evidence"), dict)
                else {}
            )
            detail.update(
                {
                    "support_status": "explicit_recommendation",
                    "reason_label": _reason_label_from_code(matched_rec.get("code")),
                    "message": matched_rec.get("message"),
                    "rationale": matched_rec.get("rationale"),
                    "evidence_id": evidence.get("evidence_id"),
                    "code": matched_rec.get("code"),
                }
            )
        elif matched_finding is not None:
            evidence = (
                matched_finding.get("evidence")
                if isinstance(matched_finding.get("evidence"), dict)
                else {}
            )
            detail.update(
                {
                    "support_status": "matched_finding",
                    "reason_label": _reason_label_from_code(
                        matched_finding.get("code")
                    ),
                    "message": matched_finding.get("message"),
                    "rationale": None,
                    "evidence_id": evidence.get("evidence_id"),
                    "code": matched_finding.get("code"),
                }
            )
        else:
            rule_reason = _rule_based_reason(
                origin=origin,
                age=age,
                observed_a2a=observed.get("a2a"),
                rule_list=rule_list,
            )
            if rule_reason is not None:
                detail.update(rule_reason)
        details.append(detail)
    return details


def _selected_row_map(selected_rows: object) -> dict[tuple[str, int], dict[str, Any]]:
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    if not isinstance(selected_rows, list):
        return mapping
    for item in selected_rows:
        if not isinstance(item, dict):
            continue
        pair = _normalize_drop_pair(
            [item.get("origin"), item.get("development_period")]
        )
        if pair is None:
            continue
        mapping[pair] = dict(item)
    return mapping


def _normalize_drop_pair(item: object) -> tuple[str, int] | None:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return None
    origin = str(item[0])
    try:
        age = int(item[1])
    except (TypeError, ValueError):
        return None
    return (origin, age)


def _match_drop_recommendation(
    recommendations: object,
    *,
    origin: str,
    age: int,
) -> dict[str, Any] | None:
    if not isinstance(recommendations, list):
        return None
    target = [origin, age]
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        proposed = (
            item.get("proposed_parameters")
            if isinstance(item.get("proposed_parameters"), dict)
            else {}
        )
        drops = proposed.get("drop") if isinstance(proposed, dict) else None
        if not isinstance(drops, list):
            continue
        if any(_normalize_drop_pair(candidate) == (origin, age) for candidate in drops):
            return item
    return None


def _match_drop_finding(
    findings: object,
    *,
    origin: str,
    age: int,
) -> dict[str, Any] | None:
    if not isinstance(findings, list):
        return None
    suffix = f"_{origin}_{age}"
    for item in findings:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", ""))
        if code.endswith(suffix):
            return item
    return None


def _rule_list(rule: object) -> list[dict[str, Any]]:
    if not isinstance(rule, dict):
        return []
    if isinstance(rule.get("rules"), list):
        return [item for item in rule.get("rules", []) if isinstance(item, dict)]
    if isinstance(rule.get("primary"), dict):
        return [rule.get("primary")]
    return [rule]


def _rule_based_reason(
    *,
    origin: str,
    age: int,
    observed_a2a: object,
    rule_list: list[dict[str, Any]],
) -> dict[str, Any] | None:
    a2a = None
    try:
        if observed_a2a is not None:
            a2a = float(observed_a2a)
    except (TypeError, ValueError):
        a2a = None
    for rule in rule_list:
        operator = str(rule.get("threshold_operator", "") or "").strip().lower()
        threshold = rule.get("threshold_value")
        try:
            threshold_value = float(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            threshold_value = None
        if (
            operator == "lt"
            and threshold_value == 1.0
            and a2a is not None
            and a2a < 1.0
        ):
            return {
                "support_status": "explicit_rule",
                "reason_label": "rule_threshold_lt_1.0",
                "message": "Selected by an explicit rule targeting observed factors below 1.0.",
                "rationale": "This reason comes from the configured selection rule, not from an inferred narrative label.",
                "evidence_id": None,
                "code": None,
            }
    return None


def _reason_label_from_code(code: object) -> str | None:
    text = str(code or "")
    if text.startswith("RECOMMEND_DROP_"):
        return "drop_recommendation"
    if text.startswith("LINK_RATIO_OUTLIER_"):
        return "link_ratio_outlier"
    if text.startswith("LARGE_LOSS_PROXY_"):
        return "large_loss_proxy"
    return None


def extract_result_row_detail(
    results_payload: dict[str, Any] | None,
    *,
    uwy: str,
) -> dict[str, Any]:
    if not isinstance(results_payload, dict):
        return {"error": "No results payload available"}
    payload = results_payload.get("results")
    if not isinstance(payload, dict):
        payload = (
            results_payload
            if isinstance(results_payload.get("results_table_rows"), list)
            else None
        )
    if not isinstance(payload, dict):
        return {"error": "No results payload available"}
    rows = payload.get("results_table_rows")
    if not isinstance(rows, list):
        return {"error": "No results rows available"}
    target = str(uwy).strip()
    for item in rows:
        if not isinstance(item, dict):
            continue
        if str(item.get("uwy", "")).strip() == target:
            return dict(item)
    return {"error": f"Underwriting year not found: {uwy}"}


def build_analysis_basis(
    *,
    session_id: str | None,
    basis_type: str,
    parameters: dict[str, Any] | None,
    scenario_id: str | None = None,
    candidate_id: str | None = None,
    source_tool: str | None = None,
    source_review_type: str | None = None,
    is_active_session: bool | None = None,
) -> dict[str, Any]:
    normalized_parameters = _normalize_basis_parameters(parameters)
    signature = _scenario_signature(normalized_parameters)
    return {
        "basis_key": basis_key_from_parameters(normalized_parameters),
        "basis_type": str(basis_type or "baseline"),
        "scenario_label": scenario_label_from_basis_payload(
            {
                "basis_type": basis_type,
                "scenario_id": scenario_id,
                "candidate_id": candidate_id,
            }
        ),
        "session_id": str(session_id or "").strip(),
        "scenario_id": str(scenario_id or "").strip() or None,
        "candidate_id": str(candidate_id or "").strip() or None,
        "scenario_signature": signature,
        "source_tool": str(source_tool or "").strip() or None,
        "source_review_type": str(source_review_type or "").strip() or None,
        "is_active_session": bool(is_active_session),
        "parameters": normalized_parameters,
    }


def build_baseline_analysis_basis(
    session_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(session_summary, dict):
        return {}
    params = (
        session_summary.get("params")
        if isinstance(session_summary.get("params"), dict)
        else {}
    )
    parameters = {
        "average": params.get("average", "volume"),
        "drop": params.get("drop_store", []),
        "drop_valuation": [],
        "tail": {
            "curve": params.get("tail_curve", "weibull"),
            "attachment_age": params.get("tail_attachment_age"),
            "projection_period": params.get("tail_projection_months", 0),
            "fit_period": params.get("tail_fit_period_selection", []),
        },
        "bf_apriori": params.get("bf_apriori_by_uwy", {}),
        "final_ultimate": "chainladder",
        "selected_ultimate_by_uwy": params.get("selected_ultimate_by_uwy", {}),
    }
    return build_analysis_basis(
        session_id=str(session_summary.get("session_id", "")).strip() or None,
        basis_type="baseline",
        scenario_id="baseline",
        source_tool="tool_get_session_summary",
        is_active_session=True,
        parameters=parameters,
    )


def merge_scenario_basis_cache(
    *,
    session_summary: dict[str, Any] | None,
    iteration_summary: dict[str, Any] | None,
    review_summary: dict[str, Any] | None,
    existing_cache: dict[str, Any] | None,
) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    if isinstance(existing_cache, dict):
        for value in existing_cache.values():
            if not isinstance(value, dict):
                continue
            normalized = build_analysis_basis(
                session_id=value.get("session_id"),
                basis_type=str(value.get("basis_type") or "baseline"),
                parameters=value.get("parameters"),
                scenario_id=value.get("scenario_id"),
                candidate_id=value.get("candidate_id"),
                source_tool=value.get("source_tool"),
                source_review_type=value.get("source_review_type"),
                is_active_session=bool(value.get("is_active_session")),
            )
            basis_key = str(normalized.get("basis_key") or "").strip()
            if basis_key:
                cache[basis_key] = normalized

    baseline_basis = build_baseline_analysis_basis(session_summary)
    if baseline_basis:
        baseline_key = str(baseline_basis.get("basis_key") or "").strip()
        if baseline_key:
            cache[baseline_key] = baseline_basis

    iteration = iteration_summary if isinstance(iteration_summary, dict) else {}
    session_id = (
        str(
            iteration.get("session_id") or baseline_basis.get("session_id") or ""
        ).strip()
        or None
    )
    baseline = iteration.get("baseline")
    if isinstance(baseline, dict):
        _store_basis_candidate(
            cache,
            session_id=session_id,
            item=baseline,
            basis_type="scenario",
            source_tool="tool_iterate_diagnostics_summary",
        )
    scenarios = iteration.get("top_scenarios")
    if isinstance(scenarios, list):
        for item in scenarios:
            _store_basis_candidate(
                cache,
                session_id=session_id,
                item=item,
                basis_type="scenario",
                source_tool="tool_iterate_diagnostics_summary",
            )

    review = review_summary if isinstance(review_summary, dict) else {}
    review_type = str(review.get("review_type", "")).strip() or None
    session_id = str(review.get("session_id") or session_id or "").strip() or None
    for key in ("top_candidates", "top_ranked"):
        items = review.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            _store_basis_candidate(
                cache,
                session_id=session_id,
                item=item,
                basis_type="review_candidate",
                source_tool=_review_source_tool(review_type),
                source_review_type=review_type,
            )
    recommendation = review.get("recommendation")
    if isinstance(recommendation, dict):
        recommended_changes = recommendation.get("recommended_changes")
        if isinstance(recommended_changes, list):
            for item in recommended_changes:
                _store_basis_candidate(
                    cache,
                    session_id=session_id,
                    item=item,
                    basis_type="review_candidate",
                    source_tool=_review_source_tool(review_type),
                    source_review_type=review_type,
                )
    return cache


def build_memory_snapshot(
    *,
    session_summary: dict[str, Any] | None = None,
    diagnostics_summary: dict[str, Any] | None = None,
    iteration_summary: dict[str, Any] | None = None,
    results_summary: dict[str, Any] | None = None,
    data_view_summary: dict[str, Any] | None = None,
    movement_summary: dict[str, Any] | None = None,
    reserve_change_summary: dict[str, Any] | None = None,
    review_summary: dict[str, Any] | None = None,
    existing_scenario_ledger: list[dict[str, Any]] | None = None,
    existing_accepted_analysis_basis: dict[str, Any] | None = None,
    existing_scenario_basis_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario_ledger = list(existing_scenario_ledger or [])
    scenario_basis_cache = merge_scenario_basis_cache(
        session_summary=session_summary,
        iteration_summary=iteration_summary,
        review_summary=review_summary,
        existing_cache=existing_scenario_basis_cache,
    )
    accepted_analysis_basis = dict(existing_accepted_analysis_basis or {})
    if isinstance(iteration_summary, dict):
        entries = iteration_summary.get("top_scenarios")
        baseline = iteration_summary.get("baseline")
        scenario_ledger = _merge_scenario_entries(scenario_ledger, baseline, entries)
    if isinstance(review_summary, dict):
        scenario_ledger = _merge_review_scenario_entries(
            scenario_ledger, review_summary
        )
    return {
        "session_summary": session_summary or {},
        "diagnostics_summary": diagnostics_summary or {},
        "iteration_summary": iteration_summary or {},
        "results_summary": results_summary or {},
        "data_view_summary": data_view_summary or {},
        "movement_summary": movement_summary or {},
        "reserve_change_summary": reserve_change_summary or {},
        "review_summary": review_summary or {},
        "scenario_ledger": scenario_ledger,
        "accepted_analysis_basis": accepted_analysis_basis,
        "scenario_basis_cache": scenario_basis_cache,
    }


def render_memory_hint(memory: dict[str, Any] | None) -> str:
    if not isinstance(memory, dict):
        return ""
    parts: list[str] = []
    session_summary = memory.get("session_summary")
    if isinstance(session_summary, dict) and session_summary:
        params = session_summary.get("params", {})
        parts.append(
            "Session memory: "
            f"segment={session_summary.get('segment')}, average={params.get('average')}, "
            f"tail_curve={params.get('tail_curve')}, drop_count={params.get('drop_count')}"
        )
    accepted_analysis_basis = memory.get("accepted_analysis_basis")
    if not isinstance(accepted_analysis_basis, dict) or not accepted_analysis_basis:
        accepted_analysis_basis = memory.get("analysis_basis")
    if isinstance(accepted_analysis_basis, dict) and accepted_analysis_basis:
        parts.append(
            "Current accepted analysis basis: "
            f"type={accepted_analysis_basis.get('basis_type')}, "
            f"scenario_id={accepted_analysis_basis.get('scenario_id')}, "
            f"active_session={accepted_analysis_basis.get('is_active_session')}"
        )
    diagnostics = memory.get("diagnostics_summary")
    if isinstance(diagnostics, dict) and diagnostics:
        gov = diagnostics.get("governance", {})
        parts.append(
            "Latest diagnostics memory: "
            f"findings={diagnostics.get('finding_count')}, recommendations={diagnostics.get('recommendation_count')}, tier={gov.get('tier')}"
        )
    iteration = memory.get("iteration_summary")
    if isinstance(iteration, dict) and iteration:
        metrics = iteration.get("iteration_metrics", {})
        parts.append(
            "Latest scenario-search memory: "
            f"scenario_count={iteration.get('scenario_count')}, best_scenario={metrics.get('best_scenario_id')}"
        )
    data_view = memory.get("data_view_summary")
    if isinstance(data_view, dict) and data_view:
        query = data_view.get("query", {})
        parts.append(
            "Latest data-view memory: "
            f"metric={query.get('metric')}, view={query.get('view')}, latest_age={data_view.get('latest_age')}"
        )
    movement = memory.get("movement_summary")
    if isinstance(movement, dict) and movement:
        parts.append(
            "Latest movement diagnostics memory: "
            f"findings={movement.get('finding_count')}"
        )
    reserve_change = memory.get("reserve_change_summary")
    if isinstance(reserve_change, dict) and reserve_change:
        parts.append(
            "Latest reserve-change memory: "
            f"delta_ibnr={reserve_change.get('delta_ibnr')}"
        )
    review_summary = memory.get("review_summary")
    if isinstance(review_summary, dict) and review_summary:
        recommendation = (
            review_summary.get("recommendation")
            if isinstance(review_summary.get("recommendation"), dict)
            else {}
        )
        parts.append(
            "Latest composite review memory: "
            f"type={review_summary.get('review_type')}, recommendation={recommendation.get('status') or recommendation.get('recommendation_class')}"
        )
    ledger = memory.get("scenario_ledger")
    if isinstance(ledger, list) and ledger:
        top = ledger[:4]
        parts.append(
            "Scenario ledger: "
            + "; ".join(
                f"{item.get('scenario_id')} score={item.get('score')} transform={item.get('transform')}"
                for item in top
                if isinstance(item, dict)
            )
        )
    return "\n".join(part for part in parts if part)


def _store_basis_candidate(
    cache: dict[str, Any],
    *,
    session_id: str | None,
    item: object,
    basis_type: str,
    source_tool: str | None,
    source_review_type: str | None = None,
) -> None:
    if not isinstance(item, dict):
        return
    scenario_id = str(item.get("scenario_id") or item.get("candidate_id") or "").strip()
    parameters = (
        item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
    )
    if not scenario_id or not parameters:
        return
    basis = build_analysis_basis(
        session_id=session_id,
        basis_type=basis_type,
        scenario_id=scenario_id,
        candidate_id=str(item.get("candidate_id") or "").strip() or None,
        source_tool=source_tool,
        source_review_type=source_review_type,
        is_active_session=scenario_id == "baseline",
        parameters=parameters,
    )
    basis_key = str(basis.get("basis_key") or "").strip()
    if basis_key:
        cache[basis_key] = basis


def _review_source_tool(review_type: str | None) -> str | None:
    mapping = {
        "drop_review": "tool_run_drop_review",
        "tail_review": "tool_run_tail_review",
        "bf_suitability_review": "tool_run_bf_suitability_review",
        "quarter_close_review": "tool_run_quarter_close_review",
        "anomaly_triage": "tool_run_anomaly_triage",
    }
    return mapping.get(str(review_type or "").strip())


def _normalize_basis_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parameters, dict) or not parameters:
        return {}
    try:
        return json.loads(json.dumps(parameters, sort_keys=True))
    except (TypeError, ValueError):
        return dict(parameters)


def _scenario_signature(parameters: dict[str, Any] | None) -> str | None:
    normalized = _normalize_basis_parameters(parameters)
    if not normalized:
        return None
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _top_findings(items: object) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return output
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        evidence = (
            item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        )
        output.append(
            {
                "code": item.get("code"),
                "severity": item.get("severity"),
                "message": item.get("message"),
                "evidence_id": evidence.get("evidence_id"),
                "metric_id": evidence.get("metric_id"),
                "value": evidence.get("value"),
                "threshold": evidence.get("threshold"),
                "basis": evidence.get("basis"),
                "plain_explanation": _explain_metric(evidence),
            }
        )
    return output


def _top_recommendations(items: object) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return output
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        evidence = (
            item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        )
        output.append(
            {
                "code": item.get("code"),
                "priority": item.get("priority"),
                "message": item.get("message"),
                "proposed_parameters": item.get("proposed_parameters", {}),
                "evidence_id": evidence.get("evidence_id"),
                "metric_id": evidence.get("metric_id"),
                "value": evidence.get("value"),
                "threshold": evidence.get("threshold"),
                "basis": evidence.get("basis"),
                "plain_explanation": _explain_metric(evidence),
            }
        )
    return output


def _summarize_uncertainty(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    baseline = (
        payload.get("baseline")
        if isinstance(payload.get("baseline"), dict)
        else payload
    )
    bootstrap = (
        payload.get("bootstrap") if isinstance(payload.get("bootstrap"), dict) else {}
    )
    tail_model = (
        payload.get("tail_model") if isinstance(payload.get("tail_model"), dict) else {}
    )
    return {
        "process_cv": baseline.get("total_process_cv")
        if isinstance(baseline, dict)
        else None,
        "bootstrap_p50": bootstrap.get("p50"),
        "bootstrap_p90": bootstrap.get("p90"),
        "tail_instability": tail_model.get("instability_flag")
        if isinstance(tail_model, dict)
        else payload.get("instability_flag"),
    }


def _compact_scenario(item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    governance = (
        item.get("governance") if isinstance(item.get("governance"), dict) else {}
    )
    lineage = item.get("lineage") if isinstance(item.get("lineage"), dict) else {}
    parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
    return {
        "basis_key": basis_key_from_parameters(parameters),
        "scenario_id": item.get("scenario_id"),
        "scenario_label": item.get("scenario_id"),
        "score": item.get("score"),
        "summary": item.get("summary"),
        "tier": governance.get("tier"),
        "transform": lineage.get("transform"),
        "rationale_evidence_ids": lineage.get("rationale_evidence_ids", [])[:5],
        "parameters": parameters,
    }


def _compact_review_candidate(item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
    return {
        "basis_key": basis_key_from_parameters(parameters),
        "candidate_id": item.get("candidate_id"),
        "scenario_id": item.get("scenario_id"),
        "summary": item.get("summary"),
        "score": item.get("score"),
        "recommendation_class": item.get("recommendation_class"),
        "rank": item.get("rank"),
        "score_breakdown": item.get("score_breakdown", {}),
        "policy_trace": item.get("policy_trace", {}),
        "continuity_notes": _top_continuity_notes(item.get("continuity_notes")),
        "parameters": parameters,
    }


def _compact_review_recommendation(item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "recommendation_class": item.get("recommendation_class"),
        "candidate_id": item.get("candidate_id"),
        "basis_key": item.get("basis_key"),
        "scenario_id": item.get("scenario_id"),
        "summary": item.get("summary"),
        "caveats": item.get("caveats", []),
        "alternatives": item.get("alternatives", []),
        "alternative_basis_keys": item.get("alternative_basis_keys", []),
        "alternative_scenario_ids": item.get("alternative_scenario_ids", []),
    }


def _top_continuity_notes(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items[:5] if isinstance(item, dict)]


def _compact_analysis_basis(basis: object) -> dict[str, Any]:
    if not isinstance(basis, dict):
        return {}
    return {
        "basis_key": basis.get("basis_key"),
        "basis_type": basis.get("basis_type"),
        "session_id": basis.get("session_id"),
        "scenario_label": basis.get("scenario_label"),
        "scenario_id": basis.get("scenario_id"),
        "candidate_id": basis.get("candidate_id"),
        "scenario_signature": basis.get("scenario_signature"),
        "source_tool": basis.get("source_tool"),
        "source_review_type": basis.get("source_review_type"),
        "is_active_session": basis.get("is_active_session"),
        "parameter_summary": _compact_parameter_summary(basis.get("parameters")),
    }


def _compact_parameter_summary(parameters: object) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {}
    drop = parameters.get("drop") if isinstance(parameters.get("drop"), list) else []
    drop_valuation = (
        parameters.get("drop_valuation")
        if isinstance(parameters.get("drop_valuation"), list)
        else []
    )
    tail = parameters.get("tail") if isinstance(parameters.get("tail"), dict) else {}
    bf_apriori = (
        parameters.get("bf_apriori")
        if isinstance(parameters.get("bf_apriori"), dict)
        else {}
    )
    selected_methods = (
        parameters.get("selected_ultimate_by_uwy")
        if isinstance(parameters.get("selected_ultimate_by_uwy"), dict)
        else {}
    )
    return {
        "average": parameters.get("average"),
        "drop_count": len(drop),
        "drop_preview": [list(item) for item in drop[:8] if isinstance(item, list)],
        "drop_valuation_count": len(drop_valuation),
        "tail": {
            "curve": tail.get("curve"),
            "attachment_age": tail.get("attachment_age"),
            "projection_period": tail.get("projection_period"),
            "fit_period": tail.get("fit_period", []),
        },
        "bf_apriori_count": len(bf_apriori),
        "final_ultimate": parameters.get("final_ultimate"),
        "selected_ultimate_by_uwy_count": len(selected_methods),
    }


def _dict_list(items: object) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]


def _string_key_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _match_findings(
    findings: object,
    recommendations: object,
    code: str,
    evidence_id: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for group_name, items in (
        ("finding", findings),
        ("recommendation", recommendations),
    ):
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            evidence = (
                item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            )
            item_code = str(item.get("code", ""))
            item_evidence_id = str(evidence.get("evidence_id", ""))
            if code and code != item_code:
                continue
            if evidence_id and evidence_id != item_evidence_id:
                continue
            matches.append(
                {
                    "kind": group_name,
                    "code": item.get("code"),
                    "severity": item.get("severity") or item.get("priority"),
                    "message": item.get("message"),
                    "rationale": item.get("rationale"),
                    "suggested_actions": item.get("suggested_actions", []),
                    "proposed_parameters": item.get("proposed_parameters", {}),
                    "evidence": evidence,
                    "plain_explanation": _explain_metric(evidence),
                }
            )
    return matches[:5]


def _explain_metric(evidence: object) -> str:
    if not isinstance(evidence, dict):
        return ""
    metric_id = str(evidence.get("metric_id", "")).strip().lower()
    basis = str(evidence.get("basis", "")).strip()
    threshold = evidence.get("threshold")
    if "z_score" in metric_id or metric_id.endswith("zscore"):
        message = "A z-score shows how far the observation sits from the typical range; larger absolute values mean the year is more unusual."
        if threshold is not None:
            message += f" Threshold used here: {threshold}."
        return message
    if "loss_ratio" in metric_id:
        return "This compares ultimate loss to premium. Large deviations from peer years can indicate unusual emergence or pricing/mix shifts."
    if "divergence" in metric_id:
        return "This measures how far Chainladder and Bornhuetter-Ferguson estimates differ for the same year."
    if basis:
        return basis
    if threshold is not None:
        return f"Threshold used in this diagnostic: {threshold}."
    return ""


def _find_scenario(
    iteration_payload: dict[str, Any] | None, scenario_id: str
) -> dict[str, Any] | None:
    if not isinstance(iteration_payload, dict):
        return None
    baseline = iteration_payload.get("baseline")
    if isinstance(baseline, dict) and baseline.get("scenario_id") == scenario_id:
        return baseline
    scenarios = iteration_payload.get("scenarios")
    if not isinstance(scenarios, list):
        return None
    for item in scenarios:
        if isinstance(item, dict) and item.get("scenario_id") == scenario_id:
            return item
    return None


def _merge_scenario_entries(
    existing: list[dict[str, Any]],
    baseline: object,
    entries: object,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        str(item.get("basis_key") or item.get("scenario_id")): dict(item)
        for item in existing
        if isinstance(item, dict)
        and str(item.get("basis_key") or item.get("scenario_id") or "").strip()
    }
    if isinstance(baseline, dict):
        compact_baseline = _compact_scenario(baseline)
        baseline_key = str(
            compact_baseline.get("basis_key") or compact_baseline.get("scenario_id") or ""
        ).strip()
        if baseline_key:
            merged[baseline_key] = compact_baseline
    if isinstance(entries, list):
        for item in entries:
            compact = _compact_scenario(item)
            compact_key = str(
                compact.get("basis_key") or compact.get("scenario_id") or ""
            ).strip()
            if compact_key:
                merged[compact_key] = compact
    ordered = list(merged.values())
    ordered.sort(key=lambda item: float(item.get("score", 0.0) or 0.0))
    return ordered[:12]


def _merge_review_scenario_entries(
    existing: list[dict[str, Any]],
    review_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        str(item.get("basis_key") or item.get("scenario_key") or item.get("scenario_id")): dict(item)
        for item in existing
        if isinstance(item, dict)
        and str(
            item.get("basis_key") or item.get("scenario_key") or item.get("scenario_id") or ""
        ).strip()
    }
    for candidate in _review_candidates_for_ledger(review_summary):
        scenario_key = str(
            candidate.get("basis_key") or candidate.get("scenario_key") or candidate.get("scenario_id") or ""
        ).strip()
        if not scenario_key:
            continue
        merged[scenario_key] = candidate
    ordered = list(merged.values())
    ordered.sort(
        key=lambda item: float(item.get("score", 0.0) or 0.0),
        reverse=True,
    )
    return ordered[:12]


def _review_candidates_for_ledger(
    review_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    review_type = str(review_summary.get("review_type", "")).strip()
    candidates: list[dict[str, Any]] = []
    top_candidates = review_summary.get("top_candidates")
    if isinstance(top_candidates, list):
        for item in top_candidates[:5]:
            compact = _compact_review_candidate_for_ledger(item, transform=review_type)
            if compact:
                candidates.append(compact)
    top_ranked = review_summary.get("top_ranked")
    if isinstance(top_ranked, list):
        for item in top_ranked[:5]:
            compact = _compact_review_candidate_for_ledger(item, transform=review_type)
            if compact:
                candidates.append(compact)
    recommendation = (
        review_summary.get("recommendation")
        if isinstance(review_summary.get("recommendation"), dict)
        else {}
    )
    recommended_changes = recommendation.get("recommended_changes")
    if isinstance(recommended_changes, list):
        for item in recommended_changes[:5]:
            compact = _compact_review_candidate_for_ledger(item, transform=review_type)
            if compact:
                candidates.append(compact)
    deduped: dict[str, dict[str, Any]] = {}
    for item in candidates:
        scenario_key = str(
            item.get("basis_key") or item.get("scenario_key") or item.get("scenario_id") or ""
        ).strip()
        if not scenario_key:
            continue
        deduped[scenario_key] = item
    return list(deduped.values())


def _compact_review_candidate_for_ledger(
    item: object,
    *,
    transform: str,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    candidate_id = str(
        item.get("candidate_id") or item.get("scenario_id") or ""
    ).strip()
    if not candidate_id:
        return {}
    policy_trace = (
        item.get("policy_trace") if isinstance(item.get("policy_trace"), dict) else {}
    )
    recommendation_class = str(item.get("recommendation_class", "")).strip()
    continuity_notes = (
        item.get("continuity_notes")
        if isinstance(item.get("continuity_notes"), list)
        else []
    )
    continuity_text = ", ".join(
        str(note.get("code", "")).strip()
        for note in continuity_notes
        if isinstance(note, dict) and str(note.get("code", "")).strip()
    )
    summary = str(item.get("summary", "")).strip()
    if recommendation_class:
        summary = f"{summary} [{recommendation_class}]".strip()
    if continuity_text:
        summary = f"{summary} ({continuity_text})".strip()
    tier = None
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    if metrics:
        tier = metrics.get("governance_tier")
    if tier is None:
        tier = policy_trace.get("governance_tier")
    parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
    return {
        "scenario_id": candidate_id,
        "basis_key": basis_key_from_parameters(parameters),
        "scenario_key": item.get("scenario_id"),
        "score": item.get("score"),
        "tier": tier,
        "transform": transform or "composite_review",
        "summary": summary,
    }
