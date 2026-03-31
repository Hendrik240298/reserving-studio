from __future__ import annotations

from typing import Any


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
                    },
                    "required": ["session_id", "metric", "view"],
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
                    "properties": {"session_id": {"type": "string"}},
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
                    },
                    "required": ["session_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tool_explain_reserve_change",
                "description": "Compare a bespoke scenario to the current baseline session and attribute the IBNR change across development, tail, BF, and final selection steps.",
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
        {
            "type": "function",
            "function": {
                "name": "tool_rank_link_ratios",
                "description": "Rank observed age-to-age factors from the current baseline triangle using a generic rule such as highest or lowest, either per development period or globally.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                "description": "Build a drop list from a generic link-ratio rule over the current baseline triangle, then run that bespoke drop scenario against baseline.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
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
                "description": "Build a drop list by taking the highest observed age-to-age factor in each development period from the current baseline triangle, then run that bespoke drop scenario against baseline.",
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
                "description": "Get a compact summary of latest reserving results and key underwriting year rows.",
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
                "description": "Run a targeted recalculation with explicit parameters when you need to test a bespoke scenario not already covered by scenario search.",
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


def summarize_session_payload(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params_store")
    result_meta = payload.get("results_store_meta")
    summary = {
        "session_id": payload.get("session_id"),
        "segment": payload.get("segment"),
        "sync_version": payload.get("sync_version"),
        "params": {},
        "results_meta": {},
    }
    if isinstance(params, dict):
        summary["params"] = {
            "average": params.get("average"),
            "tail_curve": params.get("tail_curve"),
            "tail_attachment_age": params.get("tail_attachment_age"),
            "tail_projection_months": params.get("tail_projection_months"),
            "tail_fit_period_selection": params.get("tail_fit_period_selection", []),
            "drop_count": len(params.get("drop_store", []) or []),
            "bf_apriori_year_count": len(params.get("bf_apriori_by_uwy", {}) or {}),
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
    return summary


def summarize_diagnostics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    findings = payload.get("findings")
    recommendations = payload.get("recommendations")
    governance = payload.get("governance")
    metrics = payload.get("metrics")
    uncertainty = payload.get("uncertainty")
    summary: dict[str, Any] = {
        "session_id": payload.get("session_id"),
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
        "result_row_count": len(rows) if isinstance(rows, list) else 0,
        "top_rows": compact_rows,
        "last_updated": results.get("last_updated"),
    }


def summarize_recalculate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results_table_rows")
    return {
        "session_id": payload.get("session_id"),
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
        "input_adjustments": payload.get("input_adjustments", []),
    }


def summarize_data_view_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "session_id": payload.get("session_id"),
        "query": payload.get("query", {}),
        "shape": summary.get("shape", {}),
        "latest_age": summary.get("latest_age"),
        "top_latest_rows": summary.get("top_latest_rows", []),
        "latest_diagonal_total": summary.get("latest_diagonal_total"),
        "top_latest_diagonal_rows": summary.get("top_latest_diagonal_rows", []),
        "late_movement_candidates": summary.get("late_movement_candidates", []),
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
    return {
        "session_id": payload.get("session_id"),
        "rule": payload.get("rule", {}),
        "drop_count": len(payload.get("drop", []) or []),
        "selected_rows": (payload.get("selected_rows") or [])[:5],
        "baseline_score": baseline.get("score"),
        "candidate_score": candidate.get("score"),
        "score_delta": scenario.get("score_delta"),
        "scenario_id": scenario.get("scenario_id"),
        "summary": scenario.get("summary"),
    }


def summarize_tail_evaluation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": payload.get("session_id"),
        "tail_curve": payload.get("tail_curve"),
        "fit_period": payload.get("fit_period", []),
        "attachment_age": payload.get("attachment_age"),
        "projection_period": payload.get("projection_period"),
        "r2": payload.get("r2"),
        "rmse": payload.get("rmse"),
        "point_count": payload.get("point_count"),
        "input_adjustments": payload.get("input_adjustments", []),
        "top_residuals": (payload.get("residuals") or [])[:5],
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
        "uncertainty": _summarize_uncertainty(scenario.get("uncertainty")),
        "lineage": scenario.get("lineage", {}),
    }


def extract_result_row_detail(
    results_payload: dict[str, Any] | None,
    *,
    uwy: str,
) -> dict[str, Any]:
    if not isinstance(results_payload, dict):
        return {"error": "No results payload available"}
    payload = results_payload.get("results")
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


def build_memory_snapshot(
    *,
    session_summary: dict[str, Any] | None = None,
    diagnostics_summary: dict[str, Any] | None = None,
    iteration_summary: dict[str, Any] | None = None,
    results_summary: dict[str, Any] | None = None,
    data_view_summary: dict[str, Any] | None = None,
    movement_summary: dict[str, Any] | None = None,
    reserve_change_summary: dict[str, Any] | None = None,
    existing_scenario_ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scenario_ledger = list(existing_scenario_ledger or [])
    if isinstance(iteration_summary, dict):
        entries = iteration_summary.get("top_scenarios")
        baseline = iteration_summary.get("baseline")
        scenario_ledger = _merge_scenario_entries(scenario_ledger, baseline, entries)
    return {
        "session_summary": session_summary or {},
        "diagnostics_summary": diagnostics_summary or {},
        "iteration_summary": iteration_summary or {},
        "results_summary": results_summary or {},
        "data_view_summary": data_view_summary or {},
        "movement_summary": movement_summary or {},
        "reserve_change_summary": reserve_change_summary or {},
        "scenario_ledger": scenario_ledger,
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
    return {
        "scenario_id": item.get("scenario_id"),
        "score": item.get("score"),
        "summary": item.get("summary"),
        "tier": governance.get("tier"),
        "transform": lineage.get("transform"),
        "rationale_evidence_ids": lineage.get("rationale_evidence_ids", [])[:5],
    }


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
        str(item.get("scenario_id")): dict(item)
        for item in existing
        if isinstance(item, dict) and item.get("scenario_id")
    }
    if isinstance(baseline, dict) and baseline.get("scenario_id"):
        merged[str(baseline.get("scenario_id"))] = _compact_scenario(baseline)
    if isinstance(entries, list):
        for item in entries:
            compact = _compact_scenario(item)
            if compact.get("scenario_id"):
                merged[str(compact["scenario_id"])] = compact
    ordered = list(merged.values())
    ordered.sort(key=lambda item: float(item.get("score", 0.0) or 0.0))
    return ordered[:12]
