# AI README

## Scope
- The AI layer should use the existing `source.reserving.Reserving` workflow rather than inventing new reserving engines.
- Custom scenarios are created by changing supported `Reserving` configuration inputs and then recalculating the session.

## Supported reserving controls
- Development configuration via `Reserving.set_development(...)`:
  - `average`
  - `drop`
  - `drop_valuation`
- Tail configuration via `Reserving.set_tail(...)`:
  - `curve`
  - `attachment_age`
  - `extrap_periods`
  - `projection_period`
  - `fit_period`
- Bornhuetter-Ferguson configuration via `Reserving.set_bornhuetter_ferguson(...)`:
  - global `apriori`
  - per-UWY `apriori`
- Final result selection via `Reserving.reserve(...)`:
  - `final_ultimate`
  - `selected_ultimate_by_uwy`

## What the AI can do
- Recalculate a reserving session with bespoke parameters using the existing `Reserving` class.
- Create distinct reserving results by changing drops, tail assumptions, BF apriori, or final method selection.
- Run diagnostics on the resulting scenario.
- Run scenario iteration to compare multiple candidate configurations against the baseline.
- Run composite deterministic reviews for drop selection, tail selection, BF suitability, anomaly triage, and quarter-close.
- Explain findings, evidence, uncertainty, and scenario tradeoffs grounded in tool outputs.

## What the AI should not claim
- It does not create a brand-new reserving algorithm or estimator class on the fly.
- It does not replace the `Reserving` engine with a separate custom model unless the codebase is explicitly changed to support that.

## Current AI tools
- `tool_get_session_summary`: load the current session context for a segment.
- `tool_get_data_view_summary`: get a compact summary of a data view.
- `tool_get_data_view`: get the full detailed data view when needed.
- `tool_compare_data_views`: compare two data views.
- `tool_run_diagnostics_summary`: run deterministic diagnostics and return a compact summary.
- `tool_run_drop_review`: run the composite deterministic drop review and return ranked tested drop candidates.
- `tool_run_tail_review`: run the composite deterministic tail review and return ranked tested tail candidates.
- `tool_run_bf_suitability_review`: run the composite BF suitability review and return UWY-level and overall suitability conclusions.
- `tool_run_anomaly_triage`: run the composite anomaly triage review with reserve relevance and pause guidance.
- `tool_run_quarter_close_review`: run the composite quarter-close review and return the plan-test-conclude packet.
- `tool_get_quarter_close_pack`: build the structured quarter-close pack from the deterministic review result.
- `tool_iterate_diagnostics_summary`: run scenario search and compare generated scenarios.
- `tool_run_movement_diagnostics`: run movement diagnostics on premium, incurred, outstanding, and large-loss proxy signals.
- `tool_run_ldf_consistency_diagnostics`: compare observed age-to-age factors against selected LDFs.
- `tool_project_late_emergence_benchmark`: benchmark later emergence against older comparable underwriting years.
- `tool_evaluate_tail_fit`: evaluate a specific tail curve, fit period, and attachment setup before recommending a tail.
- `tool_explain_reserve_change`: compare a bespoke scenario to baseline and attribute the IBNR change.
- `tool_rank_link_ratios`: rank observed age-to-age factors using generic selection rules.
- `tool_run_derived_drop_scenario`: build a drop list from a rule and run that scenario.
- `tool_run_highest_a2a_drop_scenario`: run the highest-age-to-age-factor-per-development-period drop scenario.
- `tool_get_last_derived_drop_detail`: recover the exact cached drop list, parameters, and drop-support detail for the latest derived-drop or highest-a2a scenario.
- `tool_get_results_summary`: get a compact summary of the latest reserving results.
- `tool_get_finding_detail`: drill into one diagnostic finding or recommendation.
- `tool_get_scenario_detail`: drill into one scenario from the latest iteration run.
- `tool_get_result_for_uwy`: get one underwriting year result row for targeted comparison.
- `tool_recalculate`: run a bespoke recalculation with explicit parameters.

## Practical tool flow
- Start with `tool_get_session_summary` when the AI needs to orient itself to the current session.
- Use summary-style tools first: `tool_get_data_view_summary`, `tool_run_diagnostics_summary`, `tool_get_results_summary`.
- When a task-specific composite review exists, use it first rather than reconstructing the workflow from low-level tools.
- Use drill-down tools only when needed for evidence: `tool_get_data_view`, `tool_get_finding_detail`, `tool_get_scenario_detail`, `tool_get_result_for_uwy`, `tool_get_last_derived_drop_detail`.
- Use `tool_run_drop_review`, `tool_run_tail_review`, `tool_run_bf_suitability_review`, `tool_run_anomaly_triage`, or `tool_run_quarter_close_review` as the primary evidence source for those review tasks.
- Use `tool_evaluate_tail_fit` before recommending or comparing tail settings when the composite tail review does not already settle the question.
- Use `tool_iterate_diagnostics_summary`, `tool_run_derived_drop_scenario`, or `tool_run_highest_a2a_drop_scenario` when there is no more specific composite review for the user request.
- Use `tool_explain_reserve_change` or `tool_recalculate` when the user wants a bespoke scenario impact rather than a generic recommendation.

## Current control-layer behavior
- The deterministic layer is composite-review-first for drop review, tail review, BF suitability, anomaly triage, and quarter-close.
- Recommendation policy prefers composite review packets over generic `scenario_comparison` when both are available.
- Continuity checks can downgrade recommendations when a scenario was rejected before or conflicts with house preferences.
- Human approve/reject actions write structured continuity memory into per-segment YAML under the session-backed AI memory path.

## Practical interpretation
- Yes: the AI can test different Chainladder drop selections and then run diagnostics on each resulting configuration.
- Yes: the AI can test different BF apriori settings, including per-UWY factors, and then run diagnostics on those results.
- Yes: the AI can compare distinct scenarios produced from the same base triangle and reserving workflow.
- No: the AI is not currently a free-form reserving-model generator; it is a scenario runner over the existing `Reserving` implementation.
