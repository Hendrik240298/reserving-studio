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
- Explain findings, evidence, uncertainty, and scenario tradeoffs grounded in tool outputs.

## What the AI should not claim
- It does not create a brand-new reserving algorithm or estimator class on the fly.
- It does not replace the `Reserving` engine with a separate custom model unless the codebase is explicitly changed to support that.

## Current AI tool flow
- `tool_recalculate` runs a bespoke reserving configuration for the current session.
- `tool_run_diagnostics_summary` runs diagnostics on the current session state after recalculation.
- `tool_iterate_diagnostics_summary` evaluates multiple generated scenarios and compares them.
- `tool_get_results_summary`, `tool_get_finding_detail`, `tool_get_scenario_detail`, and `tool_get_result_for_uwy` support drill-down.

## Practical interpretation
- Yes: the AI can test different Chainladder drop selections and then run diagnostics on each resulting configuration.
- Yes: the AI can test different BF apriori settings, including per-UWY factors, and then run diagnostics on those results.
- Yes: the AI can compare distinct scenarios produced from the same base triangle and reserving workflow.
- No: the AI is not currently a free-form reserving-model generator; it is a scenario runner over the existing `Reserving` implementation.
