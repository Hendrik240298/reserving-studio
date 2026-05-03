# AI README

## Scope
- The AI layer should use the existing `source.reserving.Reserving` workflow rather than inventing new reserving engines.
- Custom scenarios are created by changing supported `Reserving` configuration inputs and then running a non-persisting recalculation/preview.

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

## Analysis Basis
- `Analysis Basis` is the currently accepted conversation basis the user and AI are working from.
- For basis-aware reserving tools, the AI should use the accepted `Analysis Basis` for future analysis unless the user explicitly asks for `baseline` or `current session`.
- A recommendation does not update `Analysis Basis` by itself. It can create a pending proposal that the user accepts or rejects through the proposal UI.
- Only explicit proposal acceptance updates the accepted `Analysis Basis` for that AI chat.
- A bespoke recalculation is preview-only unless it is later promoted through the accepted-basis flow. It can return scenario outputs and populate preview state, but it must not silently become the accepted basis.
- `Analysis Basis` is not the same thing as the raw Dash GUI state unless the active session and the accepted chat basis happen to match.
- Not every tool is basis-aware. Raw data exploration, movement inspection, and cached evidence-drilldown tools can remain session-scoped or evidence-scoped when that is the correct behavior.

## ID Definitions
- `chat_id` is the AI conversation identifier. It tracks one chat interaction thread and its working memory.
- `session_id` is the live reserving workspace/session identifier. It points to the active deterministic reserving state loaded in the studio/backend.
- `basis_key` is the canonical normalized basis identifier used to execute or retrieve accepted and cached basis payloads.
- `proposal_id` is the UI/action identifier used to accept or reject a pending proposal.
- `execution_id` is the identifier for one concrete deterministic tool execution record.
- `evidence_id` is the provenance identifier for evidence surfaced by deterministic diagnostics and reviews.
- `scenario_id` is a chat-analysis scenario label/identifier. It refers to a tested reserving scenario discussed inside the AI flow, such as a baseline comparison, an iterated candidate, or a derived-drop scenario.
- `scenario_id` is not the same thing as `chat_id`.
- `scenario_id` is also not a general cross-system persistent reserving-state identifier unless a future implementation explicitly promotes it to that role.
- `candidate_id` and `scenario_label` are display-oriented labels and should not drive control flow when a `basis_key` is available.

## Control-plane state objects
- `accepted_analysis_basis` stores the accepted chat-carried basis only.
- `proposal_basis` stores one pending, accepted, rejected, or superseded recommendation proposal for Yes/No action.
- `preview_basis` stores temporary bespoke recalculation context for chat preview use.
- `execution_records` store requested inputs, effective inputs, execution status, warnings, material adjustments, evidence IDs, and result summaries.
- `basis_transition_history` stores accepted/rejected proposal transitions for the chat.

## Current Runtime State
- The AI chat is currently session-aware: its baseline comes from the real active reserving session, not from a dummy or placeholder state.
- The Reserving Studio GUI state is the live active session configuration shown in the application controls.
- The AI chat `Analysis Basis` is the accepted model setup the conversation is currently reasoning from. It can match the active GUI session, or it can differ after the user accepts a proposed tested scenario.
- Current AI bespoke recalculation is preview-only for chat use: it can return scenario outputs and preview context, but it must not silently persist changes back into the active Reserving Studio GUI session or accepted `Analysis Basis`.
- Recommendation-bearing reviews can create a proposal card below the relevant assistant message. `Yes` promotes the proposal to the accepted chat basis; `No` rejects it and leaves the accepted basis unchanged.
- In short: GUI state is the live studio session; AI accepted basis state is the current chat reasoning model; proposal and preview state are separate.

## What the AI can do
- Recalculate a reserving session with bespoke parameters using the existing `Reserving` class.
- Create distinct reserving results by changing drops, tail assumptions, BF apriori, or final method selection.
- Run diagnostics on the resulting scenario.
- Run scenario iteration to compare multiple candidate configurations against the accepted `Analysis Basis` or the active baseline session when explicitly requested.
- Run composite deterministic reviews for drop selection, tail selection, BF suitability, anomaly triage, and quarter-close.
- Create pending recommendation proposals from eligible deterministic review packets and update the accepted chat basis only after explicit Yes acceptance.
- Emit lightweight execution records that distinguish requested inputs from effective inputs and classify normalization, material adjustment, partial execution, or rejection.
- Explain findings, evidence, uncertainty, and scenario tradeoffs grounded in tool outputs.

## What the AI should not claim
- It does not create a brand-new reserving algorithm or estimator class on the fly.
- It does not replace the `Reserving` engine with a separate custom model unless the codebase is explicitly changed to support that.

## Current AI tools
- This inventory should match the function specs produced by `ai.tool_payloads.build_tool_specs()`.
- `tool_get_session_summary`: load the current session context for a segment.
- `tool_get_data_view_summary`: get a compact summary of a data view.
- `tool_get_data_view`: get the full detailed data view when needed.
- `tool_get_assumption_context_detail`: get exact selected LDFs, fitted tail LDFs, BF apriori by UWY, selected methods by UWY, and one observed a2a vector for the active session, accepted basis, cached tested scenario, or bespoke parameter basis.
- `tool_compare_data_views`: compare two data views.
- `tool_run_diagnostics_summary`: run deterministic diagnostics and return a compact summary for the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_run_drop_review`: run the composite deterministic drop review and return ranked tested drop candidates from the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_run_tail_review`: run the composite deterministic tail review and return ranked tested tail candidates from the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_run_bf_suitability_review`: run the composite BF suitability review and return UWY-level and overall suitability conclusions from the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_run_anomaly_triage`: run the composite anomaly triage review with reserve relevance and pause guidance.
- `tool_run_quarter_close_review`: run the composite quarter-close review and return the plan-test-conclude packet.
- `tool_get_quarter_close_pack`: build the structured quarter-close pack from the deterministic review result.
- `tool_iterate_diagnostics_summary`: run scenario search and compare generated scenarios against the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_run_movement_diagnostics`: run movement diagnostics on premium, incurred, outstanding, and large-loss proxy signals.
- `tool_run_ldf_consistency_diagnostics`: compare observed age-to-age factors against selected LDFs for the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_project_late_emergence_benchmark`: benchmark later emergence against older comparable underwriting years for the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_evaluate_tail_fit`: evaluate a specific tail curve, fit period, and attachment setup before recommending a tail.
- `tool_explain_reserve_change`: compare a bespoke scenario to the accepted `Analysis Basis` or an explicitly requested baseline/session basis and attribute the IBNR change.
- `tool_rank_link_ratios`: rank observed age-to-age factors using generic selection rules on the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_run_derived_drop_scenario`: build a drop list from a rule and run that scenario from the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_run_highest_a2a_drop_scenario`: run the highest-age-to-age-factor-per-development-period drop scenario from the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_get_last_derived_drop_detail`: recover the exact cached drop list, parameters, and drop-support detail for the latest derived-drop or highest-a2a scenario.
- `tool_get_results_summary`: get a compact summary of reserving results for the accepted `Analysis Basis` or explicit baseline/session basis.
- `tool_get_finding_detail`: drill into one diagnostic finding or recommendation.
- `tool_get_scenario_detail`: drill into one scenario from the latest iteration run.
- `tool_get_result_for_uwy`: get one underwriting year result row for targeted comparison.
- `tool_recalculate`: run a bespoke non-persisting recalculation with explicit parameters for AI scenario preview.

## Workflow / skill layer
- The second orchestration layer is implemented as structured workflow definitions in `ai.workflow_definitions` and should stay aligned with `AI_PLAYBOOKS.md`.
- Workflow definitions are not tools. They define how the assistant should select and sequence deterministic tools for a user intent.
- Each workflow includes `workflow_name`, `intent_class`, required capabilities, required evidence, minimum evidence count, stopping rule, basis behavior, answer contract, steps, prompt hint, and tool whitelist.
- `basis_behavior` controls whether the workflow uses the accepted `Analysis Basis`, the active session only, and whether a proposal may be created.
- `answer_contract` controls the expected response shape and proposal/narration constraints.
- Recommendation-capable workflows can create proposals when deterministic evidence supports one; observational or data workflows should not create proposals.
- Current workflows are `quarter_close_review`, `data_anomaly_triage`, `drop_review`, `derived_drop_expansion`, `movement_review`, `late_emergence_review`, `reserve_change_explanation`, `tail_selection`, `method_suitability_review`, `scenario_recommendation`, and `data_exploration`.
- Proposal-capable workflows are currently `quarter_close_review`, `drop_review`, `derived_drop_expansion`, `tail_selection`, `method_suitability_review`, and `scenario_recommendation`.
- Proposal-disallowed or observational workflows are currently `data_anomaly_triage`, `movement_review`, `late_emergence_review`, `reserve_change_explanation`, and `data_exploration`.

## Basis-aware vs session-scoped tools
- Basis-aware tools are the ones that materially depend on reserving assumptions, scenario state, or modeled results. These should follow `Analysis Basis` by default.
- Session-scoped or evidence-scoped tools include raw data views, movement diagnostics, and cached drilldowns where the correct behavior is to inspect the requested data or cached evidence rather than silently reinterpret it under a different modeled basis.

## Practical tool flow
- Start with `tool_get_session_summary` when the AI needs to orient itself to the current session.
- Use summary-style tools first: `tool_get_data_view_summary`, `tool_run_diagnostics_summary`, `tool_get_results_summary`.
- When a task-specific composite review exists, use it first rather than reconstructing the workflow from low-level tools.
- Use drill-down tools only when needed for evidence: `tool_get_data_view`, `tool_get_finding_detail`, `tool_get_scenario_detail`, `tool_get_result_for_uwy`, `tool_get_last_derived_drop_detail`.
- Use `tool_run_drop_review`, `tool_run_tail_review`, `tool_run_bf_suitability_review`, `tool_run_anomaly_triage`, or `tool_run_quarter_close_review` as the primary evidence source for those review tasks.
- Use `tool_evaluate_tail_fit` before recommending or comparing tail settings when the composite tail review does not already settle the question.
- Use `tool_iterate_diagnostics_summary`, `tool_run_derived_drop_scenario`, or `tool_run_highest_a2a_drop_scenario` when there is no more specific composite review for the user request.
- Use `tool_explain_reserve_change` or `tool_recalculate` when the user wants a bespoke scenario impact rather than a generic recommendation.
- After a tested recommendation creates a pending proposal, keep using the current accepted `Analysis Basis` until the user accepts that proposal through the proposal UI.
- After a bespoke recalculation, treat the result as preview context only unless it is promoted through the accepted-basis flow.

## Current control-layer behavior
- The deterministic layer is composite-review-first for drop review, tail review, BF suitability, anomaly triage, and quarter-close.
- Recommendation policy prefers composite review packets over generic `scenario_comparison` when both are available.
- Continuity checks can downgrade recommendations when a scenario was rejected before or conflicts with house preferences.
- Human approve/reject actions write structured continuity memory into per-segment YAML under the session-backed AI memory path.
- The assistant carries forward an accepted `Analysis Basis` for later basis-aware tool calls and exact numeric follow-ups.
- Eligible recommendations create `proposal_basis` instead of auto-promoting the recommended scenario.
- Proposal acceptance/rejection is explicit: acceptance updates `accepted_analysis_basis` for the chat and records a basis transition; rejection leaves the accepted basis unchanged.
- Newer pending proposals supersede older pending proposals for the same chat.
- Deterministic tool results include execution metadata, including requested inputs, effective inputs, execution status, warnings, and material adjustments.
- Narration is constrained by the control plane: it must not say the basis changed without acceptance, and it must not describe materially adjusted or partially executed requests as if they executed exactly.
- If the user asks for `baseline` or `current session`, the assistant should switch those basis-aware calls back to the active session setup.

## Practical interpretation
- Yes: the AI can test different Chainladder drop selections and then run diagnostics on each resulting configuration.
- Yes: the AI can test different BF apriori settings, including per-UWY factors, and then run diagnostics on those results.
- Yes: the AI can compare distinct scenarios produced from the same base triangle and reserving workflow.
- Yes: the AI can recommend a tested setup and present it as a pending proposal for explicit Yes/No acceptance.
- No: a recommendation, proposal, or bespoke recalculation does not silently update the accepted `Analysis Basis` or the live GUI session.
- No: the AI is not currently a free-form reserving-model generator; it is a scenario runner over the existing `Reserving` implementation.
