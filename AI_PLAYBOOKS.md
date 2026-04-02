# AI Playbooks

## Purpose
- Use these playbooks as preferred workflows rather than inventing a new workflow on each turn.
- Pick the playbook that best matches the user intent.
- These are guides, not rigid chains. Adapt when evidence clearly requires it.
- For Phase 2 reviews, prefer one composite deterministic review tool first and use low-level tools only for drilldown or follow-up evidence.

## Movement Review
- Use for observational questions about movements, volatility, unexpected changes, current quarter, latest diagonal, or in-quarter behavior.
- Preferred workflow:
  1. Start with data-view summaries, not scenario search.
  2. For claims movement questions, treat `claims` as `incurred` unless the user explicitly says paid or outstanding.
  3. For "this quarter" or "current quarter", focus on the latest diagonal / latest valuation movement.
  4. Inspect `incremental incurred` first.
  5. Add `incurred / premium` context if it helps explain materiality.
  6. Use a2a / LDF consistency diagnostics if development distortion matters.
  7. Only bring in premium movement if the user asked about premium or it is clearly supporting context.
- When describing whether a year is early, mature, or late in development, anchor that language to the selected development pattern and tail (`LDF`/`CDF`) plus comparable prior years, not just the absolute development age label.
- Do not lead with scenario recommendations.

## Scenario Recommendation
- Use for questions asking what to drop, what to change, which scenario is best, or what configuration should be recommended.
- Preferred workflow:
  1. Run diagnostics summary.
  2. Run scenario iteration.
  3. Drill into top scenario evidence if needed.
  4. Explain tradeoffs, uncertainty, and why the preferred scenario is better than the current analysis basis unless the user explicitly asked for baseline.
- Use this playbook when there is no more specific composite review playbook for the request.
- Only recommend scenarios or parameter settings that were actually run/tested in the current conversation, unless you explicitly label them as untested ideas.
- Do not recommend drops or assumption changes without comparative evidence unless the evidence is already overwhelming.
- If the user explicitly asks to derive drops from a rule over observed age-to-age factors, first rank the link ratios using the requested rule over the current analysis basis, then run a derived drop scenario from that rule.
- The rule framework can combine multiple rules, for example: highest factor per development period plus all factors below 1.0.
- After the assistant recommends a tested scenario and the conversation moves forward on that setup, treat that tested setup as the new `Analysis Basis` for later basis-aware turns unless the user explicitly switches basis.

## Drop Review
- Use for questions asking whether any ratios should be dropped, which drop is best, or whether an existing drop recommendation is strong enough.
- Preferred workflow:
  1. Run `tool_run_drop_review`.
  2. Treat its ranked candidates, continuity notes, and policy trace as the primary evidence base.
  3. Use low-level drilldowns only if the user asks why a specific candidate is ranked the way it is.
- Comment explicitly on `rejected_before`, reserve impact, fragility, and governance penalties when they appear.

## Reserve Change Explanation
- Use for questions about why the reserve changed, what is driving movement, or how a bespoke scenario compares to the current analysis basis or an explicitly requested baseline.
- Preferred workflow:
  1. If the prior turn created a bespoke derived-drop scenario, first fetch the exact cached scenario detail rather than reconstructing it from a summary.
  2. Use reserve-change explanation against the current analysis basis unless the user explicitly asks for baseline.
  3. Use data views only to clarify component drivers.
  4. Keep the answer attribution-focused, not scenario-search focused.

## Late Emergence Review
- Use for questions about how much more could still emerge or whether current IBNR looks light/heavy.
- Preferred workflow:
  1. Use late-emergence benchmark.
  2. Compare to selected reserve / IBNR context.
  3. Mention historical range and uncertainty.

## Method Suitability Review
- Use for questions comparing CL vs BF or asking whether BF is more appropriate for newer years.
- Preferred workflow:
  1. Run `tool_run_bf_suitability_review`.
  2. Treat UWY-level suitability conclusions plus continuity notes as the primary evidence base.
  3. Use data views or diagnostics drilldown only if the user asks for supporting detail.

## Tail Selection
- Use for questions about how to set, estimate, compare, or validate the tail.
- Supported tail curve method names are `exponential`, `inverse_power`, and `weibull`.
- If the user says `power` or `power_curve`, interpret that as `inverse_power`.
- Prefer the composite tail review first: run `tool_run_tail_review` before low-level tail-fit drilldowns.
- Prefer tested tail scenarios or explicit tail-fit evaluation over untested narrative recommendations.
- Before recommending a tail method or answering a tail-fit comparison question, run a tested tail evaluation for the proposed setting if the composite review does not already answer the question.
- If the question is a follow-up after a recommended scenario, reuse the tested scenario parameters first, then vary only the tail settings being compared.
- If the tail review itself produces the new preferred setup, that tested setup becomes the new `Analysis Basis` for later basis-aware analysis unless the user explicitly switches basis.
- When discussing fit periods or maximum observed development, speak in months/quarters for development age and in years for AY/UWY. Do not treat a development age like `45` as 45 years.

## Data Anomaly Triage
- Use for questions about data quality, anomalies, impossible factors, missing diagonals, calendar distortions, or large-loss contamination.
- Preferred workflow:
  1. Run `tool_run_anomaly_triage`.
  2. Treat pause guidance as binding for the recommendation layer.
  3. Use movement or LDF drilldowns only to explain the anomaly, not to replace the triage result.

## Quarter-Close Review
- Use for quarter-close, close pack, sign-off review, or full plan-test-conclude prompts.
- Preferred workflow:
  1. Run `tool_run_quarter_close_review`.
  2. Treat the quarter-close review packet as the primary deterministic evidence base.
  3. If the user wants a review-ready artifact, run `tool_get_quarter_close_pack`.
  4. Use low-level tools only for follow-up drilldown into a specific assumption or diagnostic.
- Comment explicitly on continuity notes, recommended changes, caveats, judgment items, and sign-off questions.

## Data Exploration
- Use for direct requests to inspect a triangle, ratio view, or compare two quantities.
- Preferred workflow:
  1. Start with `tool_get_data_view_summary`.
  2. Use `tool_compare_data_views` for direct comparisons.
  3. Only request `tool_get_data_view` if the detailed rows are needed.
- Treat these as primarily observational/data workflows. Do not force a scenario-recommendation pattern into them unless the user is clearly asking about modeled assumption changes.

## Terminology Guardrails
- Before answering, distinguish whether the user is asking about:
  - observed triangle `a2a` factors
  - selected `LDF` vector
  - cumulative development / `CDF`
- When the user asks about monotonicity of the selected `LDF` vector, default to the decay interpretation `LDF_i >= LDF_{i+1}` unless they explicitly define another meaning.
- When discussing factors below 1 after threshold-based triangle rules, do not accidentally recast those as selected `LDF` values unless the user explicitly shifts to the selected vector.
- If the user specifies a notation preference such as monthly notation for development age, keep using that notation consistently in later turns.
- After a user requests monthly notation, phrase development ages as `month X` or `months X to Y`, not as quarter counts.

## Priority Rules
- Prefer the smallest workflow that answers the question.
- Use summary tools first.
- Prefer composite review tools before low-level chains when a playbook-specific review tool exists.
- Do not jump from an observational question to recommendations unless the user asks.
- Keep recommendations evidence-led.
- Preserve the distinction between raw data questions and basis-aware reserving questions. `Analysis Basis` should drive model-selection workflows, not automatically relabel every observational data request as a modeled scenario question.
