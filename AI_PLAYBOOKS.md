# AI Playbooks

## Purpose
- Use these playbooks as preferred workflows rather than inventing a new workflow on each turn.
- Pick the playbook that best matches the user intent.
- These are guides, not rigid chains. Adapt when evidence clearly requires it.

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
  4. Explain tradeoffs, uncertainty, and why the preferred scenario is better than baseline.
- Only recommend scenarios or parameter settings that were actually run/tested in the current conversation, unless you explicitly label them as untested ideas.
- Do not recommend drops or assumption changes without comparative evidence unless the evidence is already overwhelming.
- If the user explicitly asks to derive drops from a rule over observed age-to-age factors, first rank the link ratios using the requested rule, then run a derived drop scenario from that rule.
- The rule framework can combine multiple rules, for example: highest factor per development period plus all factors below 1.0.

## Reserve Change Explanation
- Use for questions about why the reserve changed, what is driving movement, or how a bespoke scenario compares to baseline.
- Preferred workflow:
  1. If the prior turn created a bespoke derived-drop scenario, first fetch the exact cached scenario detail rather than reconstructing it from a summary.
  2. Use reserve-change explanation against baseline.
  2. Use data views only to clarify component drivers.
  3. Keep the answer attribution-focused, not scenario-search focused.

## Late Emergence Review
- Use for questions about how much more could still emerge or whether current IBNR looks light/heavy.
- Preferred workflow:
  1. Use late-emergence benchmark.
  2. Compare to selected reserve / IBNR context.
  3. Mention historical range and uncertainty.

## Method Suitability Review
- Use for questions comparing CL vs BF or asking whether BF is more appropriate for newer years.
- Preferred workflow:
  1. Use diagnostics summary.
  2. Use a2a / LDF consistency.
  3. Use incurred / premium context.
  4. If the user asks for a change recommendation, run scenario iteration before final recommendation.

## Tail Selection
- Use for questions about how to set, estimate, compare, or validate the tail.
- Supported tail curve method names are `exponential`, `inverse_power`, and `weibull`.
- If the user says `power` or `power_curve`, interpret that as `inverse_power`.
- Prefer tested tail scenarios or explicit tail-fit evaluation over untested narrative recommendations.
- Before recommending a tail method or answering a tail-fit comparison question, run a tested tail evaluation for the proposed setting.
- If the question is a follow-up after a recommended scenario, reuse the tested scenario parameters first, then vary only the tail settings being compared.
- When discussing fit periods or maximum observed development, speak in months/quarters for development age and in years for AY/UWY. Do not treat a development age like `45` as 45 years.

## Data Exploration
- Use for direct requests to inspect a triangle, ratio view, or compare two quantities.
- Preferred workflow:
  1. Start with `tool_get_data_view_summary`.
  2. Use `tool_compare_data_views` for direct comparisons.
  3. Only request `tool_get_data_view` if the detailed rows are needed.

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
- Do not jump from an observational question to recommendations unless the user asks.
- Keep recommendations evidence-led.
