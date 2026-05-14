# AI Examples

## Example: Claims Movement This Quarter
User: Are there unusual and unexpected claims movements this quarter?

Preferred behavior:
- Treat `claims` as `incurred`.
- Treat `this quarter` as the latest diagonal / most recent valuation period.
- Start with:
  - `tool_get_data_view_summary` for `metric=incurred`, `view=incremental`
  - `tool_get_data_view_summary` for `metric=incurred`, `view=cumulative`, `denominator=premium`
  - `tool_run_ldf_consistency_diagnostics`
- Answer the movement question directly.
- Describe development stage using the selected development/tail pattern (`LDF`/`CDF`), not absolute age labels alone.
- Do not lead with premium movement.
- Do not lead with drop recommendations.

## Example: Premium Movement In Older Years
User: Do we have unusual premium movement in older years?

Preferred behavior:
- Start with premium incremental summary.
- Focus on mature years with non-trivial latest-diagonal movement.
- Explain why late premium movement is unusual.
- No scenario search unless the user asks for impact or recommendations.

## Example: What Should We Drop?
User: Which link ratios should we consider dropping?

Preferred behavior:
- Run diagnostics summary.
- Run scenario iteration.
- Explain the best scenario vs the current analysis basis.
- Mention tradeoffs and uncertainty.
- If the assistant recommends a tested drop scenario and the conversation continues on that setup, that tested scenario becomes the new `Analysis Basis` for later basis-aware analysis.

## Example: Drop Highest A2A In Each Development Period
User: How would the analysis perform when I drop the highest age-to-age factor in each development period?

Preferred behavior:
- Do not approximate this with generic diagnostics recommendations.
- Use the generic link-ratio ranking and derived-drop framework.
- Rank observed a2a factors by `selection_mode=max` and `scope=per_development_period` over the current analysis basis.
- Build the drop list from the current analysis basis by taking the largest observed a2a factor in each development period column.
- Run that bespoke scenario against the current analysis basis and summarize the score change, major tradeoffs, and selected drop list.

## Example: Combine Rule-Based Drops
User: What if next to the highest I also remove all factors below 1?

Preferred behavior:
- Reuse the generic link-ratio framework.
- Combine rules rather than inventing a bespoke workflow.
- Example rule set:
  - `selection_mode=max`, `scope=per_development_period`
  - `selection_mode=min`, `scope=global`, `threshold_operator=lt`, `threshold_value=1.0`
- Run a derived drop scenario using both rules together and compare against the current analysis basis.
- In this context, "factors below 1" refers to observed triangle `a2a` factors below 1, not the selected `LDF` vector.

## Example: Reserve Impact After Derived Scenario
User: What is the impact on the reserves?

Preferred behavior:
- If the previous turn ran a derived-drop scenario, do not reconstruct the scenario from the summary.
- First use `tool_get_last_derived_drop_detail` to recover the exact drop list and candidate parameters.
- Then run reserve-change explanation against the current analysis basis using those exact parameters unless the user explicitly asks for baseline.

## Example: Monotone LDF Clarification
User: Is the LDF development in this setting mostly monotone?

Preferred behavior:
- Distinguish selected `LDF` vector from observed triangle `a2a` factors.
- If the user does not define monotone explicitly, interpret monotone for the selected `LDF` vector as monotone decay: `LDF_i >= LDF_{i+1}`.
- Do not answer that the selected `LDF` vector is monotone merely because all selected factors are `>= 1.0`.
- If useful, mention separately:
  - whether the selected `LDF` vector is non-increasing
  - whether all selected `LDF` values are at least 1

## Example: Compare Baseline To Better Alternatives
User: Compare the current setup with better alternatives.

Preferred behavior:
- Use scenario iteration.
- Summarize the current analysis basis, best scenario, and one or two alternatives.
- Keep the answer comparative.

## Example: Why Did Reserve Change?
User: Why does reserve increase when I drop this factor?

Preferred behavior:
- Use reserve-change explanation against the current analysis basis unless the user explicitly asks for baseline.
- Attribute movement across development, tail, BF, and selection components.
- Use data views only if they clarify the change.

## Example: How Much More Could Still Emerge?
User: How much more claims could still come after this quarter?

Preferred behavior:
- Use late-emergence benchmark.
- Compare historical continuation with current selected IBNR.
- Discuss range and uncertainty, not just one point estimate.

## Example: Tail Method Names
User: Use a Power Curve tail.

Preferred behavior:
- Recognize that supported tail methods are `exponential`, `inverse_power`, and `weibull`.
- Interpret `Power Curve` as `inverse_power`.
- If recommending a tail setting, run or compare a tested scenario before presenting it as a recommendation.

## Example: Compare Weibull Tail
User: How is the assessment if using weibull?

Preferred behavior:
- Do not answer from narrative memory.
- Reuse the active/recommended tested scenario parameters from the current `Analysis Basis`.
- Run `tool_evaluate_tail_fit` with `tail.curve=weibull` and the current fit period/attachment assumptions.
- Answer using tested metrics such as `R^2`, `RMSE`, residual pattern, and any input adjustments.

## Example: Recommendation Becomes New Basis
User: Please also remove statistically large a2a outliers. Fit the tail again. Then check BF.

Preferred behavior:
- Run the appropriate review and scenario tools.
- If the assistant recommends a tested refined setup, state the basis used clearly.
- Treat that tested recommendation as the new `Analysis Basis` for later basis-aware questions.
- If the user then asks a follow-up like "what is the final IBNR per UWY?", answer from that carried-forward basis rather than drifting back to the earlier session baseline.

## Example: Development Age Units
User: Are you sure? In chainladder I can see data up to period 132-135.

Preferred behavior:
- Clarify that development labels are month-based in this quarterly setup.
- `132-135` means month 132 to month 135 of development, i.e. one quarter.
- Do not confuse that with AY/UWY, which are annual labels.
- If discussing fit period `12-45`, explain it as months 12 to 45 of development, not years 12 to 45.

## Example: Sticky Monthly Notation
User: I want you to reference it with the monthly notation, eg. 6 is second quarter and 45 is after 45 months.

Preferred behavior:
- Acknowledge the preference and keep it for the rest of the conversation.
- Refer to development ages as `month 6`, `month 45`, `month 135`, or `months 12 to 45`.
- Do not switch back to phrasing like `15th quarter` unless the user asks for quarter notation again.
