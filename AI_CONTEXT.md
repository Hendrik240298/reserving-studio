# AI Context

## Reserving terminology
- "current quarter" means the most recent valuation period in the loaded data.
- In triangle terms, "current quarter" refers to the latest diagonal.
- If a user asks about movements "this quarter" or "in the quarter", interpret that as movement into the latest valuation period unless they explicitly say otherwise.
- Development periods / chainladder ages are month-based labels, not underwriting-year counts.
- In this quarterly setup, one development step is 3 months.
- A label like `3-6` means development from month 3 to month 6, i.e. one quarter.
- A label like `45` means 45 months of development, not 45 years and not 45 underwriting periods.
- When converting development age to years, divide months by 12.
- AY / UWY labels are always annual underwriting-year labels, not development-period labels.
- If the user asks you to reference development using monthly notation, continue using monthly notation for the rest of the conversation.
- After a user sets a notation preference, do not switch back to quarter-count notation unless they explicitly ask for it.
- In monthly notation, prefer phrases like `month 45`, `month 135`, or `months 12 to 45` rather than `15th quarter` or `45 quarters`.
- If a user asks about "claims movements" without specifying paid or outstanding, interpret that as `incurred` claims movement.
- For questions about "claims movements this quarter", prioritize the latest-diagonal / in-quarter `incremental incurred` view first.
- Do not switch to premium movement unless the user explicitly asks about premium or the answer clearly requires premium as secondary context.
- Do not describe a year as "late", "mature", or "very late in development" from absolute development age alone.
- Judge maturity relative to the selected development pattern and tail in the reserving workflow, especially the implied `LDF`/`CDF` position and comparable prior years.
- If the selected `CDF` or tail implies substantial development is still outstanding, avoid language that implies the year is already near ultimate.
- Prefer referring to the reserving class's selected development and tail assumptions rather than explicit emergence percentages unless the user asks for emergence directly.

## Development factor terminology
- Distinguish clearly between observed triangle `a2a` factors and the selected `LDF` vector.
- `a2a` factors are the raw observed age-to-age/link-ratio cells in the triangle. There can be many of them per development period.
- The selected `LDF` vector is the single selected development factor per development period used by the reserving workflow.
- Do not casually switch between `a2a` and `LDF` as if they are the same object.
- Do not confuse development-period labels with AY/UWY labels.
- If a user refers to age `132-135`, interpret that as months 132 to 135 of development, i.e. one quarterly step late in the triangle.
- If the user asks about factors "below 1" after discussing triangle link ratios or threshold rules, default to interpreting that as observed `a2a` factors below 1 unless they explicitly say selected `LDF` vector.
- If the user asks whether development is monotone and is referring to the selected `LDF` vector, the default actuarial interpretation should be monotone decay: `LDF_i >= LDF_{i+1}`.
- Do not answer "monotone" for an `LDF` vector using only the weaker condition `LDF >= 1`. If both meanings could matter, state the distinction explicitly.
- If the user says `LDF` but the operational request is row/column-wise over the triangle, that may actually imply observed `a2a` factors by development period. In that case, either clarify briefly or state the interpretation you are using.

## Data-view vocabulary
- Allowed metric names for AI tool calls:
  - `incurred`
  - `paid`
  - `outstanding`
  - `premium`
- Allowed view names for AI tool calls:
  - `cumulative`
  - `incremental`

## Metric aliases
- "claims" means `incurred` unless the user explicitly says paid claims or outstanding claims
- "incurred claims" means `incurred`
- "paid claims" means `paid`
- "outstanding claims" means `outstanding`
- "earned premium" means `premium`
- "gross written premium" means `premium`
- "GWP" means `premium`

## Scenario and reserving rules
- Use the existing `source.reserving.Reserving` class and its supported parameters.
- Supported tail curve method names are: `exponential`, `inverse_power`, `weibull`.
- Map common tail aliases as follows: `power` -> `inverse_power`, `power_curve` -> `inverse_power`.
- If the user asks for exact numeric values, vectors, or tables about selected `LDF`, fitted tail factors, observed `a2a`, BF apriori, or method selection by UWY, load an exact-detail tool result first and answer from that payload only.
- Do not answer exact numeric factor questions from narrative memory or from a recommendation summary.
- If a user asks for a tail recommendation, prefer tested tail scenarios or explicitly label an idea as untested.
- Do not present a bespoke scenario recommendation as validated unless it has actually been run through the available tools in the current conversation.
- Do not claim to create a brand-new reserving algorithm unless code explicitly supports it.
- Prefer summary tools first, then request detailed views only if the extra detail is useful.
