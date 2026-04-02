# AI implementation roadmap and current status

Remark: already implemended is ~~crossed out~~ 

# Goal

The goal is:

**a professional specialty-industrial reserving copilot that reliably drives a deterministic plan-test-conclude workflow, remembers segment context across valuation cycles, and produces review-ready recommendations that a responsible actuary can accept, modify, or reject.**

That direction fits both your operating model and the themes in the Readwise excerpt you shared: the biggest gains come from multi-step planning, persistent segment memory, structured scenario optimization, uncertainty-aware outputs, and evidence trails - not from trying to make the base model magically smarter. It also fits what `chainladder` already gives you through workflow composition, pipelines, tail estimators, adjustment tooling, and optional volatility primitives if you decide to use them as secondary evidence rather than as your booking method. ([chainladder-python.readthedocs.io][1])

# Executive summary

Your best roadmap is to evolve the assistant from a **tool-using analyst** into a **structured reserving copilot** with five concrete properties:

1. It follows fixed reserving playbooks instead of turn-by-turn improvisation.
2. It keeps segment memory across quarters.
3. It ranks scenarios with a deterministic policy, not only LLM prose.
4. It always distinguishes evidence, inference, and judgment.
5. It stays deterministic-first, with stochastic features only as optional sensitivity and volatility overlays. `chainladder` supports workflow composition via `Pipeline`, multiple tail estimators, and bootstrap/Mack components, so you can add structure without replacing your engine. ([chainladder-python.readthedocs.io][1])

For specialty industrial reserving, I would **not** start with full RAG. I would start with **curated markdown context plus structured memory plus strong tool orchestration**. RAG becomes worthwhile later when you have enough internal methodology content, prior committee notes, and reserving policy documents that retrieval improves decisions rather than adding noise. Retrieval helps when governance and policy text matter, but it only pays off once the source base is stable and curated. ([Actuarial Standards Board][2])

# Current implementation status (Apr 2026)

This document started as a forward-looking plan. Parts of Phase 1 and Phase 2 are now implemented, so the current AI state is ahead of some wording below.

Implemented today:

* deterministic playbook planner / executor / reviewer / recommendation-policy control layer
* compact segment memory with continuity and human-decision writeback
* composite reviews for drop review, tail review, BF suitability, anomaly triage, and quarter-close
* quarter-close pack generation
* scenario ledger and scenario-detail drilldown
* bespoke recalculation, reserve-change explanation, and rule-based derived-drop workflows
* `Analysis Basis` / current conversation model behavior for basis-aware reserving workflows

`Analysis Basis` means the current reserving setup the user and AI are actively working from in the conversation. For basis-aware reserving tools, future analysis should reuse that basis unless the user explicitly switches to `baseline` or `current session`.

Historical wording note:

* older references below to comparing a scenario "against baseline" should now be read as "against the current analysis basis unless baseline/current session is explicitly requested" for basis-aware workflows
* raw data exploration and movement-inspection workflows are still intentionally more session/data scoped and should not be forced into scenario-carry-forward behavior when that would blur observed data with modeled assumption state

# Recommended target architecture

## 1. Core design principle

The LLM should never be the source of reserving truth. It should do four jobs:

* choose the right playbook
* plan the evidence-gathering sequence
* interpret tool outputs
* draft a recommendation package

The reserving engine remains the computational authority. That aligns with ASOP 56's emphasis on intended purpose and responsible use of models, and with ASOP 23's focus on reviewing data and disclosing limitations. ([Actuarial Standards Board][2])

## 2. The architecture I would build

### ~~Layer A - Playbook planner~~

Input: user request, session state, segment memory, available tools.

Output:

* playbook type
* step plan
* required evidence checklist
* stopping criteria

Example playbooks:

* movement review
* drop review
* tail review
* BF suitability review
* quarter-close pack
* data anomaly triage
* segment comparison

The planner should output a machine-readable plan like:

```json
{
  "playbook": "drop_review",
  "segment": "industrial_liability_us",
  "goal": "recommend whether to drop any ratios",
  "required_evidence": [
    "diagnostics_summary",
    "link_ratio_ranking",
    "ldf_consistency",
    "late_emergence",
    "movement_diagnostics",
    "scenario_comparison"
  ],
  "decision_rule": "no recommendation unless at least 5/6 evidence items collected"
}
```

### ~~Layer B - Tool executor~~

Runs tools in sequence and stores normalized outputs.

This should be boring and deterministic:

* execute
* validate response shape
* summarize to structured facts
* cache result
* ledger result

### ~~Layer C - Reviewer pass~~

Not a heavy "auditor". More like a **professional reliability gate**.

Reviewer checks:

* Did we run the minimum required tools?
* Is every numeric claim tied to a tool result?
* Did we separate observation from recommendation?
* Did we disclose any unresolved data concerns?
* Are we recommending only supported parameters and methods?

This is the piece that makes the assistant feel reliable.

### ~~Layer D - Recommendation policy engine~~

This should be code, not prompt text.

The LLM may suggest candidate scenarios, but recommendation status should come from deterministic rules such as:

```python
if data_quality_red_flag:
    status = "hold_for_review"
elif scenario_improves_consistency and not highly_sensitive and not excessive_drop_count:
    status = "recommended"
elif improvement_is_marginal:
    status = "watch"
else:
    status = "not_recommended"
```

### Layer E - Memory

Three memory types are enough at first:

**1. Segment memory**

* prior selected method
* prior selected tail settings
* prior accepted/rejected drop logic
* recurring anomalies
* special caveats

**2. User preference memory**

* preferred wording
* preferred summary depth
* whether user likes per-UWY detail by default
* tolerance for clarifying questions

**3. Quarter memory**

* what changed this quarter
* open questions
* unresolved issues
* final human decisions

### ~~Layer F - Context layer~~

Start with:

* `AI_CONTEXT.md`
* `AI_PLAYBOOKS.md`
* `AI_POLICY.md`
* `AI_SEGMENT_NOTES/<segment>.md`

Not with full RAG.

Reason: your use case is narrow and stable. A small, curated markdown corpus will usually outperform an immature retrieval stack early on. Add retrieval later only when your internal source volume grows enough that manual curation becomes the bottleneck. Retrieval-augmented generation is useful, but only when the corpus and citation discipline are mature. ([Actuarial Standards Board][2])

# My opinion on markdown context vs RAG

## Start with markdown-first

For your stage, I would use:

* one global operating manual
* one reserving policy file
* one playbook file
* one segment note per major portfolio
* one method suitability note
* one terminology guardrail file

That gives you:

* predictability
* easy version control
* low complexity
* easy review by actuaries
* no retrieval ranking failures

## Add RAG later only for these use cases

RAG becomes high-value when you need to search across:

* many prior quarter memos
* methodology notes
* committee decisions
* prior accepted/rejected scenario rationales
* data issue logs
* claims handling change logs

My threshold would be:

* more than 50 living internal docs, or
* more than 10 segments with materially different handling, or
* frequent need to cite prior quarter reasoning verbatim

Until then, markdown plus structured memory is cleaner.

# Actuarial roadmap

## Phase 1 - Deterministic orchestration and professional controls

This is the highest ROI phase.

### ~~1. Standardize decision playbooks~~

For each major reserving task, define:

* required tools
* required evidence
* acceptable outputs
* escalation rules
* recommendation template

You already have proto-playbooks. The next step is turning them into machine-executable plans.

### ~~2. Add a reviewer gate and recommendation policy~~

Before adding more actuarial intelligence, add a deterministic reliability layer around the tools you already have.

Minimum reviewer checks:

* required evidence coverage by playbook
* unsupported method / parameter block
* numeric claim provenance present
* unresolved critical data-quality finding disclosure
* recommendation language matched to evidence strength

The output should be a machine-readable decision such as:

* `pass`
* `pass_with_caveats`
* `hard_fail`

This uses the existing diagnostics, governance, and evidence-id plumbing rather than replacing it.

### ~~3. Add segment-aware persistent memory v1~~

This moves into Phase 1 because the codebase already has chat working memory, scenario ledger summaries, and per-segment YAML persistence.

Phase 1 memory should stay compact and curated:

* prior selected method and tail settings
* prior accepted / rejected scenario hashes
* recurring segment caveats
* open issues carried into the next valuation
* last recorded human disposition

Memory writeback from explicit human decisions can still deepen in Phase 2, but the read path should exist in Phase 1.

### ~~4. Add a data quality triage playbook~~

This is critical in specialty industrial work because odd data is often more important than model elegance.

Phase 1 scope is the deterministic triage shell: select the playbook, run the minimum diagnostics, classify the issue, and pause recommendations when governance requires it.

Phase 2 scope is deeper actuarial triage intelligence: stronger issue typing, better next-step diagnostics, and tighter linkage from anomaly class to assumption-selection impact.

Minimum checks:

* missing diagonal values
* negative or impossible link ratios
* valuation-to-valuation discontinuities
* abrupt case reserve shifts
* sparse maturity areas
* segment definition changes
* calendar year distortions
* one-off large loss contamination

The output should not be "data is bad". It should be:

* issue
* likely reserve relevance
* recommended next diagnostic
* whether parameter selection should be paused

This aligns very closely with ASOP 23's emphasis on review of data, reasonable effort, and disclosure of limitations. ([Actuarial Standards Board][3])

### 5. Build structured drop recommendation logic

Today you have good primitives. Next level means combining them in a policy.

For each candidate drop:

* outlier score from ranked link ratios
* consistency improvement score
* late emergence support score
* reserve impact score
* overfit penalty
* governance penalty if too many drops or if driven only by one volatile point

Recommendation classes:

* recommend
* reasonable alternative
* watch only
* avoid

### 6. Build a real tail recommendation engine

For specialty industrial lines, tail is often where the professional value is.

You should evaluate:

* curve type suitability
* attachment age sensitivity
* fit period stability
* reserve impact
* sensitivity to a slightly earlier/later attachment
* consistency with portfolio knowledge

`chainladder` already supports multiple tail approaches and attachment logic, including `TailCurve`, `TailConstant`, `TailBondy`, and `TailClark`. That is enough to materially improve your tail workflow without inventing new methods. Tail estimation is also explicitly sensitive to assumptions, so the assistant should present tail recommendations as conditional, not absolute. ([chainladder-python.readthedocs.io][4])

### 7. Add BF suitability guidance by segment and UWY

Not just "BF for immature years."

Score BF suitability using:

* maturity
* recent diagonal volatility
* susceptibility of CL to latest valuation movement
* reasonableness of apriori source
* consistency of implied percent reported

Then let the assistant say:

* CL preferred
* BF preferred
* mixed by UWY preferred
* inconclusive without apriori review

## Phase 2 - Decision quality and actuarial depth

Check out: ai_implementation_plan_v2_april_2026_phase2.md

Once Phase 1 control-layer reliability is stable, deepen the actuarial engines and quarter-close workflow.

### 8. Deepen segment-aware persistent memory

Memory schema example:

```json
{
  "segment_id": "industrial_liability_us",
  "last_selection": {
    "valuation_date": "2026-03-31",
    "method_by_uwy": {
      "2019": "chainladder",
      "2020": "chainladder",
      "2021": "bornhuetter_ferguson"
    },
    "tail": {
      "estimator": "TailCurve_exponential",
      "attachment_age": 60,
      "fit_period": [24, 84]
    }
  },
  "rejected_scenarios": [
    {
      "scenario_hash": "abc123",
      "reason": "improved fit but too dependent on dropping single high ratio"
    }
  ],
  "known_issues": [
    "2021 UWY impacted by large refinery loss",
    "case adequacy changed after TPA transition in 2025Q2"
  ],
  "house_preferences": [
    "prefer stable tail over slightly lower selected reserve",
    "avoid more than 2 dropped ratios unless clearly justified"
  ]
}
```

Phase 1 already establishes compact segment memory read/write and basic human-decision writeback.

Phase 2 expands that into richer continuity logic: rejected-before checks, house-preference checks, longer-horizon quarter memory, and stronger contradiction detection against prior accepted selections.

This is where the assistant stops feeling forgetful over longer operating cycles.

### 9. Build quarter-close workflow orchestration

The assistant should be able to run:

1. compare prior vs current valuation
2. run data triage
3. run movement diagnostics
4. identify suspect assumptions
5. test targeted scenarios
6. rank scenarios
7. draft summary pack

This is the core "plan-test-conclude" transformation you want.

### 10. Introduce scenario optimization, not just enumeration

Do not jump to fancy Bayesian optimization immediately.

Start with constrained structured search:

* search only supported parameter combinations
* seed from baseline plus domain-reasonable perturbations
* stop when marginal improvement plateaus
* heavily penalize fragile scenarios

Objective function example:

```python
score = (
    0.30 * consistency_score
    + 0.20 * tail_fit_score
    + 0.20 * movement_explanation_score
    + 0.15 * stability_score
    + 0.15 * business_reasonableness_score
    - 0.20 * overfit_penalty
    - 0.20 * data_quality_penalty
)
```

This is not "optimization" in a research sense. It is structured actuarial triage.

## Phase 3 - Professional polish and optional uncertainty layer

### 9. Add uncertainty-aware overlays

Because you said you do not use stochastic reserving methods for actual reserving, I would position this as:

**secondary decision support, not primary booking support**

Use optional overlays for:

* sensitivity ranges
* scenario fragility
* relative volatility flags
* process/parameter risk indicators for explanation only

`chainladder` includes `MackChainladder`, bootstrap sampling, and workflow composition that can support this kind of volatility lens. For your use case, I would not surface percentile outputs as "the answer". I would use them to answer questions like:

* Is this scenario materially more fragile?
* Is this tail choice much more uncertainty-sensitive?
* Is CL for this immature UWY unusually volatile relative to BF? ([chainladder-python.readthedocs.io][1])

### 10. Add industrial-reserving-specific hypothesis engine

This is where the copilot becomes actually valuable for your niche.

Hypotheses it should test:

* large loss emergence
* claims handling change
* policy wording or attachment point shift
* portfolio mix change
* settlement speed change
* case strengthening or weakening
* reporting lag shift

The output should be:

* observed signal
* possible explanation
* diagnostic test run
* strength of support
* impact on assumption selection

# Technical roadmap

## Phase 1 - 0 to 6 weeks

Build the control system, not new math.

### Deliverables

* ~~planner/executor/reviewer skeleton~~
* ~~structured playbook specs~~
* ~~tool output normalizer~~
* ~~recommendation policy v1~~
* ~~segment memory store~~
* ~~markdown context files~~
* ~~feature-flagged assistant integration over the existing AI/tool stack~~

### Key engineering tasks

**~~A. Tool contract layer~~**
Every tool should return:

* summary text
* structured metrics
* provenance metadata
* segment / valuation identifiers

**~~B. Scenario ledger~~**
Store every scenario as:

* data snapshot id
* engine version
* parameter set
* metrics
* scenario score
* recommendation status
* human disposition

Remark: a lightweight scenario ledger already exists in chat memory. Phase 1 should standardize and persist it rather than rebuild it from scratch.

**~~C. Reviewer rules~~**
Hard fail if:

* unsupported method proposed
* recommendation without minimum evidence
* numeric statement missing provenance
* final language stronger than evidence level

**~~D. UI changes~~**
For every recommendation, show:

* ~~conclusion~~
* ~~evidence used~~
* ~~alternative considered~~
* ~~key caveat~~
* ~~next best question~~

Remark: expose these first in the AI chat/API response shape. Dash recommendation cards can follow immediately after the API contract is stable.

## Phase 2 - 6 to 12 weeks

Improve actuarial intelligence.

### Deliverables

* drop engine v2
* tail engine v2
* BF suitability scoring
* data anomaly triage v2
* quarter-close pack generator
* richer memory writeback from human decisions
* rejected-before / house-preference memory checks in policy decisions

### Key engineering tasks

* add scenario scoring service
* add sensitivity runner
* add segment comparison workflows
* add human override reason capture
* add "rejected before" memory checks
* deepen anomaly triage classification and next-step guidance

## Phase 3 - 3 to 6 months

Selective sophistication.

### Deliverables

* optional volatility overlay
* richer industrial hypothesis testing
* markdown plus retrieval hybrid
* benchmark harness
* release gating dashboard

### Key engineering tasks

* bootstrap/Mack secondary tools
* retrieval over curated internal docs
* benchmark datasets and scoring scripts
* regression suite for assistant behavior

# What the assistant should look like in practice

## Example workflow: drop review

### User asks

"Review whether any ratios should be dropped for Industrial Liability."

### Planner outputs

* run diagnostics summary
* run ranked link ratios
* run LDF consistency
* run late emergence benchmark
* generate 3-5 candidate drop scenarios
* compare results
* produce recommendation

### Recommendation output

**Recommendation:** Do not apply a drop to baseline selection.
**Reason:** The 24-36 ratio is an outlier, but dropping it improves fit only marginally and materially changes 2022-2023 UWY ultimates. The improvement appears fragile.
**Evidence used:** ranked link ratios, LDF consistency, late emergence benchmark, candidate scenarios D1-D3.
**Alternative:** D2 is a reasonable sensitivity scenario for management discussion.
**Caveat:** Outlier likely linked to a single refinery loss. Consider large-loss segmented view before revisiting.

That is professional and usable.

## Example workflow: quarter-close pack

Output sections:

1. What changed in the data
2. What assumptions were retested
3. Which scenarios were considered
4. Recommended selection changes
5. Why the recommendation is reasonable
6. What remains judgmental
7. Suggested actuarial sign-off questions

# Evaluation plan you can actually run

## Benchmark categories

Use fixed datasets and fixed expected evidence paths.

1. Data quality triage
2. Drop recommendation
3. Tail recommendation
4. BF suitability
5. Quarter-over-quarter movement explanation
6. Segment comparison
7. Large loss / portfolio shift reasoning
8. Unsupported request refusal

## Success criteria

For each test, score:

**Tool discipline**

* correct playbook selected
* required tools run
* no unsupported tools skipped

**Grounding**

* numeric claims traceable
* no invented methods
* no invented diagnostics

**Actuarial usefulness**

* recommendation actionable
* caveats appropriate
* alternatives surfaced
* reasoning matches what a reserving actuary would expect

**Continuity**

* remembers rejected prior scenarios
* remembers segment caveats
* does not contradict prior accepted selections without explanation

**Professional reliability**

* separates observation from recommendation
* discloses limitations when data issues exist
* asks for clarification only when decision-critical

## Suggested scoring rubric

0 to 2 each:

* playbook correctness
* tool completeness
* numeric grounding
* actuarial usefulness
* continuity/memory use
* recommendation quality
* limitation disclosure

Pass threshold:

* no zero on grounding
* no zero on tool completeness
* total score >= 10/14

# Risks and how to avoid them

## 1. Over-engineering RAG too early

Risk: complexity without better decisions.
Fix: markdown-first until corpus size justifies retrieval.

## 2. Letting LLM prose become policy

Risk: inconsistent recommendations.
Fix: deterministic scoring and rule layer.

## 3. Too much "confidence" language

Risk: false authority.
Fix: evidence tiers:

* observed
* inferred
* judgmental

## 4. Treating stochastic tools as if they are your selected method

Risk: workflow mismatch with your practice.
Fix: label all stochastic outputs as secondary sensitivity aids only.

## 5. Memory becoming messy

Risk: stale or contradictory context.
Fix: only store compact, curated, structured memory with timestamps and dispositions.

# Top 10 highest-ROI improvements for your actual use case

1. ~~Convert playbooks into executable planner specs.~~
2. ~~Add segment memory with accepted/rejected scenario history.~~
3. ~~Build deterministic recommendation policy scoring.~~
4. ~~Add data quality triage as a first-class workflow.~~
5. Upgrade drop logic from heuristics to evidence-weighted scoring.
6. Build a serious tail recommendation engine around supported tail estimators.
7. Add BF suitability scoring by UWY.
8. Add quarter-close pack generation.
9. Add optional uncertainty overlays as secondary evidence only.
10. Add benchmark-based release gating before every production upgrade.

# 90-day implementation plan

## Days 1-30

**Goal:** reliability foundation

Ship:

* ~~playbook planner v1~~
* ~~reviewer pass v1~~
* ~~segment memory v1~~
* ~~markdown context pack~~
* ~~recommendation policy v1~~
* ~~tool-output contract / normalizer v1~~
* ~~assistant integration behind a feature flag~~

Definition of done:

* ~~every recommendation references evidence ids~~
* ~~no recommendation without minimum evidence~~
* ~~no unsupported method leakage~~
* ~~assistant execution path is deterministic-first for supported playbooks~~

## Days 31-60

**Goal:** better actuarial decisions

Ship:

* drop engine v2
* tail engine v2
* BF suitability v1
* ~~data anomaly triage v1~~
* quarter-close playbook v1
* ~~memory writeback from human dispositions v1~~

Definition of done:

* assistant can run end-to-end drop review and tail review with no manual prompting of each step
* quarter-close workflow uses segment memory and reviewer gates

## Days 61-90

**Goal:** professional state

Ship:

* quarter-close pack generator
* memory writeback from human dispositions
* scenario optimization v1
* benchmark harness
* optional uncertainty overlays
* release gate dashboard

Definition of done:

* assistant is measurably more stable, more grounded, and more useful on fixed benchmark tasks than current baseline

# Final recommendation

If I were building this for your exact situation, I would do it in this order:

**First**: planner + reviewer + policy engine + segment memory
**Second**: stronger drop/tail/BF workflows
**Third**: quarter-close package automation
**Fourth**: optional stochastic overlays for sensitivity only
**Fifth**: retrieval, but only after markdown context and memory start to hit scale limits

That gets you to a professional and reliable state much faster than chasing generic agent sophistication.

If you want, I can turn this into a very concrete build artifact next - for example a **feature backlog with epics, acceptance criteria, architecture components, and milestone sequencing**.

[1]: https://chainladder-python.readthedocs.io/en/latest/user_guide/workflow.html?utm_source=chatgpt.com "Workflow — Chainladder - Python"
[2]: https://www.actuarialstandardsboard.org/wp-content/uploads/2020/01/asop056_195.pdf?utm_source=chatgpt.com "Actuarial Standard"
[3]: https://www.actuarialstandardsboard.org/asops/asop-no-23-revision-data-quality/?utm_source=chatgpt.com "ASOP No. 23 Revision - Data Quality - Actuarial Standards BoardActuarial Standards Board"
[4]: https://chainladder-python.readthedocs.io/en/master/user_guide/tails.html?utm_source=chatgpt.com "Tail Estimators — Chainladder - Python"
