# AI Control Plane Refactor

This document defines the target architecture for the AI chat/control layer.

It is not a roadmap for adding more AI surface area. It is a refactor guide for making the current AI behavior reliable, transparent, and extensible.

Implementation companion:

- `docs/ai-control-plane-implementation-plan.md`

The main design goal is:

- deterministic APIs do the real work
- reusable workflow/skill definitions tell the AI how to use them
- the control plane owns state, guardrails, and traceability
- chat text never silently mutates application state

## 1. Core decisions

These are the key design decisions that should shape the refactor.

### 1.1 Analysis Basis purpose

`Analysis Basis` exists to answer one question:

- what exact basis is the AI currently reasoning from?

Its purpose is:

- transparency
- traceability
- shared context between user and AI

It is not the place where a fresh recommendation should appear automatically.

### 1.2 Recommendation, proposal, and acceptance are different

These concepts are related, but they are not the same.

- `recommendation`
  - semantic conclusion produced by deterministic review/evidence
  - example: "drop_combo_1 is a reasonable alternative"
- `proposal`
  - a UI/state object derived from a recommendation that can be accepted or rejected
  - example: a small element below the latest AI message with `Yes` / `No`
- `acceptance`
  - explicit user action that promotes a proposal into the accepted chat-carried basis

Required rule:

- recommendation does not update basis
- proposal does not update basis
- only acceptance updates basis

### 1.3 Chat state must never mutate reserving session state

The AI chat can carry its own accepted analysis basis.

It must not write changes into the live reserving-studio reserving session.

Later the chat may read and understand the current actual reserving session, but it must not silently modify it.

### 1.4 Local session scope

When a user accepts a proposal, the new accepted basis should update only the local AI chat session state.

It should not become a cross-system persistent reserving state by default.

## 2. The main architectural problem

The current control plane collapses too many responsibilities into one mutable blob, mostly through `analysis_basis` and related memory fields.

Today the same state object can mean:

- current accepted reasoning basis
- latest recommended scenario
- latest review candidate
- temporary bespoke preview basis
- active reserving session baseline

This creates state drift and misleading narration.

The most important refactor goal is to detangle these responsibilities.

## 3. Target layer architecture

The architecture should be layered.

### 3.1 Layer 1: Capability Layer

Purpose:

- expose deterministic capabilities as typed Python/API contracts

This is the API-first layer.

It should contain narrow, explicit operations with stable schemas.

Examples:

- `run_drop_review`
- `run_tail_review`
- `run_bf_suitability_review`
- `run_anomaly_triage`
- `iterate_diagnostics`
- `get_results_summary`
- `explain_reserve_change`
- `get_assumption_context`
- `preview_scenario`

Design rules:

- each tool has one clear purpose
- input and output schemas are explicit
- deterministic code does the work
- no chat semantics inside this layer
- no basis auto-promotion inside this layer

This layer should be generic enough that later domains such as pricing or risk management can expose their own typed capabilities using the same contract style.

### 3.2 Layer 2: Workflow / Skill Layer

Purpose:

- define reusable building blocks for how the AI should work

This is where skills and workflows belong.

These are not the raw tools themselves.
They are reusable procedures that orchestrate tools.

Examples:

- `drop_review_workflow`
- `tail_selection_workflow`
- `reserve_change_explanation_workflow`
- `anomaly_triage_workflow`
- `quarter_close_workflow`

This layer should answer:

- what evidence is required?
- what tool sequence is preferred?
- what stopping rule applies?
- what answer structure is expected?
- what guardrails weaken or block a recommendation?

This layer is the right place for building-block behavior.

It is skill-like, but should stay structured and inspectable rather than becoming a pile of long free-form prompts.

Recommended shape:

- small versioned workflow definitions
- explicit input contract
- explicit evidence contract
- explicit output contract

### 3.3 Layer 3: Control Plane

Purpose:

- coordinate execution
- own state transitions
- enforce guardrails
- ensure that chat behavior stays correct

This is the architecture layer and the most important one to harden.

Recommended sub-components:

- `Planner`
  - chooses the workflow/skill
- `BasisManager`
  - owns accepted analysis basis only
- `ProposalManager`
  - stores recommendations that can be accepted or rejected
- `ExecutionManager`
  - runs tool calls and emits execution records
- `PolicyEngine`
  - enforces hold, pause, caveats, and narration constraints
- `NarrationAssembler`
  - builds user-facing answer content from validated outputs
- `TraceStore`
  - stores execution, provenance, and basis transition records

Layer 3 should guarantee:

- no silent basis mutation
- no narration that contradicts actual executed inputs
- no recommendation stronger than policy allows
- no confusion between accepted basis and proposed change

### 3.4 Layer 4: Durable Trace / Workflow History

Purpose:

- record what happened and why

This is not the immediate priority, but the design should prepare for it.

Needed concepts:

- execution history
- basis transition history
- proposal acceptance/rejection history
- provenance links to evidence and run IDs

For now this can remain lightweight and local.

### 3.5 Layer 5: UI / Integration Layer

Purpose:

- present basis, proposals, and execution traces clearly

This is not about low-code. It is about UI surfaces and optional integrations.

Recommended UI split:

- `Analysis Basis`
  - accepted AI reasoning basis only
- `Proposal`
  - displayed below latest AI message when relevant
  - includes `Yes` / `No`
- `Execution Trace`
  - requested vs effective execution summary
  - warnings and run links

## 4. State model

The current mutable basis state should be replaced by explicit state objects.

### 4.1 Required state objects

- `active_session_basis`
  - read-only view of the actual reserving session state
  - chat can inspect it, not mutate it
- `accepted_analysis_basis`
  - the current basis the AI uses for future basis-aware reasoning in this chat session
- `proposal_basis`
  - a pending suggested basis change derived from recommendation output
- `preview_basis`
  - temporary bespoke basis used for one-off impact or scenario analysis

### 4.2 State rules

- `accepted_analysis_basis` drives future basis-aware analysis
- `proposal_basis` never drives future analysis until accepted
- `preview_basis` never mutates `accepted_analysis_basis`
- `active_session_basis` never changes because of chat actions

### 4.3 Basis transitions

Allowed transitions:

- baseline or accepted basis -> preview basis for one execution
- recommendation -> proposal basis
- proposal basis + explicit user acceptance -> accepted analysis basis
- explicit user request -> reset accepted analysis basis to baseline/current-session view

Disallowed transitions:

- recommendation -> accepted basis
- tool result -> accepted basis
- review completion -> accepted basis

## 5. Simplified identifier model

The current identifier model is harder to reason about than it needs to be.

### 5.1 Keep

- `chat_id`
  - identity of the AI conversation
- `session_id`
  - identity of the reserving workspace/session
- `execution_id`
  - identity of a concrete tool/workflow run
- `evidence_id`
  - provenance identifier

### 5.2 Introduce one canonical basis identifier

Use one stable identity for a normalized parameter set.

Recommended name:

- `basis_key`

This should be the canonical execution identity for a basis/scenario and replace most of the overloaded usage of:

- `scenario_id`
- `scenario_signature`

### 5.3 Demote labels from identities

Human labels are still useful, but they should not be treated as the canonical state key.

Recommended display fields:

- `scenario_label`
  - human-readable scenario name such as `tail_weibull_27_12_108`
- `candidate_label`
  - review-local label such as `drop_3`

Rules:

- execute by `basis_key`
- display by `scenario_label`
- never use `candidate_label` as a durable identity

## 6. Execution outcome must be first-class

This is one of the most important structural upgrades.

The system must treat the outcome of a tool execution as its own explicit object, not as a side effect hidden inside a summarized tool result.

### 6.1 Problem with the current model

Today the system can:

- receive a request with invalid or ambiguous inputs
- silently normalize or drop parts of it
- execute something different from what the user asked for
- still narrate the run as if the original request succeeded exactly

That is how you end up with answers that sound coherent but are not faithful to the actual execution.

### 6.2 Required execution record

Each materially important tool/workflow run should produce an `ExecutionRecord`.

Recommended fields:

- `execution_id`
- `tool_name` or `workflow_name`
- `requested_inputs`
- `effective_inputs`
- `execution_status`
- `warnings`
- `material_adjustments`
- `provenance`
- `result_summary`

### 6.3 Recommended statuses

- `executed_exactly`
- `executed_with_non_material_normalization`
- `executed_with_material_adjustment`
- `partially_executed`
- `rejected`

### 6.4 Material adjustment rules

Harmless normalization may still allow normal narration.

Examples:

- alias mapping like `power_curve` -> `inverse_power`

Material changes must block success-style narration and basis mutation.

Examples:

- dropping requested drop entries
- defaulting to a different average after invalid input
- executing without part of the requested assumption set

### 6.5 Why this matters for general architecture

This pattern should be generalized across future domains.

It is not specific to reserving.

If later you add pricing, risk, or another deterministic domain, the same execution pipeline should still work:

- request
- validation/compilation
- execution
- explicit execution record
- narration constrained by execution record

That is the correct reusable architecture boundary.

## 7. Adapter redesign

The current adapter approach should be upgraded.

### 7.1 Current weakness

Weak points today:

- duplicated normalization/sanitization logic
- silent mutation of user intent
- no first-class execution record
- too much logic mixed into raw tool adapters

### 7.2 Target adapter pipeline

Each tool call should flow through the same stages.

1. `RequestCompiler`
   - parse and normalize raw inputs into typed command objects
2. `Validator`
   - decide exact / adjusted / partial / rejected
3. `Executor`
   - call deterministic backend API
4. `ExecutionRecorder`
   - emit `ExecutionRecord`
5. `ResultProjector`
   - create small AI-facing summary payload

This should become a standard contract shared across tools.

### 7.3 Design intent

The adapter layer should make it obvious to a human reader:

- what the tool does
- what input it accepts
- what counts as invalid
- what happens on normalization
- what is returned

## 8. Workflow and skill model

The workflow layer should be building-block oriented.

### 8.1 What belongs in a workflow definition

- workflow name
- user intent class
- required capabilities
- required evidence
- stopping rule
- answer template
- recommendation policy hooks
- basis behavior rules

### 8.2 What does not belong there

- live mutable chat state
- direct basis mutation
- hidden side effects

### 8.3 Relationship to AI skills

These workflow definitions are skill-like, but should be more structured than generic prompt-only skills.

Good pattern:

- workflows define the process contract
- prompts provide wording and explanation style
- tools provide deterministic capabilities

This gives building-block reuse without turning the system into opaque prompt orchestration.

## 9. Target user flow

This is the desired chat flow.

1. User asks for review
2. System runs deterministic review workflow
3. Chat answer explains:
   - what was reviewed
   - what is recommended
   - what caveats exist
4. `Analysis Basis` still shows the previously accepted basis
5. System stores the suggestion as `proposal_basis`
6. UI shows a small proposal element below the latest AI answer with `Yes` / `No`
7. User accepts
8. `BasisManager` promotes `proposal_basis` to `accepted_analysis_basis`
9. Future basis-aware analysis uses the new accepted basis

## 10. Migration order

The migration should focus on Layer 3 first.

### Phase A: state separation

- stop recommendation auto-promotion into `analysis_basis`
- introduce explicit accepted/proposal/preview basis states
- keep UI showing only accepted basis in the basis panel

### Phase B: identifier cleanup

- introduce `basis_key`
- demote `candidate_id` to display-only usage
- make labels explicit and separate from keys

### Phase C: execution record pipeline

- add shared request compiler / validator / execution record pipeline
- distinguish exact vs adjusted vs rejected runs
- constrain narration from execution status

### Phase D: workflow/skill formalization

- refactor playbooks into structured workflow definitions
- make basis rules explicit per workflow

### Phase E: UI proposal element

- add proposal panel below latest AI message
- support `Yes` / `No` acceptance flow
- wire acceptance only to local chat-carried basis state

## 11. Near-term success criteria

The refactor is succeeding if the following become true.

- `Analysis Basis` always reflects the actual accepted AI reasoning basis
- recommendations never silently change basis
- chat never claims an execution succeeded when effective inputs differ materially from requested inputs
- basis-aware follow-up questions use the accepted basis, not the latest recommendation
- review candidates and display labels no longer behave like durable execution IDs
- future domains can plug into the same capability/workflow/control-plane pattern

## 12. Summary

The foundation should be:

- API-first capabilities in Layer 1
- workflow/skill building blocks in Layer 2
- a robust control plane in Layer 3

Layer 3 is the critical architecture layer.

If Layer 3 is reliable, then Layers 1 and 2 can grow safely.

If Layer 3 remains fuzzy, the system will keep producing plausible but untrustworthy chat behavior.
