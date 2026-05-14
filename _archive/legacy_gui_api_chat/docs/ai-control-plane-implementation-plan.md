# AI Control Plane Implementation Plan

Note: this file now uses `~~strike-through~~` to mark implementation items that have been completed. Informational narrative, risks, and later/future-work sections are left unstruck unless they were themselves execution items.

This document translates `docs/ai-control-plane-refactor.md` into an implementation-oriented plan.

It answers a narrower question:

- what additional context is required so the refactor can be implemented successfully and consistently?

Short answer:

- `ai-control-plane-refactor.md` is sufficient as a target architecture document
- it is not sufficient by itself as a full implementation plan
- this companion document fills the missing implementation context

## 1. What the refactor document already covers

`docs/ai-control-plane-refactor.md` already defines the important architecture direction:

- the purpose of `Analysis Basis`
- the separation of recommendation, proposal, and acceptance
- the layer model
- the state model at a conceptual level
- the direction for identifier simplification
- the need for first-class execution records
- the target user flow
- the high-level migration order

That is enough to align design decisions.

It is not enough to implement the refactor safely without follow-up interpretation.

## 2. What is still needed for successful implementation

To implement this cleanly, the following additional context must be explicit.

### 2.1 Locked decisions

These should be treated as settled unless deliberately changed later.

- ~~`Analysis Basis` shows accepted AI reasoning basis only~~
- ~~recommendations stay in chat text and proposal UI~~
- ~~only explicit acceptance updates chat-carried basis~~
- ~~AI chat never mutates the live reserving session~~
- ~~accepted basis changes are local to the AI chat session~~
- ~~proposal UI should sit below the latest relevant AI answer~~
- ~~governance/approval workflows are not a first-phase priority~~

### 2.2 Concrete state schemas

The implementation needs typed schemas, not just concepts.

At minimum define:

- ~~`AcceptedAnalysisBasis`~~
- ~~`ProposalBasis`~~
- ~~`PreviewBasis`~~
- ~~`ExecutionRecord`~~
- ~~`BasisTransitionRecord`~~

Recommended fields:

#### `AcceptedAnalysisBasis`

- ~~`basis_key`~~
- ~~`basis_type`~~
- ~~`scenario_label`~~
- ~~`parameters`~~
- ~~`source`~~
- ~~`accepted_at`~~
- ~~`accepted_from_execution_id`~~
- ~~`matches_active_session`~~

#### `ProposalBasis`

- ~~`proposal_id`~~
- ~~`basis_key`~~
- ~~`basis_type`~~
- ~~`scenario_label`~~
- ~~`parameters`~~
- ~~`origin_execution_id`~~
- ~~`origin_workflow`~~
- ~~`recommendation_strength`~~
- ~~`caveats`~~
- ~~`status` with values such as `pending`, `accepted`, `rejected`, `superseded`~~
- ~~`created_at`~~

#### `PreviewBasis`

- ~~`basis_key`~~
- ~~`parameters`~~
- ~~`origin_execution_id`~~
- ~~`expires_after_turn`~~

#### `ExecutionRecord`

- ~~`execution_id`~~
- ~~`workflow_name`~~
- ~~`tool_name`~~
- ~~`requested_inputs`~~
- ~~`effective_inputs`~~
- ~~`execution_status`~~
- ~~`warnings`~~
- ~~`material_adjustments`~~
- ~~`result_summary`~~
- ~~`session_id`~~
- ~~`chat_id`~~
- ~~`evidence_ids`~~
- ~~`created_at`~~

### 2.3 Canonical identifier contract

The implementation needs one explicit identifier policy.

Recommended policy:

- ~~`chat_id`: AI conversation identity~~
- ~~`session_id`: reserving workspace identity~~
- ~~`basis_key`: canonical normalized basis identity~~
- ~~`execution_id`: concrete run identity~~
- ~~`proposal_id`: UI/action identity for accept/reject~~
- ~~`evidence_id`: provenance identity~~

Display-only labels:

- ~~`scenario_label`~~
- ~~`candidate_label`~~

Rules:

- ~~execute by `basis_key`~~
- ~~store proposals by `proposal_id`~~
- ~~show labels in UI~~
- ~~never drive control flow from `candidate_label`~~

### 2.4 Current-to-target module mapping

The refactor will be easier if module responsibility changes are explicit.

Recommended mapping:

- ~~`ai/assistant_service.py`~~
  - ~~keep as top-level orchestration entrypoint~~
  - ~~remove basis auto-promotion responsibility~~
- ~~`ai/tool_payloads.py`~~
  - ~~reduce state-merging and identifier-shaping responsibility~~
  - ~~keep payload compaction/projector behavior only~~
- ~~`ai/recommendation_policy.py`~~
  - ~~keep recommendation decision logic~~
  - ~~do not let it imply basis mutation~~
- ~~`ai/backend_tools.py`~~
  - ~~convert toward request compiler + validator + executor + record emission~~
- ~~`ai/api_tools.py`~~
  - ~~share the same request-compile/validation logic as backend tools~~
- ~~`source/ai_dashboard.py`~~
  - ~~split accepted basis UI from proposal UI~~
  - ~~stop presenting basis as implicitly updated by recommendation~~

Recommended new modules:

- ~~`ai/basis_manager.py`~~
- ~~`ai/proposal_manager.py`~~
- ~~`ai/execution_records.py`~~
- ~~`ai/request_validation.py`~~
- ~~`ai/workflow_definitions.py`~~

### 2.5 UI interaction specification

The implementation needs a more concrete UI contract.

Required behavior:

- ~~accepted basis panel shows only the currently accepted basis~~
- ~~proposal UI renders only when a new proposal exists~~
- ~~proposal UI belongs to the message that introduced the recommendation~~
- ~~proposal UI offers `Yes` / `No`~~
- ~~`Yes` promotes proposal to accepted basis~~
- ~~`No` marks proposal as rejected but does not change basis~~
- ~~if a newer proposal appears, older pending proposals should be marked `superseded`~~

Recommended first-phase simplification:

- ~~only one pending proposal per chat at a time~~

### 2.6 Narration constraints

The implementation needs hard response rules.

Examples:

- ~~never say a requested change was applied if execution status is not `executed_exactly` or `executed_with_non_material_normalization`~~
- ~~never say the basis changed unless acceptance occurred~~
- ~~never describe a dropped invalid input as part of the executed scenario~~
- ~~if a proposal exists but is not accepted, the answer should distinguish:~~
  - ~~current basis~~
  - ~~recommended change~~

### 2.7 Execution classification rules

To make execution outcome first-class, the implementation needs a reusable severity model.

Recommended split:

- ~~`non-material normalization`~~
  - ~~harmless alias normalization~~
  - ~~formatting cleanup~~
- ~~`material adjustment`~~
  - ~~requested drops removed~~
  - ~~fallback to different assumptions~~
  - ~~partial scenario execution~~
- ~~`rejected`~~
  - ~~request cannot be executed faithfully enough to support narrative success~~

~~This classification should be implemented once and shared by all adapters.~~

### 2.8 Workflow definition contract

The workflow layer should have a stable shape before implementation expands.

Each workflow definition should include:

- ~~`workflow_name`~~
- ~~`intent_class`~~
- ~~`required_capabilities`~~
- ~~`required_evidence`~~
- ~~`minimum_evidence_count`~~
- ~~`stopping_rule`~~
- ~~`basis_behavior`~~
- ~~`answer_contract`~~

Recommended `basis_behavior` values:

- ~~`use_accepted_basis`~~
- ~~`use_active_session_only`~~
- ~~`allow_preview_only`~~
- ~~`proposal_possible`~~
- ~~`proposal_disallowed`~~

## 3. What should be implemented first

The work should be sequenced to stabilize Layer 3 before broadening capability/workflow power.

### Step 1: Basis/proposal state split

Implement first:

- ~~`accepted_analysis_basis`~~
- ~~`proposal_basis`~~
- ~~removal of auto-promotion behavior~~

This is the most important foundational change.

### Step 2: Proposal UI flow

Implement:

- ~~proposal rendering under latest relevant AI message~~
- ~~`Yes` / `No` interaction~~
- ~~local accepted-basis update on `Yes`~~
- ~~rejected proposal state on `No`~~

### Step 3: Execution record pipeline

Implement:

- ~~shared request validation layer~~
- ~~shared execution classification layer~~
- ~~explicit `ExecutionRecord`~~
- ~~narration constrained by execution status~~

### Step 4: Identifier simplification

Implement:

- ~~`basis_key`~~
- ~~label/key separation~~
- ~~phased removal of `candidate_id` from state-driving behavior~~

### Step 5: Workflow formalization

Implement:

- ~~structured workflow definitions~~
- ~~explicit basis behavior metadata~~
- ~~explicit answer contract metadata~~

## 4. What can wait until later

These are useful but should not block the main refactor.

- broader test overhaul
- multi-domain expansion beyond reserving
- durable workflow engine adoption
- durable audit completeness beyond lightweight local execution/proposal/basis records
- richer governance/approval model
- external integrations and operator automation
- public API completion for chat control-plane state and proposal actions

Audit deferral decision:

- first-class local records should still exist for execution outcomes, proposals, and accepted-basis transitions
- full audit guarantees can wait, including replayable command/event history, cross-system persistence, external approval integration, and complete `chat_id` lineage on every execution record
- this means near-term work should prioritize predictable behavior and clear state transitions over audit-grade durability

API completion deferral decision:

- the current in-process Dash flow can use `AIChatService` directly for `Yes` / `No` proposal acceptance
- public API endpoints should later expose the same control-plane state and actions for external clients and adapters
- later API work should include response fields for `accepted_analysis_basis`, `proposal_basis`, `preview_basis`, `execution_records`, and `basis_transition_history`
- later API work should include explicit accept/reject proposal endpoints, driven by `proposal_id`
- this is architecture/integration work, not a blocker for the current local UI proposal flow

## 5. Risks during implementation

Main risks:

- keeping old `analysis_basis` behavior alive indirectly through memory merging
- using display labels as execution identifiers
- leaving silent sanitization in adapters while changing only UI/state wording
- changing UI semantics without changing execution semantics
- partially implementing proposal logic while old basis auto-promotion still exists

Rule:

- basis auto-promotion must be removed before proposal UI is trusted

## 6. Practical implementation checklist

Before implementation starts, confirm the following are present in code or written down as specs.

- ~~typed schema for accepted basis~~
- ~~typed schema for proposal basis~~
- ~~typed schema for execution record~~
- ~~one canonical `basis_key` policy~~
- ~~proposal acceptance/rejection flow~~
- ~~explicit current-to-target module ownership plan~~
- ~~narration rules tied to execution status~~

If these are in place, the implementation can proceed without guessing.

## 7. Bottom line

`docs/ai-control-plane-refactor.md` is the right architecture anchor.

For successful implementation, it needs this additional context:

- concrete schemas
- identifier policy
- module ownership changes
- UI interaction contract
- execution classification rules
- implementation sequencing

That combination is enough to start a disciplined refactor without slipping back into small fix after small fix.
