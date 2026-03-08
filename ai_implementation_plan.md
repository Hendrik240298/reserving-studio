# AI Implementation Plan

This document is the cornerstone plan for introducing AI capabilities into this project while preserving strict separation of concerns:

- The reserving backend remains deterministic and independent from AI implementation details.
- The GUI remains independent from AI implementation details.
- The AI layer interacts with reserving functionality only through stable API contracts.

This plan aligns with `Plan.md` and extends it with a production-oriented AI architecture focused on reserving diagnostics, commentary, and workflow acceleration.

## Status snapshot (Mar 2026)

### Current delivery plan (updated 2026-03-08)

This section supersedes earlier "immediate next actions" ordering and is the current implementation sequence.

#### What we will build next

- Move from finding-only diagnostics to an evidence-object + governance workflow model.
- Harden high-risk diagnostics first (portfolio-shift semantics, negative development triage, scenario score explainability).
- Add deterministic narrative conflict/language guardrails before expanding AI autonomy.
- Add uncertainty services after governance hardening (MSEP, bootstrap, tail uncertainty), then strategic modeling depth.

#### Sprint plan

1. **Sprint 1 (completed 2026-03-08): Foundation and schema migration**
   - Add Evidence V2 schema fields with backward compatibility.
   - Add immutable run metadata and reproducibility payloads (diagnostic version, scenario generator version, data fingerprint, run timestamp/id).
   - Keep v1 response compatibility while exposing the new metadata.
   - Add/update tests for schema and metadata presence.

2. **Sprint 2 (completed 2026-03-08): P0 diagnostic hardening**
   - Portfolio-shift corroboration + guarded language downgrade when evidence is mixed.
   - Negative-development triage with diagonal clustering and escalation flags.
   - Severity score decomposition for scenario explainability and auditability.
   - Deterministic conflict checks + narrative language gating.

3. **Sprint 3 (current): Calibration and governance wiring**
   - Backtest-driven threshold calibration by segment/maturity regime.
   - Governance tiering (green/amber/red) and deterministic escalation triggers.
   - Scenario lineage (parent, transform, rationale evidence IDs).

4. **Sprint 4: Uncertainty layer**
   - CL Mack MSEP and BF prediction error baseline service.
   - Bootstrap predictive distributions for scenario robustness.
   - Tail model averaging and instability flags.

5. **Sprint 5: Decision UX and reporting**
   - Scenario matrix, evidence trace, and conflict panel in GUI.
   - Structured override/sign-off capture and exportable decision packet.

#### Recommended execution policy

- Complete Sprints 1-3 before broadening stochastic complexity.
- Keep deterministic services as source of truth; AI remains synthesis and workflow support.
- Preserve contract-first changes with optional fields and compatibility defaults.

Implemented:

- API scaffolding and contracts in `source/api/` with deterministic backend adapter.
- AI tool-calling assistant in `ai/` using OpenRouter by default (`minimax/minimax-m2.5`).
- Deterministic diagnostics engine with maturity-aware, link-ratio, loss-ratio, latest-diagonal, portfolio-shift, tail, backtest, calendar-drift, paid/incurred, and data-quality checks.
- Iterative diagnostics endpoint (`/v1/diagnostics/iterate`) for scenario search over drops, tail settings, and BF apriori.
- Observability-by-default for AI tool calls and API scenario execution traces.

Still in-progress:

- Production hardening of threshold calibration by segment/line.
- Broader actuarial validation/backtesting governance and report UX integration.

Recently completed (2026-03-08):

- Evidence V2 schema extension with backward-compatible API responses.
- Run-level reproducibility metadata in diagnostics and scenario evaluation outputs.
- Portfolio-shift corroboration/guardrail diagnostics and unconfirmed-signal handling.
- Negative-development triage diagnostic with materiality and diagonal clustering signals.
- Scenario severity decomposition (`data_quality`, `stability`, `backtest`, `coherence`, `tail`, `other`) and governance tiering.
- Assistant-side deterministic narrative guardrails for coherence conflicts and causal shift language softening.
- Unit test coverage for Sprint 1 and Sprint 2 deliverables.

## 1) Vision and non-goals

### Vision

Build an AI Reserving Assistant that delivers high-value actuarial support by:

- Running deterministic triangle diagnostics.
- Detecting anomalies and instability patterns.
- Comparing reserving methods and assumptions.
- Producing evidence-grounded commentary for reports.
- Supporting collaboration via session-based outputs.

### Product principle

The assistant does **not** replace reserving logic. It improves speed, consistency, and explainability of actuarial review.

### Non-goals

- No hidden or autonomous changes to actuarial assumptions.
- No direct AI access to internal files, chainladder classes, or data stores.
- No coupling of AI provider SDKs into core reserving or GUI modules.

## 2) Core architecture decisions

### Decision A: Hard separation between domain, API, and AI

We enforce 4 layers:

1. **Domain layer (existing)**: reserving and data logic in `source/` (deterministic).
2. **Service/API layer (new)**: typed HTTP contracts that expose domain capabilities.
3. **GUI layer (existing/evolving)**: Dash app calls service/API, not AI internals.
4. **AI layer (new)**: external orchestrator that calls approved API tools only.

### Decision B: Contract-first integration

All integration between GUI, AI, and backend uses versioned request/response schemas.

### Decision C: Deterministic-first diagnostics

Diagnostics are computed by deterministic code in backend services. LLM usage is limited to synthesis, explanations, and narrative formatting.

### Decision D: Provider abstraction with OpenRouter

The AI layer must support OpenRouter as the default model gateway while remaining provider-agnostic.

- Use environment variables for model endpoint and key.
- Keep adapter interface generic (`chat_with_tools`, `structured_output`).
- Avoid vendor-specific logic in application business code.

## 3) Target system topology

###+ Runtime components

- `reserving-core` (current): chainladder wrappers + project logic.
- `reserving-api` (new): FastAPI app exposing deterministic capabilities.
- `dash-gui` (current): app interface and callbacks.
- `ai-assistant` (new): tool-calling orchestrator and commentary engine.
- `eval-and-observability` (new): traces, audit records, quality scoring.

### Interaction model

- GUI <-> API
- AI <-> API
- AI does **not** call core code directly.
- GUI does **not** depend on AI package imports.

## 4) API surface (v1)

These endpoints are the only supported execution boundary for AI and GUI.

### `POST /v1/workflows/from-dataframes`

Purpose: initialize session/workflow from claims + premium payloads.

### `GET /v1/sessions/{segment}`

Purpose: load active session metadata and parameters.

### `POST /v1/sessions/{segment}/save`

Purpose: persist session state via `ConfigManager` and sync services.

### `POST /v1/reserving/recalculate`

Purpose: run reserve recalculation using selected assumptions.

### `POST /v1/diagnostics/run`

Purpose: execute deterministic diagnostics and return structured findings.

### `POST /v1/diagnostics/iterate`

Purpose: run iterative scenario diagnostics and return ranked scenario evaluations with iteration metrics.

### `GET /v1/results/{session_id}`

Purpose: fetch latest computed outputs and assumptions snapshot.

## 5) Canonical schema strategy

Create a shared schema module with strict typing (Pydantic models) for:

- Session metadata
- Recalculation request/response
- Diagnostics findings
- Results payload references

Every schema should include:

- `schema_version`
- stable IDs (`session_id`, `finding_code`, `metric_id`)
- explicit numeric typing (`float` fields, nullable where needed)

### Standard diagnostics finding shape

Each finding must include:

- `code` (stable machine identifier)
- `severity` (`low|medium|high|critical`)
- `message` (human-readable statement)
- `evidence` (metric, threshold, value, basis)
- `suggested_actions` (bounded actionable list)

## 6) AI tool contract and permissions

The AI assistant receives a strict allowlist of tools that map to backend API endpoints:

- `tool_get_session(segment)`
- `tool_recalculate(params)`
- `tool_run_diagnostics(session_id, profile, include_recommendations)`
- `tool_iterate_diagnostics(session_id, max_scenarios, include_baseline)`
- `tool_get_results(session_id)`

Session save remains API-supported but is not currently exposed in the default AI tool allowlist.

Rules:

- Read-only by default.
- Writes require explicit action type and sync checks.
- No filesystem, shell, or unrestricted HTTP tools exposed by default.
- Tool arguments must be schema-validated before execution.

## 7) Guardrails and safety model

### Policy defaults

- AI outputs are advisory.
- AI cannot silently change reserving assumptions.
- Every commentary claim must cite evidence IDs from diagnostics/results.
- Missing evidence -> explicit uncertainty statement.

### Execution constraints

- Max tool calls per request (e.g., 8).
- Max tool-call loop steps per request: 14 (current prototype default).
- Max orchestration runtime budget (e.g., 20 seconds).
- Request timeout and retry strategy at API boundary.
- Circuit-break behavior for repeated endpoint failures.

### Mutating operations

- Use optimistic concurrency (`expected_sync_version`).
- Use idempotency keys for writes.
- Log who/what triggered mutation and resulting version.

## 8) OpenRouter compatibility plan

OpenRouter is fully compatible with this approach because provider details are isolated in the AI adapter.

### Environment variables

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` (if custom)
- `AI_MODEL`
- `AI_FALLBACK_MODEL` (optional)

### Adapter interface

Define a provider-neutral client interface:

- `generate_structured(prompt, schema)`
- `chat_with_tools(messages, tools, limits)`

The adapter implementation can use OpenRouter-compatible endpoints and models while preserving the same internal contract.

### Provider failover (optional)

Design for fallback to another provider later by swapping adapter implementation only.

## 9) Implementation phases

### Phase 0 - Foundations and alignment

Deliverables:

- Finalize this plan and acceptance criteria.
- Define API and schema contracts (`v1`).
- Define diagnostic rules catalog and thresholds.
- Decide service deployment mode (single process vs separate API service).

Exit criteria:

- Signed off contract spec.
- Named owners for API, diagnostics, AI, and QA.

Status: complete.

### Phase 1 - Reserving API extraction

Deliverables:

- Add `source/api/` with FastAPI app and routers.
- Add schema models for all v1 endpoints.
- Wire API handlers to existing services (`reserving_service`, `params_service`, `session_sync_service`).
- Add endpoint tests with deterministic fixtures.

Exit criteria:

- GUI can complete core flows through API contracts.
- No AI dependencies in domain or GUI code.

Status: complete for prototype scope.

### Phase 2 - Deterministic diagnostics engine

Deliverables:

- Add `source/services/diagnostics_service.py`.
- Implement first rule set:
  - development instability
  - AY anomaly detection
  - CL vs BF divergence
  - tail sensitivity checks
- Return strict `DiagnosticFinding` outputs.
- Add unit tests and golden-case fixtures.

Exit criteria:

- Diagnostics endpoint stable and covered by tests.
- Findings reproducible for same inputs.

Status: complete for V1/V2 rule set, with ongoing calibration.

### Phase 3 - AI orchestrator (guardrailed)

Deliverables:

- Create `ai/assistant_service.py` as separate module/process.
- Implement tool allowlist and schema validation.
- Implement bounded tool loop with execution limits.
- Implement evidence citation enforcement in generated commentary.
- Add OpenRouter adapter implementation.

Exit criteria:

- Assistant can run diagnostics and produce evidence-grounded commentary.
- No direct imports from AI into GUI/core layers.

Status: complete for prototype scope.

### Phase 4 - UX integration and report outputs

Deliverables:

- Add GUI entry points for:
  - "Run Diagnostics"
  - "Generate Commentary"
  - "Export report"
- Persist AI outputs in session under explicit keys:
  - `ai_findings`
  - `ai_commentary`
  - `ai_evidence_refs`
  - `ai_model_meta`
- Add export templates (markdown first, PDF optional).

Exit criteria:

- Users can run end-to-end diagnostics + commentary workflow from GUI.

Status: partially complete (CLI + API flow complete; GUI integration pending).

### Phase 5 - Observability, evals, and hardening

Deliverables:

- Trace all AI/tool/API calls (request IDs, latency, model, token usage).
- Add offline eval suite for commentary quality and factual grounding.
- Add production dashboards for error rate, latency, and drift signals.
- Add red-team tests for prompt injection and tool misuse attempts.

Exit criteria:

- Defined SLOs are met.
- Quality and safety gates pass consistently.

Status: partially complete (observability implemented; eval/SLO hardening pending).

## 10) Testing strategy

### Backend/API tests

- Unit tests for each schema and rule.
- Integration tests per endpoint.
- Contract tests to prevent schema regressions.

### AI integration tests

- Tool-call correctness tests.
- Citation completeness tests (every claim references evidence).
- Bounded autonomy tests (step/time/tool limits enforced).
- Failure mode tests (API timeout, invalid tool args, missing data).

### E2E tests

- GUI -> API -> diagnostics -> commentary -> export.
- Persist/load session including AI artifacts.

## 11) Observability and audit requirements

Every AI-assisted request should log:

- `request_id`, `session_id`, `user_id` (if available)
- model/provider metadata
- tool calls (name, args hash, duration, status)
- endpoint calls (path, status, latency)
- safety policy decisions (blocked/approved)
- output evidence references

Prefer OpenTelemetry-compatible spans where possible for future portability.

## 12) Data governance and compliance posture

- Minimize payloads sent to model provider (only needed aggregates/findings).
- Avoid sending raw sensitive rows if not required for task.
- Document retention policy for prompts, tool traces, and outputs.
- Provide clear user-facing labeling: AI-generated commentary + evidence links.

## 13) Risks and mitigations

### Risk: Hallucinated commentary

Mitigation:

- Deterministic diagnostics first, LLM synthesis second.
- Enforce citation policy and schema checks.

### Risk: API contract drift

Mitigation:

- Versioned schemas and contract tests in CI.

### Risk: Tool misuse or excessive autonomy

Mitigation:

- Strict allowlist, step budgets, and blocked dangerous actions.

### Risk: Provider outages or model instability

Mitigation:

- Retry logic, fallback model, graceful degradation to deterministic-only output.

## 14) Delivery milestones and acceptance criteria

### Milestone M1: API boundary complete

- All core GUI workflows supported via v1 API.
- No AI imports in `source/` core/GUI modules.

### Milestone M2: Diagnostics production-ready

- Deterministic diagnostics endpoint available and tested.
- Findings include evidence and action suggestions.

### Milestone M3: AI assistant beta

- OpenRouter-backed assistant calls tools safely.
- Commentary generated with full evidence references.

### Milestone M4: Report and team workflow

- Exportable diagnostics/commentary package.
- Session persistence includes AI artifacts for sharing/review.

### Milestone M5: Hardening complete

- Evals, tracing, and safety checks operational.
- SLO and quality thresholds met.

## 15) Definition of done (overall)

The AI implementation is considered complete when:

- Architecture separation is enforced (core + GUI independent of AI internals).
- AI uses backend only via stable API contracts.
- Deterministic diagnostics power all key AI insights.
- Commentary is evidence-grounded, auditable, and session-persisted.
- OpenRouter integration is operational through a provider-neutral adapter.
- Tests, evals, and observability prove reliable operation.

## 16) Immediate next actions

1. Calibrate diagnostics thresholds by segment and maturity cohorts using backtesting.
2. Add richer data-quality triage outputs (top offending cells and cause classification).
3. Expand iterative scenario design (wider tail-fit grids and method-comparison constraints).
4. Add structured report payloads for direct GUI rendering/export.
5. Add offline evaluation suite for recommendation quality and factual grounding.
6. Implement GUI hooks for scenario comparison and evidence drill-down.
