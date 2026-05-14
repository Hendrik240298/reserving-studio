Remark: already implemended is ~~crossed out~~ 

Assumptions
- Phase 2 remains the completed foundation; Phase 3 builds on top of the existing memory schema, composite reviews, quarter-close scaffolding, uncertainty primitives, and benchmark/test base.
- Phase 3 is not the place for true prior-quarter replay. Real prior/current valuation continuity remains Phase 4 work.
- The sidebar memory authoring UI should write into the existing segment-memory YAML shape rather than inventing a second persistence model.
- For Milestone 1, the user-facing memory authoring and proposal-approval UI should live in `source/ai_dashboard.py` only, not the main reserving workspace in `source/dashboard.py`.
- AI-proposed memory updates must always require explicit human approval before persistence.
- Uncertainty outputs remain secondary decision support rather than the booking basis.
- Retrieval should begin as a curated hybrid over markdown and selected internal docs, not as a broad ungoverned search layer.
- Release gating should reuse the existing benchmark/test posture and extend it rather than replacing it.

## Milestone Order
1. Memory authoring UI and persistence hardening
2. Uncertainty overlays and explanation integration
3. Industrial hypothesis engine
4. Retrieval hybrid and assistant grounding upgrades
5. Release gating dashboard and regression expansion


## Milestone 1: Memory Authoring UI And Persistence Hardening
Remark: the structured memory schema already exists; this milestone turns it into a practical user-facing editing and approval workflow.

1. Epic: ~~Structured segment memory authoring surface~~
Goal: expose the existing segment memory as editable workspace fields instead of YAML-only infrastructure.
Tasks:
- ~~Add editable sidebar fields in `source/ai_dashboard.py` for `segment_overview`, `known_issues`, `house_preferences`, `recent_quarter_notes`, and `open_items`.~~
- ~~Define how each field maps into the existing segment memory payload without breaking backward compatibility.~~
- ~~Keep the UI compact, segment-scoped, and confined to the AI dashboard rather than introducing a separate document editor or modifying the main reserving workspace.~~
- ~~Load saved values on workspace startup and refresh them when the segment changes.~~
Files to add:
- optionally `source/services/memory_authoring_service.py`
Files to change:
- `source/config_manager.py`
- `source/services/segment_memory_service.py`
- `source/ai_dashboard.py`
Tests:
- `tests/unit/test_segment_memory_service.py`
- new `tests/unit/test_memory_authoring_service.py` if a new service is added
- new `tests/unit/test_ai_dashboard_memory_sidebar.py`
Exit criteria:
- ~~A user can view and edit structured segment memory from the AI dashboard without touching YAML directly.~~
- ~~Saved memory remains schema-valid and segment-scoped.~~

2. Epic: ~~AI-proposed memory updates with human approval~~
Goal: let the assistant suggest memory updates without silently rewriting institutional context.
Tasks:
- ~~Define a structured proposal format for AI-suggested memory updates.~~
- ~~Add assistant output support for memory-update suggestions tied to specific fields.~~
- ~~Add UI controls in `source/ai_dashboard.py` to review, accept, reject, or edit the proposed update before save.~~
- ~~Record approved changes in a simple audit-friendly way such as updated timestamp and source.~~
Files to add:
- optionally `ai/memory_update_policy.py`
Files to change:
- `ai/assistant_service.py`
- `ai/deterministic_packet.py`
- `ai/reviewer.py`
- `source/ai_review.py`
- `source/ai_dashboard.py`
- `source/services/segment_memory_service.py`
Tests:
- new `tests/unit/test_ai_memory_update_proposals.py`
- `tests/unit/test_ai_dashboard.py`
- new `tests/unit/test_ai_dashboard_memory_proposals.py`
Exit criteria:
- ~~The assistant can propose a memory update.~~
- ~~No proposed memory update is persisted without an explicit human action.~~

3. Epic: ~~Segment memory read-path expansion for assistant context~~
Goal: make the richer memory fields reliably available to the assistant and deterministic summaries.
Tasks:
- ~~Extend memory normalization to include the new authoring fields.~~
- ~~Surface those fields in compact assistant-readable context packets built from the canonical stored memory.~~
- ~~Render each memory field with both its stored content and a short field-definition line so the model can interpret the role of each memory block correctly.~~
- ~~Keep the full stored values for `segment_overview`, `known_issues`, `house_preferences`, and `open_items` in assistant context unless they become genuinely too large in practice.~~
- ~~Bound `recent_quarter_notes` to the latest four entries in assistant context to keep the prompt efficient while preserving roughly one year of continuity.~~
- ~~Keep these memory fields soft/contextual in prompt rendering rather than automatically treating free-text memory as deterministic policy.~~
- ~~Keep continuity checks and recommendation logic stable when richer memory is present.~~
Files to change:
- `source/services/segment_memory_service.py`
- `source/services/memory_authoring_service.py`
- `ai/memory_store.py`
- `ai/assistant_service.py`
- `ai/context_loader.py`
- `AI_README.md`
- optionally `AI_CONTEXT.md`
Tests:
- `tests/unit/test_segment_memory_service.py`
- `tests/unit/test_ai_phase2_control_layer.py`
- new `tests/unit/test_ai_memory_context_rendering.py`
Exit criteria:
- ~~The assistant uses the richer memory fields as context without losing deterministic control-layer behavior.~~

## Milestone 2: Uncertainty Overlays And Explanation Integration
Remark: uncertainty primitives already exist; this milestone makes them more usable, comparative, and explainable.

4. Epic: Scenario-level uncertainty overlays
Goal: expose uncertainty as a comparative overlay for scenarios rather than as a detached metric blob.
Tasks:
- Extend uncertainty outputs to support scenario-to-scenario comparison and relative fragility interpretation.
- Add compact scenario uncertainty summaries to diagnostics, iteration, and reserve-change explanation outputs where helpful.
- Define consistent labels for stability, fragility, and volatility flags so they can be explained cleanly in UI and AI responses.
- Keep the outputs explicitly secondary to deterministic recommendation logic.
Files to add:
- optionally `source/services/uncertainty_overlay_service.py`
Files to change:
- `source/services/uncertainty_service.py`
- `source/services/scenario_evaluation_service.py`
- `source/api/adapters/reserving_adapter.py`
- `source/api/schemas.py`
- `ai/tool_payloads.py`
- `ai/api_tools.py`
- `ai/backend_tools.py`
Tests:
- `tests/unit/test_uncertainty_service.py`
- `tests/unit/test_api_uncertainty_endpoints.py`
- new `tests/unit/test_uncertainty_overlay_service.py`
- update `tests/unit/test_sprint3_calibration_governance_lineage.py`
Exit criteria:
- A scenario review can say not only what changed, but also whether the candidate appears materially more fragile than the current analysis basis.

5. Epic: Uncertainty-aware explanation layer
Goal: make uncertainty understandable and usable in AI and UI explanations.
Tasks:
- Add plain-language explanation templates for predictive ranges, tail instability, and relative volatility flags.
- Include uncertainty interpretation in deterministic packets and recommendation summaries when available.
- Ensure the assistant distinguishes observed deterministic evidence from optional uncertainty overlays.
- Add UI presentation for the most important uncertainty flags without overwhelming the primary results view.
Files to change:
- `ai/assistant_service.py`
- `ai/reviewer.py`
- `ai/recommendation_policy.py`
- `ai/tool_payloads.py`
- `source/dashboard.py`
- `source/ai_dashboard.py`
- `AI_PLAYBOOKS.md`
- `AI_POLICY.md`
Tests:
- new `tests/unit/test_ai_uncertainty_explanations.py`
- `tests/unit/test_ai_phase2_control_layer.py`
- new `tests/unit/test_dashboard_uncertainty_panels.py`
Exit criteria:
- Uncertainty is explained in plain English and clearly labeled as secondary support rather than selection authority.

## Milestone 3: Industrial Hypothesis Engine
Remark: this milestone should improve domain usefulness without turning the assistant into an ungrounded free-form storyteller.

6. Epic: Hypothesis catalog and evidence mapping
Goal: define a deterministic hypothesis layer for common industrial reserving explanations.
Tasks:
- Define supported hypotheses such as large-loss emergence, claims handling change, wording/attachment shift, portfolio mix change, settlement speed change, case reserve strengthening/weakening, and reporting lag shift.
- Map each hypothesis to the diagnostics, movement signals, and scenario evidence that can support or weaken it.
- Add strength-of-support labels and alternative-explanation handling.
- Keep hypotheses grounded in observed signals and tool outputs.
Files to add:
- `source/services/hypothesis_engine_service.py`
- optionally `source/services/hypothesis_catalog.py`
Files to change:
- `source/services/movement_diagnostics_service.py`
- `source/services/diagnostics_service.py`
- `source/services/assumption_review_service.py`
- `source/api/schemas.py`
- `source/api/adapters/reserving_adapter.py`
Tests:
- new `tests/unit/test_hypothesis_engine_service.py`
- update `tests/unit/test_data_view_and_movement_services.py`
- update `tests/unit/test_diagnostics_service.py`
Exit criteria:
- The backend can return a structured list of supported hypotheses with evidence and support strength rather than only raw findings.

7. Epic: Hypothesis-aware assistant workflow
Goal: teach the assistant when and how to use the hypothesis layer.
Tasks:
- Add or extend playbooks so the assistant can request hypothesis analysis when the user asks for causal explanation or change drivers.
- Add reviewer checks to prevent unsupported causal claims.
- Make the final assistant answer separate observed signal, possible explanation, evidence run, and residual uncertainty.
- Keep the hypothesis layer optional so simple review tasks do not become overcomplicated.
Files to change:
- `ai/planner.py`
- `ai/playbook_registry.py`
- `ai/reviewer.py`
- `ai/assistant_service.py`
- `ai/deterministic_packet.py`
- `AI_PLAYBOOKS.md`
- `AI_EXAMPLES.md`
Tests:
- new `tests/unit/test_ai_hypothesis_playbooks.py`
- `tests/unit/test_assistant_service_iteration_policy.py`
- new `tests/unit/test_phase3_hypothesis_benchmarks.py`
Exit criteria:
- The assistant can produce grounded hypothesis-style explanations without inventing unsupported causes.

## Milestone 4: Retrieval Hybrid And Assistant Grounding Upgrades
Remark: Phase 3 retrieval should stay narrow, curated, and citation-friendly.

8. Epic: Curated retrieval layer
Goal: add retrieval only where curated docs improve decisions over static markdown context alone.
Tasks:
- Define the first retrieval corpus, likely selected markdown notes, methodology documents, and curated prior review notes.
- Add ingestion/indexing for a bounded set of approved documents with stable identifiers.
- Add retrieval metadata such as source path, title, snippet, and freshness/version information.
- Keep retrieval results inspectable and bounded in number.
Files to add:
- `ai/retrieval_store.py`
- `ai/retrieval_service.py`
- optionally `ai/retrieval_indexer.py`
Files to change:
- `ai/context_loader.py`
- `ai/assistant_service.py`
- `AI_README.md`
- `AI_CONTEXT.md`
Tests:
- new `tests/unit/test_retrieval_service.py`
- new `tests/unit/test_retrieval_indexer.py`
- new `tests/unit/test_ai_retrieval_context.py`
Exit criteria:
- The assistant can retrieve from a curated corpus and include source-aware context without broadening into uncontrolled search.

9. Epic: Retrieval-aware policy and citation discipline
Goal: keep retrieval useful and safe rather than noisy.
Tasks:
- Add rules for when retrieval is allowed and when deterministic tool outputs must remain primary.
- Require assistant answers to distinguish tool-grounded reserving evidence from retrieved policy/methodology context.
- Add citation formatting or source naming that is readable to actuarial users.
- Add reviewer checks to detect unsupported retrieval-only numeric claims.
Files to change:
- `ai/reviewer.py`
- `ai/recommendation_policy.py`
- `ai/assistant_service.py`
- `AI_POLICY.md`
- `AI_PLAYBOOKS.md`
- `AI_EXAMPLES.md`
Tests:
- new `tests/unit/test_ai_retrieval_policy.py`
- new `tests/unit/test_phase3_retrieval_benchmarks.py`
Exit criteria:
- Retrieval enriches policy and continuity context without displacing deterministic reserving tools as the source of truth.

## Milestone 5: Release Gating Dashboard And Regression Expansion
Remark: this milestone makes Phase 3 measurable and safe to evolve.

10. Epic: Phase 3 benchmark and regression expansion
Goal: extend the benchmark harness to cover new memory UI, uncertainty, hypothesis, and retrieval behavior.
Tasks:
- Add benchmark fixtures for memory-aware continuity behavior, uncertainty-aware recommendation language, and grounded hypothesis explanations.
- Add regression coverage that locks in stable review-candidate identity across reruns so follow-up questions do not rebound to recycled display labels such as `drop_3`.
- Add regression coverage that locks in explicit tail-state semantics so assistant and UI behavior continue to distinguish `tail_active=false` reference-only fits from truly attached tail selections.
- Add retrieval benchmarks that test citation discipline and refusal of unsupported retrieval-only conclusions.
- Add regression tests for memory proposal approval flow and stale-context contradictions.
- Extend the benchmark scoring rubric where necessary while preserving comparability with the Phase 2 baseline.
Files to add:
- `tests/unit/test_phase3_benchmarks.py`
- `tests/fixtures/phase3/` fixture files if needed
Files to change:
- `tests/unit/test_phase2_benchmarks.py`
- `tests/unit/test_ai_phase2_control_layer.py`
- `tests/unit/test_ai_dashboard.py`
Exit criteria:
- Phase 3 behavior is measurable against fixed scenarios before release.
- Scenario identity and tail-state interpretation remain stable under iterative review reruns and exact numeric follow-up prompts.

11. Epic: Release gating dashboard
Goal: make benchmark and regression health visible before shipping changes.
Tasks:
- Define a compact dashboard or report view for key benchmark categories, failures, and drift indicators.
- Surface release-gate status for AI behavior regressions in a developer-facing view.
- Include benchmark timestamps, model/config identifiers, and relevant test summaries.
- Keep the first version simple and local rather than building a large observability platform.
Files to add:
- optionally `source/services/release_gate_service.py`
Files to change:
- `source/dashboard.py`
- `source/ai_dashboard.py`
- `ai/assistant_service.py`
- `AI_README.md`
Tests:
- new `tests/unit/test_release_gate_service.py`
- new `tests/unit/test_ai_release_gate_dashboard.py`
Exit criteria:
- A developer can see whether the current AI build is within acceptable benchmark and regression limits before release.

12. Epic: Prompt and documentation refresh for Phase 3
Goal: keep the markdown guidance aligned with the actual implementation.
Tasks:
- Update AI docs for editable memory authoring, uncertainty overlays, retrieval limits, and hypothesis behavior.
- Refresh playbook examples so they reflect the new Phase 3 workflows.
- Update any public-facing capability notes to distinguish Phase 3 features from future Phase 4 quarter-close continuity work.
Files to change:
- `AI_README.md`
- `AI_PLAYBOOKS.md`
- `AI_POLICY.md`
- `AI_EXAMPLES.md`
- `whitepaper/revision.md`
Exit criteria:
- The prompt corpus and capability docs describe the actual implemented Phase 3 behavior and its boundaries.

## Recommended Build Sequence
1. Build the editable memory authoring UI over the existing schema.
2. Add AI-proposed memory updates with human approval.
3. Extend the memory read path so the assistant can use the richer fields.
4. Add scenario-level uncertainty overlays.
5. Add uncertainty-aware explanation support in UI and assistant outputs.
6. Build the industrial hypothesis engine backend.
7. Wire the hypothesis engine into assistant playbooks and reviewer logic.
8. Add the curated retrieval layer.
9. Add retrieval-aware policy and citation discipline.
10. Expand benchmarks and regression tests for Phase 3.
11. Add the release gating dashboard/report.
12. Refresh AI docs and policy/playbook guidance.

## Minimal First Cut
If you want the smallest high-value slice before the full Phase 3 package, do this subset first:
1. Editable sidebar memory fields over the existing memory schema
2. AI-proposed memory updates with explicit human approval
3. Scenario uncertainty comparison overlays
4. Plain-language uncertainty explanation in assistant responses
5. Benchmark coverage for those behaviors

That gives you the strongest product-level upgrade before taking on retrieval and hypothesis work.

## One Small Design Recommendation
For the editable memory authoring surface, keep the stored structure compact and field-based rather than free-form documents. A good first cut is:

```json
{
  "segment_overview": "Short persistent description of the segment.",
  "known_issues": [
    "Large refinery loss still affects 2021 comparability"
  ],
  "house_preferences": [
    "Prefer stable tail over slightly lower reserve"
  ],
  "recent_quarter_notes": [
    {
      "period": "2026Q1",
      "note": "Case reserve strengthening after claims review"
    }
  ],
  "open_items": [
    "Review whether TPA change has shifted reporting lag"
  ]
}
```

That remains readable in YAML, easy to summarize for AI context, and compatible with later Phase 4 quarter-close continuity.

Implementation note:
- Store the full canonical values in segment memory.
- Render a compact assistant context packet from that stored memory rather than appending raw YAML or long note blobs directly into prompts.
- In prompt rendering, pair each field's content with a short description of what that field means so the AI can use the memory more consistently.
- For v1 context rendering, keep full `segment_overview`, `known_issues`, `house_preferences`, and `open_items`, while limiting `recent_quarter_notes` to the latest four entries.
