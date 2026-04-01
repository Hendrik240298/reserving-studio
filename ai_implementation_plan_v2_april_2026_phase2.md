Remark: already implemended is ~~crossed out~~ 

Assumptions
- “Prior quarter” starts as a proxy view built by excluding the latest valuation diagonal, not as a true historical stored valuation.
- That proxy should be summarized into compact YAML, not persisted as a full cloned triangle/workflow state.
- House preferences are YAML-managed per segment for Phase 2.
- Session persistence should continue to follow the current ConfigManager.save_session_with_version(...) and segment-keyed YAML pattern.

## Milestone Order
1. ~~Foundation extraction and memory schema~~
2. ~~Decision engines v2~~
3. Quarter-close orchestration
4. Assistant/API/UI integration
5. Regression and release gating


## ~~Milestone 1: Foundation Extraction And Memory Schema~~
1. ~~Epic: Extract scenario evaluation and scoring out of reserving_adapter~~
~~Goal: stop growing source/api/adapters/reserving_adapter.py and create reusable domain services.~~
Tasks:
- ~~Extract scenario evaluation logic from _evaluate_scenario(...) into a dedicated service.~~
- ~~Extract governance, calibration, uncertainty aggregation, and evidence mapping into reusable units.~~
- ~~Replace the current simple scenario score formula with a dedicated scoring service interface, even if the first extracted version still uses existing weights.~~
- ~~Keep adapter behavior unchanged while moving logic behind service calls.~~
Files to add:
- ~~source/services/scenario_evaluation_service.py~~
- ~~source/services/scenario_scoring_service.py~~
Files to change:
- ~~source/api/adapters/reserving_adapter.py~~
- ~~source/services/__init__.py~~
- ~~source/api/schemas.py~~
Tests:
- ~~tests/unit/test_scenario_evaluation_service.py~~
- ~~tests/unit/test_scenario_scoring_service.py~~
- ~~update tests/unit/test_sprint3_calibration_governance_lineage.py~~
Exit criteria:
- ~~reserving_adapter.py delegates scenario evaluation/scoring instead of owning the logic directly.~~
- ~~Existing iteration and diagnostics tests still pass.~~
2. ~~Epic: Segment memory schema v2~~
~~Goal: turn current loose AI memory into versioned continuity memory.~~
Tasks:
- ~~Add schema versioning to per-segment AI memory YAML.~~
- ~~Define a compact v2 shape with house_preferences, known_issues, last_selection, last_human_decision, scenario_dispositions, and valuation_history.~~
- ~~Add migration rules from the current minimal memory format.~~
- ~~Add helpers for reading/writing structured memory without overwriting unrelated keys.~~
- ~~Add support for storing scenario signatures and valuation fingerprints.~~
Files to add:
- ~~source/services/segment_memory_service.py~~
Files to change:
- ~~ai/memory_store.py~~
- source/config_manager.py
- ~~ai/assistant_service.py~~
Tests:
- ~~tests/unit/test_segment_memory_service.py~~
- update tests/unit/test_ai_phase1_control_layer.py
Exit criteria:
- ~~Existing memory files still load.~~
- ~~New memory writes are schema-versioned and stable.~~
3. ~~Epic: Valuation snapshot proxy~~
~~Goal: create a deterministic “last quarter proxy” from current data by excluding the latest diagonal.~~
Tasks:
- ~~Build a compact valuation snapshot structure for current and proxy-prior views.~~
- ~~Implement a helper/service that derives a prior proxy by removing the most recent valuation diagonal.~~
- ~~Store only summary metrics and fingerprints in YAML, not full raw data.~~
- ~~Add comparison_basis metadata such as latest_diagonal_excluded_proxy.~~
Files to add:
- ~~source/services/valuation_snapshot_service.py~~
Files to change:
- ~~source/services/scenario_evaluation_service.py~~
- ~~source/services/segment_memory_service.py~~
- possibly source/reserving.py or source/triangle.py if a small helper is needed to rebuild from trimmed data
Tests:
- ~~tests/unit/test_valuation_snapshot_service.py~~
Exit criteria:
- ~~A service can produce current_snapshot and prior_proxy_snapshot deterministically from one session.~~

## ~~Milestone 2: Decision Engines V2~~
Remark: backend/domain implementation is complete; assistant/API composite tool integration remains in Milestone 4.

4. ~~Epic: Drop review engine v2~~
~~Goal: convert current drop heuristics into a deterministic review workflow.~~
Tasks:
- ~~Create a composite drop-review service using ranked link ratios, LDF consistency, late emergence, reserve change attribution, and scenario comparison.~~
- ~~Add component scoring for outlier support, consistency improvement, reserve impact, fragility, and governance penalties.~~
- ~~Add recommendation classes: recommend, reasonable_alternative, watch, avoid.~~
- ~~Add continuity penalties for previously rejected scenarios and house-preference conflicts.~~
Files to add:
- ~~source/services/assumption_review_service.py~~
Files to change:
- ~~source/services/scenario_scoring_service.py~~
- ~~source/api/adapters/reserving_adapter.py~~
- ~~source/api/schemas.py~~
Tests:
- ~~tests/unit/test_drop_review_service.py~~
Exit criteria:
- ~~Drop review produces a ranked result with score breakdown and recommendation class.~~
5. ~~Epic: Tail review engine v2~~
~~Goal: promote existing tail heuristics plus evaluate_tail_fit(...) into a real ranked tail review.~~
Tasks:
- ~~Build a composite tail review using current tail recommendation heuristics, tested tail-fit evaluation, continuity gap checks, sub-1.0 late factor checks, reserve impact, and instability signals.~~
- ~~Rank candidate curve/attachment/fit-period combinations.~~
- ~~Add explicit penalties for material attachment cuts and unstable tail scenarios.~~
- ~~Return a recommendation package with alternatives and caveats.~~
Files to change:
- ~~source/services/assumption_review_service.py~~
- ~~source/services/scenario_scoring_service.py~~
- ~~source/api/adapters/reserving_adapter.py~~
- ~~source/api/schemas.py~~
Tests:
- ~~tests/unit/test_tail_review_service.py~~
- ~~update tests/unit/test_diagnostics_service.py~~
- tests/unit/test_backend_tools_new_tools.py
Exit criteria:
- ~~Tail review returns ranked tested candidates, not only one heuristic recommendation.~~
6. ~~Epic: BF suitability v1~~
~~Goal: separate “BF apriori recommendation” from true BF suitability assessment.~~
Tasks:
- ~~Score BF suitability by maturity, volatility, CL sensitivity, percent reported reasonableness, and apriori readiness.~~
- ~~Return classification by UWY and overall: cl_preferred, bf_preferred, mixed, inconclusive.~~
- ~~Keep apriori recommendation logic separate from suitability classification.~~
Files to change:
- ~~source/services/assumption_review_service.py~~
- ~~source/services/scenario_scoring_service.py~~
- ~~source/api/schemas.py~~
- ~~source/api/adapters/reserving_adapter.py~~
Tests:
- ~~tests/unit/test_bf_suitability_service.py~~
Exit criteria:
- ~~BF review explains why BF is or is not suitable, not just what apriori to use.~~
7. ~~Epic: Data anomaly triage v2~~
~~Goal: strengthen anomaly triage from flagging into classification and next-step guidance.~~
Tasks:
- ~~Classify anomalies into types such as data quality, calendar distortion, large-loss contamination, sparse maturity, case reserve shift, and segment definition change.~~
- ~~Attach reserve relevance, next diagnostic, and pause_recommendation guidance.~~
- ~~Feed anomaly classification into scenario scoring penalties and reviewer gating.~~
Files to change:
- ~~source/services/assumption_review_service.py~~
- source/services/movement_diagnostics_service.py
- source/services/diagnostics_service.py
- ~~source/api/schemas.py~~
Tests:
- ~~tests/unit/test_anomaly_triage_service.py~~
Exit criteria:
- ~~Triage result is structured and actionable, not just a list of findings.~~

## Milestone 3: Quarter-Close Orchestration
8. Epic: Quarter-close workflow service
Goal: deliver the end-to-end plan-test-conclude workflow.
Tasks:
- Orchestrate current snapshot vs prior proxy snapshot comparison.
- Run anomaly triage, movement diagnostics, assumption retests, targeted scenarios, and ranking in one deterministic service.
- Use segment memory continuity checks before final recommendation.
- Return a structured workflow result, not narrative only.
Files to add:
- source/services/quarter_close_service.py
Files to change:
- source/services/valuation_snapshot_service.py
- source/services/assumption_review_service.py
- source/services/segment_memory_service.py
- source/api/adapters/reserving_adapter.py
- source/api/schemas.py
Tests:
- tests/unit/test_quarter_close_service.py
Exit criteria:
- One service call can produce a reproducible quarter-close review packet.
9. Epic: Quarter-close pack generator
Goal: generate a review-ready structured pack from deterministic results.
Tasks:
- Build output sections for data changes, assumptions retested, scenarios considered, recommended changes, caveats, judgment items, and sign-off questions.
- Include evidence ids, scenario ids, policy trace, and continuity notes.
- Add packet metadata for generated-at time, comparison basis, and memory version.
Files to change:
- source/services/quarter_close_service.py
- source/api/schemas.py
- optionally source/ai_review.py if you want a shared packet format
Tests:
- tests/unit/test_quarter_close_pack_generator.py
Exit criteria:
- Pack output is deterministic and exportable as JSON-compatible data.

## Milestone 4: Assistant, API, And UI Integration
10. Epic: Composite API endpoints and AI tools
Goal: expose Phase 2 review workflows as first-class tools instead of chaining many low-level calls in prompts.
Tasks:
- Add endpoints for drop review, tail review, BF suitability review, anomaly triage v2, and quarter-close pack.
- Add matching AI tool specs and summarized payload builders.
- Keep current drilldown tools for evidence follow-up.
Files to change:
- source/api/schemas.py
- source/api/main.py
- source/api/adapters/reserving_adapter.py
- ai/tool_payloads.py
- ai/api_tools.py
- ai/backend_tools.py
Tests:
- tests/unit/test_api_app.py
- tests/unit/test_backend_tools_new_tools.py
Exit criteria:
- Planner can invoke one composite tool per review instead of reconstructing the workflow through ad hoc tool chains.
11. Epic: Planner and policy integration
Goal: make the assistant use the new composite deterministic reviews.
Tasks:
- Extend playbooks to use composite Phase 2 tools.
- Add new playbook(s) for quarter-close and stronger tail/BF/drop reviews.
- Update reviewer logic to check continuity-memory coverage where relevant.
- Update recommendation policy to consume score breakdowns, continuity penalties, and house-preference checks.
Files to change:
- ai/planner.py
- ai/playbook_registry.py
- ai/reviewer.py
- ai/recommendation_policy.py
- ai/assistant_service.py
- ai/deterministic_packet.py
Tests:
- new tests/unit/test_ai_phase2_control_layer.py
- update tests/unit/test_ai_phase1_control_layer.py
Exit criteria:
- Deterministic packet includes score breakdown, continuity notes, and stronger recommendation rationale.
12. Epic: YAML-managed house preferences and human disposition writeback
Goal: turn human decisions into structured continuity memory.
Tasks:
- Extend saved human decision payload to include scenario signature, disposition, rationale, approver, signed-off time, and valuation fingerprint.
- Write accepted/rejected scenario dispositions into AI memory v2.
- Store YAML-managed house_preferences in the per-segment AI memory file.
- Make policy check rejected_before and house_preference_conflict.
Files to change:
- source/dashboard.py
- source/ai_review.py
- source/config_manager.py
- source/services/segment_memory_service.py
- ai/assistant_service.py
Tests:
- tests/unit/test_ai_review_service.py
- tests/unit/test_dashboard_ai_review.py
- tests/unit/test_segment_memory_service.py
Exit criteria:
- A saved approve/reject decision updates segment continuity memory in a reusable way.

## Milestone 5: Regression And Release Gating
13. Epic: Benchmark and regression harness
Goal: make Phase 2 measurable and stable.
Tasks:
- Add fixed benchmark fixtures for drop review, tail review, BF suitability, anomaly triage, unsupported requests, and quarter-close.
- Score playbook correctness, tool completeness, grounding, continuity use, and recommendation quality.
- Add regression tests for memory contradictions and rejected-before behavior.
Files to add:
- tests/unit/test_phase2_benchmarks.py
- tests/fixtures/phase2/ fixture files if needed
Files to change:
- tests/unit/test_ai_phase2_control_layer.py
- tests/unit/test_quarter_close_service.py
Exit criteria:
- Phase 2 behavior can be compared against a fixed baseline before release.
14. Epic: AI prompt/policy docs refresh
Goal: keep the markdown context aligned with the implemented control layer.
Tasks:
- Update playbook guidance for composite tools and quarter-close workflow.
- Update policy text to reflect continuity checks, evidence tiers, and YAML-managed preferences.
- Update README/tool documentation for new endpoints and limits.
Files to change:
- AI_PLAYBOOKS.md
- AI_POLICY.md
- AI_README.md
- optionally AI_CONTEXT.md
Exit criteria:
- The prompt corpus describes the actual implemented Phase 2 behavior.

## Recommended Build Sequence
1. Extract scenario evaluation/scoring services.
2. Add segment memory v2 and valuation snapshot proxy.
3. Build drop review v2.
4. Build tail review v2.
5. Build BF suitability and anomaly triage v2.
6. Build quarter-close service and pack.
7. Expose composite API/tools.
8. Update planner/reviewer/policy integration.
9. Add human decision writeback and preference checks.
10. Add benchmark/regression suite.
11. Refresh markdown policy/playbook docs.
Minimal First Cut
If you want the smallest high-value slice before full Phase 2, do this subset first:
1. Scenario evaluation extraction
2. Memory schema v2
3. Drop review v2
4. Tail review v2
5. Composite AI tools for those two reviews
That gives you the strongest improvement in recommendation quality without waiting for quarter-close automation.
One Small Design Recommendation
For the prior-quarter proxy, store this in YAML as summary-only:
{
  "valuation_date": "2026-03-31",
  "comparison_basis": "latest_diagonal_excluded_proxy",
  "data_fingerprint": "...",
  "summary": {
    "total_ultimate": 0,
    "total_ibnr": 0,
    "governance_tier": "amber"
  }
}
Not full raw triangles. That keeps the memory simple and close to your current session-YAML philosophy.
