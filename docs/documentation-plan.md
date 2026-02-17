# Documentation Plan (Actuary-First)

## Goal

Build practical project documentation for actuarial users first, with clear steps for running the reserving workflow, adjusting methods in the UI, and exporting finalized outputs back into scripts.

## Primary audience

- Actuaries and reserving analysts using Python locally.
- Secondary audience: developers extending data adapters and UI behavior.

## Documentation principles

- Task-first over architecture-first.
- Show copy/paste commands and expected outcomes.
- Explain actuarial implications of controls (drops, tail, BF apriori, selected method).
- Keep source-code internals minimal unless needed for troubleshooting.

## Proposed structure

1. `README.md` (entry point)
   - install, run, 2-minute quickstart
   - links to detailed docs
2. `docs/actuary-quickstart.md`
   - fastest path from clone to working UI
3. `docs/actuary-workflow.md`
   - practical workflow from input to finalized output
4. `docs/config-practical-reference.md`
   - YAML options and examples for quarterly, CLRD, SQL
5. `docs/practical-notes.md`
   - common pitfalls, troubleshooting, performance notes
6. Existing `docs/cross-tab-sync.md`
   - technical detail for multi-tab behavior

## Delivery phases

### Phase 1 (now)

- Publish core actuary docs (quickstart + workflow + config + practical notes).
- Link docs from `README.md`.

### Phase 2

- Add recipe-style pages for common real-world scenarios:
  - mixed claims feeds, column mapping, and period coercion issues
  - yearly vs quarterly impact examples
- Add screenshots that mirror current UI tabs.

### Phase 3

- Add a short analyst-to-production handoff checklist:
  - validated inputs
  - session capture
  - reproducible run metadata
  - test evidence references

## Definition of done for v1

- A first-time actuarial user can run the app in under 10 minutes.
- A user can complete one reserving cycle (set parameters, review results, finalize) without reading source code.
- Common data and config errors have a direct fix path.
