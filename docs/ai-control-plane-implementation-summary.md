# AI Control Plane Implementation Summary

This summary records the implementation status of the AI control-plane refactor defined in:

- `docs/ai-control-plane-refactor.md`
- `docs/ai-control-plane-implementation-plan.md`

## Status

The planned control-plane refactor has been implemented through the five execution steps in `docs/ai-control-plane-implementation-plan.md` Section 3:

- Step 1: basis/proposal state split
- Step 2: proposal UI flow
- Step 3: execution record pipeline
- Step 4: identifier simplification
- Step 5: workflow formalization

## Delivered

- accepted analysis basis is now separate from proposal and preview state
- recommendations no longer auto-promote into accepted basis
- proposal acceptance/rejection is explicit and chat-local
- execution records are first-class and visible in the UI trace
- `basis_key` is the stable basis identity used for control-plane resolution
- workflow definitions now provide the structured source of truth for workflow behavior

## Main Modules Added Or Formalized

- `ai/control_plane_types.py`
- `ai/basis_manager.py`
- `ai/proposal_manager.py`
- `ai/request_validation.py`
- `ai/execution_records.py`
- `ai/workflow_definitions.py`

## Main Existing Modules Refactored

- `ai/assistant_service.py`
- `ai/chat_service.py`
- `ai/chat_store.py`
- `ai/backend_tools.py`
- `ai/api_tools.py`
- `ai/playbook_registry.py`
- `ai/planner.py`
- `ai/reviewer.py`
- `source/ai_dashboard.py`

## Validation

- unit coverage was updated across control-plane, dashboard, adapter, workflow, and trace behavior
- the full unit suite passes after the refactor

## Notes

- `docs/ai-control-plane-implementation-plan.md` now uses `~~strike-through~~` to mark implementation items that were completed
- items under "What can wait until later" remain future-facing and are intentionally left unstruck
