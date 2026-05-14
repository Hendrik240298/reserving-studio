# AGENTS

## Purpose
- This repository is now centered on **Harness Native Reserving Studio**.
- `source/` holds the active deterministic reserving core and domain services.
- `harness/` holds the active harness interface: executable tools, tool inventory, skills, templates, and generated artifacts.
- `_archive/legacy_gui_api_chat/` holds the old dashboard/API/chat/session shell.
- `_salvage/legacy_useful_parts/` holds useful old pieces that are not active architecture.

## Start Here
- Read this file first.
- For harness work, then read:
- `harness/README.md`
- `harness/tools/README.md`
- the relevant tool doc under `harness/tools/`
- the relevant skill under `harness/skills/`
- the relevant template under `harness/templates/`

## Active Architecture
- Deterministic actuarial truth lives in:
- `source/claims_collection.py`
- `source/premium_repository.py`
- `source/triangle.py`
- `source/reserving.py`
- `source/services/`
- The active drop-review tool flow is:
- user request
- harness instructions
- tool inventory
- skill
- `python -m harness.cli`
- input layer
- deterministic source core
- domain service
- markdown template
- generated packet
- harness summary

## Boundaries
- Do not add new dependencies from active code into `_archive/` or `_salvage/`.
- Do not use the old API adapter, dashboard shell, AI chat loop, session store, basis keys, or scenario ids for new harness tools unless explicitly requested.
- Harness tools should call deterministic `source/` builders and services directly.
- If something useful is discovered in `_salvage/`, promote it into `source/` or `harness/` before reuse.

## Skill Promotion Rules
- Do not create, modify, or promote a skill automatically.
- If a workflow looks reusable, you may suggest promotion to the user.
- Only create or update a skill when the user explicitly asks.
- If the user asks to turn something into a skill, prefer guided promotion over guessing: use the request details the user gave, then ask only for the missing high-impact fields.
- High-impact missing fields are: skill name, when to use it, which existing tool(s) it should use, stop rules, and what it must not do.
- A new skill must use active deterministic tools or active `source/` services. Do not build a skill on archived code.

## Tool Versus Skill Versus Note
- A `tool` is a deterministic executable capability.
- A `skill` is an agent-facing workflow that chooses and sequences existing tools, explains stop rules, and defines answer shape.
- A `note` is one-off context, rationale, or history that should not be executed as a workflow.
- If the reusable part is mainly executable logic, it should become a tool.
- If the reusable part is mainly procedure over existing tools, it should become a skill.
- If the reusable part is mainly explanation, background, or temporary learning, it should stay a note.

## Promotion Workflow
- When the user explicitly requests skill promotion, consult:
- `harness/skills/README.md`
- `harness/templates/skill_template.md`
- `harness/templates/skill_promotion_request.md`
- Create or update `harness/skills/<skill-name>/SKILL.md`.
- Update `harness/skills/README.md` if discovery or promotion guidance changes.
- Update `harness/tools/README.md` only if tool discovery changes.
- Update this `AGENTS.md` only if repository-wide routing or policy changes.
- Verify that every referenced command or tool path still works.

## Commands
- Drop review:
- `uv run python -m harness.cli drop-review --config examples/config_quarterly.yml --candidate-limit 5`
- Local fast tests:
- `uv run pytest tests/unit/test_harness_markdown.py -q`
- `uv run pytest tests/unit/test_drop_review_service.py -q`
- `uv run pytest tests/unit/test_scenario_evaluation_service.py -q`

## Code Rules
- Keep changes minimal and deterministic.
- Follow pandas/sklearn-style conventions in `source/`.
- Prefer small typed helpers and explicit inputs.
- Use `logging` in library code and keep it quiet by default.
- Avoid formatting-only churn.

## Validation
- After changing active harness code, run the relevant harness/unit tests.
- After changing the CLI or packet renderer, run the drop-review smoke command and read the generated markdown packet.
