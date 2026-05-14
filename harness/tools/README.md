# Harness Tool Inventory

This file is the deterministic tool discovery layer for AI harnesses.

When a user asks for reserving work, first check this inventory, then use the matching tool doc and skill. If no tool matches, do not invent a workflow.

## Available Tools

| Tool | Status | Use When | Command / Doc |
| --- | --- | --- | --- |
| Drop review | Implemented v1 | The user asks to review/test development drops or asks for a drop review on the quarterly example data. | `harness/tools/drop_review.md` |
| Triangle to Markdown | Implemented v1 | The user asks to render a reserving triangle as a markdown table and optionally strike out bolted cells. | `harness/tools/triangle_to_markdown.md` |

## Planned Tools

| Tool | Status | Use When | Notes |
| --- | --- | --- | --- |
| Diagnostics summary | Planned | The user asks for a compact deterministic diagnostics overview. | Not yet exposed through the harness CLI. |
| Tail review | Planned | The user asks to review/test tail assumptions. | Existing backend concept, not yet harness CLI. |
| Reserve-change explanation | Planned | The user asks why reserves changed between bases or scenarios. | Not yet harness CLI. |

## Selection Rules

- Use **Drop review** for requests like "run a drop review", "test development drops", or "review drop candidates".
- Use **Triangle to Markdown** for requests like "show the triangle as markdown", "render incurred/premium/a2a as a table", or "strike out bolted cells in a triangle".
- Use the quarterly config by default when the user says "quarterly data" or does not provide another config: `examples/config_quarterly.yml`.
- If the user provides another config path, use that config with the same capability command if it follows `reserving-studio` input conventions.
- If the user asks for a planned tool, explain that it is planned but not yet implemented in the harness and suggest drop review only if it fits the request.
- Never manually calculate actuarial results outside deterministic `reserving-studio` capabilities.

## Current V1 Command

```bash
uv run python -m harness.cli drop-review --config examples/config_quarterly.yml --candidate-limit 5
```

After running it, read the generated markdown packet under `harness/artifacts/` before answering.

## Promotion Boundary

- Tools are deterministic executable capabilities.
- Skills are reusable workflows over those tools.
- Notes are one-off context or rationale.
- Do not create a new skill automatically just because a tool run was useful.
- Only promote a workflow to a skill when the user explicitly asks.
