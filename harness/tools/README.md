# Harness Tool Inventory

This file is the deterministic tool discovery layer for AI harnesses.

When a user asks for reserving work, first check this inventory, then use the matching tool doc and skill. If no tool matches, do not invent a workflow.

## Available Tools

| Tool | Status | Use When | Command / Doc |
| --- | --- | --- | --- |
| Drop review | Implemented v2 | The user asks to review/test development drops or asks for a native drop analysis on reserving data. | `harness/tools/drop_review.md` |
| Triangle to Markdown | Implemented v1 | The user asks to render a reserving triangle as a markdown table and optionally strike out bolted cells. | `harness/tools/triangle_to_markdown.md` |
| Final Report | Implemented v2 | The user wants one final markdown report that combines multiple tool result files and artifact references into a single deliverable. | `harness/tools/final_report.md` |

## Planned Tools

| Tool | Status | Use When | Notes |
| --- | --- | --- | --- |
| Diagnostics summary | Planned | The user asks for a compact deterministic diagnostics overview. | Not yet exposed through the harness CLI. |
| Tail review | Planned | The user asks to review/test tail assumptions. | Existing backend concept, not yet harness CLI. |
| Reserve-change explanation | Planned | The user asks why reserves changed between bases or scenarios. | Not yet harness CLI. |

## Selection Rules

- Use **Drop review** for requests like "run a drop review", "test development drops", "review drop candidates", or "show the direct reserve impact of dropping this factor".
- Use **Triangle to Markdown** for requests like "show the triangle as markdown", "render incurred/premium/a2a as a table", or "strike out bolted cells in a triangle".
- Use **Final Report** for requests like "save this as one markdown report", "combine these tool outputs into one deliverable", "store the final results and artifacts", or "enforce one markdown per chat/conversation".
- Use the quarterly config by default when the user says "quarterly data" or does not provide another config: `examples/config_quarterly.yml`.
- If the user provides another config path, use that config with the same capability command if it follows `reserving-studio` input conventions.
- If the user asks for a planned tool, explain that it is planned but not yet implemented in the harness and suggest drop review only if it fits the request.
- Never manually calculate actuarial results outside deterministic `reserving-studio` capabilities.

## Current Commands

```bash
uv run python -m harness.cli drop-review --config examples/config_quarterly.yml --candidate-limit 5
uv run python -m harness.cli final-report --config examples/config_quarterly.yml --conversation-id quarterly-drop-review --title "Quarterly Drop Review" --input-file "drop-review:harness/artifacts/drop_review_quarterly.md"
```

After running a result-producing tool, read the generated markdown file before answering. When the user wants one stored deliverable, use **Final Report** to compose the final markdown output.

## Promotion Boundary

- Tools are deterministic executable capabilities.
- Skills are reusable workflows over those tools.
- Notes are one-off context or rationale.
- Do not create a new skill automatically just because a tool run was useful.
- Only promote a workflow to a skill when the user explicitly asks.
