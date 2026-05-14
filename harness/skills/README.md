# Harness Skills

This directory contains portable agent skills for Harness Native Reserving Studio.

Skills are agent-facing workflow packages. They tell a harness when to use deterministic tools, which command to run, which artifact to read, how to stop safely, and how to summarize the result.

Deterministic executable tools live under `harness/tools/`. Skills may call those tools, but should not reimplement actuarial calculations.

Tool-specific adapter directories such as `.agents/skills/`, `.claude/skills/`, or `.opencode/skills/` can later copy or symlink these skill folders.

## Promotion Contract

- Do not create, modify, or promote a skill automatically.
- If a workflow looks reusable, the harness may suggest promotion.
- Only create or update a skill when the user explicitly asks.
- If the request is incomplete, ask only for the missing high-impact fields.

## Tool Versus Skill Versus Note

- `tool`: deterministic executable capability.
- `skill`: reusable workflow over one or more existing tools.
- `note`: one-off explanation, background, or execution history.

Promote to a tool when:

- the reusable value is mainly executable logic
- the workflow currently depends on ad hoc code or copy-pasted commands
- the capability should be callable from several skills

Promote to a skill when:

- the reusable value is mainly procedure over existing tools
- the workflow needs explicit stop rules or answer shape
- the same operational pattern is likely to recur

Keep it as a note when:

- it is mainly context, rationale, or learning
- it is too immature or too narrow to justify a reusable workflow
- there is no stable tool path yet

## Required Skill Properties

Every promoted skill should:

- use active deterministic tools or active `source/` services only
- say when to use it
- say which tool doc or command to use
- say what artifact to read
- include stop rules
- include what it must not do
- define the expected answer shape

## Promotion Workflow

When the user explicitly asks to promote a workflow into a skill:

1. Read `harness/templates/skill_promotion_request.md`.
2. Use the user-provided guidance first.
3. Ask only for the missing high-impact fields.
4. Create or update `harness/skills/<skill-name>/SKILL.md` from `harness/templates/skill_template.md`.
5. Update this file if promotion guidance or discovery rules changed.
6. Update `harness/tools/README.md` only if tool discovery changed.
7. Verify the referenced tool or command works.

## High-Impact Fields

If missing, ask for:

- skill name
- when to use it
- which existing tool(s) it should use
- expected output or answer shape
- stop rules
- what it must not do

If obvious from the conversation, the harness may fill minor details itself, but should not guess the workflow boundary.

## Suggested User Prompt

Users can guide promotion with:

```text
Please turn this into a skill.

Name:
Use when:
Use these existing tools:
Inputs:
Expected output:
Stop rules:
What it must not do:
Should discovery docs be updated too? yes/no
```

## Available Skills

| Skill | Path | Use When |
| --- | --- | --- |
| Drop Review | `drop-review/SKILL.md` | User asks for a drop review, development drop testing, or to analyze drop candidates. |
| Triangle to Markdown | `triangle-to-markdown/SKILL.md` | User wants to visualize a reserving triangle (A2A, incurred, premium, etc.) as a markdown table, optionally with bolted drops marked. |

## Templates

- `harness/templates/skill_template.md`
- `harness/templates/skill_promotion_request.md`
