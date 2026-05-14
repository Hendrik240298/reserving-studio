# <Skill Name>

## Purpose

State the reusable workflow purpose in one or two sentences.

## Use When

- Describe the user requests or situations where this skill should be used.

## Do Not Use When

- Describe requests that should route to another tool, another skill, or remain a note.

## Required Tools

- `harness/tools/<tool-doc>.md`
- `uv run python -m harness.cli <command> ...`

## Inputs

- List the expected user inputs or defaults.

## Steps

1. Read the relevant tool inventory and tool doc.
2. Run the deterministic tool or command.
3. Read the generated artifact or output.
4. Summarize only from the deterministic result.

## Stop Rules

- State when to stop, caveat, or ask the user before continuing.

## Output Contract

- State the expected answer shape.

## Must Not Do

- Do not invent actuarial calculations.
- Do not bypass deterministic tools.
- Do not treat the result as approval or sign-off unless explicitly designed for that.

## Notes

- Include short implementation or routing notes only if needed.
