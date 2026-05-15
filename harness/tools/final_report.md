# Final Report Tool

## Purpose

Compose one final markdown report per conversation from existing tool result files plus optional AI-authored scaffolding.

This tool does not require other tools to publish special manifests or custom contracts. The AI can run existing deterministic tools, collect their markdown outputs, and pass those files into `final-report` for generic composition.

## Command

```bash
uv run python -m harness.cli final-report \
  --config examples/config_quarterly.yml \
  --conversation-id quarterly-drop-review-20260515 \
  --title "Quarterly Drop Review" \
  --input-file "drop-review:chats/quarterly_drop_review.md" \
  --input-file "incurred-triangle:chats/quarterly_incurred_triangle.md" \
  --input-file "a2a-triangle:chats/quarterly_a2a_triangle.md"
```

With AI-authored scaffolding:

```bash
uv run python -m harness.cli final-report \
  --config examples/config_quarterly.yml \
  --conversation-id quarterly-drop-review-20260515 \
  --title "Quarterly Drop Review" \
  --input-file "drop-review:chats/quarterly_drop_review.md" \
  --input-file "incurred-triangle:chats/quarterly_incurred_triangle.md" \
  --scaffold-file /tmp/opencode/final_report_scaffold.md
```

## Inputs

- `--config`: reserving config path. Default: `examples/config_quarterly.yml`.
- `--conversation-id`: stable conversation identifier. Required.
- `--title`: markdown title. Required.
- `--input-file`: repeated `LABEL:PATH` markdown input from existing tool results.
- `--scaffold-file`: optional AI-authored markdown scaffold. Supports `{{input:label}}` and `{{inputs}}`.
- `--body-file`: optional legacy direct-body mode. Use when the AI already drafted the full report body.
- `--artifact`: optional repeated `LABEL:PATH` appendix artifact reference.
- `--warning`: optional repeated warning line.
- `--output`: explicit markdown output path.
- `--output-dir`: directory for `<conversation-id>.md`. Defaults to `ai.final_reports.path` from config.

## Output

The command writes one markdown file with:

- title and conversation metadata
- AI-authored or tool-composed report body
- artifacts appendix
- included input file list
- execution details
- warnings
- reproducibility fields

The tool is for final recommended results only. It does not store the full transcript.

## Composition Rules

- If `--body-file` is provided, the tool writes that body inside the final scaffold.
- If `--scaffold-file` is provided, the tool replaces any `{{input:label}}` tokens with the matching input file content.
- `{{inputs}}` expands to the default grouped rendering of every supplied input file.
- If no scaffold is provided, the tool builds a default report body by inserting each input file under its own section heading.

Input labels are normalized to lowercase safe keys for token lookup. For example:

- `drop-review` -> `{{input:drop-review}}`
- `incurred triangle` -> `{{input:incurred_triangle}}`

## Implementation Boundary

The tool may read existing result files, compose them, and record deterministic metadata, but it must not:

- re-run or reinterpret actuarial outputs on its own
- depend on archived chat/session/control-plane code
- require every current or future tool to implement a custom report contract
- store raw transcript history as the primary artifact

## Future Direction

If the active tool set later benefits from shared fragment or manifest contracts, those should remain optional enhancements. The core final-report workflow must continue to work over existing tool result markdowns without per-tool composer wiring.

## Non-Use Cases

- Do not use this command to calculate reserving outputs.
- Do not use this command as a transcript logger.
- Do not use this command to bypass tool-level warnings or provenance.
