# Triangle to Markdown Tool

## Purpose

Render a reserving triangle as a markdown table using the deterministic active data-loading path. The table can represent age-to-age factors or triangle values such as incurred, paid, outstanding, or premium-selected values.

## Command

```bash
uv run python -m harness.cli triangle-to-markdown \
  --config examples/config_quarterly.yml \
  --triangle-type a2a
```

## Inputs

- `--config`: reserving config path. Required. No default.
- `--triangle-type`: triangle type to render. Required.
- `--view`: `cumulative` or `incremental` for value triangles. Default: `cumulative`. `a2a` supports `cumulative` only.
- `--bolt`: optional repeated `origin:development` pairs to strike out with markdown `~~...~~`.
- `--output`: optional path for writing the markdown table. If omitted, the command prints the table to stdout. If provided, the CLI prints a short status line instead of the full table.

Development labels are generic integer development ages from the deterministic triangle, not quarter-specific display labels. For example, quarterly data may show `3, 6, 9, ...`, while yearly data may show `12, 24, 36, ...`.

## Examples

Quarterly A2A with bolted cells:

```bash
uv run python -m harness.cli triangle-to-markdown \
  --config examples/config_quarterly.yml \
  --triangle-type a2a \
  --bolt 2002:21 \
  --bolt 2003:9
```

Cumulative incurred triangle:

```bash
uv run python -m harness.cli triangle-to-markdown \
  --config examples/config_quarterly.yml \
  --triangle-type incurred \
  --view cumulative
```

Incremental incurred triangle:

```bash
uv run python -m harness.cli triangle-to-markdown \
  --config examples/config_quarterly.yml \
  --triangle-type incurred \
  --view incremental
```

Yearly incurred triangle:

```bash
uv run python -m harness.cli triangle-to-markdown \
  --config examples/config_clrd.yml \
  --triangle-type incurred \
  --view cumulative
```

## Output

The command returns only the markdown table text when writing to stdout. It does not add a narrative summary or recalculate actuarial outputs.

## Implementation Boundary

The tool loads active `source/` data, builds the deterministic triangle through the same core path as active reserving code, and formats observed values into markdown.

It must not modify session state, invent labels, or calculate new actuarial selections.

## Non-Use Cases

- Do not use this command for drop review or candidate ranking.
- Do not use this command when there is no config path or triangle type.
- Do not use this command to persist bolted drops.

## Final Report Use

When the user wants one stored deliverable that combines triangle output with other tool results, write the triangle markdown to a file and pass that file into `harness/tools/final_report.md`.
