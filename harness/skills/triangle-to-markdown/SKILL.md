---
name: triangle-to-markdown
description: Generate a markdown table visualization of a reserving triangle (A2A factors, incurred, premium, etc.) with optional strikethrough marking for bolted cells.
compatibility: Harness Native Reserving Studio; portable Agent Skills format.
---

# Triangle to Markdown Skill

Use this skill when the user wants to visualize a reserving triangle as a markdown table, optionally with bolted drops marked with strikethrough.

## Purpose

Convert a reserving triangle (A2A factors, incurred, premium, etc.) into a formatted markdown table for insertion into reports, packets, or documentation. Optionally mark specified `(origin, development)` pairs with strikethrough to indicate bolted/excluded data points.

## Use When

- User asks to "show the triangle as markdown"
- User wants to visualize link ratios, incurred, or premium data in table format
- User wants to mark certain data points as bolted/excluded in the table
- User wants a formatted triangle table to insert into a markdown document

## Do Not Use When

- User just needs raw triangle data (suggest direct data export instead)
- User asks for calculations or LDF changes (use drop-review or other analysis tools)
- User wants to modify or persist the triangle (this is visualization only)
- User asks for incremental `a2a` output

## Required Inputs

- `config_path`: Path to the reserving config YAML (e.g., `examples/config_quarterly.yml`). No default — must be provided.
- `triangle_type`: Type of triangle to visualize. Required. Supported examples include `a2a`, `incurred`, `paid`, `outstanding`, and `premium`.
- `triangle_view` (optional): `cumulative` or `incremental` for value triangles. Default is `cumulative`. `a2a` supports `cumulative` only.
- `bolted_drops` (optional): List of `[origin, development]` pairs to mark with strikethrough. If not provided, no strikethrough marks are applied.

## Required Tools

- `harness/tools/triangle_to_markdown.md`
- `uv run python -m harness.cli triangle-to-markdown --config <config> --triangle-type <type> [--view cumulative|incremental] [--bolt ORIGIN:DEVELOPMENT]...`

## Steps

1. Load config using `ConfigManager.from_yaml(config_path)`.
2. Load claims and premium data using `load_inputs_from_config(config, repo_root=Path('.'))`.
3. Construct the triangle through the active deterministic `ClaimsCollection -> PremiumRepository -> Triangle.from_claims(...)` path.
4. Call `Triangle.get_triangle(triangle_type)` for value triangles, optionally convert to incremental view, or fit the development model and extract link ratios for `a2a`.
5. Extract origin and development labels.
6. Build a markdown table with:
  - Header row: Origin | Dev1 | Dev2 | ... | DevN |
  - Separator row: --- | --- | ... | --- |
  - Data rows: one per origin, with values or empty cells for NaN
  - Apply strikethrough `~~value~~` to cells matching `bolted_drops`
7. Return only the formatted markdown table.

## Output Contract

- Markdown table as text block, ready to insert into a larger document.
- Example output:
  ```
  | Origin | 3 | 6 | 9 | 12 | ... |
  | --- | --- | --- | --- | --- | ... |
  | 1995 | 2.182 | 2.021 | 2.165 | 1.479 | ... |
  | 1996 | 3.238 | 1.485 | ~~1.807~~ | 1.482 | ... |
  ```

## Stop Rules

- **Config load error**: If config cannot be loaded, report the error and stop. Do not invent data.
- **Data load error**: If claims/premium data cannot be loaded, report the error and stop.
- **Triangle build error**: If triangle construction fails, report the error and stop.
- **Invalid triangle type**: If requested triangle_type is not available, report available types and stop.
- **Invalid view**: If requested `triangle_view` is unsupported, report supported views and stop.
- **Incomplete or missing data**: If triangle has insufficient rows/columns or cannot be computed, report the issue and stop.

## Must Not Do

- Do not recalculate link ratios, LDF, or any actuarial values.
- Do not modify the session YAML or config files.
- Do not invent origin or development period labels.
- Do not treat the markdown output as approved data or sign-off.
- Do not bypass deterministic triangle building; always use `Triangle.get_triangle()` and observed values.
- Do not fail silently; always report errors clearly to the user.

## Notes

- This skill is purely for visualization and formatting. All data comes directly from the deterministic triangle builder.
- Bolted drops are user-provided; this skill does not calculate or discover which drops to mark.
- Empty cells represent NaN or missing values in the triangle and are rendered as blank.
- Development labels are generic integer development ages from the deterministic triangle, not quarter-specific labels.
- The skill can be used standalone or integrated into larger workflows (e.g., as part of drop-review packet generation).
