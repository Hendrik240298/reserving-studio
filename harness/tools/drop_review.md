# Drop Review Tool

## Purpose

Run native drop analysis on the deterministic `Triangle` / `Reserving` backbone and produce a human-readable markdown packet for AI harness review.

## Command

```bash
uv run python -m harness.cli drop-review \
  --config examples/config_quarterly.yml \
  --candidate-limit 5
```

Defaults:

- config-driven quarterly example when no config is provided
- diagnostic scan uses observed link ratios with no pre-applied drops
- all returned candidate signals are applied together in one combined drop scenario

Optional output path:

```bash
uv run python -m harness.cli drop-review \
  --output harness/artifacts/drop_review_quarterly.md
```

## Inputs

- `--config`: reserving config path. Default: `examples/config_quarterly.yml`.
- `--candidate-limit`: minimum number of candidate signals to include. If more high-priority signals are found, all high-priority signals are included.
- `--output`: markdown packet path. Default: timestamped file in `harness/artifacts/`.

## Output

The command writes a markdown drop-review packet with:

- summary
- run status
- candidate signal count and applied drop count
- ultimates impact by origin for the combined drop scenario
- candidate signals table
- warnings

The packet is the durable artifact for AI and human review. The CLI only prints the output path.

When summarizing this tool in chat, include two markdown tables:

- the full applied drops / candidate signals table returned in the packet, including origin year, period, link ratio, selected LDF, signal score, and priority
- the full ultimates impact by origin/UWY year table returned in the packet, including baseline ultimate, drop-scenario ultimate, and ultimate delta

Do not summarize the drops table by showing only the first few rows. If the packet contains 27 candidate signals, the chat answer should include all 27 candidate-signal rows.

## Runtime Warnings

Chainladder/numpy runtime warnings about overflow, accumulation, or invalid intermediate values may appear during the run. If the command still writes the markdown packet successfully, treat those warnings as non-blocking implementation noise for this prototype and summarize the generated packet. If the command fails without writing a packet, report the failure and stop.

## Implementation Boundary

The active tool loads config/data, builds the deterministic triangle backbone, creates `Reserving` runs directly, detects link-ratio outlier signals, applies all returned signals as drops in one combined scenario, and reports ultimate impact by origin.

It should not route through `InMemoryReservingBackend`, old REST schemas, GUI session state, chat control-plane state, or basis/scenario identifiers.

The native analysis path should:

- use `ClaimsCollection`, `PremiumRepository`, and `Triangle.from_claims(...)`
- use `Reserving.set_development`, `Reserving.set_tail`, `Reserving.set_bornhuetter_ferguson`, and `Reserving.reserve`
- generate candidate drops directly from observed link-ratio outliers in the real triangle
- apply all selected candidate signals together for the combined scenario

## Non-Use Cases

- Do not use this command to book reserves.
- Do not use this command to create a new reserving method.
- Do not use this command when the user asks for an unsupported review workflow.
- Do not manually calculate reserves if this command fails.

## Final Report Use

When the user wants one stored deliverable that combines this packet with other tool outputs, pass the generated markdown file into `harness/tools/final_report.md` rather than changing the drop-review output contract.
