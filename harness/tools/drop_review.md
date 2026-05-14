# Drop Review Tool

## Purpose

Run the deterministic `reserving-studio` drop-review tool and produce a human-readable markdown packet for AI harness review.

## Command

```bash
uv run python -m harness.cli drop-review \
  --config examples/config_quarterly.yml \
  --candidate-limit 5
```

Optional output path:

```bash
uv run python -m harness.cli drop-review \
  --output harness/artifacts/drop_review_quarterly.md
```

## Inputs

- `--config`: reserving config path. Default: `examples/config_quarterly.yml`.
- `--candidate-limit`: number of candidates to consider/display. Default: `5`.
- `--output`: markdown packet path. Default: timestamped file in `harness/artifacts/`.

## Output

The command writes a markdown drop-review packet with:

- executive summary
- recommendation/result
- key evidence
- candidate ranking
- actuarial interpretation
- caveats
- execution details
- requested and effective inputs
- tool calls
- data lineage
- warnings
- reproducibility fields

The candidate ranking should show every candidate returned by the deterministic drop review for the requested `--candidate-limit`. The packet renderer should not silently cap the ranking at a smaller display limit.

The document structure is defined in `harness/templates/drop_review_packet.md`. Python should only fill deterministic values into that template, not make hidden product decisions about what the packet should contain.

## Implementation Boundary

The v1 tool loads config/data, builds the deterministic `Reserving` workflow, and calls `AssumptionReviewService.review_drops` directly.

It should not route through `InMemoryReservingBackend`, old REST schemas, GUI session state, chat control-plane state, or basis/scenario identifiers.

## Non-Use Cases

- Do not use this command to book reserves.
- Do not use this command to create a new reserving method.
- Do not use this command when the user asks for an unsupported review workflow.
- Do not manually calculate reserves if this command fails.
