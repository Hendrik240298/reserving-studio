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

- method: `chainladder`
- tail effect: off
- monotone tail correction: off

Optional output path:

```bash
uv run python -m harness.cli drop-review \
  --output harness/artifacts/drop_review_quarterly.md
```

## Inputs

- `--config`: reserving config path. Default: `examples/config_quarterly.yml`.
- `--candidate-limit`: number of candidates to consider/display. Default: `5`.
- `--method`: `chainladder` or `bornhuetter_ferguson`. Default: `chainladder`.
- `--use-tail`: apply the session tail settings. Default: off.
- `--enforce-monotone-tail`: enable the legacy monotone tail correction. Default: off.
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

The candidate ranking should show every candidate returned by the deterministic native drop analysis for the requested `--candidate-limit`. The packet renderer should not silently cap the ranking at a smaller display limit.

The document structure is defined in `harness/templates/drop_review_packet.md`. Python should only fill deterministic values into that template, not make hidden product decisions about what the packet should contain.

## Implementation Boundary

The active tool loads config/data, builds the deterministic triangle backbone, creates `Reserving` runs directly, and compares reserve impact candidate by candidate.

It should not route through `InMemoryReservingBackend`, old REST schemas, GUI session state, chat control-plane state, or basis/scenario identifiers.

The native analysis path should:

- use `ClaimsCollection`, `PremiumRepository`, and `Triangle.from_claims(...)`
- use `Reserving.set_development`, `Reserving.set_tail`, and `Reserving.reserve`
- default to `chainladder`
- default to no tail effect
- default to no monotone tail correction

## Non-Use Cases

- Do not use this command to book reserves.
- Do not use this command to create a new reserving method.
- Do not use this command when the user asks for an unsupported review workflow.
- Do not manually calculate reserves if this command fails.
