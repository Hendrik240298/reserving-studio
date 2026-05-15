# Drop Review Workflow

## When To Use

Use this workflow when reviewing whether development/drop assumptions should be challenged or tested directly on the deterministic triangle and reserving backbone.

## Steps

- Run the deterministic native drop-review CLI command.
- Open or read the generated markdown packet.
- Summarize the recommendation, key evidence, candidate ranking, caveats, and next human decision.
- Keep the answer grounded in the markdown packet and deterministic outputs.

## Required Command

```bash
uv run python -m harness.cli drop-review \
  --config examples/config_quarterly.yml \
  --candidate-limit 5
```

## Stop Rules

Stop or caveat when:

- the config or data cannot be loaded
- deterministic workflow construction fails
- drop review returns no candidates
- deterministic output contains warnings or material caveats
- the user asks for a reserve selection or booking action rather than a review artifact
- the generated markdown packet cannot be found or read

## Answer Shape

- State the generated packet path.
- State the recommendation class and candidate id if present.
- Summarize the top evidence.
- Summarize the main caveats.
- Identify what still requires human actuarial judgment.
