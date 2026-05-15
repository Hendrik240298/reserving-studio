---
name: drop-review
description: Run the native deterministic drop-review tool for reserving data, then read and summarize the generated markdown packet. Use when the user asks for a drop review, development drop review, or to test development drop candidates directly on the triangle and reserving backbone.
compatibility: Harness Native Reserving Studio; portable Agent Skills format.
---

# Drop Review Skill

Use this skill when the user asks for a drop review or asks to test development drop candidates directly on the deterministic triangle and reserving backbone.

## Steps

1. Read the tool inventory at `harness/tools/README.md`.
2. Read the drop-review tool doc at `harness/tools/drop_review.md`.
3. Run the deterministic CLI command for the requested config/data.
4. Read the generated markdown packet before answering.
5. Summarize the recommendation, candidate ranking, key evidence, caveats, and remaining human actuarial judgment.

## Default Command

```bash
uv run python -m harness.cli drop-review --config examples/config_quarterly.yml --candidate-limit 5
```

Use `--candidate-limit` from the user request when provided.

Default analysis mode:

- `--method chainladder`
- no tail effect
- no monotone tail correction

The active implementation generates candidate drops directly from observed link-ratio outliers in the real triangle, then reruns `Reserving` candidate by candidate to compare reserve impact.

## Boundaries

- Do not calculate reserves manually.
- Do not invent drop candidates outside the deterministic tool output.
- Do not treat the generated packet as booking approval.
- If the deterministic command fails, report the failure and stop.
