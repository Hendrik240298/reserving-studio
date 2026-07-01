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
5. Summarize the candidate signals, applied drops, ultimates impact by origin, warnings, and remaining human actuarial judgment.
6. In the chat answer, include two markdown tables copied from the packet:
   - the full applied drops / candidate signals table: origin year, period, link ratio, selected LDF, signal score, priority
   - the full ultimates impact by origin/UWY year table: baseline ultimate, drop-scenario ultimate, ultimate delta
7. Do not truncate the drops table to only the first few rows. If the packet contains 27 candidate signals, include all 27 candidate-signal rows in the chat answer.

If chainladder/numpy runtime warnings appear but the command writes the markdown packet, ignore those runtime warnings for the answer and rely on the generated packet. If the command fails without writing a packet, report the failure and stop.

## Default Command

```bash
uv run python -m harness.cli drop-review --config examples/config_quarterly.yml --candidate-limit 5
```

Use `--candidate-limit` from the user request when provided.

Default analysis mode:

- config-driven input loading
- diagnostic scan over observed link ratios without pre-applied drops
- combined drop scenario applying all returned candidate signals

The active implementation generates candidate drops directly from observed link-ratio outliers in the real triangle, applies the returned candidates together, and reports ultimate impact by origin.

## Boundaries

- Do not calculate reserves manually.
- Do not invent drop candidates outside the deterministic tool output.
- Do not treat the generated packet as booking approval.
- If the deterministic command fails, report the failure and stop.
