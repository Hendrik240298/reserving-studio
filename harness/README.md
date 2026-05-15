# Harness Native Reserving Studio

This directory contains the markdown-facing scaffold for the Harness Native Reserving Studio prototype.

The harness is only an orchestration shell. Deterministic `reserving-studio` code, `source.reserving.Reserving`, domain services in `source/`, and `chainladder-python` remain the actuarial source of truth.

## Supported V1 Workflow

- Native drop analysis on the quarterly example config using the deterministic `Triangle` / `Reserving` backbone.
- Triangle to markdown for cumulative or incremental value triangles, plus cumulative A2A rendering.

Run:

```bash
uv run python -m harness.cli drop-review \
  --config examples/config_quarterly.yml \
  --candidate-limit 5
```

The command writes a markdown review packet under `harness/artifacts/` unless `--output` is provided.

## Read Next

- `harness/tools/README.md`
- `harness/tools/drop_review.md`
- `harness/tools/triangle_to_markdown.md`
- `harness/skills/README.md`
- `harness/skills/drop-review/SKILL.md`
- `harness/skills/triangle-to-markdown/SKILL.md`
- `harness/workflows/drop_review.md`
- `harness/templates/drop_review_packet.md`

Use `harness/tools/README.md` as the deterministic tool inventory before deciding which tool or skill to use.

## Rules

- Use deterministic CLI capabilities only.
- Do not invent reserving calculations.
- New harness tools may depend on deterministic `source/` core/services, but not legacy GUI/API/chat/session/control-plane modules.
- Do not create or modify skills automatically; only do that when the user explicitly asks.
- Do not treat the markdown packet as human acceptance or booking approval.
- If the deterministic command fails, report the failure instead of filling in missing results.
