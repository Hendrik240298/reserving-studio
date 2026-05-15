# reserving-studio

## Harness Native Reserving Studio

This branch is now centered on a harness-native actuarial workbench.

- `source/` contains the active deterministic reserving core.
- `harness/` contains the active harness interface: executable tools, tool inventory, skills, templates, and generated artifacts.
- `_archive/legacy_gui_api_chat/` contains the archived dashboard, REST API, AI chat loop, and related tests/docs.
- `_salvage/legacy_useful_parts/` contains useful old pieces that are not active architecture.

## Start Here

Read `AGENTS.md` first.

For active non-archived docs, start with `docs/README.md`.

Then for active harness work read:

- `harness/README.md`
- `harness/tools/README.md`
- `harness/tools/drop_review.md`
- `harness/tools/final_report.md`
- `harness/skills/drop-review/SKILL.md`
- `harness/templates/drop_review_packet.md`

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Active Command

```bash
uv run python -m harness.cli drop-review --config examples/config_quarterly.yml --candidate-limit 5
```

The command writes a markdown review packet to `harness/artifacts/` unless `--output` is provided.

## Active Deterministic Core

- `source/claims_collection.py`
- `source/premium_repository.py`
- `source/triangle.py`
- `source/reserving.py`

## Validation

```bash
uv run pytest tests/unit/test_harness_markdown.py -q
uv run pytest tests/unit/test_triangle_markdown.py -q
uv run pytest tests/unit/test_native_drop_analysis.py -q
```

## Notes

- The old dashboard/API/chat shell is archived and should not receive new active dependencies.
- New harness tools should call deterministic `source/` builders and core reserving classes directly.
