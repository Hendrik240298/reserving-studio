# Practical Notes

This page captures high-value operational and troubleshooting notes for day-to-day reserving work.

## Session behavior

- UI assumptions are persisted to segment session YAML.
- Session writes are atomic (`.tmp` then replace) to reduce corruption risk.
- A `sync_version` is used to coordinate updates across multiple browser tabs.

If a session YAML is corrupted, the app attempts to rename it to a `.corrupt.*` backup and continue with defaults.

## Multi-tab sync

- Updates from one tab propagate to other tabs for the same segment.
- Stale updates are ignored using version checks.
- Technical detail: `docs/cross-tab-sync.md`.

## Recalculation and performance

- Recalculation is parameter-driven and cached by model/visual keys.
- Typical sample-data interactions should update quickly (sub-second to low seconds depending on host).
- If performance degrades, reduce concurrent tabs and ensure local machine resources are available.

## Common validation errors and fixes

- **Claims missing required columns**: include `id`, `uw_year`, `period`, `paid`, `outstanding` (or map them).
- **Premium schema rejected**: supply one supported schema or add `column_map`.
- **Invalid period formats**: use parseable dates, quarter labels, or numeric development lags as expected.
- **Invalid BF apriori**: values must be numeric and non-negative.

## Practical actuarial checks before finalizing

- Review dropped cells and ensure each drop has business justification.
- Check tail assumptions (curve and fit range) against maturity profile.
- Compare CL vs BF outcomes by UWY and confirm selected method rationale.
- Verify IBNR direction and loss ratio level are consistent with known portfolio context.

## Testing shortcuts

- Unit suite:

```bash
uv run python -m pytest tests/unit -q
```

- E2E suite:

```bash
uv run python -m pytest tests/e2e -m e2e -q
```

On E2E failure, inspect artifacts in `tests/artifacts/e2e/` (`.png` + `.zip` trace).
