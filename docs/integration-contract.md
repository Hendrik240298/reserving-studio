# Integration Contract

This page defines the recommended integration boundary for plugging Reserving Studio into another Python project.

## Preferred contract

The reserving workflow should start from three initialized objects:

1. `ConfigManager`
2. `ClaimsCollection`
3. `PremiumRepository`

Once those objects exist and contain the intended data, pass them into:

- `source.app.build_workflow_from_collections(...)`

This keeps responsibilities explicit:

- your project owns data access and shaping
- `ClaimsCollection` owns claims-side validation and canonicalization
- `PremiumRepository` owns premium-side validation and canonicalization
- Reserving Studio owns triangle building, reserving logic, session-backed assumptions, and UI behavior

## Recommended flow

Use this sequence in external integrations:

1. load or build raw claims and premium `DataFrame`s
2. initialize `ConfigManager`
3. initialize `ClaimsCollection`
4. initialize `PremiumRepository`
5. call `build_workflow_from_collections(...)`
6. launch or run the interactive session

```python
from source.app import build_workflow_from_collections, run_interactive_session
from source.claims_collection import ClaimsCollection
from source.config_manager import ConfigManager
from source.premium_repository import PremiumRepository


config = ConfigManager.from_yaml("config.yml")

claims = ClaimsCollection(
    claims_df,
    values_are_cumulative=False,
)
premium = PremiumRepository.from_dataframe(config, premium_df)

reserving = build_workflow_from_collections(
    claims=claims,
    premium=premium,
    config=config,
)

finalized = run_interactive_session(
    reserving,
    config=config,
)
```

## Why this is the preferred boundary

- It separates raw data extraction from reserving execution.
- It makes validation happen before triangle construction and model execution.
- It lets another project keep control over repository classes and data acquisition.
- It creates a stable interface that is easier to reason about than passing partially prepared data into the workflow.
- It makes debugging simpler because raw data, normalized domain objects, and reserving outputs are distinct stages.

## When to use the dataframe shortcut

`source.app.build_workflow_from_dataframes(...)` still exists and is useful when:

- you are prototyping quickly
- your integration point is only raw dataframes
- you do not need explicit control over object initialization

That shortcut is convenient, but it is not the preferred long-term contract for cross-project integration.

## Object responsibilities

### `ConfigManager`

- loads YAML defaults and session settings
- owns the segment/session persistence boundary
- provides workflow configuration such as granularity and input settings

### `ClaimsCollection`

- expects canonical claims columns
- stores claims data in workflow-ready form
- carries whether claim values are incremental or cumulative

Expected claims columns:

- `uw_year`
- `period`
- `paid`
- `outstanding`

`id` is recommended and is required for claim-level helper methods such as `iter_claims()` and `get_claim_amounts()`.

### `PremiumRepository`

- accepts supported premium schemas
- normalizes them to the canonical premium structure used by triangle construction
- provides premium data through `get_premium()`

Supported premium schemas:

1. `UnderwritingYear`, `Premium`
2. `origin`, `development`, `Premium_selected`
3. `uw_year`, `period`, `Premium_selected`

## Integration patterns

### A) Your project already has repository classes

This is the target case.

- let your own repository layer fetch and shape data
- build `ClaimsCollection` from your claims dataframe
- build `PremiumRepository` from your premium dataframe
- pass those objects into `build_workflow_from_collections(...)`

### B) Your project only has raw dataframes

- start from the dataframes
- initialize `ClaimsCollection` and `PremiumRepository` directly
- then use the same object-driven workflow

### C) You want config-driven CSV or SQL loading inside Reserving Studio

- use `source.input_loader.load_inputs_from_config(...)`
- then initialize `ClaimsCollection` and `PremiumRepository`
- then call `build_workflow_from_collections(...)`

This is the pattern now used by the example runners.

## Non-goals of the integration contract

This contract does not require:

- a one-call wrapper that hides config, claims, and premium setup
- direct use of internal dashboard services
- replacing your existing repository layer

The goal is controlled integration, not maximum abstraction.

## Related docs

- `docs/start-with-your-data.md`
- `docs/actuary-workflow.md`
- `docs/technical-architecture.md`
- `docs/api-reference.md`
