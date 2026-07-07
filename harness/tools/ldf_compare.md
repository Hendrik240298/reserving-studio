# LDF Compare Tool

## Purpose

Compare deterministic LDF vectors from explicit reserving scenario settings against the implicit default baseline and write a focused review packet.

The tool is analysis-only evidence. It does not book reserves, store accepted basis state, or use session/scenario ids.

## Command

From a scenarios file:

```bash
uv run python -m harness.cli ldf-compare \
  --config examples/config_quarterly.yml \
  --scenarios-file examples/ldf_compare_scenarios.yml
```

Direct CLI scenario:

```bash
uv run python -m harness.cli ldf-compare \
  --config examples/config_quarterly.yml \
  --scenario-json 'drop_2000_2002_6:{"development":{"drop":[["2000",6],["2002",6]]}}'
```

## Baseline

The baseline is the implicit `default` reserving run, matching the drop-review baseline:

```python
reserving.set_development()
reserving.set_tail()
reserving.set_bornhuetter_ferguson()
reserving.reserve()
```

Explicit scenarios are overrides on top of those defaults.

## Scenario Settings Contract

Canonical JSON shape:

```json
{
  "development": {
    "average": "volume",
    "drop": [["2000", 6], ["2002", 6]]
  },
  "tail": {
    "curve": "weibull",
    "fit_period": [12, 100]
  },
  "bornhuetter": {
    "apriori": 0.6
  },
  "reserve": {
    "final_ultimate": "chainladder"
  }
}
```

Missing fields use defaults. `development.drop` uses the same shape as the active reserving API, except JSON/YAML uses two-item arrays instead of Python tuples.

YAML scenarios file:

```yaml
scenarios:
  drop_2000_2002_6:
    development:
      drop:
        - ["2000", 6]
        - ["2002", 6]

  simple_average:
    development:
      average: simple
```

## Inputs

- `--config`: reserving config path. Default: `examples/config_quarterly.yml`.
- `--scenarios-file`: YAML file with a `scenarios` mapping.
- `--scenario-json`: repeated `NAME:JSON` scenario settings.
- `--delta-threshold`: absolute LDF delta threshold for red plot markers. Default: `0.01`.
- `--output`: optional markdown output path. If omitted, a timestamped markdown packet is written under `harness/artifacts/`. The PNG plot is written beside the markdown packet.

## Output

The command writes:

- markdown packet with scenario settings, the LDF plot, the full LDF comparison table, warnings, and execution details
- PNG line plot for plotted ages

The CLI prints the written markdown packet path.

## Implementation Boundary

The tool loads data and runs deterministic reserving directly through:

- `ConfigManager.from_yaml`
- `load_inputs_from_config`
- `ClaimsCollection`, `PremiumRepository`, and `Triangle.from_claims(...)`
- `Reserving.set_development`, `Reserving.set_tail`, `Reserving.set_bornhuetter_ferguson`, and `Reserving.reserve`
- `Reserving.get_ldf`

It must not route through archived GUI/API/chat/session state, old basis ids, or scenario ids.

## Non-Use Cases

- Do not use this command to book reserves.
- Do not use this command to persist accepted basis state.
- Do not manually calculate LDFs if this command fails.
