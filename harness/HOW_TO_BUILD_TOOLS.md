# How To Build Harness Tools

This guide describes the current preferred pattern for building human-readable deterministic tools in Harness Native Reserving Studio.

Use `harness/drop_review.py` as the first concrete template.

## Goal

A harness tool should be easy for a human actuary-engineer to read:

1. load config and data,
2. build the deterministic reserving context,
3. run one clear analysis,
4. render one markdown artifact,
5. return the artifact path to the CLI.

The durable output is the markdown artifact. The CLI should stay thin.

## Recommended Tool Shape

```text
harness/cli.py
  -> calls one run_*_packet(...) function

harness/<tool>.py
  -> load config/data
  -> run analysis
  -> render markdown
  -> write artifact
  -> return output path
```

For example:

```python
def run_drop_review_packet(
    config_path: Path,
    candidate_limit: int = 5,
    output_path: Path | None = None,
) -> Path:
    # 1. Load data
    ...

    # 2. Analysis
    analysis = analyze_drop_effects(...)

    # 3. Render and write packet
    output_path = output_path or default_output_path()
    output_path.write_text(render_drop_review_packet(analysis), encoding="utf-8")

    return output_path
```

## Design Rules

### 1. Keep the orchestration linear

Prefer readable script-like code over hiding the workflow too early behind helpers.

Good:

```python
config = ConfigManager.from_yaml(file_name=config_path)
claims_df, premium_df = load_inputs_from_config(config, repo_root=REPO_ROOT)
claims = ClaimsCollection(claims_df, values_are_cumulative=...)
premium = PremiumRepository.from_dataframe(config_manager=config, dataframe=premium_df)
triangle = Triangle.from_claims(claims=claims, premium=premium)
```

Only extract helpers when the main workflow becomes harder to read.

### 2. Separate analysis from rendering

The analysis should return structured Python data.

The renderer should turn that data into markdown.

The CLI should not know analysis details.

```text
analyze_*        -> Python data
render_*_packet  -> markdown string
run_*_packet     -> writes artifact and returns Path
```

### 3. Use config as the main tool contract

Prefer:

```bash
uv run python -m harness.cli drop-review --config examples/config_quarterly.yml
```

The config should identify data, session, and project context. Add CLI flags only for small, intentional user-facing choices such as `--candidate-limit` or `--output`.

### 4. Return the artifact path

For now, a tool runner should return `Path`, not a large result object.

The AI harness reads the markdown artifact, not Python objects.

```python
output_path = run_drop_review_packet(...)
print(f"Wrote drop review packet: {output_path}")
```

### 5. Make the markdown packet the evidence artifact

The packet should contain the information a human or AI needs to review the result:

- summary,
- run status,
- key metrics,
- relevant tables,
- warnings or caveats.

For drop review, this includes candidate signals and ultimate impact by origin.

## Drop Review Template

The current drop-review tool follows this shape:

1. `run_drop_review_packet(...)`
   - loads config-driven claims and premium data,
   - builds `ClaimsCollection`, `PremiumRepository`, and `Triangle`,
   - calls `analyze_drop_effects(...)`,
   - writes the markdown packet,
   - returns the output path.

2. `analyze_drop_effects(...)`
   - runs a diagnostic reserving basis without pre-applied drops,
   - detects link-ratio outlier signals,
   - converts returned signals into drop tuples,
   - reruns reserving with all returned drops applied together,
   - returns signals, drops, baseline results, and drop-scenario results.

3. `render_drop_review_packet(...)`
   - renders summary,
   - run status,
   - ultimates impact,
   - candidate signals,
   - warnings.

## Testing And Validation

During development:

1. First make the tool run with a small smoke command.
2. Inspect the generated artifact.
3. Then add or update focused tests around stable behavior.

Useful commands:

```bash
uv run python -m harness.cli drop-review \
  --config examples/config_quarterly.yml \
  --candidate-limit 5

uv run pytest tests/unit/test_native_drop_analysis.py tests/unit/test_harness_markdown.py -q
```

## What To Avoid

- Do not route new tools through old GUI/API/chat/session architecture.
- Do not make the CLI responsible for rendering or actuarial logic.
- Do not create broad generic abstractions before two or three tools show the same repeated shape.
- Do not put large DataFrames into CLI return objects; render them into markdown artifacts.
- Do not manually invent actuarial results outside deterministic `source/` and `harness/` code.

## Future Generalization

After several tools use the same pattern, consider extracting only the repeated infrastructure:

- common output-path generation,
- common markdown section helpers,
- optional generic `ToolRunResult`,
- shared smoke-test pattern.

Until then, prefer explicit, readable tool files.
