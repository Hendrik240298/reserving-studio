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

### 0. Start single-purpose and stay narrow

Each harness tool should do one job that is easy to explain in one sentence.

Before adding a table, metric, output format, helper, or class, ask whether it is part of that single job or whether another existing tool already covers it. For example, if `drop-review` already reports ultimate impact, a follow-up LDF visualization tool should not duplicate ultimate tables by default.

Prefer the smallest useful artifact:

- markdown when humans or the AI need to read the result,
- PNG when the purpose is visual comparison,
- avoid CSV/JSON/extra machine payloads unless a real downstream consumer exists.

The AI can reformat markdown tables when needed. Do not add machine-readable outputs just because they are easy to generate.

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

Do not introduce dataclasses, result objects, parsers, or normalization helpers by default. Add them only when they remove real duplication or make the main path easier to understand. A simple `Path` return is often better than a broad result object.

Normalizations should stay close to the boundary they serve. For example, converting JSON/YAML `drop: [["2002", 6]]` into the `Reserving.set_development(drop=[("2002", 6)])` shape is useful. A general normalization layer for every possible reserving setting is not useful until the system clearly needs it.

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

Use direct CLI or explicit scenario files for concrete one-off analysis inputs. Do not introduce temporary session YAML, basis ids, or scenario ids unless the current system clearly uses them as the primary deterministic state contract. Session YAML can become a future baseline source, but should not be used as hidden state for new tools before that workflow is clear.

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

For narrow follow-up tools, include only the evidence needed for that purpose. For example, an LDF comparison tool used after drop review should focus on scenario settings, the LDF plot, the full LDF table, warnings, and execution details. It should not add ultimate impact, summary tables, JSON payloads, or CSV exports unless the user explicitly needs them.

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
