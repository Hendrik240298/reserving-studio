# AGENTS

## Scope
- This workspace has two code areas.
- `chainladder-python/` is the upstream chainladder library with tests and docs.
- `source/` is local application code that builds on chainladder for reserving.
- Prefer upstream conventions when touching shared logic.

## Project layout
- `chainladder-python/chainladder/` contains library code.
- `chainladder-python/chainladder/**/tests/` contains pytest tests.
- `chainladder-python/docs/` contains Jupyter Book docs and notebooks.
- `source/` contains local reserving workflow modules.
- `Plan.md` is the main project plan and should be consulted before making changes.

## Project plan (must-read)
- `Plan.md` is the heart of this repository; keep work aligned to its phases and scope.
- Current workflow modules (as described in `Plan.md`):
- `source/config_manager.py` reads settings from YAML.
- `source/claims_collection.py` loads claims movements.
- `source/premium_repository.py` loads premium data.
- `source/triangle.py` builds the reserving triangle.
- `source/reserving.py` runs reserving (chainladder + Bornhuetter-Ferguson).
- `source/dashboard.py` visualizes results (to be replaced by Dash GUI).
- The orchestrator is usually a `main.py` entrypoint (currently not in the repo).

### Plan priorities
- Phase 1: make `source/` runnable, remove legacy ThresholdOptimizer, add Dash UI for drops.
- Phase 2: persist UI selections to YAML in `sessions/` and add a load-session dropdown.
- Phase 3: expand parameters (average method, tail curve, BF apriori, thresholds) with caching.

## Build, lint, and test
- Run commands from `chainladder-python/` unless explicitly working in `source/`.
- Use `uv` for environment management; `uv.lock` is present.

### Root environment setup (reserving)
- `uv venv`
- `source .venv/bin/activate`
- `uv pip install -r requirements.txt`
- Run app: `uv run python -m source.app`

### Environment setup
- `uv sync --extra all`
- Run tools via `uv run ...` to ensure the right env.

### Tests
- Full suite:
- `uv run pytest chainladder`
- Single file:
- `uv run pytest chainladder/methods/tests/test_predict.py`
- Single test:
- `uv run pytest chainladder/methods/tests/test_predict.py::test_predict`
- By keyword:
- `uv run pytest chainladder -k "predict"`
- Exclude R-marked tests (marked `r`):
- `uv run pytest chainladder -m "not r"`
- Dashboard E2E suite (Playwright):
- `uv run python -m playwright install chromium`
- `uv run pytest tests/e2e -m e2e -q`
- Local unit suite (source-level fast checks):
- `uv run pytest tests/unit -q`
- Detailed testing guidance and workflow: `TESTING.md`

### AI testing workflow
- After implementing a new feature, or iterating on an in-progress feature, run relevant tests before reporting back.
- After changing `source/presentation/*.py` or other source-level helpers, run impacted unit test(s) in `tests/unit` (or full `uv run pytest tests/unit -q` if uncertain).
- After changing `source/app.py`, `source/dashboard.py`, `source/reserving.py`, or `source/triangle.py`, run impacted E2E test(s) in `tests/e2e`.
- If impact is unclear, run full `uv run pytest tests/e2e -m e2e -q`.
- On E2E failure, report artifact paths from `tests/artifacts/e2e/` (`.png` screenshot and `.zip` trace) and summarize failing interaction.

### Lint / format
- No lint/format configuration found in this repo.
- Do not add tooling without agreement; avoid formatting-only churn.

### Build / docs
- Docs build (Jupyter Book):
- `uv run jb build docs`
- Package build (if needed):
- `uv run python -m build`

## Code style and conventions

### API design (from upstream contributing rules)
- Follow pandas + scikit-learn patterns.
- Estimators/transformers should follow sklearn estimator API.
- `Triangle` methods should mirror pandas/NumPy naming and signatures.
- Methods are non-mutating by default; add `inplace` only when mutation is required.

### Imports
- Order groups: standard library, third-party, local.
- Keep a single blank line between groups.
- Prefer absolute imports within a package.
- Avoid circular imports; refactor shared utilities when needed.

### Formatting
- PEP 8 layout: 4 spaces, consistent indentation.
- Use multi-line function signatures when lines are long.
- Use trailing commas in multi-line literals and call sites.
- Match the existing file’s quoting style.
- Avoid reformatting unrelated code.

### Types
- Use type hints on public APIs and non-trivial helpers.
- Prefer `Optional[T]`, `Literal`, `Tuple`, and `list[T]` where precise.
- Annotate pandas objects as `pd.DataFrame`/`pd.Series`.
- Avoid `Any` unless interoperability forces it.
- Keep return types explicit when the function returns a DataFrame copy.

### Naming
- Classes: `PascalCase`.
- Functions/methods/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Public API names should align with pandas/sklearn conventions.
- Keep abbreviations meaningful and consistent (`uwy`, `ldf`, `ibnr`).

### Error handling
- Validate inputs early; raise `ValueError` for invalid parameters.
- Use `FileNotFoundError` for missing files or paths.
- Include the invalid value in error messages.
- Use `logging` for diagnostics; avoid `print` in library code.
- Catch broad exceptions only at boundaries (CLI, UI, IO).

### Data handling
- Prefer immutable operations; return copies rather than mutating inputs.
- Use `.copy()` when returning internal DataFrames.
- Keep column names consistent with chainladder: `incurred`, `paid`, `outstanding`, `Premium_selected`.
- Coerce numeric columns to floats and NaN-safe values before modeling.
- Avoid hidden state in model objects; expose results via accessors.

### Modeling patterns
- Use `cl.Pipeline` for chained development/tail/model steps.
- Keep parameter validation in the public API methods.
- Prefer vectorized pandas/NumPy operations over loops when practical.
- Keep transformations reproducible and deterministic.

### Docs and notebooks
- Doc changes should update both docstrings and relevant docs pages.
- Notebooks live under `chainladder-python/docs/` and should run top to bottom.
- Keep example data small and deterministic when adding new examples.

### Tests
- Use pytest; place tests under `chainladder/**/tests`.
- Use `@pytest.mark.parametrize` for variations.
- Use `pytest.raises` for error conditions.
- Keep tests fast and isolated; avoid network or large data dependencies.

## Local `source/` conventions
- Code here is WIP; some imports may be broken.
- Keep changes minimal and focused; avoid large refactors unless requested.
- Favor small, testable functions with explicit inputs.
- Keep domain-specific logic in `source/` and avoid changing upstream unless needed.
- When adding new modules, mirror the existing `source/` layout and naming.
- Use `logging` consistently; prefer `logging.info`/`warning` over `print`.

## Configuration and IO
- `source/config_manager.py` reads YAML config and creates needed paths.
- Use `pathlib.Path` for file paths; avoid hard-coded absolute paths.
- Prefer `yaml.safe_load` and `yaml.safe_dump` for config files.
- Add new config keys with sane defaults and document them.

## Data schema hints
- Claims data uses `uw_year` and `period` columns.
- Premium data uses `GWP`, `EPI`, `GWP_Forecast`, `Premium_selected`.
- Keep column naming consistent when merging datasets.

## Logging
- Library code should be quiet by default.
- Prefer module-level loggers: `logging.getLogger(__name__)`.
- Avoid logging sensitive or personal data.

## Common pitfalls
- `chainladder` objects expect cumulative vs incremental triangles; keep the mode explicit.
- Tail curves and development drops should be validated before fitting.
- Be cautious when coercing date/period fields; preserve UWY semantics.

## Repository notes
- No Cursor rules found in `.cursor/rules/` or `.cursorrules`.
- No Copilot instructions found in `.github/copilot-instructions.md`.
- If you add tooling or workflow changes, update `AGENTS.md`.
