# Actuary Quickstart

This guide gets you from clone to a working reserving UI quickly.

If you want the full own-data onboarding (data preparation, config patterns, and roadblocks), start with `docs/start-with-your-data.md`.

## 1) Install and start

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -m source.app
```

Open `http://127.0.0.1:8050`.

## 2) What loads by default

- Claims: chainladder quarterly sample claims.
- Premium: `data/quarterly_premium.csv`.
- Config: `config.yml` (or `RESERVING_CONFIG` if set).
- Session state: file in `sessions/` for the active segment.

## 3) Basic actuarial workflow in the UI

1. **Data tab**: inspect claims and premium triangles.
2. **Chainladder tab**: adjust drops/averaging/tail assumptions.
3. **Bornhuetter-Ferguson tab**: set apriori factors by UWY.
4. **Results tab**: compare outcomes and set selected method by UWY.

The UI recalculates on updates and persists parameters in the session YAML.

## 4) Finalize from script-driven mode

If you launch from an example script, click **Finalize & Continue** in Results.
The script resumes with finalized payloads and a numeric results dataframe.

Example launchers:

```bash
uv run python examples/run_quarterly_interactive.py
uv run python examples/run_clrd_interactive.py
uv run python examples/run_sql_interactive.py
```

## 5) Most common first issues

- Missing Playwright browser (for E2E tests only):

```bash
uv run python -m playwright install chromium
```

- Wrong config path: set `RESERVING_CONFIG` explicitly.

```bash
RESERVING_CONFIG=config.yml uv run python -m source.app
```
