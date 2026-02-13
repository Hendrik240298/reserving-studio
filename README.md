# reserving-studio

This is currently a personal project to build a reserving workflow, including a minimalistic GUI, to familiarize myself with and learn the amazing `chainladder-python` package for actuarial loss reserving.

## Preview

The Data tab shows claims and premium triangles. You can toggle between incremental and cumulative triangles. Each triangle can also be viewed in relation to another one, for example incurred in relation to premium.

<img src=".github/images/data-tab.png" alt="Data tab preview" width="100%" />

The Chainladder tab displays link ratios, loss development factors, and fitted projections. Different weighting schemes can be applied, and selected link ratios can be dropped. For tail estimation, you can choose different methods, set the fitting interval, and define the starting point.

<img src=".github/images/chainladder-tab.png" alt="Chainladder tab preview" width="100%" />

<img src=".github/images/bornhuetter-tab.png" alt="Bornhuetter tab preview" width="100%" />

<img src=".github/images/results-tab.png" alt="Results tab preview" width="100%" />

# Get started

## Install (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Activate environment

```bash
source .venv/bin/activate
```

## Run server (Dash)

```bash
python -m source.app
```

Open http://127.0.0.1:8050

## Run dashboard E2E tests

The E2E suite uses Playwright to open the Dash app in a real Chromium browser and verify key user flows deterministically (drop selection recalculation and BF apriori-driven results updates).

```bash
uv run python -m playwright install chromium
uv run pytest tests/e2e -m e2e -q
```

## Optional: custom config path

```bash
RESERVING_CONFIG=config.yml uv run python -m source.app
```
