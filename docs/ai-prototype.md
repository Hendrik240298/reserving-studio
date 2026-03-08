# AI Prototype Quickstart

This guide runs the first end-to-end prototype:

- Reserving backend exposed through API contracts.
- OpenRouter-powered assistant calling API tools.
- Deterministic diagnostics returned to the assistant.

## 1) Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2) Run the reserving API

```bash
uv run python -m source.api
```

By default this starts on `http://127.0.0.1:8000`.

Optional environment overrides:

- `RESERVING_API_HOST`
- `RESERVING_API_PORT`
- `RESERVING_CONFIG` (path to config yaml)

## 3) Configure OpenRouter

```bash
export OPENROUTER_API_KEY="<your-key>"
export AI_MODEL="minimax/minimax-m2.5"
```

Recommended local setup is a `.env` file in repo root:

```dotenv
OPENROUTER_API_KEY=<your-key>
AI_MODEL=minimax/minimax-m2.5
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_APP_TITLE=reserving-studio
```

The assistant CLI auto-loads `.env` and `.env.local`.

Optional:

- `OPENROUTER_BASE_URL` (defaults to `https://openrouter.ai/api/v1`)
- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_TITLE`
- `AI_OBSERVABILITY` (`1` default, set `0` to disable verbose tool-call logs)
- `RESERVING_OBSERVABILITY` (`1` default, set `0` to disable API scenario trace logs)

## 4) Run assistant CLI with sample data

```bash
uv run python -m ai.assistant_cli --use-chainladder-sample --segment motor
```

This command will:

1. Create a reserving workflow through `POST /v1/workflows/from-dataframes`.
2. Start an AI loop with tool calling.
3. Ask the model to fetch session details, run diagnostics, and run iterative scenario search.
4. Print workflow metadata and final commentary.

## 4b) Run the quarterly config dataset example

```bash
uv run python -m examples.run_quarterly_ai_assistant
```

This uses `examples/config_quarterly.yml` with `source.input_loader` to mirror the same dataset pathing as `examples/run_quarterly_interactive.py`.

## 5) Run assistant CLI with your own data

```bash
uv run python -m ai.assistant_cli \
  --claims-csv path/to/claims.csv \
  --premium-csv path/to/premium.csv \
  --segment your_segment
```

### Expected claims columns

- `uw_year`
- `period`
- `incurred`
- `paid`
- `outstanding`

### Expected premium columns

- `uw_year`
- `period`
- `Premium_selected`

Other premium columns can be present.

## Notes

- AI output is advisory and grounded on deterministic backend diagnostics.
- Iterative diagnostics search (`POST /v1/diagnostics/iterate`) tests drop, tail-fit, and BF apriori candidates.
- Diagnostics include latest diagonal actual-vs-expected checks and incurred-on-premium development comparisons by age.
- Incurred-on-premium diagnostics run on both cumulative and incremental bases at matched development ages.
- Tail recommendations include candidate fit intervals and a deterministic recommended fit period.
- BF apriori recommendations are auto-completed across origins during scenario runs.
- Full observability mode is enabled by default and logs tool calls plus scenario-by-scenario diagnostics iteration traces.
- API and GUI/backend remain AI-agnostic by design.
- This is a prototype; figures are currently returned as payload data records rather than full plotting objects.
