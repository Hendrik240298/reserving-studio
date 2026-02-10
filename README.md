# reserving-studio

This is currently a personal project to build a reserving workflow, including a minimalistic GUI, to familiarize myself with and learn the amazing ´chainladder-python´ package for actuarial loss reserving.

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

## Optional: custom config path

```bash
RESERVING_CONFIG=config.yml uv run python -m source.app
```
