from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(
    path: str | Path = ".env",
    *,
    override: bool = False,
) -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = value.strip().strip('"').strip("'")
        if not env_key:
            continue
        if env_key in os.environ and not override:
            continue
        os.environ[env_key] = env_value
