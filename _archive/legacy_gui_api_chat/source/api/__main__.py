from __future__ import annotations

import logging
import os


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = os.environ.get("RESERVING_API_HOST", "127.0.0.1")
    port = int(os.environ.get("RESERVING_API_PORT", "8000"))
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "uvicorn is required to run the API server. Install it with 'uv pip install uvicorn'."
        ) from error

    uvicorn.run("source.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
