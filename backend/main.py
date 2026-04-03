from __future__ import annotations

from pathlib import Path

import uvicorn

from thesis.api import app

__all__ = ["app"]


def main() -> None:
    backend_dir = Path(__file__).resolve().parent
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)],
    )


if __name__ == "__main__":
    main()
