from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

__all__ = ["load_project_env"]


def load_project_env(
    env_path: str | Path | None = None,
    *,
    override: bool = False,
) -> None:
    if env_path is None:
        resolved_env_path = Path(__file__).resolve().parents[1] / ".env"
    else:
        resolved_env_path = Path(env_path).resolve()

    if resolved_env_path.exists():
        load_dotenv(dotenv_path=resolved_env_path, override=override)
