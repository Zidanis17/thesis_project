from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .examples import SHOWCASE_EXAMPLES
from .runner import InputModeHint, ScenarioDomainError, ShowcaseRuntime

__all__ = ["app", "create_app"]


class ScenarioRunRequest(BaseModel):
    input: str | dict[str, Any]
    input_mode_hint: InputModeHint = "auto"

    model_config = ConfigDict(extra="forbid")


def get_runtime(request: Request) -> ShowcaseRuntime:
    return request.app.state.runtime


def create_app(runtime: ShowcaseRuntime | None = None) -> FastAPI:
    app = FastAPI(
        title="AV Ethics Pipeline API",
        version="1.0.0",
        description="FastAPI backend for the AV ethics thesis showcase.",
    )
    app.state.runtime = runtime or ShowcaseRuntime()

    @app.get("/api/v1/health")
    def health(runtime: ShowcaseRuntime = Depends(get_runtime)) -> dict[str, Any]:
        return runtime.health_payload()

    @app.get("/api/v1/examples")
    def examples() -> dict[str, Any]:
        return {"examples": SHOWCASE_EXAMPLES}

    @app.post("/api/v1/scenario/run", response_model=None)
    def run_scenario(
        request_payload: ScenarioRunRequest,
        runtime: ShowcaseRuntime = Depends(get_runtime),
    ) -> Any:
        try:
            return runtime.run(request_payload.input, request_payload.input_mode_hint)
        except ScenarioDomainError as exc:
            return JSONResponse(status_code=400, content=exc.payload)

    return app


app = create_app()
