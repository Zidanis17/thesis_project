from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .examples import SHOWCASE_EXAMPLES, SHOWCASE_SUBDIVISIONS, get_examples_by_subdivision
from .runner import InputModeHint, ScenarioDomainError, ShowcaseRuntime
from .storage import ScenarioRunStore

__all__ = ["app", "create_app"]


class ScenarioRunRequest(BaseModel):
    input: str | dict[str, Any]
    input_mode_hint: InputModeHint = "auto"

    model_config = ConfigDict(extra="forbid")


class SubdivisionRunRequest(BaseModel):
    subdivision_id: str

    model_config = ConfigDict(extra="forbid")


def get_runtime(request: Request) -> ShowcaseRuntime:
    return request.app.state.runtime


def get_run_store(request: Request) -> ScenarioRunStore:
    return request.app.state.run_store


def create_app(
    runtime: ShowcaseRuntime | None = None,
    run_store: ScenarioRunStore | None = None,
) -> FastAPI:
    app = FastAPI(
        title="AV Ethics Pipeline API",
        version="1.0.0",
        description="FastAPI backend for the AV ethics thesis showcase.",
    )
    app.state.runtime = runtime or ShowcaseRuntime()
    app.state.run_store = run_store or ScenarioRunStore()

    @app.get("/api/v1/health")
    def health(runtime: ShowcaseRuntime = Depends(get_runtime)) -> dict[str, Any]:
        return runtime.health_payload()

    @app.get("/api/v1/examples")
    def examples() -> dict[str, Any]:
        return {
            "examples": SHOWCASE_EXAMPLES,
            "subdivisions": SHOWCASE_SUBDIVISIONS,
        }

    @app.post("/api/v1/scenario/run", response_model=None)
    def run_scenario(
        request_payload: ScenarioRunRequest,
        runtime: ShowcaseRuntime = Depends(get_runtime),
        run_store: ScenarioRunStore = Depends(get_run_store),
    ) -> Any:
        model_name = runtime.reasoning_llm.model_name if runtime.reasoning_llm is not None else None
        try:
            payload = runtime.run(request_payload.input, request_payload.input_mode_hint)
            record = run_store.save_run(
                request_input=request_payload.input,
                input_mode_hint=request_payload.input_mode_hint,
                payload=payload,
                status="success",
                model_name=model_name,
            )
            return {
                "run": record.to_dict(),
                **payload,
            }
        except ScenarioDomainError as exc:
            record = run_store.save_run(
                request_input=request_payload.input,
                input_mode_hint=request_payload.input_mode_hint,
                payload=exc.payload,
                status="error",
                model_name=model_name,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "run": record.to_dict(),
                    **exc.payload,
                },
            )

    @app.get("/api/v1/scenario/runs")
    def list_scenario_runs(
        limit: int = Query(default=25, ge=1, le=200),
        run_store: ScenarioRunStore = Depends(get_run_store),
    ) -> dict[str, Any]:
        return run_store.list_runs(limit=limit)

    @app.get("/api/v1/scenario/runs/{run_id}")
    def get_scenario_run(
        run_id: str,
        run_store: ScenarioRunStore = Depends(get_run_store),
    ) -> Any:
        payload = run_store.get_run(run_id)
        if payload is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "run_not_found",
                        "message": f"Run '{run_id}' was not found.",
                    }
                },
            )
        return payload

    @app.post("/api/v1/scenario/subdivision/run", response_model=None)
    def run_subdivision(
        request_payload: SubdivisionRunRequest,
        runtime: ShowcaseRuntime = Depends(get_runtime),
    ) -> Any:
        subdivision = next(
            (item for item in SHOWCASE_SUBDIVISIONS if item["id"] == request_payload.subdivision_id),
            None,
        )
        if subdivision is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "unknown_subdivision",
                        "message": f"Subdivision '{request_payload.subdivision_id}' was not found.",
                    }
                },
            )

        try:
            return runtime.run_subdivision(
                subdivision=subdivision,
                examples=get_examples_by_subdivision(subdivision["id"]),
            )
        except ScenarioDomainError as exc:
            return JSONResponse(status_code=400, content=exc.payload)

    return app


app = create_app()
