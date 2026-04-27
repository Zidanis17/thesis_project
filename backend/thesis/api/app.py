from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .examples import SHOWCASE_EXAMPLES, SHOWCASE_SUBDIVISIONS, get_examples_by_subdivision, get_scenario_bank_examples
from .runner import EvaluationVariant, InputModeHint, ScenarioDomainError, ShowcaseRuntime
from .storage import ScenarioRunStore

__all__ = ["app", "create_app"]


class ScenarioRunRequest(BaseModel):
    input: str | dict[str, Any]
    input_mode_hint: InputModeHint = "auto"

    model_config = ConfigDict(extra="forbid")


class SubdivisionRunRequest(BaseModel):
    subdivision_id: str
    variant: EvaluationVariant = "full_system"

    model_config = ConfigDict(extra="forbid")


class ScenarioBankRunRequest(BaseModel):
    variant: EvaluationVariant = "full_system"

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
        run_store: ScenarioRunStore = Depends(get_run_store),
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
            payload = runtime.run_subdivision(
                subdivision=subdivision,
                examples=get_examples_by_subdivision(subdivision["id"]),
                variant=request_payload.variant,
            )
            model_name = runtime.reasoning_llm.model_name if runtime.reasoning_llm is not None else None
            return run_store.save_evaluation_run(
                payload=payload,
                scope="subdivision",
                subdivision_id=subdivision["id"],
                variant=request_payload.variant,
                model_name=model_name,
            )
        except ScenarioDomainError as exc:
            return JSONResponse(status_code=400, content=exc.payload)

    @app.post("/api/v1/scenario/bank/run", response_model=None)
    def run_scenario_bank(
        request_payload: ScenarioBankRunRequest | None = None,
        runtime: ShowcaseRuntime = Depends(get_runtime),
        run_store: ScenarioRunStore = Depends(get_run_store),
    ) -> Any:
        request_payload = request_payload or ScenarioBankRunRequest()
        try:
            payload = runtime.run_scenario_bank(
                examples=get_scenario_bank_examples(),
                variant=request_payload.variant,
            )
            model_name = runtime.reasoning_llm.model_name if runtime.reasoning_llm is not None else None
            return run_store.save_evaluation_run(
                payload=payload,
                scope="full_bank",
                subdivision_id=None,
                variant=request_payload.variant,
                model_name=model_name,
            )
        except ScenarioDomainError as exc:
            return JSONResponse(status_code=400, content=exc.payload)

    @app.get("/api/v1/evaluations")
    def list_evaluation_runs(
        limit: int = Query(default=25, ge=1, le=200),
        run_store: ScenarioRunStore = Depends(get_run_store),
    ) -> dict[str, Any]:
        return run_store.list_evaluation_runs(limit=limit)

    @app.get("/api/v1/evaluations/{evaluation_id}")
    def get_evaluation_run(
        evaluation_id: str,
        run_store: ScenarioRunStore = Depends(get_run_store),
    ) -> Any:
        payload = run_store.get_evaluation_run(evaluation_id)
        if payload is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "evaluation_not_found",
                        "message": f"Evaluation run '{evaluation_id}' was not found.",
                    }
                },
            )
        return payload

    return app


app = create_app()
