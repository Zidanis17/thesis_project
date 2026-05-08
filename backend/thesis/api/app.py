from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .examples import SHOWCASE_EXAMPLES, SHOWCASE_SUBDIVISIONS, get_examples_by_subdivision, get_scenario_bank_examples
from .runner import EvaluationVariant, InputModeHint, ScenarioDomainError, ShowcaseRuntime, apply_field_ablation
from .storage import ScenarioRunStore

__all__ = ["app", "create_app"]

ABLATABLE_FIELDS: list[dict[str, Any]] = [
    {"path": "ego_vehicle.speed_kmh",        "label": "Ego speed",           "group": "ego_vehicle"},
    {"path": "ego_vehicle.acceleration_ms2", "label": "Ego acceleration",    "group": "ego_vehicle"},
    {"path": "ego_vehicle.mass_kg",          "label": "Ego mass",            "group": "ego_vehicle"},
    {"path": "ego_vehicle.braking_distance_m","label": "Braking distance",   "group": "ego_vehicle"},
    {"path": "ego_vehicle.passenger_at_risk","label": "Passenger at risk",   "group": "ego_vehicle"},
    {"path": "obstacles.mass_kg",            "label": "Obstacle mass",       "group": "obstacles"},
    {"path": "obstacles.time_to_impact_s",   "label": "Time to impact",      "group": "obstacles"},
    {"path": "obstacles.responsible_for_risk","label": "Obstacle responsibility","group": "obstacles"},
    {"path": "sensor_confidence.lidar",      "label": "LIDAR confidence",    "group": "sensor"},
    {"path": "sensor_confidence.camera",     "label": "Camera confidence",   "group": "sensor"},
    {"path": "sensor_confidence.radar",      "label": "Radar confidence",    "group": "sensor"},
    {"path": "sensor_confidence.overall_scene_confidence","label": "Overall scene confidence","group": "sensor"},
    {"path": "collision_unavoidable",        "label": "Collision unavoidable flag","group": "meta"},
]


class ScenarioRunRequest(BaseModel):
    input: str | dict[str, Any]
    input_mode_hint: InputModeHint = "auto"
    variant: EvaluationVariant = "full_system"
    ablation_fields: list[str] | None = None

    model_config = ConfigDict(extra="forbid")


class SubdivisionRunRequest(BaseModel):
    subdivision_id: str
    variant: EvaluationVariant = "full_system"

    model_config = ConfigDict(extra="forbid")


class ScenarioBankRunRequest(BaseModel):
    variant: EvaluationVariant = "full_system"

    model_config = ConfigDict(extra="forbid")


class FieldAblationCompareRequest(BaseModel):
    input: str | dict[str, Any]
    input_mode_hint: InputModeHint = "auto"
    ablation_groups: list[list[str]]

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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

    @app.get("/api/v1/ablation/fields")
    def list_ablatable_fields() -> dict[str, Any]:
        return {"fields": ABLATABLE_FIELDS}

    @app.post("/api/v1/ablation/field-compare", response_model=None)
    def field_ablation_compare(
        request_payload: FieldAblationCompareRequest,
        runtime: ShowcaseRuntime = Depends(get_runtime),
    ) -> Any:
        groups: list[list[str]] = [[], *request_payload.ablation_groups]
        group_labels = ["Baseline (all fields)", *(
            ", ".join(g) if g else "empty" for g in request_payload.ablation_groups
        )]

        results: list[dict[str, Any]] = []
        for i, fields_to_remove in enumerate(groups):
            started = perf_counter()
            raw_input = request_payload.input
            if fields_to_remove and isinstance(raw_input, dict):
                raw_input = apply_field_ablation(raw_input, fields_to_remove)
            try:
                pipeline_payload = runtime.run(
                    raw_input,
                    request_payload.input_mode_hint,
                    variant="full_system",
                )
                duration_ms = max(1, int(round((perf_counter() - started) * 1000)))
                summary = pipeline_payload.get("summary", {})
                reasoning = pipeline_payload.get("artifacts", {}).get("reasoning_result", {})
                math = pipeline_payload.get("artifacts", {}).get("mathematical_layer_result", {})
                results.append({
                    "ablation_label": group_labels[i],
                    "fields_removed": fields_to_remove,
                    "dominant_framework": summary.get("dominant_framework"),
                    "deterministic_best_action": summary.get("deterministic_best_action"),
                    "confidence": reasoning.get("confidence"),
                    "rationale": reasoning.get("rationale"),
                    "weights": reasoning.get("weights") or {},
                    "math_runtime_available": summary.get("math_runtime_available", True),
                    "reasoning_runtime_available": summary.get("reasoning_runtime_available", False),
                    "rag_runtime_available": summary.get("rag_runtime_available", False),
                    "status": "success",
                    "error_code": None,
                    "error_message": None,
                    "duration_ms": duration_ms,
                    "best_action_by_total_risk": math.get("best_action_by_total_risk"),
                })
            except ScenarioDomainError as exc:
                duration_ms = max(1, int(round((perf_counter() - started) * 1000)))
                error = exc.payload.get("error", {})
                results.append({
                    "ablation_label": group_labels[i],
                    "fields_removed": fields_to_remove,
                    "dominant_framework": None,
                    "deterministic_best_action": None,
                    "confidence": None,
                    "rationale": None,
                    "weights": {},
                    "math_runtime_available": False,
                    "reasoning_runtime_available": False,
                    "rag_runtime_available": False,
                    "status": "error",
                    "error_code": error.get("code"),
                    "error_message": error.get("message"),
                    "duration_ms": duration_ms,
                    "best_action_by_total_risk": None,
                })

        return {"baseline": results[0], "ablations": results[1:]}

    @app.post("/api/v1/scenario/run", response_model=None)
    def run_scenario(
        request_payload: ScenarioRunRequest,
        runtime: ShowcaseRuntime = Depends(get_runtime),
        run_store: ScenarioRunStore = Depends(get_run_store),
    ) -> Any:
        model_name = runtime.reasoning_llm.model_name if runtime.reasoning_llm is not None else None
        run_input = request_payload.input
        if request_payload.ablation_fields and isinstance(run_input, dict):
            run_input = apply_field_ablation(run_input, request_payload.ablation_fields)
        try:
            payload = runtime.run(
                run_input,
                request_payload.input_mode_hint,
                variant=request_payload.variant,
            )
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
