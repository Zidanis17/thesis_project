from __future__ import annotations

from collections import Counter
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from ..mathematical_layer import DeterministicMathematicalLayer
from ..pipeline import ScenarioPipelineResult
from ..rag import DeterministicRAGRetriever
from ..reasoning_llm import EthicalReasoningLLM, EthicalReasoningResult
from ..scenario_parser import DeterministicScenarioParser, ScenarioParseError
from .serializers import (
    build_summary_payload,
    coerce_input_snapshot,
    summarize_rag_result,
    summarize_reasoning_result,
)

InputModeHint = Literal["auto", "json", "text"]

__all__ = [
    "InputModeHint",
    "ScenarioDomainError",
    "ShowcaseRuntime",
]


_FRAMEWORK_LABELS: dict[str, str] = {
    "EF-01": "Utilitarian Risk Minimization",
    "EF-02": "Deontological Rule-Based Safety",
    "EF-03": "Rawlsian Maximin",
    "EF-04": "Ethics of Risk",
    "EF-05": "Ethical Valence Theory",
    "EF-06": "Virtue Ethics",
}
_UNRESOLVED_FRAMEWORK_KEY = "__unresolved__"


class ScenarioDomainError(ValueError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("error", {}).get("message", "Scenario domain error"))
        self.payload = payload


@dataclass(slots=True)
class ShowcaseRuntime:
    parser: DeterministicScenarioParser
    mathematical_layer: DeterministicMathematicalLayer
    rag_retriever: DeterministicRAGRetriever | None
    reasoning_llm: EthicalReasoningLLM | None

    def __init__(
        self,
        *,
        parser: DeterministicScenarioParser | None = None,
        mathematical_layer: DeterministicMathematicalLayer | None = None,
        rag_retriever: DeterministicRAGRetriever | None = None,
        reasoning_llm: EthicalReasoningLLM | None = None,
    ) -> None:
        self.parser = parser or DeterministicScenarioParser()
        self.mathematical_layer = mathematical_layer or DeterministicMathematicalLayer()
        self.rag_retriever = rag_retriever if rag_retriever is not None else DeterministicRAGRetriever()
        self.reasoning_llm = (
            reasoning_llm
            if reasoning_llm is not None
            else EthicalReasoningLLM(model_name="gpt-5.4-mini", temperature=0.0)
        )

    def health_payload(self) -> dict[str, Any]:
        rag_available = bool(self.rag_retriever and self.rag_retriever.vector_store is not None)
        reasoning_available = bool(self.reasoning_llm and self.reasoning_llm.client is not None)

        warnings: list[str] = []
        if self.rag_retriever and self.rag_retriever._runtime_error is not None:
            warnings.append(str(self.rag_retriever._runtime_error))
        if self.reasoning_llm and self.reasoning_llm._runtime_error is not None:
            warnings.append(str(self.reasoning_llm._runtime_error))

        return {
            "status": "ok",
            "knowledge_base_path": (
                str(self.rag_retriever.knowledge_base_path) if self.rag_retriever is not None else None
            ),
            "rag": {
                "runtime_available": rag_available,
                "runtime_error": str(self.rag_retriever._runtime_error)
                if self.rag_retriever and self.rag_retriever._runtime_error is not None
                else None,
            },
            "reasoning": {
                "runtime_available": reasoning_available,
                "runtime_error": str(self.reasoning_llm._runtime_error)
                if self.reasoning_llm and self.reasoning_llm._runtime_error is not None
                else None,
                "model_name": self.reasoning_llm.model_name if self.reasoning_llm is not None else None,
            },
            "warnings": warnings,
        }

    def run(self, payload: str | dict[str, Any], input_mode_hint: InputModeHint = "auto") -> dict[str, Any]:
        replay: list[dict[str, Any]] = []
        started_at = perf_counter()

        snapshot: dict[str, Any] = {
            "input": coerce_input_snapshot(payload, input_mode_hint=input_mode_hint),
        }
        replay.append(
            _stage(
                stage_id="input",
                label="Input",
                status="success",
                started_at=started_at,
                ended_at=started_at,
                headline=_input_headline(payload),
                snapshot=snapshot,
                previous_snapshot={},
                metrics=_input_metrics(payload),
            )
        )

        try:
            prepared_input = _prepare_input(payload, input_mode_hint)
        except ScenarioDomainError as exc:
            error_snapshot = {
                **snapshot,
                "error": exc.payload["error"],
            }
            replay.append(
                _stage(
                    stage_id="parser",
                    label="Parser",
                    status="error",
                    started_at=started_at,
                    ended_at=perf_counter(),
                    headline=exc.payload["error"]["message"],
                    snapshot=error_snapshot,
                    previous_snapshot=snapshot,
                    metrics={},
                )
            )
            exc.payload["replay"] = replay
            raise

        parser_started = perf_counter()
        try:
            parser_result = self.parser.parse(prepared_input)
        except ScenarioParseError as exc:
            parser_snapshot = {
                **snapshot,
                "error": {
                    "code": "scenario_parse_error",
                    "message": str(exc),
                },
            }
            replay.append(
                _stage(
                    stage_id="parser",
                    label="Parser",
                    status="error",
                    started_at=parser_started,
                    ended_at=perf_counter(),
                    headline=str(exc),
                    snapshot=parser_snapshot,
                    previous_snapshot=snapshot,
                    metrics={},
                )
            )
            raise ScenarioDomainError(
                {
                    "error": {
                        "code": "scenario_parse_error",
                        "message": str(exc),
                    },
                    "replay": replay,
                }
            ) from exc

        parser_payload = parser_result.to_dict()
        parser_snapshot = {**snapshot, "parser_result": parser_payload}
        replay.append(
            _stage(
                stage_id="parser",
                label="Parser",
                status="success",
                started_at=parser_started,
                ended_at=perf_counter(),
                headline=(
                    f"Parsed {len(parser_result.scenario.obstacles)} obstacle(s) from "
                    f"{parser_result.input_mode} input."
                ),
                snapshot=parser_snapshot,
                previous_snapshot=snapshot,
                metrics={
                    "input_mode": parser_result.input_mode,
                    "obstacles": len(parser_result.scenario.obstacles),
                    "warnings": len(parser_result.warnings),
                },
            )
        )
        snapshot = parser_snapshot

        math_started = perf_counter()
        math_result = self.mathematical_layer.analyze(parser_result.scenario)
        math_payload = math_result.to_dict()
        math_snapshot = {**snapshot, "mathematical_layer_result": math_payload}
        replay.append(
            _stage(
                stage_id="math",
                label="Math",
                status="success",
                started_at=math_started,
                ended_at=perf_counter(),
                headline=f"Computed action risks. Best deterministic action: {math_result.best_action_by_total_risk}.",
                snapshot=math_snapshot,
                previous_snapshot=snapshot,
                metrics={
                    "best_action": math_result.best_action_by_total_risk,
                    "actions": len(math_result.action_assessments),
                    "violations": len(math_result.violated_rules),
                },
            )
        )
        snapshot = math_snapshot

        rag_started = perf_counter()
        rag_result = None
        rag_payload: dict[str, Any]
        rag_status = "skipped"
        rag_headline = "RAG stage skipped."
        rag_metrics: dict[str, Any] = {}

        if self.rag_retriever is not None:
            try:
                rag_result = self.rag_retriever.retrieve(parser_result.scenario, math_result)
                rag_payload = summarize_rag_result(rag_result)
                rag_status = "success" if rag_payload.get("runtime_available") else "warning"
                rag_headline = (
                    f"Retrieved {rag_payload.get('frameworks_retrieved', 0)} framework(s) "
                    f"and {rag_payload.get('supporting_docs_retrieved', 0)} supporting document(s)."
                )
                if not rag_payload.get("runtime_available"):
                    rag_headline = rag_payload.get("runtime_error") or "RAG runtime unavailable."
                rag_metrics = {
                    "frameworks": rag_payload.get("frameworks_retrieved", 0),
                    "supporting_docs": rag_payload.get("supporting_docs_retrieved", 0),
                    "runtime_available": rag_payload.get("runtime_available", False),
                }
            except Exception as exc:
                rag_payload = {
                    "runtime_available": False,
                    "runtime_error": str(exc),
                    "frameworks_retrieved": 0,
                    "supporting_docs_retrieved": 0,
                    "frameworks": [],
                    "supporting_documents": [],
                }
                rag_status = "warning"
                rag_headline = str(exc)
                rag_metrics = {
                    "frameworks": 0,
                    "supporting_docs": 0,
                    "runtime_available": False,
                }
        else:
            rag_payload = {"runtime_status": "not_requested", "reason": "RAG stage not provided."}

        rag_snapshot = {**snapshot, "rag_retrieval_result": rag_payload}
        replay.append(
            _stage(
                stage_id="rag",
                label="RAG",
                status=rag_status,
                started_at=rag_started,
                ended_at=perf_counter(),
                headline=rag_headline,
                snapshot=rag_snapshot,
                previous_snapshot=snapshot,
                metrics=rag_metrics,
            )
        )
        snapshot = rag_snapshot

        reasoning_started = perf_counter()
        reasoning_result: EthicalReasoningResult | None = None
        reasoning_payload: dict[str, Any]
        reasoning_status = "skipped"
        reasoning_headline = "Reasoning stage skipped."
        reasoning_metrics: dict[str, Any] = {}

        if self.reasoning_llm is not None:
            try:
                reasoning_result = self.reasoning_llm.reason(parser_result, math_result, rag_result)
                reasoning_payload = summarize_reasoning_result(reasoning_result)
                reasoning_status = "success" if reasoning_payload.get("runtime_available") else "warning"
                if reasoning_payload.get("runtime_available"):
                    reasoning_headline = (
                        f"Resolved dominant framework: {reasoning_payload.get('dominant_framework')}."
                    )
                else:
                    reasoning_headline = reasoning_payload.get("runtime_error") or "Reasoning runtime unavailable."
                reasoning_metrics = {
                    "dominant_framework": reasoning_payload.get("dominant_framework"),
                    "runtime_available": reasoning_payload.get("runtime_available", False),
                }
            except Exception as exc:
                reasoning_payload = {
                    "runtime_available": False,
                    "runtime_error": str(exc),
                    "dominant_framework": None,
                    "contributing_frameworks": [],
                    "weights": {},
                    "weights_reasoning": "",
                    "risk_scores_per_action": math_result.risk_score_matrix,
                    "rationale": "",
                    "confidence": None,
                    "violated_constraints": [],
                }
                reasoning_status = "warning"
                reasoning_headline = str(exc)
                reasoning_metrics = {
                    "dominant_framework": None,
                    "runtime_available": False,
                }
        else:
            reasoning_payload = {"runtime_status": "not_requested", "reason": "Reasoning stage not provided."}

        reasoning_snapshot = {**snapshot, "reasoning_result": reasoning_payload}
        replay.append(
            _stage(
                stage_id="reasoning",
                label="Reasoning",
                status=reasoning_status,
                started_at=reasoning_started,
                ended_at=perf_counter(),
                headline=reasoning_headline,
                snapshot=reasoning_snapshot,
                previous_snapshot=snapshot,
                metrics=reasoning_metrics,
            )
        )
        snapshot = reasoning_snapshot

        result = ScenarioPipelineResult(
            parser_result=parser_result,
            mathematical_layer_result=math_result,
            rag_retrieval_result=rag_result,
            reasoning_result=reasoning_result,
        )
        summary_payload = build_summary_payload(
            result,
            rag_payload=rag_payload,
            reasoning_payload=reasoning_payload,
        )
        artifacts = {
            "parser_result": parser_payload,
            "mathematical_layer_result": math_payload,
            "rag_retrieval_result": rag_payload,
            "reasoning_result": reasoning_payload,
        }

        completed_snapshot = {**snapshot, "summary": summary_payload}
        replay.append(
            _stage(
                stage_id="complete",
                label="Complete",
                status="success",
                started_at=started_at,
                ended_at=perf_counter(),
                headline="Pipeline response ready for replay.",
                snapshot=completed_snapshot,
                previous_snapshot=snapshot,
                metrics={
                    "deterministic_best_action": summary_payload.get("deterministic_best_action"),
                    "dominant_framework": summary_payload.get("dominant_framework"),
                },
            )
        )

        return {
            "summary": summary_payload,
            "artifacts": artifacts,
            "replay": replay,
        }

    def run_subdivision(
        self,
        *,
        subdivision: dict[str, Any],
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        subdivision_id = str(subdivision["id"])
        if not examples:
            raise ScenarioDomainError(
                {
                    "error": {
                        "code": "unknown_subdivision",
                        "message": f"No scenarios found for subdivision '{subdivision_id}'.",
                    },
                    "replay": [],
                }
            )

        started_at = perf_counter()
        scenario_results: list[dict[str, Any]] = []
        framework_counts: Counter[str] = Counter()
        completed_runs = 0
        reasoning_runtime_ready = 0
        rag_runtime_ready = 0

        for example in examples:
            scenario_started = perf_counter()
            try:
                result = self.run(example["value"], example.get("mode", "auto"))
                summary = result["summary"]
                duration_ms = max(1, int(round((perf_counter() - scenario_started) * 1000)))
                dominant_framework = summary.get("dominant_framework")
                framework_key = dominant_framework or _UNRESOLVED_FRAMEWORK_KEY
                framework_counts[framework_key] += 1
                completed_runs += 1

                if summary.get("reasoning_runtime_available"):
                    reasoning_runtime_ready += 1
                if summary.get("rag_runtime_available"):
                    rag_runtime_ready += 1

                scenario_results.append(
                    {
                        "scenario_id": example["id"],
                        "scenario_label": example["label"],
                        "subdivision_id": example.get("subdivision_id"),
                        "subdivision_label": example.get("subdivision_label"),
                        "expected_framework": example.get("expected_framework"),
                        "status": "success",
                        "duration_ms": duration_ms,
                        "deterministic_best_action": summary.get("deterministic_best_action"),
                        "dominant_framework": dominant_framework,
                        "reasoning_runtime_available": summary.get("reasoning_runtime_available", False),
                        "rag_runtime_available": summary.get("rag_runtime_available", False),
                        "error_code": None,
                        "error_message": None,
                    }
                )
            except ScenarioDomainError as exc:
                duration_ms = max(1, int(round((perf_counter() - scenario_started) * 1000)))
                error = exc.payload.get("error", {})
                framework_counts[_UNRESOLVED_FRAMEWORK_KEY] += 1
                scenario_results.append(
                    {
                        "scenario_id": example["id"],
                        "scenario_label": example["label"],
                        "subdivision_id": example.get("subdivision_id"),
                        "subdivision_label": example.get("subdivision_label"),
                        "expected_framework": example.get("expected_framework"),
                        "status": "error",
                        "duration_ms": duration_ms,
                        "deterministic_best_action": None,
                        "dominant_framework": None,
                        "reasoning_runtime_available": False,
                        "rag_runtime_available": False,
                        "error_code": error.get("code"),
                        "error_message": error.get("message"),
                    }
                )

        total_scenarios = len(examples)
        failed_runs = total_scenarios - completed_runs
        total_duration_ms = max(1, int(round((perf_counter() - started_at) * 1000)))

        framework_distribution: list[dict[str, Any]] = []
        for framework_key, count in framework_counts.most_common():
            framework_id = None if framework_key == _UNRESOLVED_FRAMEWORK_KEY else framework_key
            framework_distribution.append(
                {
                    "framework_id": framework_id,
                    "framework_label": (
                        "Unresolved / Unavailable"
                        if framework_id is None
                        else _FRAMEWORK_LABELS.get(framework_id, framework_id)
                    ),
                    "count": count,
                    "percentage": round((count / total_scenarios) * 100, 1) if total_scenarios else 0.0,
                }
            )

        top_framework = framework_distribution[0] if framework_distribution else None

        return {
            "subdivision": {
                **subdivision,
                "scenario_count": total_scenarios,
            },
            "summary": {
                "scenario_count": total_scenarios,
                "completed_runs": completed_runs,
                "failed_runs": failed_runs,
                "completion_rate_pct": round((completed_runs / total_scenarios) * 100, 1) if total_scenarios else 0.0,
                "reasoning_runtime_ready_pct": (
                    round((reasoning_runtime_ready / total_scenarios) * 100, 1) if total_scenarios else 0.0
                ),
                "rag_runtime_ready_pct": (
                    round((rag_runtime_ready / total_scenarios) * 100, 1) if total_scenarios else 0.0
                ),
                "top_framework": top_framework["framework_id"] if top_framework else None,
                "top_framework_label": top_framework["framework_label"] if top_framework else None,
                "top_framework_percentage": top_framework["percentage"] if top_framework else 0.0,
                "total_duration_ms": total_duration_ms,
            },
            "framework_distribution": framework_distribution,
            "scenario_results": scenario_results,
        }


def _prepare_input(payload: str | dict[str, Any], input_mode_hint: InputModeHint) -> str | dict[str, Any]:
    if input_mode_hint == "auto":
        return payload

    if input_mode_hint == "text":
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                raise ScenarioDomainError(
                    {
                        "error": {
                            "code": "empty_input",
                            "message": "Text input cannot be empty.",
                        },
                        "replay": [],
                    }
                )
            return stripped
        raise ScenarioDomainError(
            {
                "error": {
                    "code": "input_mode_mismatch",
                    "message": "Text mode requires a string input.",
                },
                "replay": [],
            }
        )

    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        raise ScenarioDomainError(
            {
                "error": {
                    "code": "input_mode_mismatch",
                    "message": "JSON mode requires a JSON object or a JSON string.",
                },
                "replay": [],
            }
        )

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ScenarioDomainError(
            {
                "error": {
                    "code": "invalid_json",
                    "message": f"JSON mode requires valid JSON: {exc.msg}.",
                },
                "replay": [],
            }
        ) from exc

    if not isinstance(decoded, dict):
        raise ScenarioDomainError(
            {
                "error": {
                    "code": "invalid_json",
                    "message": "JSON mode requires a top-level JSON object.",
                },
                "replay": [],
            }
        )
    return decoded


def _stage(
    *,
    stage_id: str,
    label: str,
    status: str,
    started_at: float,
    ended_at: float,
    headline: str,
    snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "label": label,
        "status": status,
        "duration_ms": max(1, int(round((ended_at - started_at) * 1000))),
        "headline": headline,
        "snapshot": snapshot,
        "highlight_paths": _diff_paths(previous_snapshot, snapshot),
        "metrics": metrics,
    }


def _diff_paths(previous: Any, current: Any, path: str = "") -> list[str]:
    if previous is current:
        return []

    if path == "":
        path = "$"

    if previous is None:
        return [path]

    if type(previous) is not type(current):
        return [path]

    if isinstance(current, dict):
        changes: list[str] = []
        for key in sorted(set(previous) | set(current)):
            next_path = f"{path}.{key}" if path != "$" else f"$.{key}"
            if key not in previous:
                changes.append(next_path)
                continue
            if key not in current:
                changes.append(next_path)
                continue
            changes.extend(_diff_paths(previous[key], current[key], next_path))
        return changes

    if isinstance(current, list):
        if len(previous) != len(current):
            return [path]
        changes: list[str] = []
        for index, item in enumerate(current):
            next_path = f"{path}[{index}]"
            changes.extend(_diff_paths(previous[index], item, next_path))
        return changes

    if previous != current:
        return [path]
    return []


def _input_headline(payload: str | dict[str, Any]) -> str:
    if isinstance(payload, dict):
        return f"Received structured JSON with {len(payload)} top-level field(s)."
    return f"Received natural-language input ({len(payload)} characters)."


def _input_metrics(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {"submitted_kind": "json", "top_level_fields": len(payload)}
    return {"submitted_kind": "text", "characters": len(payload)}
