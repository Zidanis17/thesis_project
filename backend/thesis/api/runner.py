from __future__ import annotations

from collections import Counter
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from ..mathematical_layer import DeterministicMathematicalLayer, MathematicalLayerResult
from ..pipeline import ScenarioPipelineResult
from ..rag import DeterministicRAGRetriever
from ..reasoning_llm import EthicalReasoningLLM, EthicalReasoningResult
from ..scenario_parser import DeterministicScenarioParser, ScenarioParseError
from .serializers import (
    build_summary_payload,
    coerce_input_snapshot,
    extract_framework_id,
    summarize_rag_result,
    summarize_reasoning_result,
    strip_payload_metadata,
)

InputModeHint = Literal["auto", "json", "text"]
EvaluationVariant = Literal["full_system", "no_rag", "no_math", "no_rag_no_math"]

__all__ = [
    "InputModeHint",
    "EvaluationVariant",
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
_VALID_DOMINANT_FRAMEWORKS = {"EF-01", "EF-02", "EF-03", "EF-05", "EF-06"}
_CONFUSION_LABELS = [
    "EF-01",
    "EF-02",
    "EF-03",
    "EF-05",
    "EF-06",
    "EF-04_invalid",
    "unresolved",
    "other_invalid",
]


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

    def run(
        self,
        payload: str | dict[str, Any],
        input_mode_hint: InputModeHint = "auto",
        *,
        variant: EvaluationVariant = "full_system",
        disable_rag: bool | None = None,
        disable_math: bool | None = None,
    ) -> dict[str, Any]:
        disable_rag = _variant_disables_rag(variant) if disable_rag is None else disable_rag
        disable_math = _variant_disables_math(variant) if disable_math is None else disable_math
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
        math_result: MathematicalLayerResult | None = None
        math_status = "skipped"
        math_headline = "Mathematical layer skipped."
        math_metrics: dict[str, Any] = {}

        if disable_math:
            math_payload = {
                "runtime_status": "not_requested",
                "reason": "Mathematical layer disabled for evaluation variant.",
                "risk_score_matrix": None,
                "violated_rules": [],
                "action_assessments": [],
                "best_action_by_total_risk": None,
                "best_action_by_ethical_cost": None,
            }
        else:
            math_result = self.mathematical_layer.analyze(parser_result.scenario)
            math_payload = math_result.to_dict()
            math_status = "success"
            math_headline = (
                "Computed action risks. Best deterministic action: "
                f"{math_result.best_action_by_total_risk}."
            )
            math_metrics = {
                "best_action": math_result.best_action_by_total_risk,
                "actions": len(math_result.action_assessments),
                "violations": len(math_result.violated_rules),
            }

        math_snapshot = {**snapshot, "mathematical_layer_result": math_payload}
        replay.append(
            _stage(
                stage_id="math",
                label="Math",
                status=math_status,
                started_at=math_started,
                ended_at=perf_counter(),
                headline=math_headline,
                snapshot=math_snapshot,
                previous_snapshot=snapshot,
                metrics=math_metrics,
            )
        )
        snapshot = math_snapshot

        rag_started = perf_counter()
        rag_result = None
        rag_payload: dict[str, Any]
        rag_status = "skipped"
        rag_headline = "RAG stage skipped."
        rag_metrics: dict[str, Any] = {}

        if disable_rag:
            rag_payload = {
                "runtime_available": False,
                "runtime_status": "not_requested",
                "reason": "RAG stage disabled for evaluation variant.",
                "frameworks_retrieved": 0,
                "supporting_docs_retrieved": 0,
                "frameworks": [],
                "supporting_documents": [],
            }
        elif self.rag_retriever is not None:
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
                    "risk_scores_per_action": math_result.risk_score_matrix if math_result is not None else {},
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
        summary_payload["variant"] = variant
        summary_payload["math_runtime_available"] = math_result is not None
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
                    "variant": variant,
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
        variant: EvaluationVariant = "full_system",
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

        evaluation = self._run_evaluation(
            examples=examples,
            scope="subdivision",
            variant=variant,
            subdivision=subdivision,
        )
        return {
            **evaluation,
            "subdivision": {
                **subdivision,
                "scenario_count": len(examples),
            },
        }

    def run_scenario_bank(
        self,
        *,
        examples: list[dict[str, Any]],
        variant: EvaluationVariant = "full_system",
    ) -> dict[str, Any]:
        if not examples:
            raise ScenarioDomainError(
                {
                    "error": {
                        "code": "empty_scenario_bank",
                        "message": "No JSON scenarios were found in the scenario bank.",
                    },
                    "replay": [],
                }
            )

        return self._run_evaluation(
            examples=examples,
            scope="full_bank",
            variant=variant,
            subdivision=None,
        )

    def _run_evaluation(
        self,
        *,
        examples: list[dict[str, Any]],
        scope: Literal["subdivision", "full_bank"],
        variant: EvaluationVariant,
        subdivision: dict[str, Any] | None,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        scenario_results: list[dict[str, Any]] = []

        for example in examples:
            scenario_started = perf_counter()
            try:
                result = self.run(
                    example["value"],
                    example.get("mode", "auto"),
                    variant=variant,
                )
                duration_ms = max(1, int(round((perf_counter() - scenario_started) * 1000)))
                scenario_results.append(_scenario_evaluation_result(example, result, duration_ms=duration_ms))
            except ScenarioDomainError as exc:
                duration_ms = max(1, int(round((perf_counter() - scenario_started) * 1000)))
                error = exc.payload.get("error", {})
                scenario_results.append(_failed_scenario_evaluation_result(example, duration_ms=duration_ms, error=error))
            except Exception as exc:
                duration_ms = max(1, int(round((perf_counter() - scenario_started) * 1000)))
                scenario_results.append(
                    _failed_scenario_evaluation_result(
                        example,
                        duration_ms=duration_ms,
                        error={
                            "code": "scenario_run_error",
                            "message": str(exc),
                        },
                    )
                )

        total_scenarios = len(examples)
        total_duration_ms = max(1, int(round((perf_counter() - started_at) * 1000)))
        summary = _evaluation_summary(
            scenario_results,
            total_duration_ms=total_duration_ms,
        )

        return {
            "evaluation_id": None,
            "created_at": _utc_now(),
            "scope": scope,
            "variant": variant,
            "subdivision_id": subdivision.get("id") if subdivision else None,
            "subdivision_label": subdivision.get("label") if subdivision else None,
            "total_scenarios": summary["scenario_count"],
            "completed_runs": summary["completed_runs"],
            "failed_runs": summary["failed_runs"],
            "completion_rate_pct": summary["completion_rate_pct"],
            "correct_predictions": summary["correct_predictions"],
            "incorrect_predictions": summary["incorrect_predictions"],
            "overall_accuracy_pct": summary["accuracy_pct"],
            "total_duration_ms": total_duration_ms,
            "expected_framework_distribution": _framework_distribution(
                scenario_results,
                key="expected_framework",
            ),
            "framework_distribution": _framework_distribution(
                scenario_results,
                key="dominant_framework",
            ),
            "per_framework_accuracy": _per_framework_accuracy(scenario_results),
            "confusion_matrix": _confusion_matrix(scenario_results),
            "rag_retrieval_hit_rate_pct": summary["rag_retrieval_hit_rate_pct"],
            "reasoning_contract_validity_rate_pct": summary["reasoning_contract_validity_rate_pct"],
            "risk_matrix_preservation_rate_pct": summary["risk_matrix_preservation_rate_pct"],
            "weights_validity_rate_pct": summary["weights_validity_rate_pct"],
            "summary": summary,
            "scenario_results": scenario_results,
        }


def _scenario_evaluation_result(
    example: dict[str, Any],
    pipeline_payload: dict[str, Any],
    *,
    duration_ms: int,
) -> dict[str, Any]:
    summary = pipeline_payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    artifacts = pipeline_payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    rag_payload = artifacts.get("rag_retrieval_result", {})
    if not isinstance(rag_payload, dict):
        rag_payload = {}
    reasoning_payload = artifacts.get("reasoning_result", {})
    if not isinstance(reasoning_payload, dict):
        reasoning_payload = {}
    math_payload = artifacts.get("mathematical_layer_result", {})
    if not isinstance(math_payload, dict):
        math_payload = {}

    expected_framework = _clean_framework_id(example.get("expected_framework"))
    dominant_framework = _clean_framework_id(summary.get("dominant_framework"))
    retrieval_metrics = _retrieval_metrics_from_rag_payload(rag_payload, expected_framework)
    contract_checks = _reasoning_contract_checks(reasoning_payload, math_payload)
    confidence = _coerce_float(reasoning_payload.get("confidence"))

    correct_prediction = (
        expected_framework == dominant_framework
        if expected_framework is not None and dominant_framework is not None
        else False
    )

    return {
        "scenario_id": example["id"],
        "scenario_label": example["label"],
        "subdivision_id": example.get("subdivision_id"),
        "subdivision_label": example.get("subdivision_label"),
        "expected_framework": expected_framework,
        "dominant_framework": dominant_framework,
        "correct_prediction": correct_prediction,
        "confidence": confidence,
        "contributing_frameworks": _string_list(reasoning_payload.get("contributing_frameworks")),
        "weights": reasoning_payload.get("weights") if isinstance(reasoning_payload.get("weights"), dict) else {},
        "retrieved_framework_ids": retrieval_metrics["retrieved_framework_ids"],
        "expected_framework_retrieved": retrieval_metrics["expected_framework_retrieved"],
        "top_retrieved_framework": retrieval_metrics["top_retrieved_framework"],
        "top_retrieval_score": retrieval_metrics["top_retrieval_score"],
        "reasoning_contract_valid": contract_checks["reasoning_contract_valid"],
        "dominant_framework_valid": contract_checks["dominant_framework_valid"],
        "weights_sum_to_one": contract_checks["weights_sum_to_one"],
        "risk_matrix_preserved": contract_checks["risk_matrix_preserved"],
        "no_recommended_action": contract_checks["no_recommended_action"],
        "violated_constraints_supported": contract_checks["violated_constraints_supported"],
        "deterministic_best_action": summary.get("deterministic_best_action"),
        "status": "success",
        "duration_ms": duration_ms,
        "reasoning_runtime_available": summary.get("reasoning_runtime_available", False),
        "rag_runtime_available": summary.get("rag_runtime_available", False),
        "error_code": None,
        "error_message": None,
    }


def _failed_scenario_evaluation_result(
    example: dict[str, Any],
    *,
    duration_ms: int,
    error: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_id": example["id"],
        "scenario_label": example["label"],
        "subdivision_id": example.get("subdivision_id"),
        "subdivision_label": example.get("subdivision_label"),
        "expected_framework": _clean_framework_id(example.get("expected_framework")),
        "dominant_framework": None,
        "correct_prediction": False,
        "confidence": None,
        "contributing_frameworks": [],
        "weights": {},
        "retrieved_framework_ids": [],
        "expected_framework_retrieved": False,
        "top_retrieved_framework": None,
        "top_retrieval_score": None,
        "reasoning_contract_valid": False,
        "dominant_framework_valid": False,
        "weights_sum_to_one": False,
        "risk_matrix_preserved": False,
        "no_recommended_action": False,
        "violated_constraints_supported": False,
        "deterministic_best_action": None,
        "status": "error",
        "duration_ms": duration_ms,
        "reasoning_runtime_available": False,
        "rag_runtime_available": False,
        "error_code": error.get("code"),
        "error_message": error.get("message"),
    }


def _retrieval_metrics_from_rag_payload(
    rag_payload: dict[str, Any],
    expected_framework: str | None,
) -> dict[str, Any]:
    frameworks = rag_payload.get("frameworks", [])
    retrieved_framework_ids: list[str] = []
    top_retrieved_framework: str | None = None
    top_retrieval_score: float | None = None

    if isinstance(frameworks, list):
        for index, framework in enumerate(frameworks):
            framework_id = extract_framework_id(framework)
            if framework_id is not None and framework_id not in retrieved_framework_ids:
                retrieved_framework_ids.append(framework_id)
            if index == 0:
                top_retrieved_framework = framework_id
                if isinstance(framework, dict):
                    top_retrieval_score = _coerce_float(framework.get("score"))

    return {
        "retrieved_framework_ids": retrieved_framework_ids,
        "expected_framework_retrieved": (
            expected_framework in retrieved_framework_ids if expected_framework is not None else False
        ),
        "top_retrieved_framework": top_retrieved_framework,
        "top_retrieval_score": top_retrieval_score,
    }


def _reasoning_contract_checks(
    reasoning_payload: dict[str, Any],
    mathematical_layer_payload: dict[str, Any],
) -> dict[str, bool]:
    dominant_framework = _clean_framework_id(reasoning_payload.get("dominant_framework"))
    dominant_framework_valid = dominant_framework in _VALID_DOMINANT_FRAMEWORKS
    weights_sum_to_one = _weights_sum_to_one(reasoning_payload.get("weights"))
    risk_matrix_preserved = _risk_matrix_preserved(
        reasoning_payload.get("risk_scores_per_action"),
        mathematical_layer_payload.get("risk_score_matrix"),
    )
    no_recommended_action = "recommended_action" not in reasoning_payload
    violated_constraints_supported = _violated_constraints_supported(
        reasoning_payload.get("violated_constraints"),
        mathematical_layer_payload,
    )
    reasoning_runtime_available = bool(reasoning_payload.get("runtime_available", False))
    reasoning_contract_valid = all(
        (
            reasoning_runtime_available,
            dominant_framework_valid,
            weights_sum_to_one,
            risk_matrix_preserved,
            no_recommended_action,
            violated_constraints_supported,
        )
    )

    return {
        "reasoning_contract_valid": reasoning_contract_valid,
        "dominant_framework_valid": dominant_framework_valid,
        "weights_sum_to_one": weights_sum_to_one,
        "risk_matrix_preserved": risk_matrix_preserved,
        "no_recommended_action": no_recommended_action,
        "violated_constraints_supported": violated_constraints_supported,
    }


def _weights_sum_to_one(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    total = 0.0
    for key in ("bayesian", "equality", "maximin"):
        numeric = _coerce_float(value.get(key))
        if numeric is None:
            return False
        total += numeric
    return abs(total - 1.0) <= 0.01


def _risk_matrix_preserved(reasoning_matrix: Any, math_matrix: Any, *, tolerance: float = 1e-6) -> bool:
    if not isinstance(reasoning_matrix, dict) or not isinstance(math_matrix, dict):
        return False
    if set(reasoning_matrix) != set(math_matrix):
        return False
    for action, expected_scores in math_matrix.items():
        actual_scores = reasoning_matrix.get(action)
        if not isinstance(actual_scores, dict) or not isinstance(expected_scores, dict):
            return False
        if set(actual_scores) != set(expected_scores):
            return False
        for stakeholder_id, expected_value in expected_scores.items():
            actual_value = _coerce_float(actual_scores.get(stakeholder_id))
            expected_numeric = _coerce_float(expected_value)
            if actual_value is None or expected_numeric is None:
                return False
            if abs(actual_value - expected_numeric) > tolerance:
                return False
    return True


def _violated_constraints_supported(value: Any, mathematical_layer_payload: dict[str, Any]) -> bool:
    if value is None:
        value = []
    if not isinstance(value, list):
        return False

    supported = set(_string_list(mathematical_layer_payload.get("violated_rules")))
    action_assessments = mathematical_layer_payload.get("action_assessments", [])
    if isinstance(action_assessments, list):
        for assessment in action_assessments:
            if isinstance(assessment, dict):
                supported.update(_string_list(assessment.get("constraint_flags")))

    for constraint in value:
        if not isinstance(constraint, str) or constraint not in supported:
            return False
    return True


def _evaluation_summary(
    scenario_results: list[dict[str, Any]],
    *,
    total_duration_ms: int,
) -> dict[str, Any]:
    total_scenarios = len(scenario_results)
    completed_runs = sum(1 for result in scenario_results if result.get("status") == "success")
    failed_runs = total_scenarios - completed_runs
    correct_predictions = sum(1 for result in scenario_results if result.get("correct_prediction") is True)
    incorrect_predictions = total_scenarios - correct_predictions
    reasoning_runtime_ready = sum(1 for result in scenario_results if result.get("reasoning_runtime_available") is True)
    rag_runtime_ready = sum(1 for result in scenario_results if result.get("rag_runtime_available") is True)
    rag_hits = sum(1 for result in scenario_results if result.get("expected_framework_retrieved") is True)
    risk_preserved = sum(1 for result in scenario_results if result.get("risk_matrix_preserved") is True)
    contract_valid = sum(1 for result in scenario_results if result.get("reasoning_contract_valid") is True)
    weights_valid = sum(1 for result in scenario_results if result.get("weights_sum_to_one") is True)

    framework_distribution = _framework_distribution(scenario_results, key="dominant_framework")
    top_framework = framework_distribution[0] if framework_distribution else None
    expected_counter = Counter(
        result.get("expected_framework")
        for result in scenario_results
        if result.get("expected_framework")
    )
    expected_framework, expected_framework_total = (
        expected_counter.most_common(1)[0] if expected_counter else (None, 0)
    )

    confidence_values = [_coerce_float(result.get("confidence")) for result in scenario_results]
    correct_confidence_values = [
        _coerce_float(result.get("confidence"))
        for result in scenario_results
        if result.get("correct_prediction") is True
    ]
    incorrect_confidence_values = [
        _coerce_float(result.get("confidence"))
        for result in scenario_results
        if result.get("correct_prediction") is not True
    ]

    return {
        "scenario_count": total_scenarios,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "completion_rate_pct": _pct(completed_runs, total_scenarios),
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy_pct": _pct(correct_predictions, total_scenarios),
        "expected_framework": expected_framework,
        "expected_framework_total": expected_framework_total,
        "average_confidence": _average(confidence_values),
        "average_confidence_correct": _average(correct_confidence_values),
        "average_confidence_incorrect": _average(incorrect_confidence_values),
        "reasoning_runtime_ready_pct": _pct(reasoning_runtime_ready, total_scenarios),
        "rag_runtime_ready_pct": _pct(rag_runtime_ready, total_scenarios),
        "rag_retrieval_hit_rate_pct": _pct(rag_hits, total_scenarios),
        "risk_matrix_preservation_rate_pct": _pct(risk_preserved, total_scenarios),
        "reasoning_contract_validity_rate_pct": _pct(contract_valid, total_scenarios),
        "weights_validity_rate_pct": _pct(weights_valid, total_scenarios),
        "top_framework": top_framework["framework_id"] if top_framework else None,
        "top_framework_label": top_framework["framework_label"] if top_framework else None,
        "top_framework_percentage": top_framework["percentage"] if top_framework else 0.0,
        "total_duration_ms": total_duration_ms,
    }


def _framework_distribution(
    scenario_results: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    total_scenarios = len(scenario_results)
    counts: Counter[str] = Counter()
    for result in scenario_results:
        framework_key = _distribution_key(result.get(key))
        counts[framework_key] += 1

    distribution: list[dict[str, Any]] = []
    for framework_key, count in counts.most_common():
        framework_id = None if framework_key == _UNRESOLVED_FRAMEWORK_KEY else framework_key
        distribution.append(
            {
                "framework_id": framework_id,
                "framework_label": _framework_label(framework_id),
                "count": count,
                "percentage": _pct(count, total_scenarios),
            }
        )
    return distribution


def _per_framework_accuracy(scenario_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for framework_id in sorted(_VALID_DOMINANT_FRAMEWORKS):
        expected_count = sum(1 for result in scenario_results if result.get("expected_framework") == framework_id)
        correct_count = sum(
            1
            for result in scenario_results
            if result.get("expected_framework") == framework_id and result.get("correct_prediction") is True
        )
        incorrect_count = expected_count - correct_count
        metrics.append(
            {
                "framework_id": framework_id,
                "expected_count": expected_count,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "accuracy_pct": _pct(correct_count, expected_count),
            }
        )
    return metrics


def _confusion_matrix(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for expected_label in _CONFUSION_LABELS:
        predictions = {label: 0 for label in _CONFUSION_LABELS}
        for result in scenario_results:
            if _confusion_bucket(result.get("expected_framework")) != expected_label:
                continue
            predictions[_confusion_bucket(result.get("dominant_framework"))] += 1
        rows.append(
            {
                "expected": expected_label,
                "predictions": predictions,
            }
        )
    return {
        "labels": list(_CONFUSION_LABELS),
        "rows": rows,
    }


def _confusion_bucket(value: Any) -> str:
    framework_id = _clean_framework_id(value)
    if framework_id is None:
        return "unresolved"
    if framework_id == "EF-04":
        return "EF-04_invalid"
    if framework_id in _VALID_DOMINANT_FRAMEWORKS:
        return framework_id
    return "other_invalid"


def _distribution_key(value: Any) -> str:
    framework_id = _clean_framework_id(value)
    return framework_id if framework_id else _UNRESOLVED_FRAMEWORK_KEY


def _framework_label(framework_id: str | None) -> str:
    if framework_id is None:
        return "Unresolved / Unavailable"
    if framework_id == "EF-04_invalid":
        return "EF-04 used as invalid dominant framework"
    if framework_id == "other_invalid":
        return "Other invalid framework"
    if framework_id == "unresolved":
        return "Unresolved / Unavailable"
    return _FRAMEWORK_LABELS.get(framework_id, framework_id)


def _clean_framework_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped.upper()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average(values: list[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 3)


def _pct(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 1) if denominator else 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _variant_disables_rag(variant: EvaluationVariant) -> bool:
    return variant in {"no_rag", "no_rag_no_math"}


def _variant_disables_math(variant: EvaluationVariant) -> bool:
    return variant in {"no_math", "no_rag_no_math"}


def _prepare_input(payload: str | dict[str, Any], input_mode_hint: InputModeHint) -> str | dict[str, Any]:
    if input_mode_hint == "auto":
        return strip_payload_metadata(payload)

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
        return strip_payload_metadata(payload)
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
    return strip_payload_metadata(decoded)


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
        return f"Received structured JSON with {len(strip_payload_metadata(payload))} top-level field(s)."
    return f"Received natural-language input ({len(payload)} characters)."


def _input_metrics(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        return {"submitted_kind": "json", "top_level_fields": len(strip_payload_metadata(payload))}
    return {"submitted_kind": "text", "characters": len(payload)}
