from __future__ import annotations

import json
import re
from typing import Any

from ..pipeline import ScenarioPipelineResult
from .payload import strip_payload_metadata

__all__ = [
    "build_summary_payload",
    "coerce_input_snapshot",
    "extract_framework_id",
    "summarize_rag_result",
    "summarize_reasoning_result",
    "strip_payload_metadata",
]


_FRAMEWORK_ID_PATTERN = re.compile(r"(?<![A-Z0-9])EF-\d{2}(?!\d)", re.IGNORECASE)


def summarize_rag_result(rag_result: Any) -> dict[str, Any]:
    if rag_result is None:
        return {"runtime_status": "not_requested", "reason": "RAG stage not provided."}

    framework_docs: list[dict[str, Any]] = []
    supporting_docs: list[dict[str, Any]] = []

    for doc in rag_result.retrieved_documents:
        entry = {
            "framework_id": extract_framework_id(
                {
                    "title": doc.title,
                    "path": doc.path,
                    "content": doc.full_content,
                }
            ),
            "title": doc.title,
            "path": doc.path,
            "score": doc.score,
            "excerpt": doc.excerpt[:400] if doc.excerpt else "",
        }
        if doc.category == "ethical_frameworks":
            framework_docs.append(entry)
        else:
            entry["category"] = doc.category
            supporting_docs.append(entry)

    fallback_docs = [
        {
            "framework_id": extract_framework_id(
                {
                    "title": doc.title,
                    "path": doc.path,
                    "content": doc.content,
                }
            ),
            "title": doc.title,
            "path": doc.path,
            "score": "fallback",
            "content_chars": len(doc.content),
        }
        for doc in rag_result.always_included_documents
        if doc.category == "ethical_frameworks"
    ]

    return {
        "runtime_available": rag_result.runtime_available,
        "runtime_error": rag_result.runtime_error,
        "indexed_chunks": rag_result.indexed_chunks,
        "query": rag_result.query,
        "frameworks_retrieved": len(framework_docs) + len(fallback_docs),
        "supporting_docs_retrieved": len(supporting_docs),
        "frameworks": framework_docs or fallback_docs,
        "supporting_documents": supporting_docs,
    }


def summarize_reasoning_result(reasoning_result: Any) -> dict[str, Any]:
    if reasoning_result is None:
        return {"runtime_status": "not_requested", "reason": "Reasoning stage not provided."}

    payload = reasoning_result.to_dict()
    payload.pop("system_prompt", None)
    return payload


def extract_framework_id(value: str | dict[str, Any] | None) -> str | None:
    if value is None:
        return None

    candidates: list[Any] = []
    if isinstance(value, dict):
        for key in ("framework_id", "id", "title", "path", "source", "name", "content", "excerpt"):
            candidates.append(value.get(key))
    else:
        candidates.append(value)

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, dict):
            nested = extract_framework_id(candidate)
            if nested is not None:
                return nested
            continue
        text = str(candidate)
        if text.strip().startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                nested = extract_framework_id(payload)
                if nested is not None:
                    return nested
        match = _FRAMEWORK_ID_PATTERN.search(text)
        if match:
            return match.group(0).upper()
    return None


def build_summary_payload(
    result: ScenarioPipelineResult,
    *,
    rag_payload: dict[str, Any],
    reasoning_payload: dict[str, Any],
) -> dict[str, Any]:
    parser_result = result.parser_result
    math_result = result.mathematical_layer_result
    violated_rules = list(math_result.violated_rules) if math_result is not None else []
    deterministic_best_action = (
        math_result.best_action_by_total_risk if math_result is not None else None
    )
    return {
        "input_mode": parser_result.input_mode,
        "parser_warnings": list(parser_result.warnings),
        "violated_rules": violated_rules,
        "deterministic_best_action": deterministic_best_action,
        "dominant_framework": reasoning_payload.get("dominant_framework"),
        "agentic_scenario_class": (
            result.agentic_assessment.retrieval_intent.scenario_class
            if result.agentic_assessment is not None
            else None
        ),
        "agentic_candidate_frameworks": (
            list(result.agentic_assessment.candidate_frameworks)
            if result.agentic_assessment is not None
            else []
        ),
        "agentic_validation_valid": (
            result.agentic_validation_result.is_valid
            if result.agentic_validation_result is not None
            else None
        ),
        "agentic_validation_errors": (
            list(result.agentic_validation_result.errors)
            if result.agentic_validation_result is not None
            else []
        ),
        "agentic_validation_warnings": (
            list(result.agentic_validation_result.warnings)
            if result.agentic_validation_result is not None
            else []
        ),
        "rag_runtime_available": rag_payload.get("runtime_available", False),
        "reasoning_runtime_available": reasoning_payload.get("runtime_available", False),
        "reasoning_runtime_error": reasoning_payload.get("runtime_error"),
        "rag_runtime_error": rag_payload.get("runtime_error"),
    }


def coerce_input_snapshot(payload: Any, *, input_mode_hint: str) -> dict[str, Any]:
    submitted_kind = "json" if isinstance(payload, dict) else "text"
    return {
        "input_mode_hint": input_mode_hint,
        "submitted_kind": submitted_kind,
        "submitted": strip_payload_metadata(payload),
    }
