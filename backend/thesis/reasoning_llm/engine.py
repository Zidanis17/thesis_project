from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .._env import load_project_env
from ..mathematical_layer import MathematicalLayerResult
from ..models import ParserResult, Scenario
from ..rag import RAGRetrievalResult
from .prompt import ETHICAL_REASONING_SYSTEM_PROMPT

__all__ = [
    "EthicalReasoningLLM",
    "EthicalReasoningResult",
]


@dataclass(slots=True)
class EthicalReasoningResult:
    model_name: str
    system_prompt: str
    runtime_available: bool
    dominant_framework: str | None = None
    contributing_frameworks: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    weights_reasoning: str = ""
    risk_scores_per_action: dict[str, dict[str, float]] = field(default_factory=dict)
    rationale: str = ""
    confidence: float | None = None
    violated_constraints: list[str] = field(default_factory=list)
    runtime_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EthicalReasoningLLM:
    DEFAULT_MODEL_NAME = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.0

    FRAMEWORKS = {
        "EF-01",
        "EF-02",
        "EF-03",
        "EF-04",
        "EF-05",
        "EF-06",
    }
    DOMINANT_FRAMEWORKS = FRAMEWORKS - {"EF-04"}
    WEIGHT_KEYS = ("bayesian", "equality", "maximin")
    FRAMEWORK_CONTEXT_FIELDS = (
        "framework_id",
        "name",
        "foundation",
        "decision_logic",
        "pros",
        "cons",
        "tradeoffs",
    )
    FRAMEWORK_FIT_FIELDS = (
        "best_fit_scenarios",
        "poor_fit_scenarios",
    )
    FRAMEWORK_SELECTION_FIELDS = (
        "use_when",
        "avoid_when",
        "dominant_when",
    )

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = ETHICAL_REASONING_SYSTEM_PROMPT,
    ) -> None:
        self.model_name = model_name
        self.temperature = float(temperature)
        self.system_prompt = system_prompt.strip()
        self._runtime_error: RuntimeError | None = None
        self.client: Any | None = None

        try:
            self.client = self._build_client()
        except Exception as exc:
            self._runtime_error = RuntimeError(
                "LangChain OpenAI reasoning model could not be initialized. "
                "Install langchain-openai and provide OPENAI_API_KEY via the process or .env."
            )
            self._runtime_error.__cause__ = exc

    def reason(
        self,
        parser_result: ParserResult,
        mathematical_layer_result: MathematicalLayerResult,
        rag_retrieval_result: RAGRetrievalResult | None = None,
    ) -> EthicalReasoningResult:
        risk_scores_per_action = self._normalize_risk_scores(
            mathematical_layer_result.risk_score_matrix
        )

        if self._runtime_error is not None or self.client is None:
            return EthicalReasoningResult(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                runtime_available=False,
                risk_scores_per_action=risk_scores_per_action,
                runtime_error=str(self._runtime_error or "Reasoning runtime is unavailable"),
            )

        try:
            user_prompt = self._build_user_prompt(
                parser_result,
                mathematical_layer_result,
                rag_retrieval_result,
            )
            print(f"Prompt: {len(user_prompt)} chars, ~{len(user_prompt)//4} tokens")
            response = self.client.invoke(
                [
                    ("system", self.system_prompt),
                    ("user", user_prompt),
                ]
            )
            payload = self._parse_json_response(self._message_text(response))
            return self._build_result(
                payload=payload,
                risk_scores_per_action=risk_scores_per_action,
            )
        except Exception as exc:
            return EthicalReasoningResult(
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                runtime_available=False,
                risk_scores_per_action=risk_scores_per_action,
                runtime_error=str(exc),
            )

    def _build_user_prompt(
        self,
        parser_result: ParserResult,
        mathematical_layer_result: MathematicalLayerResult,
        rag_retrieval_result: RAGRetrievalResult | None,
    ) -> str:
        scenario = parser_result.scenario
        
        prompt_payload = {
            "input_mode": parser_result.input_mode,
            "parser_warnings": list(parser_result.warnings),
            "scenario": scenario.to_dict(),
            "mathematical_layer": mathematical_layer_result.to_dict(),
            "rag_context": self._rag_context_payload(rag_retrieval_result, scenario),
            "required_output_schema": {
                "dominant_framework": "EF-01 | EF-02 | EF-03 | EF-05 | EF-06",
                "contributing_frameworks": ["EF-01", "EF-02", "EF-03", "EF-04"],
                "weights": {
                    "bayesian": "number",
                    "equality": "number",
                    "maximin": "number",
                },
                "weights_reasoning": "string - why these weights suit this scenario",
                "risk_scores_per_action": "copy mathematical_layer.risk_score_matrix exactly",
                "rationale": "string - cite retrieved framework_ids and their fields",
                "confidence": "number between 0 and 1",
                "violated_constraints": "list of input-supported constraint flags that shaped the reasoning, or []",
            },
        }
        return (
            "Produce the final ethical analysis as a JSON object.\n"
            "Copy risk_scores_per_action from mathematical_layer.risk_score_matrix exactly.\n"
            "Do not include a recommended_action field.\n"
            "Cite retrieved EKB framework_ids in your rationale.\n\n"
            f"{json.dumps(prompt_payload, separators=(',', ':'))}"
        )

    def _rag_context_payload(
        self,
        rag_retrieval_result: RAGRetrievalResult | None,
        scenario: Scenario,
    ) -> dict[str, Any]:
        if rag_retrieval_result is None:
            return {
                "runtime_available": False,
                "runtime_error": "RAG stage not provided",
                "frameworks": [],
                "supporting_documents": [],
            }

        framework_entries: list[dict[str, Any]] = []
        supporting_docs: list[dict[str, Any]] = []

        for doc in rag_retrieval_result.retrieved_documents:
            if doc.category == "ethical_frameworks":
                framework_entries.append(
                    {
                        "title": doc.title,
                        "score": doc.score,
                        "content": self._compress_framework(doc.full_content, scenario),
                    }
                )
            else:
                supporting_docs.append(
                    {
                        "title": doc.title,
                        "category": doc.category,
                        "score": doc.score,
                        "excerpt": doc.excerpt,
                    }
                )

        for doc in rag_retrieval_result.always_included_documents:
            if doc.category == "ethical_frameworks":
                framework_entries.append(
                    {
                        "title": doc.title,
                        "score": None,
                        "content": self._compress_framework(doc.content, scenario),
                    }
                )

        return {
            "runtime_available": rag_retrieval_result.runtime_available,
            "runtime_error": rag_retrieval_result.runtime_error,
            "query": rag_retrieval_result.query,
            "frameworks": framework_entries,
            "supporting_documents": supporting_docs,
        }

    def _compress_framework(self, full_content: str, scenario: Scenario) -> dict[str, Any]:
        try:
            payload = json.loads(full_content)
        except (json.JSONDecodeError, TypeError):
            return {"raw_text": full_content[:2000]}

        if not isinstance(payload, dict):
            return {"raw_text": full_content[:2000]}

        compressed: dict[str, Any] = {}
        for field_name in (
            self.FRAMEWORK_CONTEXT_FIELDS
            + self.FRAMEWORK_FIT_FIELDS
            + self.FRAMEWORK_SELECTION_FIELDS
        ):
            value = payload.get(field_name)
            if value is not None:
                compressed[field_name] = value

        return compressed

    def _build_result(
        self,
        *,
        payload: dict[str, Any],
        risk_scores_per_action: dict[str, dict[str, float]],
    ) -> EthicalReasoningResult:
        contributing_frameworks = self._framework_list(
            payload.get("contributing_frameworks", [])
        )
        dominant_framework = self._dominant_framework(
            payload.get("dominant_framework"),
            contributing_frameworks=contributing_frameworks,
        )
        weights = self._normalize_weights(payload.get("weights"))
        weights_reasoning = self._required_text(
            payload.get("weights_reasoning"), "weights_reasoning"
        )
        rationale = self._required_text(payload.get("rationale"), "rationale")
        confidence = self._confidence(payload.get("confidence"))
        violated_constraints = self._string_list(
            payload.get("violated_constraints", []),
            field_name="violated_constraints",
        )

        return EthicalReasoningResult(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            runtime_available=True,
            dominant_framework=dominant_framework,
            contributing_frameworks=contributing_frameworks,
            weights=weights,
            weights_reasoning=weights_reasoning,
            risk_scores_per_action=risk_scores_per_action,
            rationale=rationale,
            confidence=confidence,
            violated_constraints=violated_constraints,
        )

    def _message_text(self, response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1:
            stripped = stripped[start : end + 1]
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("Reasoning model output must decode to a JSON object")
        return payload

    def _canonical_framework(self, value: Any, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

        candidate = value.strip()
        if candidate in self.FRAMEWORKS:
            return candidate

        legacy_map = {
            "utilitarianism": "EF-01",
            "utilitarian_risk_minimization": "EF-01",
            "deontology": "EF-02",
            "deontological_safety": "EF-02",
            "rawlsian_maximin": "EF-03",
            "maximin": "EF-03",
            "ethics_of_risk": "EF-04",
            "ethical_valence_theory": "EF-05",
            "evt": "EF-05",
            "virtue_ethics": "EF-06",
        }
        normalised = candidate.lower().replace(" ", "_").replace("-", "_")
        if normalised in legacy_map:
            return legacy_map[normalised]

        raise ValueError(
            f"{field_name} must be one of {sorted(self.FRAMEWORKS)}, got {candidate!r}"
        )

    def _dominant_framework(
        self,
        value: Any,
        *,
        contributing_frameworks: list[str],
    ) -> str:
        candidate = self._canonical_framework(value, field_name="dominant_framework")
        if candidate in self.DOMINANT_FRAMEWORKS:
            return candidate

        for framework in contributing_frameworks:
            if framework in self.DOMINANT_FRAMEWORKS:
                return framework

        raise ValueError(
            "dominant_framework cannot be EF-04 (ethics_of_risk); "
            "use EF-01, EF-02, EF-03, EF-05, or EF-06"
        )

    def _framework_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("contributing_frameworks must be a list")
        frameworks: list[str] = []
        for item in value:
            framework = self._canonical_framework(
                item,
                field_name="contributing_frameworks[]",
            )
            if framework == "EF-04":
                continue
            if framework not in frameworks:
                frameworks.append(framework)
        return frameworks

    def _normalize_weights(self, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("weights must be an object")
        raw: dict[str, float] = {}
        for key in self.WEIGHT_KEYS:
            raw_value = value.get(key, 0.0)
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"weights.{key} must be numeric") from exc
            if numeric < 0:
                raise ValueError(f"weights.{key} must be non-negative")
            raw[key] = numeric
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        return {key: round(raw[key] / total, 3) for key in self.WEIGHT_KEYS}

    def _normalize_risk_scores(
        self,
        risk_score_matrix: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        return {
            action: {
                stakeholder_id: round(float(score), 3)
                for stakeholder_id, score in scores.items()
            }
            for action, scores in risk_score_matrix.items()
        }

    def _required_text(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()

    def _string_list(self, value: Any, *, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list")
        items: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"{field_name} entries must be non-empty strings")
            items.append(entry.strip())
        return items

    def _confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        return round(confidence, 3)

    def _build_client(self) -> Any:
        from langchain_openai import ChatOpenAI

        load_project_env()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the reasoning model")
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
        )
