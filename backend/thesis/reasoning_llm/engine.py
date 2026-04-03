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
    recommended_action: str | None = None
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

    # All six EKB frameworks the LLM may reference in contributing_frameworks.
    FRAMEWORKS = {
        "EF-01",  # utilitarian_risk_minimization
        "EF-02",  # deontological_safety
        "EF-03",  # rawlsian_maximin
        "EF-04",  # ethics_of_risk (substrate / contributing only)
        "EF-05",  # ethical_valence_theory
        "EF-06",  # virtue_ethics (explanation layer / fallback dominant)
    }

    # EF-04 is the mathematical substrate — never a dominant framework.
    # FIX: EF-03 and EF-05 were incorrectly excluded — they are valid dominant frameworks.
    DOMINANT_FRAMEWORKS = FRAMEWORKS - {"EF-04"}

    WEIGHT_KEYS = ("bayesian", "equality", "maximin")

    # Fields extracted from EKB framework JSON when compressing for context.
    # Excludes source_papers, key_parameters, embedding_text — noise for the LLM.
    FRAMEWORK_CONTEXT_FIELDS = (
        "framework_id", "name", "foundation", "decision_logic",
        "pros", "cons", "tradeoffs",
    )

    # FIX: best_fit / poor_fit fields are always included, not only on unavoidable collisions.
    # These fields are precisely what tells the LLM which framework governs routine scenarios.
    FRAMEWORK_FIT_FIELDS = (
        "best_fit_scenarios",
        "poor_fit_scenarios",
    )
    
    FRAMEWORK_SELECTION_FIELDS = (   # ← add this
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

    # ── Public API ────────────────────────────────────────────────────────────

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
                parser_result=parser_result,
                mathematical_layer_result=mathematical_layer_result,
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

    # ── Prompt building ───────────────────────────────────────────────────────

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
                "recommended_action": "string — one of available_actions",
                # FIX: schema now matches system prompt — EF-05 restored, consistent with DOMINANT_FRAMEWORKS
                "dominant_framework": "EF-01 | EF-02 | EF-03 | EF-05 | EF-06",
                "contributing_frameworks": ["EF-01", "EF-02", "EF-03", "EF-04"],
                "weights": {
                    "bayesian": "number",
                    "equality": "number",
                    "maximin": "number",
                },
                "weights_reasoning": "string — why these weights suit this scenario",
                "risk_scores_per_action": "copy mathematical_layer.risk_score_matrix exactly",
                "rationale": "string — cite retrieved framework_ids and their fields",
                "confidence": "number between 0 and 1",
                "violated_constraints": "list of constraint flags for recommended_action, or []",
            },
        }
        return (
            "Produce the final ethical decision as a JSON object.\n"
            "Copy risk_scores_per_action from mathematical_layer.risk_score_matrix exactly.\n"
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
                framework_entries.append({
                    "title": doc.title,
                    "score": doc.score,
                    "content": self._compress_framework(doc.full_content, scenario),
                })
            else:
                supporting_docs.append({
                    "title": doc.title,
                    "category": doc.category,
                    "score": doc.score,
                    "excerpt": doc.excerpt,
                })

        for doc in rag_retrieval_result.always_included_documents:
            if doc.category == "ethical_frameworks":
                framework_entries.append({
                    "title": doc.title,
                    "score": None,
                    "content": self._compress_framework(doc.content, scenario),
                })

        return {
            "runtime_available": rag_retrieval_result.runtime_available,
            "runtime_error": rag_retrieval_result.runtime_error,
            "query": rag_retrieval_result.query,
            "frameworks": framework_entries,
            "supporting_documents": supporting_docs,
        }

    def _compress_framework(self, full_content: str, scenario: Scenario) -> dict[str, Any]:
        """
        Extract only the fields the LLM needs for ethical reasoning from a framework
        JSON entry. Drops source_papers, key_parameters, scenario_tags, embedding_text
        to stay within the token budget.

        FIX: best_fit_scenarios and poor_fit_scenarios are always included regardless of
        collision_unavoidable. These fields are what tells the LLM which framework governs
        routine non-dilemma scenarios — stripping them for collision_unavoidable=False was
        hiding EF-02's guidance that it governs the vast majority of AV operation.
        """
        try:
            payload = json.loads(full_content)
        except (json.JSONDecodeError, TypeError):
            return {"raw_text": full_content[:2000]}

        if not isinstance(payload, dict):
            return {"raw_text": full_content[:2000]}

        compressed: dict[str, Any] = {}
        for field_name in self.FRAMEWORK_CONTEXT_FIELDS + self.FRAMEWORK_FIT_FIELDS + self.FRAMEWORK_SELECTION_FIELDS:
            value = payload.get(field_name)
            if value is not None:
                compressed[field_name] = value

        return compressed

    # ── Result construction ───────────────────────────────────────────────────

    def _build_result(
        self,
        *,
        payload: dict[str, Any],
        parser_result: ParserResult,
        mathematical_layer_result: MathematicalLayerResult,
        risk_scores_per_action: dict[str, dict[str, float]],
    ) -> EthicalReasoningResult:
        scenario = parser_result.scenario

        recommended_action = self._required_text(
            payload.get("recommended_action"), "recommended_action"
        )
        if recommended_action not in scenario.available_actions:
            raise ValueError(
                f"recommended_action must be one of {scenario.available_actions}, "
                f"got {recommended_action!r}"
            )

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
        violated_constraints = self._constraints_for_action(
            recommended_action, mathematical_layer_result
        )

        return EthicalReasoningResult(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            runtime_available=True,
            recommended_action=recommended_action,
            dominant_framework=dominant_framework,
            contributing_frameworks=contributing_frameworks,
            weights=weights,
            weights_reasoning=weights_reasoning,
            risk_scores_per_action=risk_scores_per_action,
            rationale=rationale,
            confidence=confidence,
            violated_constraints=violated_constraints,
        )

    # ── Response parsing ──────────────────────────────────────────────────────

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
            stripped = stripped[start: end + 1]
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("Reasoning model output must decode to a JSON object")
        return payload

    # ── Validation helpers ────────────────────────────────────────────────────

    def _canonical_framework(self, value: Any, *, field_name: str) -> str:
        """
        Normalise a framework identifier to its canonical EKB form (e.g. "EF-01").
        Accepts both exact IDs ("EF-01") and old-style snake_case names
        ("utilitarianism", "deontology") for backwards compatibility.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

        candidate = value.strip()

        if candidate in self.FRAMEWORKS:
            return candidate

        _LEGACY_MAP = {
            "utilitarianism":                "EF-01",
            "utilitarian_risk_minimization": "EF-01",
            "deontology":                    "EF-02",
            "deontological_safety":          "EF-02",
            "rawlsian_maximin":              "EF-03",
            "maximin":                       "EF-03",
            "ethics_of_risk":                "EF-04",
            "ethical_valence_theory":        "EF-05",
            "evt":                           "EF-05",
            "virtue_ethics":                 "EF-06",
        }
        normalised = candidate.lower().replace(" ", "_").replace("-", "_")
        if normalised in _LEGACY_MAP:
            return _LEGACY_MAP[normalised]

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

        # EF-04 was given as dominant — fall back to first valid contributing framework
        for fw in contributing_frameworks:
            if fw in self.DOMINANT_FRAMEWORKS:
                return fw

        raise ValueError(
            f"dominant_framework cannot be EF-04 (ethics_of_risk); "
            f"use EF-01, EF-02, EF-03, EF-05, or EF-06"
        )

    def _framework_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("contributing_frameworks must be a list")
        frameworks: list[str] = []
        for item in value:
            fw = self._canonical_framework(item, field_name="contributing_frameworks[]")
            if fw == "EF-04":
                continue
            if fw not in frameworks:
                frameworks.append(fw)
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

    def _constraints_for_action(
        self,
        action: str,
        mathematical_layer_result: MathematicalLayerResult,
    ) -> list[str]:
        for assessment in mathematical_layer_result.action_assessments:
            if assessment.action == action:
                return list(assessment.constraint_flags)
        return []

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

    def _confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        return round(confidence, 3)

    # ── Client ────────────────────────────────────────────────────────────────

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