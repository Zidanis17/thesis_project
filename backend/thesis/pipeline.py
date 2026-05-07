from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .agentic_controller import (
    AgenticAssessment,
    AgenticEthicalController,
    AgenticValidationResult,
)
from .mathematical_layer import DeterministicMathematicalLayer, MathematicalLayerResult
from .models import ParserResult, Scenario
from .rag import DeterministicRAGRetriever, RAGRetrievalResult, ensure_rag_retriever
from .reasoning_llm import EthicalReasoningLLM, EthicalReasoningResult
from .scenario_parser import DeterministicScenarioParser

__all__ = [
    "ScenarioPipeline",
    "ScenarioPipelineResult",
    "DeterministicScenarioPipeline",
    "PipelineResult",
]


@dataclass(slots=True)
class ScenarioPipelineResult:
    parser_result: ParserResult
    mathematical_layer_result: MathematicalLayerResult | None
    agentic_assessment: AgenticAssessment | None = None
    rag_retrieval_result: RAGRetrievalResult | None = None
    reasoning_result: EthicalReasoningResult | None = None
    agentic_validation_result: AgenticValidationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "parser_result": self.parser_result.to_dict(),
        }
        if self.mathematical_layer_result is not None:
            payload["mathematical_layer_result"] = self.mathematical_layer_result.to_dict()
        if self.agentic_assessment is not None:
            payload["agentic_assessment"] = self.agentic_assessment.to_dict()
        if self.rag_retrieval_result is not None:
            payload["rag_retrieval_result"] = self.rag_retrieval_result.to_dict()
        if self.reasoning_result is not None:
            payload["reasoning_result"] = self.reasoning_result.to_dict()
        if self.agentic_validation_result is not None:
            payload["agentic_validation_result"] = self.agentic_validation_result.to_dict()
        return payload


class ScenarioPipeline:
    def __init__(
        self,
        parser: DeterministicScenarioParser | None = None,
        mathematical_layer: DeterministicMathematicalLayer | None = None,
        rag_retriever: DeterministicRAGRetriever | None = None,
        reasoning_llm: EthicalReasoningLLM | None = None,
        agentic_controller: AgenticEthicalController | None = None,
        auto_rag: bool = True,
    ) -> None:
        self.parser = parser or DeterministicScenarioParser()
        self.mathematical_layer = mathematical_layer or DeterministicMathematicalLayer()
        self.rag_retriever = rag_retriever
        self.reasoning_llm = reasoning_llm
        self.agentic_controller = agentic_controller or AgenticEthicalController()
        self.auto_rag = auto_rag

    def run(self, payload: str | Mapping[str, Any] | Scenario | ParserResult) -> ScenarioPipelineResult:
        if isinstance(payload, ParserResult):
            parser_result = payload
        elif isinstance(payload, Scenario):
            parser_result = ParserResult(scenario=payload, input_mode="structured_json")
        else:
            parser_result = self.parser.parse(payload)

        mathematical_layer_result = self.mathematical_layer.analyze(parser_result.scenario)
        agentic_assessment = None
        try:
            agentic_assessment = self.agentic_controller.assess(
                parser_result,
                mathematical_layer_result,
            )
        except Exception:
            agentic_assessment = None

        rag_retrieval_result = None
        try:
            retriever = ensure_rag_retriever(self.rag_retriever, enabled=self.auto_rag)
        except RuntimeError:
            retriever = None
        if retriever is not None:
            try:
                rag_retrieval_result = retriever.retrieve(
                    parser_result.scenario,
                    mathematical_layer_result,
                    retrieval_intent=(
                        agentic_assessment.retrieval_intent
                        if agentic_assessment is not None
                        else None
                    ),
                )
            except RuntimeError:
                rag_retrieval_result = None

        reasoning_result = None
        if self.reasoning_llm is not None:
            reasoning_result = self.reasoning_llm.reason(
                parser_result,
                mathematical_layer_result,
                rag_retrieval_result,
                agentic_assessment=agentic_assessment,
            )

        agentic_validation_result = None
        if reasoning_result is not None:
            try:
                agentic_validation_result = self.agentic_controller.validate_reasoning_result(
                    parser_result,
                    mathematical_layer_result,
                    reasoning_result,
                )
            except Exception:
                agentic_validation_result = None

        return ScenarioPipelineResult(
            parser_result=parser_result,
            mathematical_layer_result=mathematical_layer_result,
            agentic_assessment=agentic_assessment,
            rag_retrieval_result=rag_retrieval_result,
            reasoning_result=reasoning_result,
            agentic_validation_result=agentic_validation_result,
        )


# Backward-compatible aliases for the previous public API.
PipelineResult = ScenarioPipelineResult
DeterministicScenarioPipeline = ScenarioPipeline
