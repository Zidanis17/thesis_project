from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .mathematical_layer import DeterministicMathematicalLayer, MathematicalLayerResult
from .models import ParserResult, Scenario
from .rag import DeterministicRAGRetriever, RAGRetrievalResult
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
    rag_retrieval_result: RAGRetrievalResult | None = None
    reasoning_result: EthicalReasoningResult | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "parser_result": self.parser_result.to_dict(),
        }
        if self.mathematical_layer_result is not None:
            payload["mathematical_layer_result"] = self.mathematical_layer_result.to_dict()
        if self.rag_retrieval_result is not None:
            payload["rag_retrieval_result"] = self.rag_retrieval_result.to_dict()
        if self.reasoning_result is not None:
            payload["reasoning_result"] = self.reasoning_result.to_dict()
        return payload


class ScenarioPipeline:
    def __init__(
        self,
        parser: DeterministicScenarioParser | None = None,
        mathematical_layer: DeterministicMathematicalLayer | None = None,
        rag_retriever: DeterministicRAGRetriever | None = None,
        reasoning_llm: EthicalReasoningLLM | None = None,
    ) -> None:
        self.parser = parser or DeterministicScenarioParser()
        self.mathematical_layer = mathematical_layer or DeterministicMathematicalLayer()
        self.rag_retriever = rag_retriever
        self.reasoning_llm = reasoning_llm

    def run(self, payload: str | Mapping[str, Any] | Scenario | ParserResult) -> ScenarioPipelineResult:
        if isinstance(payload, ParserResult):
            parser_result = payload
        elif isinstance(payload, Scenario):
            parser_result = ParserResult(scenario=payload, input_mode="structured_json")
        else:
            parser_result = self.parser.parse(payload)

        mathematical_layer_result = self.mathematical_layer.analyze(parser_result.scenario)
        rag_retrieval_result = None
        retriever = self.rag_retriever
        if retriever is None:
            try:
                retriever = DeterministicRAGRetriever()
            except RuntimeError:
                retriever = None
        if retriever is not None:
            try:
                rag_retrieval_result = retriever.retrieve(
                    parser_result.scenario,
                    mathematical_layer_result,
                )
            except RuntimeError:
                rag_retrieval_result = None

        reasoning_result = None
        if self.reasoning_llm is not None:
            reasoning_result = self.reasoning_llm.reason(
                parser_result,
                mathematical_layer_result,
                rag_retrieval_result,
            )

        return ScenarioPipelineResult(
            parser_result=parser_result,
            mathematical_layer_result=mathematical_layer_result,
            rag_retrieval_result=rag_retrieval_result,
            reasoning_result=reasoning_result,
        )


# Backward-compatible aliases for the previous public API.
PipelineResult = ScenarioPipelineResult
DeterministicScenarioPipeline = ScenarioPipeline
