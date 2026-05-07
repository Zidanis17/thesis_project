from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .engine import DeterministicRAGRetriever

__all__ = ["LazyRAGRuntime", "ensure_rag_retriever"]


@dataclass(slots=True)
class LazyRAGRuntime:
    retriever: Any | None = None
    enabled: bool = True

    def ensure(self) -> DeterministicRAGRetriever | None:
        self.retriever = ensure_rag_retriever(self.retriever, enabled=self.enabled)
        return self.retriever

    @property
    def runtime_available(self) -> bool:
        return bool(self.retriever and getattr(self.retriever, "vector_store", None) is not None)

    @property
    def runtime_error(self) -> Any | None:
        if self.retriever is None:
            return None
        return getattr(self.retriever, "_runtime_error", None)

    @property
    def runtime_status(self) -> str:
        if self.enabled and self.retriever is None:
            return "lazy_not_initialized"
        return "available" if self.runtime_available else "unavailable"

    @property
    def knowledge_base_path(self) -> Any | None:
        if self.retriever is None:
            return None
        return getattr(self.retriever, "knowledge_base_path", None)


def ensure_rag_retriever(
    retriever: DeterministicRAGRetriever | None,
    *,
    enabled: bool,
) -> DeterministicRAGRetriever | None:
    if retriever is not None or not enabled:
        return retriever
    return DeterministicRAGRetriever()
