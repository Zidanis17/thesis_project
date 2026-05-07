from __future__ import annotations

from .engine import DeterministicRAGRetriever

__all__ = ["ensure_rag_retriever"]


def ensure_rag_retriever(
    retriever: DeterministicRAGRetriever | None,
    *,
    enabled: bool,
) -> DeterministicRAGRetriever | None:
    if retriever is not None or not enabled:
        return retriever
    return DeterministicRAGRetriever()
