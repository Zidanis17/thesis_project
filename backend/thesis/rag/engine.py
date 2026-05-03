from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .._env import load_project_env
from ..mathematical_layer import MathematicalLayerResult
from ..models import Scenario

__all__ = [
    "AlwaysIncludedDocument",
    "DeterministicRAGRetriever",
    "RAGRetrievalResult",
    "RetrievedDocument",
]


@dataclass(slots=True)
class AlwaysIncludedDocument:
    document_id: str
    title: str
    category: str
    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedDocument:
    document_id: str
    title: str
    category: str
    path: str
    score: float
    excerpt: str
    full_content: str  # full page_content from Chroma — used by LLM layer for compression

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RAGRetrievalResult:
    query: str
    retrieved_documents: list[RetrievedDocument]
    always_included_documents: list[AlwaysIncludedDocument]
    knowledge_base_path: str
    indexed_chunks: int
    runtime_available: bool
    runtime_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "always_included_documents": [
                doc.to_dict() for doc in self.always_included_documents
            ],
            "knowledge_base_path": self.knowledge_base_path,
            "indexed_chunks": self.indexed_chunks,
            "runtime_available": self.runtime_available,
            "runtime_error": self.runtime_error,
        }


class DeterministicRAGRetriever:
    # Frameworks go through scored retrieval — top 2 is enough for most scenarios.
    # Top-k covers the remaining general knowledge base chunks.
    DEFAULT_TOP_K = 4
    DEFAULT_FRAMEWORK_TOP_K = 2
    DEFAULT_COLLECTION_NAME = "av_ethics_knowledge_base"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    ETHICAL_FRAMEWORK_CATEGORY = "ethical_frameworks"

    # Minimum cosine-similarity-derived score to include a result.
    # Results below this are semantically too distant to be useful.
    MIN_SCORE_THRESHOLD = 0.30

    # Excerpt shown in the result summary — full_content is passed to the LLM layer.
    EXCERPT_LIMIT = 1500
    WEATHER_CANONICAL_MAP = {
        "light_rain": "rain",
        "overcast": "clear",
    }
    ROAD_TYPE_CANONICAL_MAP = {
        "urban_arterial": "urban",
        "residential_street": "residential",
        "urban_intersection": "intersection",
        "ring_road": "highway",
        "highway_merge": "highway",
        "hospital_access_road": "hospital_zone",
    }
    VRU_TYPES = {
        "pedestrian",
        "pedestrian_adult",
        "adult_pedestrian",
        "child_pedestrian",
        "elderly_pedestrian",
        "cyclist",
        "motorcyclist",
        "hidden_pedestrian",
        "hidden_cyclist",
    }
    VRU_VULNERABILITY_CLASSES = {"high", "child", "elderly", "cyclist", "pedestrian"}

    def __init__(
        self,
        knowledge_base_path: str | Path | None = None,
        *,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
        framework_top_k: int = DEFAULT_FRAMEWORK_TOP_K,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embeddings: Any | None = None,
    ) -> None:
        self.knowledge_base_path = Path(
            knowledge_base_path or self._default_knowledge_base_path()
        ).resolve()
        self.persist_directory = Path(
            persist_directory or self.knowledge_base_path / ".chroma"
        ).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.top_k = max(1, top_k)
        self.framework_top_k = max(0, min(framework_top_k, self.top_k))
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self._runtime_error: RuntimeError | None = None
        self.vector_store: Any | None = None

        # Cached once on first access — framework files rarely change between calls.
        self._always_included_cache: list[AlwaysIncludedDocument] | None = None

        # FIX: Cache collection count — Chroma count() is called once and reused.
        self._cached_collection_count: int | None = None

        try:
            self.vector_store = self._build_vector_store()
        except Exception as exc:
            self._runtime_error = RuntimeError(
                "LangChain Chroma retriever could not be initialized. Install the LangChain "
                "RAG dependencies and set OPENAI_API_KEY, or inject a custom embeddings object."
            )
            self._runtime_error.__cause__ = exc

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        scenario: Scenario,
        mathematical_layer_result: MathematicalLayerResult | None = None,
    ) -> RAGRetrievalResult:
        query = self._build_query(scenario, mathematical_layer_result)

        # If the vector store is unavailable fall back to loading framework files
        # directly from disk so the LLM always has at least some ethical context.
        if self._runtime_error is not None or self.vector_store is None:
            fallback_documents = self._heuristic_fallback_documents(scenario)
            return RAGRetrievalResult(
                query=query,
                retrieved_documents=[],
                always_included_documents=fallback_documents,
                knowledge_base_path=str(self.knowledge_base_path),
                indexed_chunks=0,
                runtime_available=False,
                runtime_error=str(self._runtime_error or "RAG retriever is unavailable"),
            )

        indexed_chunks = self._collection_count()
        if indexed_chunks == 0:
            fallback_documents = self._heuristic_fallback_documents(scenario)
            return RAGRetrievalResult(
                query=query,
                retrieved_documents=[],
                always_included_documents=fallback_documents,
                knowledge_base_path=str(self.knowledge_base_path),
                indexed_chunks=0,
                runtime_available=True,
            )

        result_limit = min(self.top_k, indexed_chunks)

        # FIX: Embed the query exactly once and reuse the vector for both searches.
        # Previously similarity_search_with_score() embedded the query string on each
        # call, causing two round trips to text-embedding-3-small (~3s each = ~6s total).
        query_embedding = self.embeddings.embed_query(query)

        # Frameworks are prioritised — they get their own filtered search pass
        # so general chunks cannot crowd them out of the top-k.
        framework_matches = self._ethical_framework_matches(query_embedding, indexed_chunks)
        general_matches = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            query_embedding,
            k=min(self.top_k + self.framework_top_k, indexed_chunks),
        )
        matches = self._merge_matches(
            framework_matches,
            general_matches,
            limit=result_limit,
        )

        retrieved_documents = [
            self._to_retrieved_document(document, distance)
            for document, distance in matches
        ]

        # always_included_documents is empty in the normal path —
        # frameworks arrive via scored retrieval above.
        return RAGRetrievalResult(
            query=query,
            retrieved_documents=retrieved_documents,
            always_included_documents=[],
            knowledge_base_path=str(self.knowledge_base_path),
            indexed_chunks=indexed_chunks,
            runtime_available=True,
        )

    def close(self) -> None:
        client = getattr(self.vector_store, "_client", None)
        if client is not None and hasattr(client, "close"):
            client.close()

    # ── Query building ────────────────────────────────────────────────────────

    def _build_query(
        self,
        scenario: Scenario,
        mathematical_layer_result: MathematicalLayerResult | None,
    ) -> str:
        """
        Build a natural-language semantic query rather than a keyword list.
        Embedding models produce better similarity scores when the query reads
        like a description of what is being looked for.
        """
        obstacle_summary = ", ".join(
            f"{o.vulnerability_class} {o.type} moving {o.trajectory}"
            for o in scenario.obstacles
        ) or "no obstacles"

        actions = ", ".join(scenario.available_actions) or "none"
        road_type = self._canonical_road_type(scenario.environment.road_type)
        weather = self._canonical_weather(scenario.environment.weather)

        query = (
            f"Autonomous vehicle ethical decision-making: "
            f"{road_type} road, "
            f"{weather} weather, "
            f"{scenario.environment.time_of_day}, "
            f"{scenario.environment.traffic_density} traffic density. "
            f"Collision unavoidable: {scenario.collision_unavoidable}. "
            f"Road users involved: {obstacle_summary}. "
            f"Available actions: {actions}. "
            f"Which ethical framework — utilitarian risk minimization, deontological RSS rules, "
            f"Rawlsian maximin, ethics of risk, ethical valence theory, virtue ethics — "
            f"is most appropriate and why?"
        )

        if mathematical_layer_result is not None:
            violated = ", ".join(mathematical_layer_result.violated_rules) or "none"
            query += (
                f" RSS rules violated: {violated}. "
                f"Best action by total risk calculation: "
                f"{mathematical_layer_result.best_action_by_total_risk}."
            )

        # Append heuristic hints so the embedding is biased toward the
        # correct framework dimension without overriding semantic search.
        query += " " + self._heuristic_hint(scenario)
        return query

    def _heuristic_hint(self, scenario: Scenario) -> str:
        """
        Append scenario-specific ethical keywords derived from the selection
        heuristic table. These bias the embedding toward the most relevant
        framework(s) without hard-coding the final selection.
        """
        hints: list[str] = []

        vru_present = self._vru_present(scenario)

        if not scenario.collision_unavoidable:
            hints.append("deontological rule-based safety RSS responsibility sensitive")

        if vru_present:
            hints.append("maximin worst-off protection vulnerable road user child elderly cyclist")

        if scenario.collision_unavoidable and not vru_present:
            hints.append("utilitarian aggregate risk minimization total harm")

        if scenario.collision_unavoidable and vru_present:
            hints.append("ethics of risk weighted distribution equity fairness")
            hints.append(
                "ethical valence theory passenger pedestrian social valence hierarchy "
                "EVT passenger vs VRU occupant dilemma moral profile altruistic"
            )

        road_type = self._canonical_road_type(scenario.environment.road_type)
        if road_type in {"highway", "motorway"}:
            hints.append("high speed highway aggregate risk utilitarian")

        if road_type in {"school_zone", "hospital_zone"}:
            hints.append("maximin vulnerable protection school zone")

        return " ".join(hints)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _ethical_framework_matches(
        self, query_embedding: list[float], indexed_chunks: int
    ) -> list[tuple[Any, float]]:
        """
        Dedicated filtered search for ethical framework documents.
        Accepts a pre-computed query embedding to avoid a redundant API call.
        Falls back to unfiltered search + manual category check if the
        Chroma filter raises (some versions have filter compatibility issues).
        Falls back further to loading from disk if the vector store returns nothing.
        """
        if self.framework_top_k == 0 or self.vector_store is None:
            return []

        fetch_limit = min(max(self.framework_top_k * 4, self.framework_top_k), indexed_chunks)

        # Attempt 1 — filtered search by vector
        try:
            matches = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k=fetch_limit,
                filter={"category": self.ETHICAL_FRAMEWORK_CATEGORY},
            )
        except Exception:
            # Attempt 2 — unfiltered, then manually keep only framework docs
            try:
                all_matches = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                    query_embedding, k=fetch_limit * 2
                )
                matches = [
                    (doc, dist)
                    for doc, dist in all_matches
                    if (doc.metadata or {}).get("category") == self.ETHICAL_FRAMEWORK_CATEGORY
                ]
            except Exception:
                matches = []

        # Attempt 3 — load directly from disk as LangChain Documents
        if not matches:
            return self._load_frameworks_from_disk_as_matches()

        # Deduplicate by source file (not just chunk) so one framework file
        # cannot fill multiple slots in the top-k.
        selected: list[tuple[Any, float]] = []
        seen_sources: set[str] = set()
        for document, distance in matches:
            meta = document.metadata or {}
            source = str(meta.get("source") or meta.get("path") or "")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            selected.append((document, distance))
            if len(selected) >= self.framework_top_k:
                break

        return selected

    def _load_frameworks_from_disk_as_matches(self) -> list[tuple[Any, float]]:
        """
        Emergency fallback: load framework JSON files from disk and wrap them
        as pseudo-match tuples with a neutral distance of 0.5.
        Used when the vector store filter returns nothing.
        """
        from langchain_core.documents import Document

        framework_dir = self.knowledge_base_path / self.ETHICAL_FRAMEWORK_CATEGORY
        if not framework_dir.exists():
            return []

        results: list[tuple[Any, float]] = []
        for path in sorted(framework_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            if path.name.startswith("."):
                continue
            relative = path.relative_to(self.knowledge_base_path).as_posix()
            content = path.read_text(encoding="utf-8", errors="ignore")
            doc = Document(
                page_content=content,
                metadata={
                    "title": self._title_from_file(path, content),
                    "category": self.ETHICAL_FRAMEWORK_CATEGORY,
                    "path": relative,
                    "source": relative,
                },
            )
            results.append((doc, 0.5))
            if len(results) >= self.framework_top_k:
                break

        return results

    def _merge_matches(
        self,
        priority_matches: list[tuple[Any, float]],
        fallback_matches: list[tuple[Any, float]],
        *,
        limit: int,
    ) -> list[tuple[Any, float]]:
        """
        Merge framework-priority matches with general matches.
        Applies the minimum score threshold and deduplicates by chunk ID.
        """
        merged: list[tuple[Any, float]] = []
        seen_ids: set[str] = set()

        for document, distance in [*priority_matches, *fallback_matches]:
            score = round(1.0 / (1.0 + float(distance)), 4)
            if score < self.MIN_SCORE_THRESHOLD:
                continue
            doc_id = self._match_key(document)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            merged.append((document, distance))
            if len(merged) >= limit:
                break

        return merged

    # ── Fallback (vector store unavailable) ──────────────────────────────────

    def _heuristic_fallback_documents(
        self, scenario: Scenario
    ) -> list[AlwaysIncludedDocument]:
        """
        When the vector store is completely unavailable, load a small set of
        framework files from disk using the selection heuristic as a guide.
        Returns at most 2 frameworks to keep the context window manageable.
        """
        all_docs = self._always_included_framework_documents()
        if not all_docs:
            return []

        vru_present = self._vru_present(scenario)

        # Priority ordering based on the selection heuristic
        def priority(doc: AlwaysIncludedDocument) -> int:
            name = doc.title.lower()
            if not scenario.collision_unavoidable and "deontolog" in name:
                return 0
            if vru_present and "maximin" in name:
                return 0
            if scenario.collision_unavoidable and not vru_present and "utilitarian" in name:
                return 0
            if scenario.collision_unavoidable and vru_present and "ethics of risk" in name:
                return 0
            if scenario.collision_unavoidable and vru_present and (
                "valence" in name or "evt" in name
            ):
                return 0
            return 1

        sorted_docs = sorted(all_docs, key=priority)
        return sorted_docs[:2]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _default_knowledge_base_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "knowledge_base"

    def _vru_present(self, scenario: Scenario) -> bool:
        return any(
            obstacle.vulnerability_class.strip().lower() in self.VRU_VULNERABILITY_CLASSES
            or obstacle.type.strip().lower() in self.VRU_TYPES
            for obstacle in scenario.obstacles
        )

    def _canonical_weather(self, weather: str) -> str:
        value = weather.strip().lower()
        return self.WEATHER_CANONICAL_MAP.get(value, value)

    def _canonical_road_type(self, road_type: str) -> str:
        value = road_type.strip().lower()
        return self.ROAD_TYPE_CANONICAL_MAP.get(value, value)

    def _build_vector_store(self) -> Any:
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        load_project_env()
        embeddings = self.embeddings
        if embeddings is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
            embeddings = OpenAIEmbeddings(model=self.embedding_model)

        # FIX: store the resolved embeddings object back on self so retrieve()
        # can call self.embeddings.embed_query() for the single-embedding optimisation.
        self.embeddings = embeddings

        client_settings = Settings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=client_settings,
        )
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
            client=client,
            client_settings=client_settings,
        )

    def _collection_count(self) -> int:
        # FIX: cache the result — the knowledge base doesn't change at runtime,
        # so calling collection.count() on every retrieve() was pure overhead.
        cached_collection_count = getattr(self, "_cached_collection_count", None)
        if cached_collection_count is not None:
            return cached_collection_count

        if self.vector_store is None:
            return 0
        collection = getattr(self.vector_store, "_chroma_collection", None)
        if collection is None or not hasattr(collection, "count"):
            return 0

        self._cached_collection_count = int(collection.count())
        return self._cached_collection_count

    def _to_retrieved_document(self, document: Any, distance: float) -> RetrievedDocument:
        metadata = dict(document.metadata or {})
        document_id = str(
            metadata.get("chunk_id")
            or metadata.get("id")
            or metadata.get("path")
            or "unknown"
        )
        return RetrievedDocument(
            document_id=document_id,
            title=str(metadata.get("title", document_id)),
            category=str(metadata.get("category", "general")),
            path=str(metadata.get("path", "")),
            score=round(1.0 / (1.0 + float(distance)), 4),
            excerpt=self._build_excerpt(document.page_content),
            full_content=document.page_content,
        )

    def _build_excerpt(self, text: str, limit: int = EXCERPT_LIMIT) -> str:
        """
        Return a meaningful excerpt up to `limit` characters.
        Tries to cut at a sentence boundary to avoid mid-sentence truncation.
        """
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        # Prefer cutting at the last sentence boundary within the limit
        cutoff = normalized[:limit].rfind(". ")
        if cutoff > limit * 0.7:
            return normalized[: cutoff + 1]
        return normalized[: limit - 3].rstrip() + "..."

    def _match_key(self, document: Any) -> str:
        metadata = dict(document.metadata or {})
        return str(
            metadata.get("chunk_id")
            or metadata.get("record_id")
            or metadata.get("id")
            or metadata.get("path")
            or id(document)
        )

    def _always_included_framework_documents(self) -> list[AlwaysIncludedDocument]:
        """
        Load all framework files from disk. Result is cached after the first call
        since framework files do not change between retrieval calls.
        This is only used as a fallback — in normal operation frameworks are
        retrieved via scored vector search.
        """
        cached_documents = getattr(self, "_always_included_cache", None)
        if cached_documents is not None:
            return cached_documents

        framework_directory = self.knowledge_base_path / self.ETHICAL_FRAMEWORK_CATEGORY
        if not framework_directory.exists():
            self._always_included_cache = []
            return []

        documents: list[AlwaysIncludedDocument] = []
        for path in sorted(framework_directory.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            relative_path = path.relative_to(self.knowledge_base_path).as_posix()
            content = path.read_text(encoding="utf-8", errors="ignore")
            documents.append(
                AlwaysIncludedDocument(
                    document_id=relative_path,
                    title=self._title_from_file(path, content),
                    category=self.ETHICAL_FRAMEWORK_CATEGORY,
                    path=relative_path,
                    content=content,
                )
            )

        self._always_included_cache = documents
        return documents

    def _title_from_file(self, path: Path, content: str) -> str:
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return path.stem.replace("_", " ")
            if isinstance(payload, dict):
                for key in ("title", "name", "source"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return path.stem.replace("_", " ")

    # Keep old name as alias so existing call sites in other files don't break
    def _context_document_title(self, path: Path, content: str) -> str:
        return self._title_from_file(path, content)
