from __future__ import annotations

import importlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .._env import load_project_env
from .engine import DeterministicRAGRetriever

__all__ = [
    "KnowledgeBaseIngester",
    "KnowledgeBaseIngestionResult",
]


@dataclass(slots=True)
class KnowledgeBaseIngestionResult:
    knowledge_base_path: str
    persist_directory: str
    collection_name: str
    ingested_files: int
    source_documents: int
    stored_chunks: int
    # How many framework files were stored whole (not chunked)
    framework_files_whole: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeBaseIngester:
    SUPPORTED_SUFFIXES = {".md", ".txt", ".json", ".pdf"}
    IGNORED_FILENAMES = {"readme.md", "readme.txt", ".gitkeep", ".keep"}
    DEFAULT_CHUNK_SIZE = 400
    DEFAULT_CHUNK_OVERLAP = 60
    DEFAULT_COLLECTION_NAME = DeterministicRAGRetriever.DEFAULT_COLLECTION_NAME
    DEFAULT_EMBEDDING_MODEL = DeterministicRAGRetriever.DEFAULT_EMBEDDING_MODEL
    ETHICAL_FRAMEWORK_CATEGORY = DeterministicRAGRetriever.ETHICAL_FRAMEWORK_CATEGORY

    # Framework JSON fields that are semantically meaningful for embedding.
    # These are concatenated into a single embedding_text at index time.
    FRAMEWORK_EMBEDDING_FIELDS = (
        "foundation",
        "decision_logic",
        "pros",
        "cons",
        "best_fit_scenarios",
        "poor_fit_scenarios",
        "tradeoffs",
        "scenario_tags",
    )

    JSON_RECORD_KEYS = ("documents", "cases", "scenarios", "items", "chunks")
    JSON_TITLE_KEYS = ("title", "name", "id", "scenario_id")
    JSON_PRIMARY_TEXT_KEYS = ("summary", "content", "text", "description", "rationale")
    JSON_PREFERRED_FIELD_ORDER = (
        "type",
        "summary",
        "content",
        "text",
        "description",
        "rationale",
        "av_implication",
        "av_applicability",
        "formula",
        "weakness",
        "open_problem",
        "use_when",
        "avoid_when",
        "constraint_list",
        "tags",
    )

    def __init__(
        self,
        knowledge_base_path: str | Path | None = None,
        *,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embeddings: Any | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.knowledge_base_path = Path(
            knowledge_base_path or self._default_knowledge_base_path()
        ).resolve()
        self.persist_directory = Path(
            persist_directory or self.knowledge_base_path / ".chroma"
        ).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.chunk_size = max(200, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self._runtime_error: RuntimeError | None = None
        self.vector_store: Any | None = None
        self.text_splitter: Any | None = None

        try:
            self.vector_store, self.text_splitter = self._build_runtime_components()
        except Exception as exc:
            self._runtime_error = RuntimeError(
                "Knowledge-base ingestion could not be initialized. Install the LangChain "
                "RAG dependencies and set OPENAI_API_KEY, or inject a custom embeddings object."
            )
            self._runtime_error.__cause__ = exc

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, *, reset_collection: bool = True) -> KnowledgeBaseIngestionResult:
        if (
            self._runtime_error is not None
            or self.vector_store is None
            or self.text_splitter is None
        ):
            raise self._runtime_error or RuntimeError("Knowledge-base ingester is unavailable")

        # Separate framework documents (stored whole) from everything else (chunked).
        framework_documents, general_source_documents = self._load_source_documents()
        chunked_documents = self._chunk_documents(general_source_documents)
        all_documents = framework_documents + chunked_documents

        if reset_collection:
            self.vector_store.delete_collection()
            self.close()
            self.vector_store, self.text_splitter = self._build_runtime_components()

        if all_documents:
            self.vector_store.add_documents(
                all_documents,
                ids=[str(doc.metadata["chunk_id"]) for doc in all_documents],
            )

        unique_paths = {
            str(doc.metadata["path"])
            for doc in framework_documents + general_source_documents
        }
        return KnowledgeBaseIngestionResult(
            knowledge_base_path=str(self.knowledge_base_path),
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
            ingested_files=len(unique_paths),
            source_documents=len(framework_documents) + len(general_source_documents),
            stored_chunks=self._collection_count(),
            framework_files_whole=len(framework_documents),
        )

    def close(self) -> None:
        client = getattr(self.vector_store, "_client", None)
        if client is not None and hasattr(client, "close"):
            client.close()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_source_documents(self) -> tuple[list[Any], list[Any]]:
        """
        Returns (framework_documents, general_documents).

        Framework documents (knowledge_base/ethical_frameworks/) are stored
        as single whole documents — no chunking. This preserves the full
        semantic content of each framework entry and prevents retrieval from
        returning an incomplete chunk that is missing e.g. tradeoffs or cons.

        Everything else is loaded as normal and will be chunked downstream.
        """
        from langchain_core.documents import Document

        if not self.knowledge_base_path.exists():
            return [], []

        framework_docs: list[Document] = []
        general_docs: list[Document] = []

        for path in sorted(self.knowledge_base_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                continue
            if path.name.lower() in self.IGNORED_FILENAMES:
                continue
            if path.is_relative_to(self.persist_directory):
                continue

            relative_path = path.relative_to(self.knowledge_base_path).as_posix()
            category = self._category_for_path(path)

            if category == self.ETHICAL_FRAMEWORK_CATEGORY:
                # Store framework files whole — do not chunk.
                docs = self._load_framework_document(path, relative_path)
                framework_docs.extend(docs)
            elif path.suffix.lower() == ".pdf":
                general_docs.extend(
                    self._load_pdf_documents(path, relative_path, category)
                )
            else:
                for item in self._load_text_documents(path, relative_path, category):
                    general_docs.append(
                        Document(
                            page_content=item["text"],
                            metadata=item["metadata"],
                        )
                    )

        return framework_docs, general_docs

    def _load_framework_document(
        self, path: Path, relative_path: str
    ) -> list[Any]:
        """
        Load a single ethical framework file as one whole document.

        For JSON files the embedding text is built from the semantically
        meaningful fields only (foundation, decision_logic, pros, cons, etc.)
        so the vector representation captures the framework's ethical content
        rather than its metadata fields (ids, sources, parameters).

        The full raw JSON is stored in page_content so the LLM layer can
        read any field it needs at inference time.
        """
        from langchain_core.documents import Document

        raw_content = path.read_text(encoding="utf-8", errors="ignore")
        title = self._title_from_file(path, raw_content)

        metadata: dict[str, Any] = {
            "title": title,
            "category": self.ETHICAL_FRAMEWORK_CATEGORY,
            "path": relative_path,
            "source": relative_path,
            "file_type": path.suffix.lower().lstrip("."),
            # Framework files are always stored as single chunks.
            "chunk_id": f"{relative_path}::whole",
            "chunk_index": 1,
        }

        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(raw_content)
            except json.JSONDecodeError:
                # Malformed JSON — fall back to raw text embedding
                embedding_text = self._compose_content(title, raw_content)
                return [Document(page_content=embedding_text, metadata=metadata)]

            if isinstance(payload, dict):
                # Enrich metadata with top-level identifiers
                for key in ("framework_id", "name", "alias"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        metadata[key] = value.strip()

                embedding_text = self._build_framework_embedding_text(title, payload)
                # page_content = full raw JSON so LLM layer gets everything
                return [
                    Document(
                        page_content=raw_content,
                        metadata={**metadata, "embedding_text": embedding_text},
                    )
                ]

        # Non-JSON framework file (markdown/txt) — embed and store whole
        embedding_text = self._compose_content(title, raw_content)
        return [Document(page_content=raw_content, metadata=metadata)]

    def _build_framework_embedding_text(
        self, title: str, payload: dict[str, Any]
    ) -> str:
        """
        Build a concise, semantically dense embedding string from the fields
        that describe the framework's ethical character. Avoids noisy fields
        like source_papers, key_parameters, embedding_vector that add tokens
        without improving retrieval relevance.
        """
        parts: list[str] = [title]
        for field in self.FRAMEWORK_EMBEDDING_FIELDS:
            value = payload.get(field)
            if value is None:
                continue
            normalized = self._normalize_json_value(value)
            if normalized:
                label = field.replace("_", " ").capitalize()
                parts.append(f"{label}: {normalized}")
        return "\n\n".join(parts)

    # ── PDF / text loading (unchanged from original) ──────────────────────────

    def _load_pdf_documents(
        self, path: Path, relative_path: str, category: str
    ) -> list[Any]:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_core.documents import Document

        loader = PyPDFLoader(str(path), mode="page")
        documents = loader.load()
        output: list[Document] = []
        for page_index, document in enumerate(documents, start=1):
            title = f"{path.stem.replace('_', ' ')} page {page_index}"
            metadata = {
                "title": title,
                "category": category,
                "path": relative_path,
                "file_type": "pdf",
                "page": page_index,
            }
            output.append(
                Document(
                    page_content=self._compose_content(title, document.page_content),
                    metadata=metadata,
                )
            )
        return output

    def _load_text_documents(
        self, path: Path, relative_path: str, category: str
    ) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._json_documents(path, relative_path, category, text)
        title = path.stem.replace("_", " ")
        return [
            {
                "text": self._compose_content(title, text),
                "metadata": {
                    "title": title,
                    "category": category,
                    "path": relative_path,
                    "file_type": suffix.lstrip("."),
                },
            }
        ]

    def _json_documents(
        self,
        path: Path,
        relative_path: str,
        category: str,
        text: str,
    ) -> list[dict[str, Any]]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            title = path.stem.replace("_", " ")
            return [
                {
                    "text": self._compose_content(title, text),
                    "metadata": {
                        "title": title,
                        "category": category,
                        "path": relative_path,
                        "file_type": "json",
                    },
                }
            ]

        records, context = self._json_records_and_context(payload)
        output: list[dict[str, Any]] = []
        for record_index, record in enumerate(records, start=1):
            title = self._json_title(
                record,
                default=f"{path.stem.replace('_', ' ')} item {record_index}",
            )
            metadata: dict[str, Any] = {
                "title": title,
                "category": category,
                "path": relative_path,
                "file_type": "json",
                "record_index": record_index,
            }
            if isinstance(record, dict):
                record_id = record.get("id")
                if isinstance(record_id, str) and record_id.strip():
                    metadata["record_id"] = record_id.strip()
                record_type = record.get("type")
                if isinstance(record_type, str) and record_type.strip():
                    metadata["record_type"] = record_type.strip()
            source_title = context.get("source")
            if isinstance(source_title, str) and source_title.strip():
                metadata["source_document"] = source_title.strip()
            output.append(
                {
                    "text": self._compose_content(
                        title, self._json_text(record, context=context)
                    ),
                    "metadata": metadata,
                }
            )
        return output

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_documents(self, source_documents: list[Any]) -> list[Any]:
        """
        Chunk general (non-framework) documents. Framework documents must NOT
        be passed here — they are stored whole in _load_source_documents.
        """
        assert self.text_splitter is not None
        chunked_documents = self.text_splitter.split_documents(source_documents)
        counters: Counter[str] = Counter()
        for document in chunked_documents:
            source_path = str(document.metadata.get("path", "unknown"))
            counters[source_path] += 1
            document.metadata["chunk_index"] = counters[source_path]
            document.metadata["chunk_id"] = (
                f"{source_path}::chunk_{counters[source_path]:03d}"
            )
        return chunked_documents

    # ── Runtime ───────────────────────────────────────────────────────────────

    def _build_runtime_components(self) -> tuple[Any, Any]:
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        load_project_env()
        embeddings = self.embeddings
        if embeddings is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
            embeddings = OpenAIEmbeddings(model=self.embedding_model)

        client_settings = Settings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=client_settings,
        )
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
            client=client,
            client_settings=client_settings,
        )
        text_splitter = self._build_text_splitter(RecursiveCharacterTextSplitter)
        return vector_store, text_splitter

    def _build_text_splitter(self, splitter_class: Any) -> Any:
        splitter_kwargs = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        try:
            importlib.import_module("tiktoken")
        except Exception:
            return splitter_class(**splitter_kwargs)
        try:
            return splitter_class.from_tiktoken_encoder(
                model_name=self.embedding_model, **splitter_kwargs
            )
        except Exception:
            try:
                return splitter_class.from_tiktoken_encoder(
                    encoding_name="cl100k_base", **splitter_kwargs
                )
            except Exception:
                return splitter_class(**splitter_kwargs)

    def _collection_count(self) -> int:
        if self.vector_store is None:
            return 0
        collection = getattr(self.vector_store, "_chroma_collection", None)
        if collection is None or not hasattr(collection, "count"):
            return 0
        return int(collection.count())

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _default_knowledge_base_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "knowledge_base"

    def _category_for_path(self, path: Path) -> str:
        parts = path.relative_to(self.knowledge_base_path).parts
        return parts[0] if len(parts) > 1 else "general"

    def _compose_content(self, title: str, text: str) -> str:
        content = text.strip()
        return title if not content else f"{title}\n\n{content}"

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

    def _json_title(self, item: Any, default: str) -> str:
        if isinstance(item, dict):
            for key in self.JSON_TITLE_KEYS:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return default

    def _json_records_and_context(
        self, payload: Any
    ) -> tuple[list[Any], dict[str, Any]]:
        if isinstance(payload, list):
            return payload, {}
        if not isinstance(payload, dict):
            return [payload], {}

        context: dict[str, Any] = {}
        source = payload.get("source")
        if isinstance(source, str) and source.strip():
            context["source"] = source.strip()
        compiled_from = payload.get("compiled_from")
        if isinstance(compiled_from, list):
            references = [
                r.strip() for r in compiled_from if isinstance(r, str) and r.strip()
            ]
            if references:
                context["compiled_from"] = references

        for key in self.JSON_RECORD_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value, context

        return [payload], context

    def _json_text(self, item: Any, *, context: dict[str, Any] | None = None) -> str:
        sections = self._json_context_sections(context or {})
        if isinstance(item, dict):
            seen_keys: set[str] = set()
            for key in self.JSON_PREFERRED_FIELD_ORDER:
                rendered = self._render_json_field(key, item.get(key))
                if rendered:
                    sections.append(rendered)
                    seen_keys.add(key)
            ignored_keys = set(self.JSON_TITLE_KEYS) | {"source", "compiled_from"}
            for key, value in item.items():
                if key in seen_keys or key in ignored_keys:
                    continue
                rendered = self._render_json_field(key, value)
                if rendered:
                    sections.append(rendered)
            if sections:
                return "\n\n".join(sections)
        serialized = json.dumps(item, indent=2, ensure_ascii=True, sort_keys=True)
        if sections:
            sections.append(serialized)
            return "\n\n".join(sections)
        return serialized

    def _json_context_sections(self, context: dict[str, Any]) -> list[str]:
        sections: list[str] = []
        source = context.get("source")
        if isinstance(source, str) and source.strip():
            sections.append(f"Source: {source.strip()}")
        compiled_from = context.get("compiled_from")
        if isinstance(compiled_from, list):
            references = [
                r.strip() for r in compiled_from if isinstance(r, str) and r.strip()
            ]
            if references:
                sections.append(f"Compiled from: {'; '.join(references)}")
        return sections

    def _render_json_field(self, key: str, value: Any) -> str:
        if key in self.JSON_TITLE_KEYS or value is None:
            return ""
        normalized = self._normalize_json_value(value)
        if not normalized:
            return ""
        if key in self.JSON_PRIMARY_TEXT_KEYS:
            return normalized
        label = self._json_field_label(key)
        return f"{label}: {normalized}"

    def _json_field_label(self, key: str) -> str:
        words: list[str] = []
        for index, part in enumerate(key.split("_")):
            if part in {"av", "llm", "pdf"}:
                words.append(part.upper())
                continue
            words.append(part.capitalize() if index == 0 else part.lower())
        return " ".join(words)

    def _normalize_json_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            items = [self._normalize_json_value(item) for item in value]
            return ", ".join(item for item in items if item)
        if isinstance(value, dict):
            parts: list[str] = []
            for k, v in value.items():
                normalized = self._normalize_json_value(v)
                if normalized:
                    parts.append(f"{k.replace('_', ' ')}: {normalized}")
            return "; ".join(parts)
        return ""