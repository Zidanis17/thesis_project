from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest the thesis knowledge base into the persisted Chroma collection.",
    )
    parser.add_argument(
        "--knowledge-base-path",
        default="knowledge_base",
        help="Path to the knowledge base root. Defaults to ./knowledge_base",
    )
    parser.add_argument(
        "--persist-directory",
        default=None,
        help="Optional Chroma persist directory. Defaults to <knowledge-base-path>/.chroma",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Optional Chroma collection name override.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional embedding model override.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional chunk size override.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Optional chunk overlap override.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the existing collection instead of resetting it first.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from thesis._env import load_project_env
    from thesis import KnowledgeBaseIngester

    load_project_env()

    def resolve_path(value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        return path

    kwargs: dict[str, object] = {
        "knowledge_base_path": resolve_path(args.knowledge_base_path),
    }
    if args.persist_directory:
        kwargs["persist_directory"] = resolve_path(args.persist_directory)
    if args.collection_name:
        kwargs["collection_name"] = args.collection_name
    if args.embedding_model:
        kwargs["embedding_model"] = args.embedding_model
    if args.chunk_size is not None:
        kwargs["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        kwargs["chunk_overlap"] = args.chunk_overlap

    ingester = KnowledgeBaseIngester(**kwargs)
    try:
        result = ingester.ingest(reset_collection=not args.append)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    finally:
        ingester.close()

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
