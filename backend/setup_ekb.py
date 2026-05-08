#!/usr/bin/env python3
"""
CLI for generating the Ethical Knowledge Base (EKB) framework files.

Runs two LLM passes:
  Pass 1 — extract governing ethical principles from the German Ethics Commission
            report and supporting academic papers.
  Pass 2 — generate EF-01 through EF-06 JSON framework definitions from those
            principles, validate schema, write files, and rebuild ChromaDB.

Usage
-----
  python setup_ekb.py                        # skip if files already exist
  python setup_ekb.py --force                # always regenerate
  python setup_ekb.py --model gpt-4o-mini    # choose a different model
  python setup_ekb.py --knowledge-base PATH  # custom knowledge base directory

Programmatic usage (from ScenarioPipeline):
  from thesis.core.pipeline import ScenarioPipeline
  result = ScenarioPipeline.setup_ekb(force=True)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the backend package importable when the script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="setup_ekb",
        description=(
            "Generate EF-01 through EF-06 ethical framework JSON files from source documents "
            "and rebuild the ChromaDB knowledge base."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate framework files even if they already exist.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="LLM model name (default: gpt-4o).",
    )
    parser.add_argument(
        "--knowledge-base",
        default=None,
        dest="knowledge_base",
        metavar="PATH",
        help="Path to the knowledge_base directory (default: auto-detected).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print the EKBGeneratorResult as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        from thesis.ekb_generator import EKBGeneratorAgent
    except ImportError as exc:
        print(
            f"ERROR: Could not import thesis package: {exc}\n"
            "Run this script from the backend/ directory or ensure the package is installed.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = args.model or EKBGeneratorAgent.DEFAULT_MODEL
    agent = EKBGeneratorAgent(
        model=model,
        knowledge_base_path=args.knowledge_base,
    )

    if not args.output_json:
        print("=" * 60)
        print("  EKB Generator — Ethical Knowledge Base setup")
        print("=" * 60)
        print(f"  Model          : {agent.model}")
        print(f"  Knowledge base : {agent.knowledge_base_path}")
        print(f"  Frameworks dir : {agent.frameworks_dir}")
        print()

    if not args.force and agent._frameworks_exist():
        if args.output_json:
            from thesis.ekb_generator import EKBGeneratorResult

            result = EKBGeneratorResult(
                skipped=True,
                frameworks_written=list(EKBGeneratorAgent.FRAMEWORK_FILES.values()),
                knowledge_base_path=str(agent.knowledge_base_path),
                model=agent.model,
            )
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(
                "All six framework files already exist — skipping generation.\n"
                "Use --force to regenerate them."
            )
        return

    if not args.output_json:
        print("Starting EKB generation …\n")

    result = agent.generate(force=args.force)

    if args.output_json:
        print(json.dumps(result.to_dict(), indent=2))
        if result.error:
            sys.exit(1)
        return

    if result.error:
        print(f"\nERROR: {result.error}", file=sys.stderr)
        sys.exit(1)

    if result.skipped:
        print("Skipped — framework files already exist. Use --force to regenerate.")
        return

    print()
    print("=" * 60)
    print("  Generation complete")
    print("=" * 60)
    print(f"  Principles extracted : {result.principles_extracted}")
    print(f"  Frameworks written   : {', '.join(result.frameworks_written)}")
    if result.manifest_path:
        print(f"  Manifest             : {result.manifest_path}")
    print()
    print("ChromaDB has been rebuilt. The pipeline is ready.")


if __name__ == "__main__":
    main()
