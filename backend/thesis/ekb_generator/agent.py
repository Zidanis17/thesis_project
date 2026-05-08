from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._env import load_project_env
from .prompts import FRAMEWORK_GENERATION_SYSTEM_PROMPT, PRINCIPLES_EXTRACTION_SYSTEM_PROMPT
from .schema import VALID_FRAMEWORK_IDS, validate_framework

__all__ = ["EKBGeneratorAgent", "EKBGeneratorResult"]


@dataclass(slots=True)
class EKBGeneratorResult:
    skipped: bool
    frameworks_written: list[str] = field(default_factory=list)
    principles_extracted: int = 0
    principles: list[dict[str, Any]] = field(default_factory=list)
    manifest_path: str = ""
    knowledge_base_path: str = ""
    model: str = ""
    generated_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EKBGeneratorAgent:
    """
    One-time setup agent that generates EF-01 through EF-06 JSON framework files by reading
    the German Ethics Commission report and academic papers via two LLM passes, then rebuilds
    ChromaDB.

    Usage:
        agent = EKBGeneratorAgent()
        result = agent.generate()          # skip if files already exist
        result = agent.generate(force=True) # always regenerate
    """

    DEFAULT_MODEL = "gpt-4o"
    MAX_RETRIES = 3

    # Character limits for PDF extraction per document type
    MAX_PDF_CHARS_COMMISSION = 80_000
    MAX_PDF_CHARS_RISK_PAPER = 40_000
    MAX_PDF_CHARS_PAPER = 30_000
    MAX_PDF_CHARS_PAPER_PASS2 = 12_000  # condensed excerpt used in Pass 2

    MANIFEST_FILENAME = ".ekb_manifest.json"

    # Output filename for each framework ID
    FRAMEWORK_FILES: dict[str, str] = {
        "EF-01": "EF-01_utilitarian.json",
        "EF-02": "EF-02_deontological.json",
        "EF-03": "EF-03_maximin.json",
        "EF-04": "EF-04_ethics_of_risk.json",
        "EF-05": "EF-05_evt.json",
        "EF-06": "EF-06_virtue_ethics.json",
    }

    # Deterministic metadata overwritten after generation to guarantee consistency
    _FIXED_METADATA: dict[str, dict[str, str]] = {
        "EF-01": {
            "name": "Utilitarian Risk Minimization",
            "alias": "Bayesian Aggregate Harm Minimization",
        },
        "EF-02": {
            "name": "Deontological Rule-Based Safety",
            "alias": "RSS Constraint-Based Ethics",
        },
        "EF-03": {
            "name": "Rawlsian Maximin",
            "alias": "Egalitarian Worst-Case Protection",
        },
        "EF-04": {
            "name": "Ethics of Risk",
            "alias": "Weighted Risk Distribution Hybrid",
        },
        "EF-05": {
            "name": "Ethical Valence Theory",
            "alias": "Social Valence and Harm Minimization",
        },
        "EF-06": {
            "name": "Virtue Ethics",
            "alias": "Skilled Driver Analogy and Reasonable Driver Standard",
        },
    }

    # EF-04 dominant_when is enforced regardless of what the LLM produces
    _EF04_DOMINANT_WHEN = [
        "never — EF-04 cannot be dominant_framework; set it only in contributing_frameworks"
    ]
    _EF04_DOMINANT_PROHIBITION = (
        "as dominant_framework — EF-04 is never dominant; "
        "it is the mathematical substrate within which EF-01, EF-02, EF-03, and EF-05 operate"
    )

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        knowledge_base_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.knowledge_base_path = Path(
            knowledge_base_path or self._default_knowledge_base_path()
        ).resolve()
        self.frameworks_dir = self.knowledge_base_path / "ethical_frameworks"

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, *, force: bool = False) -> EKBGeneratorResult:
        """
        Run the two-pass EKB generation pipeline.

        Pass 1 — extract governing principles from the German Ethics Commission report
                  and supporting academic papers.
        Pass 2 — generate the six EF-0X JSON framework definitions using those principles
                  plus paper content, validate against the schema, then write to disk and
                  rebuild ChromaDB.

        If *force* is False and all six framework files already exist the method returns
        immediately without calling the LLM.
        """
        if not force and self._frameworks_exist():
            return EKBGeneratorResult(
                skipped=True,
                frameworks_written=list(self.FRAMEWORK_FILES.values()),
                knowledge_base_path=str(self.knowledge_base_path),
                model=self.model,
            )

        generated_at = datetime.now(timezone.utc).isoformat()

        try:
            client = self._build_client()
        except Exception as exc:
            return EKBGeneratorResult(
                skipped=False,
                knowledge_base_path=str(self.knowledge_base_path),
                model=self.model,
                generated_at=generated_at,
                error=f"LLM client initialization failed: {exc}",
            )

        print(f"[EKBGenerator] model={self.model}  kb={self.knowledge_base_path}")

        # ── Load source documents ─────────────────────────────────────────────
        commission_text = self._load_pdf(
            self.knowledge_base_path
            / "german_ethics_commission"
            / "report-ethics-commission-automated-and-connected-driving.pdf",
            max_chars=self.MAX_PDF_CHARS_COMMISSION,
        )
        risk_paper_text = self._load_pdf(
            self.knowledge_base_path / "german_ethics_commission" / "Ethics_of_risk.pdf",
            max_chars=self.MAX_PDF_CHARS_RISK_PAPER,
        )
        paper_dir = self.knowledge_base_path / "papers"
        paper_paths = sorted(paper_dir.glob("*.pdf")) if paper_dir.is_dir() else []
        paper_texts_full = [
            self._load_pdf(p, max_chars=self.MAX_PDF_CHARS_PAPER) for p in paper_paths
        ]

        # ── Pass 1: extract ethical principles ───────────────────────────────
        print("[EKBGenerator] Pass 1 — extracting principles from source documents …")
        try:
            principles_result = self._run_pass1(
                client, commission_text, risk_paper_text, paper_texts_full
            )
        except Exception as exc:
            return EKBGeneratorResult(
                skipped=False,
                knowledge_base_path=str(self.knowledge_base_path),
                model=self.model,
                generated_at=generated_at,
                error=f"Pass 1 (principle extraction) failed: {exc}",
            )
        principles = principles_result.get("principles", [])
        print(f"[EKBGenerator] Pass 1 complete — {len(principles)} principles extracted.")

        # ── Pass 2: generate framework JSON definitions ───────────────────────
        paper_texts_condensed = [
            t[: self.MAX_PDF_CHARS_PAPER_PASS2] for t in paper_texts_full
        ]
        print("[EKBGenerator] Pass 2 — generating framework definitions …")
        try:
            frameworks = self._run_pass2(client, principles_result, paper_texts_condensed)
        except Exception as exc:
            return EKBGeneratorResult(
                skipped=False,
                principles_extracted=len(principles),
                principles=principles,
                knowledge_base_path=str(self.knowledge_base_path),
                model=self.model,
                generated_at=generated_at,
                error=f"Pass 2 (framework generation) failed: {exc}",
            )
        print(f"[EKBGenerator] Pass 2 complete — {len(frameworks)} frameworks generated.")

        # ── Write framework files ─────────────────────────────────────────────
        self.frameworks_dir.mkdir(parents=True, exist_ok=True)
        for fw_id, payload in frameworks.items():
            filename = self.FRAMEWORK_FILES[fw_id]
            (self.frameworks_dir / filename).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[EKBGenerator] Wrote {filename}")

        # ── Write provenance manifest ─────────────────────────────────────────
        manifest: dict[str, Any] = {
            "generated_at": generated_at,
            "model": self.model,
            "principles_extracted": len(principles),
            "principles": principles,
            "endorsed_frameworks": principles_result.get("endorsed_frameworks", []),
            "rejected_approaches": principles_result.get("rejected_approaches", []),
            "vru_protections": principles_result.get("vru_protections", []),
            "key_prohibitions": principles_result.get("key_prohibitions", []),
            "responsibility_principles": principles_result.get("responsibility_principles", []),
            "frameworks_generated": sorted(frameworks.keys()),
            "knowledge_base_path": str(self.knowledge_base_path),
            "frameworks_directory": str(self.frameworks_dir),
        }
        manifest_path = self.frameworks_dir / self.MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[EKBGenerator] Manifest written to {manifest_path}")

        # ── Rebuild ChromaDB ──────────────────────────────────────────────────
        print("[EKBGenerator] Rebuilding ChromaDB …")
        try:
            self._rebuild_chroma()
            print("[EKBGenerator] ChromaDB rebuild complete.")
        except Exception as exc:
            # Non-fatal: frameworks are written; ingestion can be run separately.
            print(f"[EKBGenerator] WARNING: ChromaDB rebuild failed ({exc}). "
                  "Run scripts/ingest_kb.py manually.")

        return EKBGeneratorResult(
            skipped=False,
            frameworks_written=sorted(frameworks.keys()),
            principles_extracted=len(principles),
            principles=principles,
            manifest_path=str(manifest_path),
            knowledge_base_path=str(self.knowledge_base_path),
            model=self.model,
            generated_at=generated_at,
        )

    # ── Idempotency guard ─────────────────────────────────────────────────────

    def _frameworks_exist(self) -> bool:
        return all(
            (self.frameworks_dir / filename).is_file()
            for filename in self.FRAMEWORK_FILES.values()
        )

    # ── PDF text extraction ───────────────────────────────────────────────────

    def _load_pdf(self, path: Path, *, max_chars: int) -> str:
        if not path.exists():
            return f"[PDF not found: {path.name}]"
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            parts: list[str] = []
            total = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                parts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
            return "\n".join(parts)[:max_chars]
        except Exception as exc:
            return f"[PDF extraction failed for {path.name}: {exc}]"

    # ── LLM client ────────────────────────────────────────────────────────────

    def _build_client(self) -> Any:
        from langchain_openai import ChatOpenAI

        load_project_env()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=api_key,
        )

    # ── Pass 1: principle extraction ──────────────────────────────────────────

    def _run_pass1(
        self,
        client: Any,
        commission_text: str,
        risk_paper_text: str,
        paper_texts: list[str],
    ) -> dict[str, Any]:
        paper_block = "\n\n".join(
            f"--- Academic Paper {i + 1} ---\n{text}"
            for i, text in enumerate(paper_texts)
        )
        user_message = (
            "Extract the governing ethical principles from these source documents.\n\n"
            "=== GERMAN ETHICS COMMISSION REPORT (2017) ===\n"
            f"{commission_text}\n\n"
            "=== ETHICS OF RISK — Supplementary Paper ===\n"
            f"{risk_paper_text}\n\n"
            "=== ACADEMIC PAPERS ===\n"
            f"{paper_block}\n\n"
            "Respond with the JSON object as specified. No prose or markdown."
        )

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.invoke(
                    [
                        ("system", PRINCIPLES_EXTRACTION_SYSTEM_PROMPT),
                        ("user", user_message),
                    ]
                )
                return self._parse_json(self._extract_text(response))
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[EKBGenerator] Pass 1 attempt {attempt + 1} failed: {exc}. Retrying …")

        raise RuntimeError(f"Pass 1 failed after {self.MAX_RETRIES} attempts") from last_exc

    # ── Pass 2: framework generation ─────────────────────────────────────────

    def _run_pass2(
        self,
        client: Any,
        principles_result: dict[str, Any],
        paper_excerpts: list[str],
    ) -> dict[str, dict[str, Any]]:
        paper_block = "\n\n".join(
            f"--- Paper {i + 1} Excerpt ---\n{text}"
            for i, text in enumerate(paper_excerpts)
        )
        base_user_message = (
            "Generate all six ethical framework JSON definitions using the principles and papers below.\n\n"
            "=== EXTRACTED ETHICAL PRINCIPLES ===\n"
            f"{json.dumps(principles_result, indent=2)}\n\n"
            "=== ACADEMIC PAPER EXCERPTS ===\n"
            f"{paper_block}\n\n"
            "Respond with a JSON object keyed EF-01 through EF-06. No prose or markdown."
        )

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            messages: list[tuple[str, str]] = [
                ("system", FRAMEWORK_GENERATION_SYSTEM_PROMPT),
                ("user", base_user_message),
            ]
            if attempt > 0 and last_exc is not None:
                messages.append(
                    (
                        "user",
                        f"Your previous response had validation errors. Fix them:\n{last_exc}",
                    )
                )
            try:
                response = client.invoke(messages)
                raw = self._parse_json(self._extract_text(response))
                frameworks = self._post_process(raw)
                errors = self._validate_all(frameworks)
                if errors:
                    raise ValueError(
                        f"Schema validation failed ({len(errors)} error(s)):\n"
                        + "\n".join(f"  • {e}" for e in errors)
                    )
                return frameworks
            except ValueError as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[EKBGenerator] Pass 2 attempt {attempt + 1} failed: {exc}. Retrying …")
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[EKBGenerator] Pass 2 attempt {attempt + 1} error: {exc}. Retrying …")

        raise RuntimeError(f"Pass 2 failed after {self.MAX_RETRIES} attempts") from last_exc

    # ── Post-processing ───────────────────────────────────────────────────────

    def _post_process(self, raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Enforce deterministic metadata and critical pipeline invariants on the LLM output.

        Overwrites framework_id, name, alias, title, source, category — these must be
        canonical regardless of what the LLM produced.  Enforces EF-04's never-dominant
        constraint deterministically so it cannot be broken by LLM drift.
        """
        result: dict[str, dict[str, Any]] = {}

        for fw_id, meta in self._FIXED_METADATA.items():
            payload = raw.get(fw_id)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"LLM output is missing framework '{fw_id}' or it is not a JSON object"
                )

            name = meta["name"]
            title = f"{fw_id} — {name}"

            payload["framework_id"] = fw_id
            payload["name"] = name
            payload["alias"] = meta["alias"]
            payload["title"] = title
            payload["source"] = title
            payload["category"] = "ethical_frameworks"

            # EF-04: enforce never-dominant invariant
            if fw_id == "EF-04":
                payload["dominant_when"] = self._EF04_DOMINANT_WHEN
                avoid_when = payload.get("avoid_when")
                if not isinstance(avoid_when, list):
                    avoid_when = []
                if not any("dominant_framework" in item for item in avoid_when if isinstance(item, str)):
                    avoid_when.insert(0, self._EF04_DOMINANT_PROHIBITION)
                payload["avoid_when"] = avoid_when

            result[fw_id] = payload

        return result

    # ── Schema validation ─────────────────────────────────────────────────────

    def _validate_all(self, frameworks: dict[str, dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        for fw_id in VALID_FRAMEWORK_IDS:
            payload = frameworks.get(fw_id)
            if payload is None:
                errors.append(f"Missing framework {fw_id}")
            else:
                errors.extend(validate_framework(payload, fw_id))
        return errors

    # ── ChromaDB rebuild ──────────────────────────────────────────────────────

    def _rebuild_chroma(self) -> None:
        from ..rag import KnowledgeBaseIngester

        ingester = KnowledgeBaseIngester(knowledge_base_path=self.knowledge_base_path)
        ingester.ingest(reset_collection=True)

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(response: Any) -> str:
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

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        # Strip markdown fences if present
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```\s*$", "", stripped)
        # Find the outermost JSON object
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object, got {type(payload).__name__}")
        return payload

    # ── Path resolution ───────────────────────────────────────────────────────

    @staticmethod
    def _default_knowledge_base_path() -> Path:
        # backend/thesis/ekb_generator/agent.py → parents[2] = backend/
        return Path(__file__).resolve().parents[2] / "knowledge_base"
