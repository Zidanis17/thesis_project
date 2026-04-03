import json
import os
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from thesis import (
    DeterministicRAGRetriever,
    DeterministicScenarioParser,
    KnowledgeBaseIngester,
    ScenarioPipeline,
)


def build_sample_payload() -> dict:
    return {
        "ego_vehicle": {
            "speed_kmh": 60,
            "acceleration_ms2": -2.1,
            "heading_deg": 270,
            "lane_position": "center",
            "braking_distance_m": 42.5,
            "mass_kg": 1800,
        },
        "environment": {
            "road_type": "residential",
            "speed_limit_kmh": 50,
            "weather": "clear",
            "visibility_m": 120,
            "time_of_day": "daytime",
            "traffic_density": "low",
        },
        "obstacles": [
            {
                "id": "obj_01",
                "type": "child_pedestrian",
                "distance_m": 10.2,
                "relative_speed_kmh": 60,
                "time_to_impact_s": 0.61,
                "trajectory": "crossing",
                "vulnerability_class": "high",
                "mass_kg": 30,
                "responsible_for_risk": False,
            },
            {
                "id": "obj_02",
                "type": "parked_vehicle",
                "distance_m": 6.5,
                "relative_speed_kmh": 60,
                "time_to_impact_s": 0.39,
                "trajectory": "stationary",
                "vulnerability_class": "low",
                "mass_kg": 1500,
                "responsible_for_risk": False,
            },
        ],
        "sensor_confidence": {
            "lidar": 0.97,
            "camera": 0.91,
            "radar": 0.95,
            "overall_scene_confidence": 0.93,
            "occluded_zones": ["left_sidewalk"],
        },
        "available_actions": [
            "brake_straight",
            "swerve_left",
            "swerve_right",
            "brake_swerve_left",
        ],
        "collision_unavoidable": True,
    }


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 24
        for token in text.lower().replace("_", " ").replace("-", " ").split():
            bucket = sha256(token.encode("utf-8")).digest()[0] % len(vector)
            vector[bucket] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


def write_text_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream.encode('latin-1'))} >> stream\n{stream}\nendstream\nendobj",
    ]

    pdf = "%PDF-1.4\n"
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(pdf.encode("latin-1")))
        pdf += obj + "\n"

    xref_offset = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n"
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
    pdf += f"startxref\n{xref_offset}\n%%EOF"
    path.write_bytes(pdf.encode("latin-1"))


class RAGRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DeterministicScenarioParser()
        self.payload = build_sample_payload()
        self.embeddings = FakeEmbeddings()

    def test_ingestion_is_separate_and_runtime_retrieval_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_base = root / "knowledge_base"
            persist_directory = root / "vector_store"
            (knowledge_base / "german_ethics_commission").mkdir(parents=True)
            (knowledge_base / "ethical_frameworks").mkdir()
            (knowledge_base / "legal_constraints").mkdir()
            (knowledge_base / "similar_scenarios").mkdir()

            (knowledge_base / "german_ethics_commission" / "guideline_07.md").write_text(
                "# Guideline 7\nProtect vulnerable road users, especially children and pedestrians, when collision is unavoidable.",
                encoding="utf-8",
            )
            (knowledge_base / "ethical_frameworks" / "risk_matrix.txt").write_text(
                "Ethics of risk uses maximin reasoning when a highly vulnerable child pedestrian is exposed to severe harm.",
                encoding="utf-8",
            )
            (knowledge_base / "legal_constraints" / "residential_speed.json").write_text(
                '[{"title": "Residential right-of-way", "summary": "Residential roads require caution, speed-limit compliance, and protection of pedestrians with priority."}]',
                encoding="utf-8",
            )
            write_text_pdf(
                knowledge_base / "similar_scenarios" / "occluded_sidewalk.pdf",
                "Hidden pedestrian risk can emerge from an occluded left sidewalk in a residential road scenario.",
            )

            scenario = self.parser.parse(self.payload).scenario

            try:
                ingester = KnowledgeBaseIngester(
                    knowledge_base_path=knowledge_base,
                    persist_directory=persist_directory,
                    embeddings=self.embeddings,
                )
                ingestion_result = ingester.ingest()
            finally:
                ingester.close()

            retriever = DeterministicRAGRetriever(
                knowledge_base_path=knowledge_base,
                persist_directory=persist_directory,
                top_k=4,
                embeddings=self.embeddings,
            )

            try:
                first_result = retriever.retrieve(scenario)
                second_result = retriever.retrieve(scenario)
            finally:
                retriever.close()

            self.assertEqual(ingestion_result.ingested_files, 4)
            self.assertEqual(ingestion_result.stored_chunks, 4)
            self.assertEqual(
                [document.document_id for document in first_result.retrieved_documents],
                [document.document_id for document in second_result.retrieved_documents],
            )
            self.assertEqual(len(first_result.retrieved_documents), 4)
            self.assertEqual(
                {document.category for document in first_result.retrieved_documents},
                {
                    "german_ethics_commission",
                    "ethical_frameworks",
                    "legal_constraints",
                    "similar_scenarios",
                },
            )
            self.assertTrue(
                any(document.path.endswith(".pdf") for document in first_result.retrieved_documents)
            )
            self.assertTrue(first_result.runtime_available)
            self.assertEqual(first_result.runtime_error, None)
            self.assertEqual(first_result.always_included_documents, [])

    def test_pipeline_uses_runtime_retriever_after_offline_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_base = root / "knowledge_base"
            persist_directory = root / "vector_store"
            (knowledge_base / "ethical_frameworks").mkdir(parents=True)
            (knowledge_base / "ethical_frameworks" / "virtue_ethics.md").write_text(
                "# Virtue Ethics\nContext-sensitive responsibility matters when hidden pedestrians may emerge from an occluded sidewalk.",
                encoding="utf-8",
            )

            try:
                ingester = KnowledgeBaseIngester(
                    knowledge_base_path=knowledge_base,
                    persist_directory=persist_directory,
                    embeddings=self.embeddings,
                )
                ingestion_result = ingester.ingest()
            finally:
                ingester.close()

            retriever = DeterministicRAGRetriever(
                knowledge_base_path=knowledge_base,
                persist_directory=persist_directory,
                top_k=2,
                embeddings=self.embeddings,
            )
            pipeline = ScenarioPipeline(rag_retriever=retriever)

            try:
                result = pipeline.run(self.payload)
            finally:
                retriever.close()

            self.assertEqual(ingestion_result.stored_chunks, 1)
            self.assertIsNotNone(result.rag_retrieval_result)
            assert result.rag_retrieval_result is not None
            self.assertEqual(result.rag_retrieval_result.indexed_chunks, 1)
            self.assertGreaterEqual(len(result.rag_retrieval_result.retrieved_documents), 1)
            self.assertEqual(result.rag_retrieval_result.always_included_documents, [])
            self.assertTrue(result.rag_retrieval_result.runtime_available)
            self.assertIn("query", result.to_dict()["rag_retrieval_result"])

    def test_chunked_framework_json_is_split_and_retains_structured_fields(self) -> None:
        ingester = KnowledgeBaseIngester.__new__(KnowledgeBaseIngester)
        payload = {
            "source": "Virtue Ethics - Philosophical Foundations and AV Application",
            "compiled_from": [
                "Aristotle - Nicomachean Ethics",
                "Geisslinger et al. (2021) - Autonomous Driving Ethics",
            ],
            "chunks": [
                {
                    "id": "ve_phronesis",
                    "type": "foundational_principle",
                    "title": "Phronesis",
                    "text": "Practical wisdom is context-sensitive judgment.",
                    "tags": ["virtue", "phronesis"],
                    "av_implication": "LLM reasoning should surface the morally salient features.",
                },
                {
                    "id": "ve_application",
                    "type": "av_application",
                    "title": "Virtue Ethics in AV Practice",
                    "text": "Virtue ethics helps with contextual calibration.",
                    "formula": "J(u) = w_B * J_B(u)",
                    "use_when": ["school_zone", "emergency_vehicle"],
                },
            ],
        }

        documents = ingester._json_documents(
            Path("knowledge_base/ethical_frameworks/virtue_ethics.json"),
            "ethical_frameworks/virtue_ethics.json",
            "ethical_frameworks",
            json.dumps(payload),
        )

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["metadata"]["record_id"], "ve_phronesis")
        self.assertEqual(documents[0]["metadata"]["record_type"], "foundational_principle")
        self.assertEqual(
            documents[0]["metadata"]["source_document"],
            "Virtue Ethics - Philosophical Foundations and AV Application",
        )
        self.assertIn("Source: Virtue Ethics - Philosophical Foundations and AV Application", documents[0]["text"])
        self.assertIn("Compiled from: Aristotle - Nicomachean Ethics", documents[0]["text"])
        self.assertIn(
            "AV implication: LLM reasoning should surface the morally salient features.",
            documents[0]["text"],
        )
        self.assertIn("Tags: virtue, phronesis", documents[0]["text"])
        self.assertIn("Formula: J(u) = w_B * J_B(u)", documents[1]["text"])
        self.assertIn("Use when: school_zone, emergency_vehicle", documents[1]["text"])

    def test_chunked_framework_json_is_retrievable_as_multiple_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_base = root / "knowledge_base"
            persist_directory = root / "vector_store"
            (knowledge_base / "ethical_frameworks").mkdir(parents=True)

            framework_payload = {
                "source": "Virtue Ethics - Philosophical Foundations and AV Application",
                "compiled_from": ["Aristotle - Nicomachean Ethics"],
                "chunks": [
                    {
                        "id": "ve_phronesis",
                        "type": "foundational_principle",
                        "title": "Phronesis",
                        "text": "Practical wisdom matters in school zones with hidden pedestrians.",
                        "av_implication": "Context-sensitive judgment should guide retrieval and reasoning.",
                    },
                    {
                        "id": "ve_role_morality",
                        "type": "av_application",
                        "title": "Role Morality",
                        "text": "Emergency vehicles have special obligations in road ethics.",
                        "use_when": ["ambulance_scenario", "school_zone"],
                    },
                ],
            }
            (knowledge_base / "ethical_frameworks" / "virtue_ethics.json").write_text(
                json.dumps(framework_payload, indent=2),
                encoding="utf-8",
            )

            try:
                ingester = KnowledgeBaseIngester(
                    knowledge_base_path=knowledge_base,
                    persist_directory=persist_directory,
                    embeddings=self.embeddings,
                )
                ingestion_result = ingester.ingest()
            finally:
                ingester.close()

            retriever = DeterministicRAGRetriever(
                knowledge_base_path=knowledge_base,
                persist_directory=persist_directory,
                top_k=2,
                embeddings=self.embeddings,
            )

            try:
                result = retriever.retrieve(self.parser.parse(self.payload).scenario)
            finally:
                retriever.close()

            self.assertEqual(ingestion_result.ingested_files, 1)
            self.assertEqual(ingestion_result.source_documents, 1)
            self.assertEqual(ingestion_result.stored_chunks, 1)
            self.assertEqual(len(result.retrieved_documents), 1)
            self.assertEqual(result.always_included_documents, [])
            self.assertEqual(
                result.retrieved_documents[0].title,
                "Virtue Ethics - Philosophical Foundations and AV Application",
            )

    def test_retriever_reserves_ethical_framework_slots_in_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_base = root / "knowledge_base"
            persist_directory = root / "vector_store"
            (knowledge_base / "ethical_frameworks").mkdir(parents=True)
            (knowledge_base / "german_ethics_commission").mkdir()

            framework_files = {
                "deontology.json": {
                    "source": "Deontology",
                    "chunks": [
                        {
                            "id": "deontology_core",
                            "title": "Deontology Core",
                            "text": "Deontology imposes hard constraints and rejects treating people merely as means.",
                        }
                    ],
                },
                "utilitarianism.json": {
                    "source": "Utilitarianism",
                    "chunks": [
                        {
                            "id": "utilitarianism_core",
                            "title": "Utilitarianism Core",
                            "text": "Utilitarianism minimises total expected harm across all affected parties.",
                        }
                    ],
                },
                "ethics_of_risk.json": {
                    "source": "Ethics of Risk",
                    "chunks": [
                        {
                            "id": "ethics_of_risk_core",
                            "title": "Ethics of Risk Core",
                            "text": "Ethics of risk compares probability distributions and vulnerability-sensitive risk.",
                        }
                    ],
                },
                "virtue_ethics.json": {
                    "source": "Virtue Ethics",
                    "chunks": [
                        {
                            "id": "virtue_ethics_core",
                            "title": "Virtue Ethics Core",
                            "text": "Virtue ethics values context-sensitive judgment and practical wisdom.",
                        }
                    ],
                },
            }
            for name, payload in framework_files.items():
                (knowledge_base / "ethical_frameworks" / name).write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )

            dense_general_text = (
                "autonomous vehicle ethics german ethics commission automated driving guidelines "
                "road_type residential weather clear time_of_day daytime traffic_density low "
                "collision_unavoidable true obstacles child_pedestrian parked_vehicle "
                "trajectories crossing stationary vulnerability_classes high low "
                "occluded_zones left sidewalk available_actions brake_straight swerve_left "
                "brake_swerve_left"
            )
            for index in range(1, 5):
                (
                    knowledge_base
                    / "german_ethics_commission"
                    / f"guideline_{index:02d}.md"
                ).write_text(
                    f"# Guideline {index}\n{dense_general_text}",
                    encoding="utf-8",
                )

            try:
                ingester = KnowledgeBaseIngester(
                    knowledge_base_path=knowledge_base,
                    persist_directory=persist_directory,
                    embeddings=self.embeddings,
                )
                ingester.ingest()
            finally:
                ingester.close()

            retriever = DeterministicRAGRetriever(
                knowledge_base_path=knowledge_base,
                persist_directory=persist_directory,
                top_k=6,
                embeddings=self.embeddings,
            )

            try:
                result = retriever.retrieve(self.parser.parse(self.payload).scenario)
            finally:
                retriever.close()

            framework_documents = [
                document for document in result.retrieved_documents if document.category == "ethical_frameworks"
            ]
            self.assertEqual(len(result.retrieved_documents), 6)
            self.assertEqual(len(framework_documents), 2)
            self.assertEqual(result.always_included_documents, [])
            self.assertTrue(
                {Path(document.path).name for document in framework_documents}.issubset(set(framework_files))
            )

    def test_retriever_returns_full_framework_files_even_when_runtime_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            knowledge_base = root / "knowledge_base"
            framework_directory = knowledge_base / "ethical_frameworks"
            framework_directory.mkdir(parents=True)

            framework_files = {
                "deontology.json": {"source": "Deontology", "chunks": [{"id": "d1", "text": "Deontology text"}]},
                "ethics_of_risk.json": {
                    "source": "Ethics of Risk",
                    "chunks": [{"id": "r1", "text": "Ethics of risk text"}],
                },
                "framework_comparison.json": {
                    "source": "Framework Comparison",
                    "chunks": [{"id": "f1", "text": "Framework comparison text"}],
                },
                "utilitarianism.json": {
                    "source": "Utilitarianism",
                    "chunks": [{"id": "u1", "text": "Utilitarianism text"}],
                },
                "virtue_ethics.json": {
                    "source": "Virtue Ethics",
                    "chunks": [{"id": "v1", "text": "Virtue ethics text"}],
                },
            }
            for name, payload in framework_files.items():
                (framework_directory / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

            retriever = DeterministicRAGRetriever.__new__(DeterministicRAGRetriever)
            retriever.knowledge_base_path = knowledge_base.resolve()
            retriever.persist_directory = (knowledge_base / ".chroma").resolve()
            retriever.collection_name = DeterministicRAGRetriever.DEFAULT_COLLECTION_NAME
            retriever.top_k = 6
            retriever.framework_top_k = 4
            retriever.embedding_model = DeterministicRAGRetriever.DEFAULT_EMBEDDING_MODEL
            retriever.embeddings = None
            retriever._runtime_error = RuntimeError("simulated unavailable runtime")
            retriever.vector_store = None

            result = retriever.retrieve(self.parser.parse(self.payload).scenario)

            self.assertFalse(result.runtime_available)
            self.assertEqual(result.runtime_error, "simulated unavailable runtime")
            self.assertEqual(result.indexed_chunks, 0)
            self.assertEqual(result.retrieved_documents, [])
            self.assertEqual(len(result.always_included_documents), 2)
            self.assertTrue(
                {Path(document.path).name for document in result.always_included_documents}.issubset(set(framework_files))
            )
            self.assertTrue(
                any('"source": "Utilitarianism"' in document.content for document in result.always_included_documents)
            )

    def test_package_import_does_not_require_rag_runtime_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            blocker_dir = Path(temp_dir)
            for module_name in ("chromadb", "langchain_chroma", "langchain_openai"):
                (blocker_dir / f"{module_name}.py").write_text(
                    "raise ImportError('blocked for import safety test')\n",
                    encoding="utf-8",
                )

            env = dict(os.environ)
            project_root = str(Path(__file__).resolve().parents[1])
            python_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                [str(blocker_dir), project_root, python_path] if python_path else [str(blocker_dir), project_root]
            )

            result = subprocess.run(
                [sys.executable, "-c", "import thesis"],
                capture_output=True,
                text=True,
                env=env,
                cwd=project_root,
            )

            self.assertEqual(result.returncode, 0, result.stderr)

    def test_text_splitter_prefers_tiktoken_when_available(self) -> None:
        class DummySplitter:
            def __init__(self, **kwargs: object) -> None:
                self.mode = "character"
                self.kwargs = kwargs

            @classmethod
            def from_tiktoken_encoder(cls, **kwargs: object) -> "DummySplitter":
                instance = cls.__new__(cls)
                instance.mode = "tokens"
                instance.kwargs = kwargs
                return instance

        ingester = KnowledgeBaseIngester.__new__(KnowledgeBaseIngester)
        ingester.chunk_size = 128
        ingester.chunk_overlap = 24
        ingester.embedding_model = "text-embedding-3-small"

        with patch("thesis.rag.ingestion.importlib.import_module", return_value=object()):
            splitter = ingester._build_text_splitter(DummySplitter)

        self.assertEqual(splitter.mode, "tokens")
        self.assertEqual(splitter.kwargs["model_name"], "text-embedding-3-small")
        self.assertEqual(splitter.kwargs["chunk_size"], 128)
        self.assertEqual(splitter.kwargs["chunk_overlap"], 24)

    def test_text_splitter_falls_back_to_character_splitter_without_tiktoken(self) -> None:
        class DummySplitter:
            def __init__(self, **kwargs: object) -> None:
                self.mode = "character"
                self.kwargs = kwargs

            @classmethod
            def from_tiktoken_encoder(cls, **kwargs: object) -> "DummySplitter":
                instance = cls.__new__(cls)
                instance.mode = "tokens"
                instance.kwargs = kwargs
                return instance

        ingester = KnowledgeBaseIngester.__new__(KnowledgeBaseIngester)
        ingester.chunk_size = 96
        ingester.chunk_overlap = 12
        ingester.embedding_model = "text-embedding-3-small"

        with patch(
            "thesis.rag.ingestion.importlib.import_module",
            side_effect=ImportError("tiktoken unavailable"),
        ):
            splitter = ingester._build_text_splitter(DummySplitter)

        self.assertEqual(splitter.mode, "character")
        self.assertEqual(splitter.kwargs["chunk_size"], 96)
        self.assertEqual(splitter.kwargs["chunk_overlap"], 12)


if __name__ == "__main__":
    unittest.main()
