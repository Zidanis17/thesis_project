from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from thesis import EthicalReasoningResult
from thesis.api import ShowcaseRuntime, create_app
from thesis.rag import RAGRetrievalResult, RetrievedDocument


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
            }
        ],
        "sensor_confidence": {
            "lidar": 0.97,
            "camera": 0.91,
            "radar": 0.95,
            "overall_scene_confidence": 0.93,
            "occluded_zones": ["left sidewalk"],
        },
        "available_actions": [
            "brake_straight",
            "swerve_left",
            "swerve_right",
            "brake_swerve_left",
        ],
        "collision_unavoidable": True,
    }


def build_sample_text() -> str:
    return (
        "An autonomous vehicle weighing 1800 kg is traveling at 60 km/h in the center lane on a "
        "residential road with a 50 km/h speed limit. It is braking at 2.1 m/s2 and has a braking "
        "distance of 42.5 m. Weather is clear, visibility is 120 m, it is daytime, and traffic is low. "
        "A child pedestrian is crossing 10.2 m ahead with time to impact 0.61 s. "
        "Lidar confidence is 0.97, camera 0.91, radar 0.95, and overall scene confidence is 0.93. "
        "The left sidewalk is occluded. Available actions are brake straight, swerve left, "
        "swerve right, and brake and swerve left. Collision is unavoidable."
    )


class FakeRetriever:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.knowledge_base_path = (Path(__file__).resolve().parents[1] / "knowledge_base").resolve()
        self.vector_store = object() if available else None
        self._runtime_error = None if available else RuntimeError("RAG disabled in test")

    def retrieve(self, *_args: object, **_kwargs: object) -> RAGRetrievalResult:
        if self.available:
            retrieved_documents = [
                RetrievedDocument(
                    document_id="EF-02",
                    title="Deontological Safety",
                    category="ethical_frameworks",
                    path="ethical_frameworks/EF-02_deontological.json",
                    score=0.93,
                    excerpt="Rule-based safety and feasible action filtering.",
                    full_content='{"framework_id":"EF-02","name":"Deontological Safety"}',
                )
            ]
            return RAGRetrievalResult(
                query="test-query",
                retrieved_documents=retrieved_documents,
                always_included_documents=[],
                knowledge_base_path=str(self.knowledge_base_path),
                indexed_chunks=12,
                runtime_available=True,
                runtime_error=None,
            )
        return RAGRetrievalResult(
            query="test-query",
            retrieved_documents=[],
            always_included_documents=[],
            knowledge_base_path=str(self.knowledge_base_path),
            indexed_chunks=0,
            runtime_available=False,
            runtime_error="RAG disabled in test",
        )


class FakeReasoner:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.client = object() if available else None
        self._runtime_error = None if available else RuntimeError("Reasoning disabled in test")
        self.model_name = "fake-ethical-model"

    def reason(self, _parser_result, math_result, _rag_result) -> EthicalReasoningResult:
        if self.available:
            return EthicalReasoningResult(
                model_name=self.model_name,
                system_prompt="test prompt",
                runtime_available=True,
                recommended_action="brake_straight",
                dominant_framework="EF-02",
                contributing_frameworks=["EF-01", "EF-03"],
                weights={"bayesian": 0.4, "equality": 0.2, "maximin": 0.4},
                weights_reasoning="Favor the feasible action that protects the vulnerable road user.",
                risk_scores_per_action=math_result.risk_score_matrix,
                rationale="EF-02 constrains the feasible set and EF-03 reinforces protection of the child.",
                confidence=0.91,
                violated_constraints=[],
            )
        return EthicalReasoningResult(
            model_name=self.model_name,
            system_prompt="test prompt",
            runtime_available=False,
            recommended_action=None,
            dominant_framework=None,
            contributing_frameworks=[],
            weights={},
            weights_reasoning="",
            risk_scores_per_action=math_result.risk_score_matrix,
            rationale="",
            confidence=None,
            violated_constraints=[],
            runtime_error="Reasoning disabled in test",
        )


def build_client(*, rag_available: bool = True, reasoning_available: bool = True) -> TestClient:
    runtime = ShowcaseRuntime(
        rag_retriever=FakeRetriever(available=rag_available),
        reasoning_llm=FakeReasoner(available=reasoning_available),
    )
    return TestClient(create_app(runtime=runtime))


def test_health_reports_backend_paths_after_repo_split() -> None:
    client = build_client()

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert Path(payload["knowledge_base_path"]).exists()
    assert Path(payload["knowledge_base_path"]).name == "knowledge_base"
    assert payload["rag"]["runtime_available"] is True
    assert payload["reasoning"]["runtime_available"] is True


def test_examples_endpoint_returns_seed_examples() -> None:
    client = build_client()

    response = client.get("/api/v1/examples")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["examples"]) >= 3
    assert {item["mode"] for item in payload["examples"]} == {"json", "text"}


def test_json_request_returns_replay_and_artifacts() -> None:
    client = build_client()

    response = client.post(
        "/api/v1/scenario/run",
        json={"input": build_sample_payload(), "input_mode_hint": "json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["deterministic_best_action"] == "brake_straight"
    assert payload["summary"]["recommended_action"] == "brake_straight"
    assert payload["artifacts"]["parser_result"]["_meta"]["input_mode"] == "structured_json"
    assert [stage["stage_id"] for stage in payload["replay"]] == [
        "input",
        "parser",
        "math",
        "rag",
        "reasoning",
        "complete",
    ]
    assert "$.parser_result" in payload["replay"][1]["highlight_paths"]


def test_natural_language_request_returns_parser_derived_json() -> None:
    client = build_client()

    response = client.post(
        "/api/v1/scenario/run",
        json={"input": build_sample_text(), "input_mode_hint": "text"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifacts"]["parser_result"]["_meta"]["input_mode"] == "natural_language"
    assert payload["artifacts"]["parser_result"]["environment"]["road_type"] == "residential"


def test_parser_validation_error_returns_400_with_partial_replay() -> None:
    client = build_client()

    response = client.post(
        "/api/v1/scenario/run",
        json={
            "input": {"environment": {"road_type": "residential"}, "obstacles": [{"type": "vehicle", "distance_m": 10}]},
            "input_mode_hint": "json",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "scenario_parse_error"
    assert [stage["stage_id"] for stage in payload["replay"]] == ["input", "parser"]
    assert payload["replay"][-1]["status"] == "error"


def test_rag_unavailable_is_warning_not_failure() -> None:
    client = build_client(rag_available=False)

    response = client.post(
        "/api/v1/scenario/run",
        json={"input": build_sample_payload(), "input_mode_hint": "json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["replay"][3]["stage_id"] == "rag"
    assert payload["replay"][3]["status"] == "warning"
    assert payload["replay"][-1]["status"] == "success"


def test_reasoning_unavailable_is_warning_not_failure() -> None:
    client = build_client(reasoning_available=False)

    response = client.post(
        "/api/v1/scenario/run",
        json={"input": build_sample_payload(), "input_mode_hint": "json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["replay"][4]["stage_id"] == "reasoning"
    assert payload["replay"][4]["status"] == "warning"
    assert payload["summary"]["reasoning_runtime_available"] is False
    assert payload["replay"][-1]["status"] == "success"
