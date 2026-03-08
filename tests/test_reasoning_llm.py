import json
import unittest

from thesis import (
    ETHICAL_REASONING_SYSTEM_PROMPT,
    DeterministicMathematicalLayer,
    DeterministicScenarioParser,
    EthicalReasoningLLM,
    EthicalReasoningResult,
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


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def invoke(self, _: object) -> object:
        class Response:
            def __init__(self, content: str) -> None:
                self.content = content

        return Response(json.dumps(self.payload))


class FakeReasoningEngine:
    def reason(self, *_: object) -> EthicalReasoningResult:
        return EthicalReasoningResult(
            model_name="fake-model",
            system_prompt=ETHICAL_REASONING_SYSTEM_PROMPT,
            runtime_available=True,
            recommended_action="brake_straight",
            dominant_framework="deontology",
            contributing_frameworks=["ethics_of_risk"],
            weights={"bayesian": 0.5, "equality": 0.2, "maximin": 0.3},
            weights_reasoning="Maximin increases because a child pedestrian is exposed.",
            risk_scores_per_action={"brake_straight": {"ego_vehicle": 1.0}},
            rationale="Brake straight minimizes harm while avoiding an active lateral steer into uncertainty.",
            confidence=0.91,
            violated_constraints=[],
        )


class ReasoningLLMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DeterministicScenarioParser()
        self.math_layer = DeterministicMathematicalLayer()
        self.payload = build_sample_payload()
        self.parser_result = self.parser.parse(self.payload)
        self.math_result = self.math_layer.analyze(self.parser_result.scenario)

    def test_reasoning_engine_uses_math_layer_for_risk_matrix_and_constraints(self) -> None:
        engine = EthicalReasoningLLM.__new__(EthicalReasoningLLM)
        engine.model_name = "gpt-4-nano"
        engine.temperature = 0.0
        engine.system_prompt = ETHICAL_REASONING_SYSTEM_PROMPT
        engine._runtime_error = None
        engine.client = FakeClient(
            {
                "recommended_action": "swerve_left",
                "dominant_framework": "ethics_of_risk",
                "contributing_frameworks": ["deontology", "utilitarianism"],
                "weights": {
                    "bayesian": 2,
                    "equality": 1,
                    "maximin": 1,
                },
                "weights_reasoning": "Occlusion and a vulnerable child make maximin and rule-sensitive reasoning salient.",
                "risk_scores_per_action": {"tampered": {"value": 999}},
                "rationale": "The response selects the left swerve for test purposes.",
                "confidence": 0.82,
                "violated_constraints": [],
            }
        )

        result = engine.reason(self.parser_result, self.math_result)

        self.assertTrue(result.runtime_available)
        self.assertEqual(result.recommended_action, "swerve_left")
        self.assertEqual(result.dominant_framework, "deontology")
        self.assertEqual(result.weights, {"bayesian": 0.5, "equality": 0.25, "maximin": 0.25})
        self.assertEqual(result.risk_scores_per_action, self.math_result.risk_score_matrix)

        expected_flags = next(
            assessment.constraint_flags
            for assessment in self.math_result.action_assessments
            if assessment.action == "swerve_left"
        )
        self.assertEqual(result.violated_constraints, expected_flags)

    def test_pipeline_includes_reasoning_result_when_engine_is_supplied(self) -> None:
        pipeline = ScenarioPipeline(reasoning_llm=FakeReasoningEngine())

        result = pipeline.run(self.payload)

        self.assertIsNotNone(result.reasoning_result)
        assert result.reasoning_result is not None
        self.assertEqual(result.reasoning_result.recommended_action, "brake_straight")
        self.assertIn("reasoning_result", result.to_dict())


if __name__ == "__main__":
    unittest.main()
