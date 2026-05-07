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
    def reason(self, *_: object, **__: object) -> EthicalReasoningResult:
        return EthicalReasoningResult(
            model_name="fake-model",
            system_prompt=ETHICAL_REASONING_SYSTEM_PROMPT,
            runtime_available=True,
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
        self.assertEqual(result.dominant_framework, "EF-03")
        self.assertEqual(result.weights, {"bayesian": 0.5, "equality": 0.25, "maximin": 0.25})
        self.assertEqual(result.risk_scores_per_action, self.math_result.risk_score_matrix)
        self.assertFalse(hasattr(result, "recommended_action"))
        self.assertEqual(result.violated_constraints, [])

    def test_unavoidable_collision_rejects_ef02_as_dominant_framework(self) -> None:
        engine = EthicalReasoningLLM.__new__(EthicalReasoningLLM)
        engine.model_name = "gpt-4-nano"
        engine.temperature = 0.0
        engine.system_prompt = ETHICAL_REASONING_SYSTEM_PROMPT
        engine._runtime_error = None
        engine.client = FakeClient(
            {
                "dominant_framework": "EF-02",
                "contributing_frameworks": ["EF-01", "EF-03"],
                "weights": {
                    "bayesian": 0.4,
                    "equality": 0.2,
                    "maximin": 0.4,
                },
                "weights_reasoning": "The model attempted to over-privilege deontological constraints.",
                "risk_scores_per_action": {"tampered": {"value": 999}},
                "rationale": "The response incorrectly prefers EF-02 for an unavoidable collision.",
                "confidence": 0.75,
                "violated_constraints": [],
            }
        )

        result = engine.reason(self.parser_result, self.math_result)

        self.assertEqual(result.dominant_framework, "EF-03")

    def test_ef05_requires_explicit_passenger_valence_signal(self) -> None:
        engine = EthicalReasoningLLM.__new__(EthicalReasoningLLM)
        engine.model_name = "gpt-4-nano"
        engine.temperature = 0.0
        engine.system_prompt = ETHICAL_REASONING_SYSTEM_PROMPT
        engine._runtime_error = None
        engine.client = FakeClient(
            {
                "dominant_framework": "EF-05",
                "contributing_frameworks": ["EF-01", "EF-03"],
                "weights": {
                    "bayesian": 0.3,
                    "equality": 0.2,
                    "maximin": 0.5,
                },
                "weights_reasoning": "The model tried to infer a passenger dilemma without evidence.",
                "risk_scores_per_action": {"tampered": {"value": 999}},
                "rationale": "The response incorrectly elevates EF-05 without a passenger-valence signal.",
                "confidence": 0.71,
                "violated_constraints": [],
            }
        )

        result = engine.reason(self.parser_result, self.math_result)

        self.assertEqual(result.dominant_framework, "EF-03")

    def test_no_math_ef05_can_use_explicit_scenario_passenger_vru_tradeoff(self) -> None:
        payload = build_sample_payload()
        payload["obstacles"][1]["type"] = "concrete_barrier"
        payload["obstacles"][1]["trajectory"] = "left_fixed_barrier"
        payload["available_actions"] = ["brake_straight", "swerve_left", "swerve_right"]
        parser_result = self.parser.parse(payload)
        engine = EthicalReasoningLLM.__new__(EthicalReasoningLLM)
        engine.model_name = "gpt-4-nano"
        engine.temperature = 0.0
        engine.system_prompt = ETHICAL_REASONING_SYSTEM_PROMPT
        engine._runtime_error = None
        engine.client = FakeClient(
            {
                "dominant_framework": "EF-05",
                "contributing_frameworks": ["EF-01", "EF-03"],
                "weights": {
                    "bayesian": 0.4,
                    "equality": 0.3,
                    "maximin": 0.3,
                },
                "weights_reasoning": "The scenario explicitly pits vehicle occupant risk against a vulnerable road user.",
                "risk_scores_per_action": {},
                "rationale": "EF-05 governs the passenger-vs-VRU trade-off without deterministic math.",
                "confidence": 0.78,
                "violated_constraints": [],
            }
        )

        result = engine.reason(parser_result, None, None)

        self.assertEqual(result.dominant_framework, "EF-05")
        self.assertEqual(result.risk_scores_per_action, {})

    def test_structured_meta_does_not_enable_ef05(self) -> None:
        payload = {
            **build_sample_payload(),
            "_meta": {
                "warnings": [
                    "passenger-protection and third-party protection create explicit valence trade-off",
                    "distinct stakeholder categories make social valence central",
                ]
            },
        }
        parser_result = self.parser.parse(payload)
        math_result = self.math_layer.analyze(parser_result.scenario)

        engine = EthicalReasoningLLM.__new__(EthicalReasoningLLM)
        engine.model_name = "gpt-4-nano"
        engine.temperature = 0.0
        engine.system_prompt = ETHICAL_REASONING_SYSTEM_PROMPT
        engine._runtime_error = None
        engine.client = FakeClient(
            {
                "dominant_framework": "EF-05",
                "contributing_frameworks": ["EF-01", "EF-03"],
                "weights": {
                    "bayesian": 0.4,
                    "equality": 0.3,
                    "maximin": 0.3,
                },
                "weights_reasoning": "The scenario metadata explicitly marks a passenger-vs-third-party dilemma.",
                "risk_scores_per_action": {"tampered": {"value": 999}},
                "rationale": "The response selects EF-05 when the structured scenario carries a valence cue.",
                "confidence": 0.86,
                "violated_constraints": [],
            }
        )

        result = engine.reason(parser_result, math_result)

        self.assertEqual(result.dominant_framework, "EF-03")

    def test_user_prompt_excludes_parser_metadata(self) -> None:
        engine = EthicalReasoningLLM.__new__(EthicalReasoningLLM)
        prompt = engine._build_user_prompt(self.parser_result, self.math_result, None)

        self.assertNotIn("_meta", prompt)
        self.assertNotIn("input_mode", prompt)
        self.assertNotIn("parser_warnings", prompt)

    def test_reasoning_defaults_and_prompt_match_thesis_configuration(self) -> None:
        self.assertEqual(EthicalReasoningLLM.DEFAULT_MODEL_NAME, "gpt-5.4-mini")
        self.assertIn("ego_vehicle, passenger, or occupant risk", ETHICAL_REASONING_SYSTEM_PROMPT)
        self.assertNotIn('"ego:passenger" plus', ETHICAL_REASONING_SYSTEM_PROMPT)

    def test_pipeline_includes_reasoning_result_when_engine_is_supplied(self) -> None:
        pipeline = ScenarioPipeline(reasoning_llm=FakeReasoningEngine(), auto_rag=False)

        result = pipeline.run(self.payload)

        self.assertIsNotNone(result.reasoning_result)
        assert result.reasoning_result is not None
        self.assertEqual(result.reasoning_result.dominant_framework, "deontology")
        self.assertIn("reasoning_result", result.to_dict())


if __name__ == "__main__":
    unittest.main()
