import os
import unittest
from unittest.mock import patch

from thesis import (
    AgenticEthicalController,
    DeterministicMathematicalLayer,
    DeterministicRAGRetriever,
    DeterministicScenarioParser,
    EthicalReasoningLLM,
    EthicalReasoningResult,
    ScenarioPipeline,
)


def build_payload() -> dict:
    return {
        "ego_vehicle": {
            "speed_kmh": 45,
            "acceleration_ms2": -1.0,
            "heading_deg": 0,
            "lane_position": "center",
            "braking_distance_m": 24,
            "mass_kg": 1800,
            "passenger_at_risk": False,
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
                "distance_m": 16,
                "relative_speed_kmh": 45,
                "time_to_impact_s": 1.2,
                "trajectory": "crossing",
                "vulnerability_class": "high",
                "mass_kg": 30,
                "responsible_for_risk": False,
            }
        ],
        "sensor_confidence": {
            "lidar": 0.96,
            "camera": 0.94,
            "radar": 0.95,
            "overall_scene_confidence": 0.95,
            "occluded_zones": [],
        },
        "available_actions": ["brake_straight", "swerve_left"],
        "collision_unavoidable": False,
    }


class FakeReasoner:
    def reason(self, parser_result, math_result, rag_result=None, agentic_assessment=None):
        return EthicalReasoningResult(
            model_name="fake",
            system_prompt="fake",
            runtime_available=True,
            dominant_framework="EF-02",
            contributing_frameworks=["EF-01"],
            weights={"bayesian": 0.4, "equality": 0.2, "maximin": 0.4},
            weights_reasoning="Test reasoning.",
            risk_scores_per_action=math_result.risk_score_matrix if math_result is not None else {},
            rationale="Test rationale.",
            confidence=0.8,
            violated_constraints=[],
        )


class AgenticControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DeterministicScenarioParser()
        self.math_layer = DeterministicMathematicalLayer()
        self.controller = AgenticEthicalController()

    def _parse_and_analyze(self, payload: dict):
        parser_result = self.parser.parse(payload)
        return parser_result, self.math_layer.analyze(parser_result.scenario)

    def test_avoidable_interpretable_scenario_prioritizes_ef02(self) -> None:
        parser_result, math_result = self._parse_and_analyze(build_payload())

        assessment = self.controller.assess(parser_result, math_result)

        self.assertEqual(assessment.retrieval_intent.scenario_class, "avoidable_rule_based_case")
        self.assertEqual(assessment.candidate_frameworks[0], "EF-02")
        self.assertNotIn("EF-04", assessment.candidate_frameworks)

    def test_unavoidable_vru_scenario_includes_ef03_and_supporting_ef04(self) -> None:
        payload = build_payload()
        payload["collision_unavoidable"] = True
        parser_result, math_result = self._parse_and_analyze(payload)

        assessment = self.controller.assess(parser_result, math_result)

        self.assertEqual(assessment.retrieval_intent.scenario_class, "unavoidable_vru_dilemma")
        self.assertIn("EF-03", assessment.candidate_frameworks)
        self.assertIn("EF-04", assessment.retrieval_intent.required_frameworks)
        self.assertNotIn("EF-04", assessment.candidate_frameworks)

    def test_unknown_object_scenario_prioritizes_ef06(self) -> None:
        payload = build_payload()
        payload["obstacles"][0]["type"] = "unknown_object"
        parser_result, math_result = self._parse_and_analyze(payload)

        assessment = self.controller.assess(parser_result, math_result)

        self.assertEqual(assessment.retrieval_intent.scenario_class, "epistemic_uncertainty_case")
        self.assertEqual(assessment.candidate_frameworks[0], "EF-06")

    def test_uninterpretable_scene_prioritizes_ef06(self) -> None:
        payload = build_payload()
        payload["sensor_confidence"] = {
            "lidar": 0.2,
            "camera": 0.2,
            "radar": 0.2,
            "overall_scene_confidence": 0.2,
            "occluded_zones": [],
        }
        parser_result, math_result = self._parse_and_analyze(payload)

        assessment = self.controller.assess(parser_result, math_result)

        self.assertFalse(assessment.scene_interpretable)
        self.assertEqual(assessment.retrieval_intent.scenario_class, "epistemic_uncertainty_case")
        self.assertEqual(assessment.candidate_frameworks[0], "EF-06")

    def test_validate_reasoning_result_rejects_ef04_as_dominant(self) -> None:
        parser_result, math_result = self._parse_and_analyze(build_payload())
        reasoning_result = EthicalReasoningResult(
            model_name="fake",
            system_prompt="fake",
            runtime_available=True,
            dominant_framework="EF-04",
            risk_scores_per_action=math_result.risk_score_matrix,
        )

        validation = self.controller.validate_reasoning_result(parser_result, math_result, reasoning_result)

        self.assertFalse(validation.is_valid)
        self.assertTrue(any("EF-04" in error for error in validation.errors))

    def test_validate_reasoning_result_rejects_ef02_when_collision_unavoidable(self) -> None:
        payload = build_payload()
        payload["collision_unavoidable"] = True
        parser_result, math_result = self._parse_and_analyze(payload)
        reasoning_result = EthicalReasoningResult(
            model_name="fake",
            system_prompt="fake",
            runtime_available=True,
            dominant_framework="EF-02",
            risk_scores_per_action=math_result.risk_score_matrix,
        )

        validation = self.controller.validate_reasoning_result(parser_result, math_result, reasoning_result)

        self.assertFalse(validation.is_valid)
        self.assertTrue(any("EF-02" in error for error in validation.errors))

    def test_validate_reasoning_result_flags_ef05_without_tradeoff_evidence(self) -> None:
        parser_result, math_result = self._parse_and_analyze(build_payload())
        reasoning_result = EthicalReasoningResult(
            model_name="fake",
            system_prompt="fake",
            runtime_available=True,
            dominant_framework="EF-05",
            risk_scores_per_action=math_result.risk_score_matrix,
        )

        validation = self.controller.validate_reasoning_result(parser_result, math_result, reasoning_result)

        self.assertFalse(validation.is_valid)
        self.assertTrue(validation.errors or validation.warnings)

    def test_controller_does_not_require_openai_api_key(self) -> None:
        parser_result, math_result = self._parse_and_analyze(build_payload())

        with patch.dict(os.environ, {}, clear=True):
            assessment = AgenticEthicalController().assess(parser_result, math_result)

        self.assertEqual(assessment.retrieval_intent.scenario_class, "avoidable_rule_based_case")

    def test_controller_does_not_instantiate_rag_or_llm_clients(self) -> None:
        parser_result, math_result = self._parse_and_analyze(build_payload())

        with patch.object(DeterministicRAGRetriever, "__init__", side_effect=AssertionError):
            with patch.object(EthicalReasoningLLM, "__init__", side_effect=AssertionError):
                assessment = AgenticEthicalController().assess(parser_result, math_result)

        self.assertEqual(assessment.candidate_frameworks[0], "EF-02")

    def test_pipeline_returns_agentic_assessment(self) -> None:
        pipeline = ScenarioPipeline(reasoning_llm=None, auto_rag=False)

        result = pipeline.run(build_payload())

        self.assertIsNotNone(result.agentic_assessment)
        assert result.agentic_assessment is not None
        self.assertEqual(
            result.agentic_assessment.retrieval_intent.scenario_class,
            "avoidable_rule_based_case",
        )
        self.assertIn("agentic_assessment", result.to_dict())

    def test_pipeline_includes_agentic_validation_when_reasoning_result_exists(self) -> None:
        pipeline = ScenarioPipeline(reasoning_llm=FakeReasoner(), auto_rag=False)

        result = pipeline.run(build_payload())

        self.assertIsNotNone(result.reasoning_result)
        self.assertIsNotNone(result.agentic_validation_result)
        assert result.agentic_validation_result is not None
        self.assertTrue(result.agentic_validation_result.is_valid)
        self.assertIn("agentic_validation_result", result.to_dict())

    def test_rag_query_building_preserves_backward_compatibility_without_intent(self) -> None:
        scenario = self.parser.parse(build_payload()).scenario
        retriever = DeterministicRAGRetriever.__new__(DeterministicRAGRetriever)

        query = retriever._build_query(scenario, None, None)

        self.assertIn("Autonomous vehicle ethical decision-making", query)
        self.assertNotIn("Agentic retrieval intent", query)


if __name__ == "__main__":
    unittest.main()
