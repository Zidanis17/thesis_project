import unittest

from thesis import (
    DeterministicMathematicalLayer,
    DeterministicScenarioParser,
    DeterministicScenarioPipeline,
    LLMScenarioParserAgentResult,
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


class FakeParserAgent:
    def extract(self, _text: str) -> LLMScenarioParserAgentResult:
        return LLMScenarioParserAgentResult(
            payload=build_sample_payload(),
            model_name="fake-parser-model",
            provider="fake",
            runtime_available=True,
            runtime_error=None,
        )


class MathematicalLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DeterministicScenarioParser()
        self.math_layer = DeterministicMathematicalLayer()

    def test_risk_matrix_contains_obstacles_occlusions_and_ego(self) -> None:
        scenario = self.parser.parse(build_sample_payload()).scenario

        result = self.math_layer.analyze(scenario)

        self.assertIn("speed_limit_exceeded", result.violated_rules)
        self.assertIn("ego_vehicle", result.risk_score_matrix["brake_straight"])
        self.assertIn("obj_01", result.risk_score_matrix["brake_straight"])
        self.assertIn("obj_02", result.risk_score_matrix["brake_straight"])
        self.assertIn("occlusion:left_sidewalk", result.risk_score_matrix["swerve_left"])

    def test_left_occlusion_penalizes_left_swerve_more_than_straight_braking(self) -> None:
        scenario = self.parser.parse(build_sample_payload()).scenario

        result = self.math_layer.analyze(scenario)

        left_swerve_occlusion = result.risk_score_matrix["swerve_left"]["occlusion:left_sidewalk"]
        brake_straight_occlusion = result.risk_score_matrix["brake_straight"]["occlusion:left_sidewalk"]

        self.assertGreater(left_swerve_occlusion, brake_straight_occlusion)
        self.assertLess(result.risk_score_matrix["brake_straight"]["obj_01"], result.risk_score_matrix["swerve_left"]["obj_01"])

    def test_non_responsible_crossing_stakeholder_triggers_constraint_flags(self) -> None:
        scenario = self.parser.parse(build_sample_payload()).scenario

        result = self.math_layer.analyze(scenario)
        flags_by_action = {
            assessment.action: assessment.constraint_flags for assessment in result.action_assessments
        }

        self.assertIn("potential_right_of_way_violation:obj_01", flags_by_action["swerve_left"])
        self.assertIn("steers_into_occluded_zone:left_sidewalk", flags_by_action["swerve_left"])

    def test_scenario_bank_canonical_values_and_maintain_lane_are_supported(self) -> None:
        payload = build_sample_payload()
        payload["environment"]["road_type"] = "urban_arterial"
        payload["environment"]["weather"] = "light_rain"
        payload["obstacles"][0]["trajectory"] = "same_lane_braking"
        payload["available_actions"] = ["maintain_lane", "brake_straight"]
        scenario = self.parser.parse(payload).scenario

        result = self.math_layer.analyze(scenario)

        self.assertIn("maintain_lane", result.risk_score_matrix)
        self.assertEqual(result.global_metrics["canonical_weather"], "rain")
        self.assertEqual(result.global_metrics["canonical_road_type"], "urban")

    def test_missing_obstacle_harm_fields_are_not_defaulted(self) -> None:
        payload = build_sample_payload()
        del payload["obstacles"][0]["trajectory"]
        del payload["obstacles"][0]["vulnerability_class"]
        del payload["obstacles"][0]["mass_kg"]
        payload["obstacles"] = [payload["obstacles"][0]]
        payload["sensor_confidence"]["occluded_zones"] = []
        scenario = self.parser.parse(payload).scenario

        result = self.math_layer.analyze(scenario)

        self.assertEqual(result.global_metrics["runtime_status"], "insufficient_data")
        skipped = result.global_metrics["skipped_obstacles"][0]["missing_fields"]
        self.assertIn("obstacles[0].trajectory", skipped)
        self.assertIn("obstacles[0].vulnerability_class", skipped)
        self.assertIn("obstacles[0].mass_kg", skipped)
        self.assertEqual(result.risk_score_matrix, {})

    def test_missing_optional_context_still_allows_partial_risk_analysis(self) -> None:
        payload = build_sample_payload()
        payload["environment"].pop("speed_limit_kmh")
        payload["sensor_confidence"].pop("camera")
        scenario = self.parser.parse(payload).scenario

        result = self.math_layer.analyze(scenario)

        self.assertEqual(result.global_metrics["runtime_status"], "partial_success")
        self.assertFalse(result.global_metrics["analysis_complete"])
        self.assertIn("environment.speed_limit_kmh", result.global_metrics["missing_optional_fields"])
        self.assertIn("sensor_confidence.camera", result.global_metrics["missing_optional_fields"])
        self.assertIn("brake_straight", result.risk_score_matrix)

    def test_vru_type_overrides_low_vulnerability_class_for_protection_mapping(self) -> None:
        self.assertFalse(self.math_layer._is_protected("low", "adult_pedestrian"))
        self.assertFalse(self.math_layer._is_protected("low", "motorcyclist"))
        self.assertFalse(self.math_layer._is_protected("high", "vehicle_sedan"))

    def test_pipeline_runs_from_natural_language_input(self) -> None:
        text = (
            "An autonomous vehicle weighing 1800 kg is traveling at 60 km/h in the center lane on a "
            "residential road with a 50 km/h speed limit. It is braking at 2.1 m/s2 and has a braking "
            "distance of 42.5 m. Weather is clear, visibility is 120 m, it is daytime, and traffic is low. "
            "A child pedestrian is crossing 10.2 m ahead with time to impact 0.61 s. "
            "A parked vehicle is 6.5 m ahead with time to impact 0.39 s. "
            "Lidar confidence is 0.97, camera 0.91, radar 0.95, and overall scene confidence is 0.93. "
            "The left sidewalk is occluded. Available actions are brake straight, swerve left, "
            "swerve right, and brake and swerve left. Collision is unavoidable."
        )
        pipeline = ScenarioPipeline(
            parser=DeterministicScenarioParser(llm_agent=FakeParserAgent()),
            auto_rag=False,
        )

        result = pipeline.run(text)

        self.assertEqual(result.parser_result.input_mode, "natural_language")
        self.assertEqual(result.mathematical_layer_result.best_action_by_total_risk, "brake_straight")
        self.assertIn(
            "occlusion:left_sidewalk",
            result.mathematical_layer_result.risk_score_matrix["swerve_left"],
        )

    def test_legacy_pipeline_alias_still_resolves(self) -> None:
        self.assertIs(DeterministicScenarioPipeline, ScenarioPipeline)


if __name__ == "__main__":
    unittest.main()
