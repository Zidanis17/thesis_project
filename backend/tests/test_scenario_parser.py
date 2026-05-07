import unittest

from thesis import (
    DeterministicScenarioParser,
    LLMScenarioParserAgent,
    LLMScenarioParserAgentResult,
    Scenario,
    ScenarioParseError,
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
    def __init__(self, *, payload: dict | None = None, error: str | None = None) -> None:
        self.payload = payload or {}
        self.error = error

    def extract(self, _text: str) -> LLMScenarioParserAgentResult:
        return LLMScenarioParserAgentResult(
            payload=self.payload if self.error is None else {},
            model_name="fake-parser-model",
            provider="fake",
            runtime_available=self.error is None,
            runtime_error=self.error,
        )


class FakeChatClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def invoke(self, _messages: list[tuple[str, str]]) -> str:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ScenarioParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DeterministicScenarioParser(llm_agent=FakeParserAgent())

    def test_structured_input_is_validated_and_normalized(self) -> None:
        result = self.parser.parse(build_sample_payload())
        scenario = result.scenario.to_dict()

        self.assertEqual(result.input_mode, "structured_json")
        self.assertEqual(scenario["ego_vehicle"]["speed_kmh"], 60.0)
        self.assertEqual(scenario["obstacles"][0]["type"], "child_pedestrian")
        self.assertEqual(scenario["available_actions"][0], "brake_straight")
        self.assertTrue(scenario["collision_unavoidable"])

    def test_natural_language_input_uses_llm_agent_json(self) -> None:
        parser = DeterministicScenarioParser(llm_agent=FakeParserAgent(payload=build_sample_payload()))

        result = parser.parse("Natural language scenario text.")
        scenario = result.scenario.to_dict()

        self.assertEqual(result.input_mode, "natural_language")
        self.assertEqual(scenario["environment"]["road_type"], "residential")
        self.assertEqual(len(scenario["obstacles"]), 2)
        self.assertEqual(scenario["obstacles"][0]["type"], "child_pedestrian")
        self.assertEqual(scenario["obstacles"][1]["type"], "parked_vehicle")
        self.assertAlmostEqual(scenario["sensor_confidence"]["overall_scene_confidence"], 0.93)
        self.assertIn("left sidewalk", scenario["sensor_confidence"]["occluded_zones"])
        self.assertIn("brake_swerve_left", scenario["available_actions"])
        self.assertTrue(scenario["collision_unavoidable"])

    def test_natural_language_does_not_merge_regex_fields_into_llm_output(self) -> None:
        payload = build_sample_payload()
        payload["obstacles"] = [
            {
                "id": "obj_01",
                "type": "vehicle",
                "distance_m": 12,
                "relative_speed_kmh": 40,
                "time_to_impact_s": 1.08,
                "trajectory": "same_lane",
                "vulnerability_class": "medium",
                "mass_kg": 1400,
                "responsible_for_risk": False,
            }
        ]
        parser = DeterministicScenarioParser(llm_agent=FakeParserAgent(payload=payload))

        result = parser.parse("A child pedestrian is crossing. Available actions include swerve left.")

        self.assertEqual(result.scenario.obstacles[0].type, "vehicle")
        self.assertEqual(
            result.scenario.available_actions,
            ["brake_straight", "swerve_left", "swerve_right", "brake_swerve_left"],
        )

    def test_natural_language_parser_agent_failure_raises(self) -> None:
        parser = DeterministicScenarioParser(llm_agent=FakeParserAgent(error="provider unavailable"))

        with self.assertRaisesRegex(ScenarioParseError, "provider unavailable"):
            parser.parse("A vehicle approaches a crossing pedestrian.")

    def test_llm_agent_extracts_json_from_provider_response(self) -> None:
        agent = LLMScenarioParserAgent(provider="disabled")
        agent._runtime_error = None
        agent.client = FakeChatClient('```json\n{"available_actions":["brake_straight"]}\n```')

        result = agent.extract("text")

        self.assertTrue(result.runtime_available)
        self.assertEqual(result.payload["available_actions"], ["brake_straight"])

    def test_llm_agent_reports_provider_errors(self) -> None:
        agent = LLMScenarioParserAgent(provider="disabled")
        agent._runtime_error = None
        agent.client = FakeChatClient(RuntimeError("provider exploded"))

        result = agent.extract("text")

        self.assertFalse(result.runtime_available)
        self.assertIn("provider exploded", result.runtime_error)

    def test_llm_agent_reports_malformed_json(self) -> None:
        agent = LLMScenarioParserAgent(provider="disabled")
        agent._runtime_error = None
        agent.client = FakeChatClient("not-json")

        result = agent.extract("text")

        self.assertFalse(result.runtime_available)
        self.assertIsNotNone(result.runtime_error)

    def test_missing_fields_are_preserved_as_blanks(self) -> None:
        payload = {
            "ego_vehicle": {
                "speed_kmh": 50,
                "acceleration_ms2": -5.0,
            },
            "environment": {
                "road_type": "residential",
            },
            "obstacles": [
                {
                    "type": "parked_vehicle",
                    "distance_m": 10,
                }
            ],
        }

        result = self.parser.parse(payload)
        scenario = result.scenario.to_dict()

        self.assertGreater(len(result.warnings), 0)
        self.assertIsNone(scenario["ego_vehicle"]["mass_kg"])
        self.assertIsNone(scenario["environment"]["speed_limit_kmh"])
        self.assertIsNone(scenario["obstacles"][0]["relative_speed_kmh"])
        self.assertIsNone(scenario["obstacles"][0]["time_to_impact_s"])
        self.assertIsNone(scenario["ego_vehicle"]["braking_distance_m"])
        self.assertIsNone(scenario["collision_unavoidable"])

    def test_missing_core_fields_do_not_raise(self) -> None:
        result = self.parser.parse(
            {"environment": {"road_type": "residential"}, "obstacles": [{"type": "vehicle", "distance_m": 10}]}
        )

        scenario = result.scenario.to_dict()

        self.assertIsNone(scenario["ego_vehicle"]["speed_kmh"])
        self.assertEqual(scenario["environment"]["road_type"], "residential")
        self.assertEqual(scenario["obstacles"][0]["type"], "vehicle")
        self.assertIsNone(scenario["collision_unavoidable"])
        self.assertGreater(len(result.warnings), 0)

    def test_direct_scenario_construction_preserves_unknown_passenger_and_drops_blank_actions(self) -> None:
        scenario = Scenario.from_dict({"available_actions": ["", None, "brake_straight"]})

        self.assertIsNone(scenario.ego_vehicle.passenger_at_risk)
        self.assertEqual(scenario.available_actions, ["brake_straight"])

    def test_markdown_wrapped_json_is_accepted(self) -> None:
        payload = """```json
        {
          "ego_vehicle": {"speed_kmh": 40, "acceleration_ms2": -4.0},
          "environment": {"road_type": "urban"},
          "obstacles": [{"type": "vehicle", "distance_m": 15}]
        }
        ```"""

        result = self.parser.parse(payload)
        self.assertEqual(result.input_mode, "structured_json")
        self.assertEqual(result.scenario.environment.road_type, "urban")

    def test_structured_meta_is_ignored_by_parser_output(self) -> None:
        payload = {
            "ego_vehicle": {
                "speed_kmh": 40,
                "acceleration_ms2": -4.0,
            },
            "environment": {
                "road_type": "urban",
            },
            "obstacles": [
                {
                    "type": "vehicle",
                    "distance_m": 15,
                }
            ],
            "_meta": {
                "warnings": [
                    "passenger-protection and third-party protection create explicit valence trade-off",
                    "distinct stakeholder categories make social valence central",
                ]
            },
        }

        result = self.parser.parse(payload)

        self.assertNotIn("_meta", result.to_dict())
        self.assertNotIn(
            "passenger-protection and third-party protection create explicit valence trade-off",
            result.warnings,
        )
        self.assertNotIn(
            "distinct stakeholder categories make social valence central",
            result.warnings,
        )


if __name__ == "__main__":
    unittest.main()
