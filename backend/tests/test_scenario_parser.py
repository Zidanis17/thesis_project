import unittest

from thesis import DeterministicScenarioParser, ScenarioParseError


class ScenarioParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DeterministicScenarioParser()

    def test_structured_input_is_validated_and_normalized(self) -> None:
        payload = {
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

        result = self.parser.parse(payload)
        scenario = result.scenario.to_dict()

        self.assertEqual(result.input_mode, "structured_json")
        self.assertEqual(scenario["ego_vehicle"]["speed_kmh"], 60.0)
        self.assertEqual(scenario["obstacles"][0]["type"], "child_pedestrian")
        self.assertEqual(scenario["available_actions"][0], "brake_straight")
        self.assertTrue(scenario["collision_unavoidable"])

    def test_natural_language_input_maps_to_schema(self) -> None:
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

        result = self.parser.parse(text)
        scenario = result.scenario.to_dict()

        self.assertEqual(result.input_mode, "natural_language")
        self.assertEqual(scenario["environment"]["road_type"], "residential")
        self.assertEqual(scenario["environment"]["weather"], "clear")
        self.assertEqual(len(scenario["obstacles"]), 2)
        self.assertEqual(scenario["obstacles"][0]["type"], "child_pedestrian")
        self.assertEqual(scenario["obstacles"][1]["type"], "parked_vehicle")
        self.assertAlmostEqual(scenario["sensor_confidence"]["overall_scene_confidence"], 0.93)
        self.assertIn("left sidewalk", scenario["sensor_confidence"]["occluded_zones"])
        self.assertIn("brake_swerve_left", scenario["available_actions"])
        self.assertTrue(scenario["collision_unavoidable"])

    def test_missing_optional_fields_are_inferred(self) -> None:
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
        self.assertEqual(scenario["ego_vehicle"]["mass_kg"], 1800.0)
        self.assertEqual(scenario["environment"]["speed_limit_kmh"], 50.0)
        self.assertEqual(scenario["obstacles"][0]["relative_speed_kmh"], 50.0)
        self.assertAlmostEqual(scenario["obstacles"][0]["time_to_impact_s"], 0.72, places=2)
        self.assertTrue(scenario["collision_unavoidable"])

    def test_missing_core_fields_raise(self) -> None:
        with self.assertRaises(ScenarioParseError):
            self.parser.parse({"environment": {"road_type": "residential"}, "obstacles": [{"type": "vehicle", "distance_m": 10}]})

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
