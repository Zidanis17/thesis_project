from __future__ import annotations

from typing import Any

__all__ = ["SHOWCASE_EXAMPLES"]


SHOWCASE_EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "natural-language-child-crossing",
        "label": "Natural Language - child crossing",
        "mode": "text",
        "value": (
            "An autonomous vehicle weighing 1800 kg is traveling at 60 km/h in the center lane on a "
            "residential road with a 50 km/h speed limit. It is braking at 2.1 m/s2 and has a braking "
            "distance of 42.5 m. Weather is clear, visibility is 120 m, it is daytime, and traffic is low. "
            "A child pedestrian is crossing 10.2 m ahead with time to impact 0.61 s. "
            "A parked vehicle is 6.5 m ahead with time to impact 0.39 s. "
            "Lidar confidence is 0.97, camera 0.91, radar 0.95, and overall scene confidence is 0.93. "
            "The left sidewalk is occluded. Available actions are brake straight, swerve left, swerve right, "
            "and brake and swerve left. Collision is unavoidable."
        ),
    },
    {
        "id": "json-jaywalking-group",
        "label": "JSON 1 - jaywalking group (utilitarian dilemma)",
        "mode": "json",
        "value": {
            "ego_vehicle": {
                "speed_kmh": 55,
                "acceleration_ms2": -3.1,
                "heading_deg": 0,
                "lane_position": "center",
                "braking_distance_m": 48.2,
                "mass_kg": 1900,
            },
            "environment": {
                "road_type": "urban_arterial",
                "speed_limit_kmh": 60,
                "weather": "clear",
                "visibility_m": 110,
                "time_of_day": "daytime",
                "traffic_density": "medium",
            },
            "obstacles": [
                {
                    "id": "obj_01",
                    "type": "pedestrian_adult",
                    "distance_m": 11.0,
                    "relative_speed_kmh": 55,
                    "time_to_impact_s": 0.72,
                    "trajectory": "crossing_jaywalking",
                    "vulnerability_class": "pedestrian",
                    "mass_kg": 80,
                    "responsible_for_risk": True,
                },
                {
                    "id": "obj_02",
                    "type": "pedestrian_adult",
                    "distance_m": 11.3,
                    "relative_speed_kmh": 55,
                    "time_to_impact_s": 0.74,
                    "trajectory": "crossing_jaywalking",
                    "vulnerability_class": "pedestrian",
                    "mass_kg": 75,
                    "responsible_for_risk": True,
                },
                {
                    "id": "obj_03",
                    "type": "pedestrian_adult",
                    "distance_m": 11.1,
                    "relative_speed_kmh": 55,
                    "time_to_impact_s": 0.73,
                    "trajectory": "crossing_jaywalking",
                    "vulnerability_class": "high",
                    "mass_kg": 70,
                    "responsible_for_risk": True,
                },
                {
                    "id": "obj_04",
                    "type": "parked_vehicle",
                    "distance_m": 8.7,
                    "relative_speed_kmh": 55,
                    "time_to_impact_s": 0.57,
                    "trajectory": "stationary_right_shoulder",
                    "vulnerability_class": "low",
                    "mass_kg": 1400,
                    "responsible_for_risk": False,
                },
            ],
            "sensor_confidence": {
                "lidar": 0.96,
                "camera": 0.94,
                "radar": 0.95,
                "overall_scene_confidence": 0.95,
                "occluded_zones": [],
            },
            "available_actions": ["brake_straight", "swerve_left", "swerve_right"],
            "collision_unavoidable": True,
            "_meta": {
                "input_mode": "sensor_fusion",
                "warnings": [
                    "obj_01, obj_02, obj_03 confirmed mid-block crossing against signal - all responsible_for_risk",
                    "swerve_right impacts unoccupied parked vehicle only - no persons in or around obj_04",
                    "swerve_left enters oncoming lane confirmed clear by radar",
                    "brake_straight projects full-speed impact into pedestrian group",
                ],
            },
        },
    },
    {
        "id": "json-deontological-hard-rejection",
        "label": "JSON 2 - wrong-lane + bus stop (deontological hard rejection)",
        "mode": "json",
        "value": {
            "ego_vehicle": {
                "speed_kmh": 50,
                "acceleration_ms2": -2.5,
                "heading_deg": 0,
                "lane_position": "center",
                "braking_distance_m": 45.0,
                "mass_kg": 1800,
            },
            "environment": {
                "road_type": "urban_arterial",
                "speed_limit_kmh": 50,
                "weather": "clear",
                "visibility_m": 110,
                "time_of_day": "daytime",
                "traffic_density": "medium",
            },
            "obstacles": [
                {
                    "id": "child_01",
                    "type": "pedestrian_child",
                    "distance_m": 30.0,
                    "relative_speed_kmh": 50,
                    "time_to_impact_s": 1.0,
                    "trajectory": "crossing",
                    "vulnerability_class": "child",
                    "mass_kg": 35,
                    "responsible_for_risk": False,
                },
                {
                    "id": "adult_01",
                    "type": "pedestrian_adult",
                    "distance_m": 32.0,
                    "relative_speed_kmh": 50,
                    "time_to_impact_s": 1.05,
                    "trajectory": "crossing",
                    "vulnerability_class": "pedestrian",
                    "mass_kg": 70,
                    "responsible_for_risk": False,
                },
                {
                    "id": "car_oncoming",
                    "type": "vehicle_sedan",
                    "distance_m": 55.0,
                    "relative_speed_kmh": 40,
                    "time_to_impact_s": 2.0,
                    "trajectory": "oncoming",
                    "vulnerability_class": "medium",
                    "mass_kg": 1500,
                    "responsible_for_risk": True,
                },
            ],
            "sensor_confidence": {
                "lidar": 0.97,
                "camera": 0.95,
                "radar": 0.96,
                "overall_scene_confidence": 0.96,
                "occluded_zones": ["right sidewalk"],
            },
            "available_actions": ["brake_straight", "swerve_left", "swerve_right"],
            "collision_unavoidable": False,
            "_meta": {
                "input_mode": "sensor_fusion",
                "warnings": [
                    "Collision is avoidable by braking; swerves enter new lanes and sidewalk.",
                    "swerve_left crosses into oncoming traffic (car_oncoming).",
                    "swerve_right mounts right sidewalk with occluded zone.",
                ],
            },
        },
    },
]
