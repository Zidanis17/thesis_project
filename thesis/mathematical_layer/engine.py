from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import cos, radians
from typing import Any

from ..models import Scenario


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(slots=True)
class StakeholderRisk:
    stakeholder_id: str
    stakeholder_type: str
    label: str
    source: str
    collision_probability: float
    harm_estimate: float
    risk_score: float
    impact_speed_kmh: float
    impact_angle_deg: float
    responsible_for_risk: bool | None = None
    constraint_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionRiskAssessment:
    action: str
    stakeholder_risks: list[StakeholderRisk]
    stakeholder_total_risk: float
    ego_vehicle_risk: float
    total_risk: float
    constraint_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "stakeholder_risks": [risk.to_dict() for risk in self.stakeholder_risks],
            "stakeholder_total_risk": self.stakeholder_total_risk,
            "ego_vehicle_risk": self.ego_vehicle_risk,
            "total_risk": self.total_risk,
            "constraint_flags": list(self.constraint_flags),
        }


@dataclass(slots=True)
class MathematicalLayerResult:
    global_metrics: dict[str, Any]
    violated_rules: list[str]
    action_assessments: list[ActionRiskAssessment]
    risk_score_matrix: dict[str, dict[str, float]]
    best_action_by_total_risk: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_metrics": dict(self.global_metrics),
            "violated_rules": list(self.violated_rules),
            "action_assessments": [assessment.to_dict() for assessment in self.action_assessments],
            "risk_score_matrix": {
                action: dict(scores) for action, scores in self.risk_score_matrix.items()
            },
            "best_action_by_total_risk": self.best_action_by_total_risk,
        }


class DeterministicMathematicalLayer:
    TIME_PRESSURE_WINDOW_S = 2.5
    VISIBILITY_REFERENCE_M = 120.0
    TRAFFIC_PRESSURE = {
        "low": 0.10,
        "medium": 0.45,
        "high": 0.80,
    }
    WEATHER_PRESSURE = {
        "clear": 0.00,
        "rain": 0.20,
        "fog": 0.35,
        "snow": 0.30,
        "storm": 0.45,
    }
    WEATHER_SPEED_DEGRADATION = {
        "clear": 1.00,
        "rain": 1.08,
        "fog": 1.05,
        "snow": 1.12,
        "storm": 1.18,
    }
    VULNERABILITY_MULTIPLIERS = {
        "high": 2.50,
        "medium": 1.50,
        "low": 0.90,
    }
    PRIORITY_TRAJECTORIES = {"crossing", "oncoming", "merging"}
    RIGHT_OF_WAY_PROBABILITY_THRESHOLD = 0.45
    ACTION_PROFILES: dict[str, dict[str, Any]] = {
        "brake_straight": {
            "speed_factor": 0.45,
            "trajectory_probability": {
                "stationary": 0.75,
                "same_lane": 0.60,
                "crossing": 0.55,
                "oncoming": 0.78,
                "merging": 0.62,
            },
            "impact_angles": {
                "stationary": 5.0,
                "same_lane": 0.0,
                "crossing": 25.0,
                "oncoming": 175.0,
                "merging": 20.0,
            },
            "occlusion_exposure": {
                "left sidewalk": 0.20,
                "right sidewalk": 0.20,
                "crosswalk": 0.45,
                "bike lane": 0.25,
                "left lane": 0.10,
                "right lane": 0.10,
                "intersection": 0.40,
            },
        },
        "swerve_left": {
            "speed_factor": 0.85,
            "trajectory_probability": {
                "stationary": 0.35,
                "same_lane": 0.50,
                "crossing": 0.90,
                "oncoming": 0.95,
                "merging": 0.72,
            },
            "impact_angles": {
                "stationary": 45.0,
                "same_lane": 35.0,
                "crossing": 70.0,
                "oncoming": 150.0,
                "merging": 50.0,
            },
            "occlusion_exposure": {
                "left sidewalk": 0.95,
                "right sidewalk": 0.05,
                "crosswalk": 0.25,
                "bike lane": 0.70,
                "left lane": 0.90,
                "right lane": 0.05,
                "intersection": 0.35,
            },
        },
        "swerve_right": {
            "speed_factor": 0.85,
            "trajectory_probability": {
                "stationary": 0.35,
                "same_lane": 0.50,
                "crossing": 0.90,
                "oncoming": 0.95,
                "merging": 0.72,
            },
            "impact_angles": {
                "stationary": 45.0,
                "same_lane": 35.0,
                "crossing": 70.0,
                "oncoming": 150.0,
                "merging": 50.0,
            },
            "occlusion_exposure": {
                "left sidewalk": 0.05,
                "right sidewalk": 0.95,
                "crosswalk": 0.25,
                "bike lane": 0.70,
                "left lane": 0.05,
                "right lane": 0.90,
                "intersection": 0.35,
            },
        },
        "brake_swerve_left": {
            "speed_factor": 0.58,
            "trajectory_probability": {
                "stationary": 0.30,
                "same_lane": 0.40,
                "crossing": 0.70,
                "oncoming": 0.82,
                "merging": 0.52,
            },
            "impact_angles": {
                "stationary": 35.0,
                "same_lane": 25.0,
                "crossing": 55.0,
                "oncoming": 160.0,
                "merging": 40.0,
            },
            "occlusion_exposure": {
                "left sidewalk": 0.75,
                "right sidewalk": 0.05,
                "crosswalk": 0.35,
                "bike lane": 0.55,
                "left lane": 0.75,
                "right lane": 0.05,
                "intersection": 0.35,
            },
        },
        "brake_swerve_right": {
            "speed_factor": 0.58,
            "trajectory_probability": {
                "stationary": 0.30,
                "same_lane": 0.40,
                "crossing": 0.70,
                "oncoming": 0.82,
                "merging": 0.52,
            },
            "impact_angles": {
                "stationary": 35.0,
                "same_lane": 25.0,
                "crossing": 55.0,
                "oncoming": 160.0,
                "merging": 40.0,
            },
            "occlusion_exposure": {
                "left sidewalk": 0.05,
                "right sidewalk": 0.75,
                "crosswalk": 0.35,
                "bike lane": 0.55,
                "left lane": 0.05,
                "right lane": 0.75,
                "intersection": 0.35,
            },
        },
    }
    OCCLUSION_ZONE_PROFILES: dict[str, dict[str, Any]] = {
        "left sidewalk": {
            "stakeholder_type": "hidden_pedestrian",
            "mass_kg": 75.0,
            "vulnerability_class": "high",
            "trajectory": "crossing",
            "zone_factor": 1.00,
        },
        "right sidewalk": {
            "stakeholder_type": "hidden_pedestrian",
            "mass_kg": 75.0,
            "vulnerability_class": "high",
            "trajectory": "crossing",
            "zone_factor": 1.00,
        },
        "crosswalk": {
            "stakeholder_type": "hidden_pedestrian",
            "mass_kg": 75.0,
            "vulnerability_class": "high",
            "trajectory": "crossing",
            "zone_factor": 1.10,
        },
        "bike lane": {
            "stakeholder_type": "hidden_cyclist",
            "mass_kg": 90.0,
            "vulnerability_class": "high",
            "trajectory": "crossing",
            "zone_factor": 1.00,
        },
        "left lane": {
            "stakeholder_type": "adjacent_vehicle",
            "mass_kg": 1500.0,
            "vulnerability_class": "medium",
            "trajectory": "same_lane",
            "zone_factor": 0.90,
        },
        "right lane": {
            "stakeholder_type": "adjacent_vehicle",
            "mass_kg": 1500.0,
            "vulnerability_class": "medium",
            "trajectory": "same_lane",
            "zone_factor": 0.90,
        },
        "intersection": {
            "stakeholder_type": "cross_traffic",
            "mass_kg": 1200.0,
            "vulnerability_class": "medium",
            "trajectory": "crossing",
            "zone_factor": 1.15,
        },
    }

    def analyze(self, scenario: Scenario) -> MathematicalLayerResult:
        global_metrics = self._compute_global_metrics(scenario)
        violated_rules = self._compute_rule_flags(scenario, global_metrics)

        action_assessments: list[ActionRiskAssessment] = []
        risk_score_matrix: dict[str, dict[str, float]] = {}

        for action in scenario.available_actions:
            assessment = self._analyze_action(scenario, action, global_metrics)
            action_assessments.append(assessment)
            risk_score_matrix[action] = {"ego_vehicle": round(assessment.ego_vehicle_risk, 3)}
            for stakeholder_risk in assessment.stakeholder_risks:
                risk_score_matrix[action][stakeholder_risk.stakeholder_id] = round(
                    stakeholder_risk.risk_score,
                    3,
                )

        best_action = min(
            action_assessments,
            key=lambda assessment: (assessment.total_risk, assessment.action),
        ).action

        return MathematicalLayerResult(
            global_metrics=global_metrics,
            violated_rules=violated_rules,
            action_assessments=action_assessments,
            risk_score_matrix=risk_score_matrix,
            best_action_by_total_risk=best_action,
        )

    def _analyze_action(
        self,
        scenario: Scenario,
        action: str,
        global_metrics: dict[str, Any],
    ) -> ActionRiskAssessment:
        profile = self._action_profile(action)
        stakeholder_risks: list[StakeholderRisk] = []
        constraint_flags: list[str] = []
        ego_vehicle_risk = 0.0

        for obstacle in scenario.obstacles:
            stakeholder_risk, ego_risk = self._analyze_obstacle(
                scenario=scenario,
                action=action,
                profile=profile,
                obstacle=obstacle,
                global_metrics=global_metrics,
            )
            stakeholder_risks.append(stakeholder_risk)
            ego_vehicle_risk += ego_risk
            constraint_flags.extend(stakeholder_risk.constraint_flags)

        for zone in scenario.sensor_confidence.occluded_zones:
            stakeholder_risk, ego_risk = self._analyze_occlusion_zone(
                scenario=scenario,
                action=action,
                profile=profile,
                zone=zone,
                global_metrics=global_metrics,
            )
            if stakeholder_risk is None:
                continue
            stakeholder_risks.append(stakeholder_risk)
            ego_vehicle_risk += ego_risk
            constraint_flags.extend(stakeholder_risk.constraint_flags)

        # FIX 1: use "swerve" in action to also catch brake_swerve_left / brake_swerve_right
        if "swerve" in action and scenario.ego_vehicle.speed_kmh > scenario.environment.speed_limit_kmh:
            constraint_flags.append("speeding_during_lateral_evasion")

        stakeholder_total_risk = sum(item.risk_score for item in stakeholder_risks)
        total_risk = stakeholder_total_risk + ego_vehicle_risk

        deduped_flags = list(dict.fromkeys(constraint_flags))
        stakeholder_risks.sort(key=lambda item: item.risk_score, reverse=True)

        return ActionRiskAssessment(
            action=action,
            stakeholder_risks=stakeholder_risks,
            stakeholder_total_risk=round(stakeholder_total_risk, 3),
            ego_vehicle_risk=round(ego_vehicle_risk, 3),
            total_risk=round(total_risk, 3),
            constraint_flags=deduped_flags,
        )

    def _analyze_obstacle(
        self,
        scenario: Scenario,
        action: str,
        profile: dict[str, Any],
        obstacle: Any,
        global_metrics: dict[str, Any],
    ) -> tuple[StakeholderRisk, float]:
        base_probability = self._base_collision_probability(
            distance_m=obstacle.distance_m,
            time_to_impact_s=obstacle.time_to_impact_s,
            scenario=scenario,
            global_metrics=global_metrics,
        )
        trajectory = obstacle.trajectory.lower()
        probability_multiplier = profile["trajectory_probability"].get(trajectory, 0.75)
        collision_probability = _clamp(base_probability * probability_multiplier)

        impact_speed_mps = self._impact_speed_mps(
            scenario.ego_vehicle.speed_kmh,
            profile["speed_factor"],
            scenario.environment.weather,
        )
        impact_angle_deg = profile["impact_angles"].get(trajectory, 35.0)
        harm_estimate = self._harm_estimate(
            ego_mass_kg=scenario.ego_vehicle.mass_kg,
            stakeholder_mass_kg=obstacle.mass_kg,
            impact_speed_mps=impact_speed_mps,
            vulnerability_class=obstacle.vulnerability_class,
            impact_angle_deg=impact_angle_deg,
        )
        ego_harm_estimate = self._ego_harm_estimate(
            ego_mass_kg=scenario.ego_vehicle.mass_kg,
            stakeholder_mass_kg=obstacle.mass_kg,
            impact_speed_mps=impact_speed_mps,
            impact_angle_deg=impact_angle_deg,
        )

        constraint_flags: list[str] = []
        if (
            not obstacle.responsible_for_risk
            and trajectory in self.PRIORITY_TRAJECTORIES
            and collision_probability >= self.RIGHT_OF_WAY_PROBABILITY_THRESHOLD
        ):
            constraint_flags.append(f"potential_right_of_way_violation:{obstacle.id}")
        # FIX 1: use "swerve" in action to also catch brake_swerve_left / brake_swerve_right
        if "swerve" in action and trajectory == "crossing" and collision_probability >= 0.60:
            constraint_flags.append(f"high_speed_swerve_toward_crossing_stakeholder:{obstacle.id}")

        stakeholder_risk = StakeholderRisk(
            stakeholder_id=obstacle.id,
            stakeholder_type=obstacle.type,
            label=obstacle.type,
            source="obstacle",
            collision_probability=round(collision_probability, 3),
            harm_estimate=round(harm_estimate, 3),
            risk_score=round(collision_probability * harm_estimate, 3),
            impact_speed_kmh=round(impact_speed_mps * 3.6, 3),
            impact_angle_deg=round(impact_angle_deg, 3),
            responsible_for_risk=obstacle.responsible_for_risk,
            constraint_flags=constraint_flags,
        )
        ego_vehicle_risk = round(collision_probability * ego_harm_estimate, 3)
        return stakeholder_risk, ego_vehicle_risk

    def _analyze_occlusion_zone(
        self,
        scenario: Scenario,
        action: str,
        profile: dict[str, Any],
        zone: str,
        global_metrics: dict[str, Any],
    ) -> tuple[StakeholderRisk | None, float]:
        canonical_zone = self._canonicalize_zone(zone)
        zone_profile = self.OCCLUSION_ZONE_PROFILES.get(canonical_zone)
        if zone_profile is None:
            return None, 0.0

        exposure = profile["occlusion_exposure"].get(canonical_zone, 0.0)
        collision_probability = self._occlusion_probability(
            exposure=exposure,
            zone_factor=zone_profile["zone_factor"],
            global_metrics=global_metrics,
            collision_unavoidable=scenario.collision_unavoidable,
        )
        impact_speed_mps = self._impact_speed_mps(
            scenario.ego_vehicle.speed_kmh,
            profile["speed_factor"],
            scenario.environment.weather,
        )
        impact_angle_deg = profile["impact_angles"].get(zone_profile["trajectory"], 40.0)
        harm_estimate = self._harm_estimate(
            ego_mass_kg=scenario.ego_vehicle.mass_kg,
            stakeholder_mass_kg=zone_profile["mass_kg"],
            impact_speed_mps=impact_speed_mps,
            vulnerability_class=zone_profile["vulnerability_class"],
            impact_angle_deg=impact_angle_deg,
        )
        ego_harm_estimate = self._ego_harm_estimate(
            ego_mass_kg=scenario.ego_vehicle.mass_kg,
            stakeholder_mass_kg=zone_profile["mass_kg"],
            impact_speed_mps=impact_speed_mps,
            impact_angle_deg=impact_angle_deg,
        )

        constraint_flags: list[str] = []
        if "swerve" in action and exposure >= 0.60:
            constraint_flags.append(f"steers_into_occluded_zone:{canonical_zone.replace(' ', '_')}")

        stakeholder_id = f"occlusion:{canonical_zone.replace(' ', '_')}"
        stakeholder_risk = StakeholderRisk(
            stakeholder_id=stakeholder_id,
            stakeholder_type=zone_profile["stakeholder_type"],
            label=canonical_zone,
            source="occlusion_zone",
            collision_probability=round(collision_probability, 3),
            harm_estimate=round(harm_estimate, 3),
            risk_score=round(collision_probability * harm_estimate, 3),
            impact_speed_kmh=round(impact_speed_mps * 3.6, 3),
            impact_angle_deg=round(impact_angle_deg, 3),
            responsible_for_risk=None,
            constraint_flags=constraint_flags,
        )
        ego_vehicle_risk = round(collision_probability * ego_harm_estimate, 3)
        return stakeholder_risk, ego_vehicle_risk

    def _compute_global_metrics(self, scenario: Scenario) -> dict[str, Any]:
        sensor_fusion_confidence = _clamp(
            (
                0.40 * scenario.sensor_confidence.overall_scene_confidence
                + 0.20 * scenario.sensor_confidence.lidar
                + 0.20 * scenario.sensor_confidence.camera
                + 0.20 * scenario.sensor_confidence.radar
            ),
        )
        scene_uncertainty = 1.0 - sensor_fusion_confidence
        speed_limit_delta_kmh = scenario.ego_vehicle.speed_kmh - scenario.environment.speed_limit_kmh
        visibility_pressure = _clamp(
            (self.VISIBILITY_REFERENCE_M - scenario.environment.visibility_m) / self.VISIBILITY_REFERENCE_M,
        )
        # FIX 2: safe fallback when obstacles list is empty
        closest_obstacle_distance_m = min(
            (obstacle.distance_m for obstacle in scenario.obstacles),
            default=float("inf"),
        )
        braking_margin_m = (
            closest_obstacle_distance_m - scenario.ego_vehicle.braking_distance_m
            if closest_obstacle_distance_m != float("inf")
            else float("inf")
        )

        return {
            "sensor_fusion_confidence": round(sensor_fusion_confidence, 3),
            "scene_uncertainty": round(scene_uncertainty, 3),
            "speed_limit_delta_kmh": round(speed_limit_delta_kmh, 3),
            "visibility_pressure": round(visibility_pressure, 3),
            "closest_obstacle_distance_m": round(closest_obstacle_distance_m, 3) if closest_obstacle_distance_m != float("inf") else None,
            "braking_margin_m": round(braking_margin_m, 3) if braking_margin_m != float("inf") else None,
        }

    def _compute_rule_flags(self, scenario: Scenario, global_metrics: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        if global_metrics["speed_limit_delta_kmh"] > 0:
            flags.append("speed_limit_exceeded")
        if scenario.ego_vehicle.braking_distance_m > scenario.environment.visibility_m:
            flags.append("cannot_stop_within_visible_distance")
        return flags

    def _base_collision_probability(
        self,
        distance_m: float,
        time_to_impact_s: float,
        scenario: Scenario,
        global_metrics: dict[str, Any],
    ) -> float:
        braking_pressure = _clamp(
            scenario.ego_vehicle.braking_distance_m / max(distance_m, 0.5),
            maximum=2.0,
        ) / 2.0
        time_pressure = _clamp(
            (self.TIME_PRESSURE_WINDOW_S - time_to_impact_s) / self.TIME_PRESSURE_WINDOW_S,
        )
        visibility_pressure = float(global_metrics["visibility_pressure"])
        scene_uncertainty = float(global_metrics["scene_uncertainty"])
        weather_pressure = self.WEATHER_PRESSURE.get(scenario.environment.weather, 0.15)
        traffic_pressure = self.TRAFFIC_PRESSURE.get(scenario.environment.traffic_density, 0.45)
        unavoidable_bonus = 0.10 if scenario.collision_unavoidable else 0.0

        return _clamp(
            0.45 * braking_pressure
            + 0.35 * time_pressure
            + 0.07 * scene_uncertainty
            + 0.05 * visibility_pressure
            + 0.04 * weather_pressure
            + 0.04 * traffic_pressure
            + unavoidable_bonus,
        )

    def _occlusion_probability(
        self,
        exposure: float,
        zone_factor: float,
        global_metrics: dict[str, Any],
        collision_unavoidable: bool,
    ) -> float:
        scene_uncertainty = float(global_metrics["scene_uncertainty"])
        visibility_pressure = float(global_metrics["visibility_pressure"])
        unavoidable_bonus = 0.02 if collision_unavoidable and exposure >= 0.50 else 0.0
        # FIX 3: removed redundant round() here — caller already rounds the result
        probability = exposure * zone_factor * (
            0.55 * scene_uncertainty
            + 0.25 * visibility_pressure
            + 0.05
            + unavoidable_bonus
        )
        return _clamp(probability, maximum=0.40)

    def _impact_speed_mps(self, ego_speed_kmh: float, speed_factor: float, weather: str) -> float:
        degraded_factor = min(
            1.0,
            speed_factor * self.WEATHER_SPEED_DEGRADATION.get(weather, 1.05),
        )
        return max(0.1, self._kmh_to_mps(ego_speed_kmh) * degraded_factor)

    def _harm_estimate(
        self,
        ego_mass_kg: float,
        stakeholder_mass_kg: float,
        impact_speed_mps: float,
        vulnerability_class: str,
        impact_angle_deg: float,
    ) -> float:
        kinetic_energy_kj = 0.5 * ego_mass_kg * (impact_speed_mps**2) / 1000.0
        mass_factor = 0.65 + 0.35 * min(1.0, stakeholder_mass_kg / max(ego_mass_kg, 1.0))
        vulnerability_multiplier = self.VULNERABILITY_MULTIPLIERS.get(vulnerability_class, 1.2)
        angle_factor = 0.70 + 0.50 * abs(cos(radians(impact_angle_deg)))
        return kinetic_energy_kj * mass_factor * vulnerability_multiplier * angle_factor

    def _ego_harm_estimate(
        self,
        ego_mass_kg: float,
        stakeholder_mass_kg: float,
        impact_speed_mps: float,
        impact_angle_deg: float,
    ) -> float:
        kinetic_energy_kj = 0.5 * ego_mass_kg * (impact_speed_mps**2) / 1000.0
        mass_ratio = stakeholder_mass_kg / max(ego_mass_kg, 1.0)
        mass_factor = 0.60 + 0.45 * min(1.5, mass_ratio)
        angle_factor = 0.80 + 0.40 * abs(cos(radians(impact_angle_deg)))
        return kinetic_energy_kj * mass_factor * angle_factor

    def _action_profile(self, action: str) -> dict[str, Any]:
        return self.ACTION_PROFILES.get(action, self.ACTION_PROFILES["brake_straight"])

    def _canonicalize_zone(self, zone: str) -> str:
        return zone.strip().lower().replace("_", " ")

    def _kmh_to_mps(self, speed_kmh: float) -> float:
        return speed_kmh / 3.6