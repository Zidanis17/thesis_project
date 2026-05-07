"""
Risk assessment engine for autonomous vehicle ethical trajectory planning.

Harm estimation (Equations 1–2) and the risk definition (R = p × H) are
implemented from Geisslinger et al. (2023). The ethical cost functions
(Bayes, equality, maximin, responsibility) follow the structure of the same
paper and are adapted to this thesis' action-level scenario representation:

    Geisslinger, M., Poszler, F., & Lienkamp, M. (2023).
    "An ethical trajectory planning algorithm for autonomous vehicles."
    Nature Machine Intelligence, 5, 137–144.
    https://doi.org/10.1038/s42256-022-00607-z

Open-source reference implementation:
    https://github.com/TUMFTM/EthicalTrajectoryPlanning
    (risk_assessment/harm_estimation.py, risk_assessment/utils/,
     risk_assessment/risk_costs.py, planner/Frenet/configs/harm_parameters.json)

---------------------------------------------------------------------------
Collision likelihood model — design rationale
---------------------------------------------------------------------------
The paper's collision probability module relies on a CommonRoad-coupled LSTM
motion predictor that outputs per-timestep bivariate Gaussian position
distributions for each obstacle (ref. 48 in Geisslinger et al. 2023). That
predictor requires CommonRoad scenario objects and per-obstacle state
covariance matrices, neither of which is available in this architecture's
Scenario representation. A direct port is therefore not feasible.

Instead, this module implements a **deterministic surrogate
collision-likelihood model** that is *inspired by* — but does not claim to
replicate — established model-based threat assessment literature:

    PRIMARY INSPIRATION (physical threat reasoning):
    Brännström, M., Coelingh, E., & Sjöberg, J. (2010).
    "Model-based threat assessment for avoiding arbitrary vehicle collisions."
    IEEE Transactions on Intelligent Transportation Systems, 11(3), 658–669.
    https://doi.org/10.1109/TITS.2010.2048314

    SECONDARY INSPIRATION (uncertainty-aware criticality assessment):
    Berthelot, A., Tamke, A., Dang, T., & Breuel, G. (2011).
    "Handling uncertainties in criticality assessment."
    2011 IEEE Intelligent Vehicles Symposium (IV), pp. 571–576.
    https://doi.org/10.1109/IVS.2011.5940560

The surrogate preserves the main physical intuition of model-based
criticality assessment — especially braking feasibility, time-to-collision,
and perceptual uncertainty — but replaces probabilistic state distributions
with a calibrated deterministic weighted formulation compatible with the
Scenario data model. Weather and traffic density are own contextual additions
not directly drawn from either paper. Full probabilistic formulations are not
adopted due to the absence of state covariance and trajectory distribution
inputs in this architecture.

See _base_collision_likelihood() for a detailed mapping of each term to
its literature counterpart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import cos, radians, exp
from typing import Any

from ..models import Scenario
from ..normalization import (
    VULNERABILITY_TO_PROTECTED as SHARED_VULNERABILITY_TO_PROTECTED,
    canonicalize_road_type,
    canonicalize_trajectory,
    canonicalize_weather,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# NHTSA logistic-regression coefficients
#
# Source: Geisslinger et al. (2023), Equations 1–2 and Methods section.
# Coefficients are trained on the NHTSA Crash Report Sampling System (CRSS)
# dataset and published in the open-source repository under
# planner/Frenet/configs/harm_parameters.json.
#
# Three models are retained here:
#   "ignore_angle"  – vehicle-to-vehicle, no impact-area distinction (LR1S)
#   "pedestrian"    – vehicle-to-unprotected road user (VRU)
# ---------------------------------------------------------------------------

# Protected road users (vehicles with crash structure): LR1S variant
# H = 1 / (1 + exp(-c0 - c1 * Δv))   [Eq. 2, Geisslinger et al. 2023]
_LR_PROTECTED_CONST: float = -4.591   # c0  [harm_parameters.json → log_reg.ignore_angle.const]
_LR_PROTECTED_SPEED: float = 0.185    # c1  [harm_parameters.json → log_reg.ignore_angle.speed]

# Unprotected road users (pedestrians, cyclists): pedestrian logistic model
# H = 1 / (1 + exp(c0 - c1 * Δv))    [Geisslinger et al. 2023, Methods]
_LR_PEDESTRIAN_CONST: float = 3.164   # c0  [harm_parameters.json → pedestrian.const]
_LR_PEDESTRIAN_SPEED: float = 0.288   # c1  [harm_parameters.json → pedestrian.speed]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StakeholderRisk:
    stakeholder_id: str
    stakeholder_type: str
    label: str
    source: str
    collision_likelihood: float
    harm_estimate: float         # H for the stakeholder  [Eq. 2, Geisslinger et al. 2023]
    ego_harm_estimate: float     # H for the ego vehicle  [Eq. 2, Geisslinger et al. 2023]
    risk_score: float            # R = p × H; here p is collision_likelihood
    ego_risk_score: float        # R = p × H; here p is collision_likelihood
    impact_speed_kmh: float
    impact_angle_deg: float
    delta_v_mps: float           # Δv used in harm model   [Eq. 1, Geisslinger et al. 2023]
    responsible_for_risk: bool | None = None
    constraint_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EthicalCosts:
    """
    Ethical cost breakdown per trajectory action.

    All three cost terms follow Geisslinger et al. (2023):
      - bayes_cost   → Eq. 9  (utilitarian / Bayes principle)
      - equality_cost → Eq. 10 (equality principle)
      - maximin_cost  → Eq. 11 (maximin / prioritise worst-off principle)
    Combined cost (Eq. 8) uses equal weights (w_B = w_E = w_M = 1).
    """
    bayes_cost: float      # J_B  [Eq. 9,  Geisslinger et al. 2023]
    equality_cost: float   # J_E  [Eq. 10, Geisslinger et al. 2023]
    maximin_cost: float    # J_M  [Eq. 11, Geisslinger et al. 2023]
    responsibility_cost: float  # J_R  [Eq. 12, Geisslinger et al. 2023]
    combined_cost: float   # J_Risk = w_B*J_B + w_E*J_E + w_M*J_M - w_B*J_R [Eq. 8]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionRiskAssessment:
    action: str
    stakeholder_risks: list[StakeholderRisk]
    stakeholder_total_risk: float
    ego_vehicle_risk: float
    total_risk: float
    ethical_costs: EthicalCosts
    constraint_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "stakeholder_risks": [risk.to_dict() for risk in self.stakeholder_risks],
            "stakeholder_total_risk": self.stakeholder_total_risk,
            "ego_vehicle_risk": self.ego_vehicle_risk,
            "total_risk": self.total_risk,
            "ethical_costs": self.ethical_costs.to_dict(),
            "constraint_flags": list(self.constraint_flags),
        }


@dataclass(slots=True)
class MathematicalLayerResult:
    global_metrics: dict[str, Any]
    violated_rules: list[str]
    action_assessments: list[ActionRiskAssessment]
    risk_score_matrix: dict[str, dict[str, float]]
    best_action_by_total_risk: str | None
    best_action_by_ethical_cost: str | None  # selected by Bayes cost [Eq. 9, Geisslinger et al. 2023]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_metrics": dict(self.global_metrics),
            "violated_rules": list(self.violated_rules),
            "action_assessments": [a.to_dict() for a in self.action_assessments],
            "risk_score_matrix": {
                action: dict(scores) for action, scores in self.risk_score_matrix.items()
            },
            "best_action_by_total_risk": self.best_action_by_total_risk,
            "best_action_by_ethical_cost": self.best_action_by_ethical_cost,
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class DeterministicMathematicalLayer:
    """
    Deterministic risk assessment layer.

    Harm model:      Geisslinger et al. (2023), Equations 1–2.
    Risk definition: Geisslinger et al. (2023), R = p × H.
    Cost functions:  follow Geisslinger et al. (2023), Equations 8–12
                     (Bayes, equality, maximin, responsibility), adapted to
                     this action-level scenario representation.

    Collision likelihood: deterministic surrogate model (own engineering),
        inspired by the physical input space of Brännström et al. (2010)
        and Berthelot et al. (2011). See _base_collision_likelihood() for
        the full rationale and term-by-term literature mapping.
    """

    # ------------------------------------------------------------------
    # Scenario-level constants
    # ------------------------------------------------------------------

    # 2.5 s reaction/intervention window consistent with the prediction
    # horizons used in Brännström et al. (2010), Section VI-B (T_max = 3 s,
    # with critical TTC typically falling between 0.8–2.0 s in their
    # rear-end and intersection evaluations).
    TIME_PRESSURE_WINDOW_S = 2.5

    # Reference visibility beyond which visibility no longer meaningfully
    # increases risk (own engineering constant).
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

    # Protection classification follows Geisslinger et al. (2023):
    # "our model distinguishes between protected (vehicles, trucks) and
    #  unprotected (pedestrians, cyclists) road users" (Methods section).
    # vulnerability_class → True = protected (vehicle), False = unprotected (VRU)
    #
    # The map covers both the abstract severity levels ("high"/"medium"/"low")
    # used internally and the explicit obstacle-type strings that may appear
    # in scenario obstacle definitions. Without explicit entries for type
    # strings, the default=True fallback in .get() would silently treat
    # pedestrians and cyclists as protected road users — incorrectly applying
    # the vehicle harm coefficients instead of the pedestrian logistic model.
    VULNERABILITY_TO_PROTECTED: dict[str, bool] = SHARED_VULNERABILITY_TO_PROTECTED

    PRIORITY_TRAJECTORIES = {"crossing", "oncoming", "merging"}
    RIGHT_OF_WAY_LIKELIHOOD_THRESHOLD = 0.45

    # ------------------------------------------------------------------
    # Action profiles (own engineering)
    # ------------------------------------------------------------------
    ACTION_PROFILES: dict[str, dict[str, Any]] = {
        "brake_straight": {
            "speed_factor": 0.45,
            "trajectory_likelihood_multiplier": {
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
            "trajectory_likelihood_multiplier": {
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
            "trajectory_likelihood_multiplier": {
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
            "trajectory_likelihood_multiplier": {
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
            "trajectory_likelihood_multiplier": {
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
        "maintain_lane": {
            "speed_factor": 1.00,
            "trajectory_likelihood_multiplier": {
                "stationary": 0.95,
                "same_lane": 0.95,
                "crossing": 0.95,
                "oncoming": 0.90,
                "merging": 0.85,
            },
            "impact_angles": {
                "stationary": 5.0,
                "same_lane": 0.0,
                "crossing": 25.0,
                "oncoming": 175.0,
                "merging": 20.0,
            },
            "occlusion_exposure": {
                "left sidewalk": 0.10,
                "right sidewalk": 0.10,
                "crosswalk": 0.35,
                "bike lane": 0.20,
                "left lane": 0.05,
                "right lane": 0.05,
                "intersection": 0.35,
            },
        },
    }

    OCCLUSION_ZONE_PROFILES: dict[str, dict[str, Any]] = {
        "left sidewalk": {
            "stakeholder_type": "hidden_pedestrian",
            "mass_kg": 75.0,
            "vulnerability_class": "high",   # unprotected [Geisslinger et al. 2023]
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
            "vulnerability_class": "high",   # unprotected [Geisslinger et al. 2023]
            "trajectory": "crossing",
            "zone_factor": 1.00,
        },
        "left lane": {
            "stakeholder_type": "adjacent_vehicle",
            "mass_kg": 1500.0,
            "vulnerability_class": "medium",  # protected [Geisslinger et al. 2023]
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, scenario: Scenario) -> MathematicalLayerResult:
        missing_fields = self._missing_required_analysis_fields(scenario)
        if missing_fields:
            return MathematicalLayerResult(
                global_metrics={
                    "runtime_status": "insufficient_data",
                    "analysis_complete": False,
                    "missing_fields": missing_fields,
                    "scene_interpretable": False,
                    "reason": "Mathematical risk analysis requires concrete action, ego-vehicle, and stakeholder fields.",
                },
                violated_rules=[],
                action_assessments=[],
                risk_score_matrix={},
                best_action_by_total_risk=None,
                best_action_by_ethical_cost=None,
            )

        global_metrics = self._compute_global_metrics(scenario)
        skipped_obstacles = self._skipped_obstacle_reasons(scenario)
        optional_missing = self._missing_optional_analysis_fields(scenario)
        global_metrics["skipped_obstacles"] = skipped_obstacles
        global_metrics["missing_optional_fields"] = optional_missing
        global_metrics["runtime_status"] = "partial_success" if optional_missing or skipped_obstacles else "success"
        global_metrics["analysis_complete"] = not optional_missing and not skipped_obstacles
        violated_rules = self._compute_rule_flags(scenario, global_metrics)

        action_assessments: list[ActionRiskAssessment] = []
        risk_score_matrix: dict[str, dict[str, float]] = {}

        for action in scenario.available_actions:
            assessment = self._analyze_action(scenario, action, global_metrics)
            action_assessments.append(assessment)
            risk_score_matrix[action] = {"ego_vehicle": round(assessment.ego_vehicle_risk, 3)}
            for sr in assessment.stakeholder_risks:
                risk_score_matrix[action][sr.stakeholder_id] = round(sr.risk_score, 3)

        if not any(assessment.stakeholder_risks for assessment in action_assessments):
            return MathematicalLayerResult(
                global_metrics={
                    **global_metrics,
                    "runtime_status": "insufficient_data",
                    "analysis_complete": False,
                    "scene_interpretable": False,
                    "reason": "No complete obstacle or occlusion-zone stakeholder could be analyzed.",
                },
                violated_rules=[],
                action_assessments=[],
                risk_score_matrix={},
                best_action_by_total_risk=None,
                best_action_by_ethical_cost=None,
            )

        # Best action by raw total risk (Bayes principle, Eq. 9)
        best_total = min(
            action_assessments,
            key=lambda a: (a.total_risk, a.action),
        ).action

        # Best action by combined ethical cost (Eq. 8, Geisslinger et al. 2023)
        best_ethical = min(
            action_assessments,
            key=lambda a: (a.ethical_costs.combined_cost, a.action),
        ).action

        return MathematicalLayerResult(
            global_metrics=global_metrics,
            violated_rules=violated_rules,
            action_assessments=action_assessments,
            risk_score_matrix=risk_score_matrix,
            best_action_by_total_risk=best_total,
            best_action_by_ethical_cost=best_ethical,
        )

    def _missing_required_analysis_fields(self, scenario: Scenario) -> list[str]:
        missing: list[str] = []

        for path, value in (
            ("ego_vehicle.speed_kmh", scenario.ego_vehicle.speed_kmh),
            ("ego_vehicle.braking_distance_m", scenario.ego_vehicle.braking_distance_m),
            ("ego_vehicle.mass_kg", scenario.ego_vehicle.mass_kg),
        ):
            if value is None:
                missing.append(path)

        if not scenario.available_actions:
            missing.append("available_actions")
        else:
            unsupported_actions = [
                action for action in scenario.available_actions if action not in self.ACTION_PROFILES
            ]
            for action in unsupported_actions:
                missing.append(f"available_actions unsupported action: {action}")
        if not scenario.obstacles and not scenario.sensor_confidence.occluded_zones:
            missing.append("obstacles")

        return missing

    def _missing_optional_analysis_fields(self, scenario: Scenario) -> list[str]:
        missing: list[str] = []
        for path, value in (
            ("environment.speed_limit_kmh", scenario.environment.speed_limit_kmh),
            ("environment.visibility_m", scenario.environment.visibility_m),
            ("environment.weather", scenario.environment.weather),
            ("environment.traffic_density", scenario.environment.traffic_density),
            ("sensor_confidence.lidar", scenario.sensor_confidence.lidar),
            ("sensor_confidence.camera", scenario.sensor_confidence.camera),
            ("sensor_confidence.radar", scenario.sensor_confidence.radar),
            (
                "sensor_confidence.overall_scene_confidence",
                scenario.sensor_confidence.overall_scene_confidence,
            ),
        ):
            if value is None or value == "":
                missing.append(path)
        return missing

    def _obstacle_missing_analysis_fields(self, obstacle: Any) -> list[str]:
        missing: list[str] = []
        for field_name, value in (
            ("id", obstacle.id),
            ("type", obstacle.type),
            ("distance_m", obstacle.distance_m),
            ("time_to_impact_s", obstacle.time_to_impact_s),
            ("trajectory", obstacle.trajectory),
            ("vulnerability_class", obstacle.vulnerability_class),
            ("mass_kg", obstacle.mass_kg),
        ):
            if value is None or value == "":
                missing.append(field_name)
        return missing

    def _skipped_obstacle_reasons(self, scenario: Scenario) -> list[dict[str, Any]]:
        skipped: list[dict[str, Any]] = []
        for index, obstacle in enumerate(scenario.obstacles):
            missing = self._obstacle_missing_analysis_fields(obstacle)
            if missing:
                skipped.append(
                    {
                        "index": index,
                        "id": obstacle.id or f"obstacles[{index}]",
                        "missing_fields": [f"obstacles[{index}].{field}" for field in missing],
                    }
                )
        return skipped

    # ------------------------------------------------------------------
    # Action-level analysis
    # ------------------------------------------------------------------

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
            if self._obstacle_missing_analysis_fields(obstacle):
                continue
            sr, ego_risk = self._analyze_obstacle(
                scenario=scenario,
                action=action,
                profile=profile,
                obstacle=obstacle,
                global_metrics=global_metrics,
            )
            stakeholder_risks.append(sr)
            ego_vehicle_risk += ego_risk
            constraint_flags.extend(sr.constraint_flags)

        for zone in scenario.sensor_confidence.occluded_zones:
            sr, ego_risk = self._analyze_occlusion_zone(
                scenario=scenario,
                action=action,
                profile=profile,
                zone=zone,
                global_metrics=global_metrics,
            )
            if sr is None:
                continue
            stakeholder_risks.append(sr)
            ego_vehicle_risk += ego_risk
            constraint_flags.extend(sr.constraint_flags)

        if (
            "swerve" in action
            and scenario.environment.speed_limit_kmh is not None
            and scenario.ego_vehicle.speed_kmh > scenario.environment.speed_limit_kmh
        ):
            constraint_flags.append("speeding_during_lateral_evasion")

        stakeholder_total_risk = sum(sr.risk_score for sr in stakeholder_risks)
        total_risk = stakeholder_total_risk + ego_vehicle_risk
        deduped_flags = list(dict.fromkeys(constraint_flags))
        stakeholder_risks.sort(key=lambda sr: sr.risk_score, reverse=True)

        ethical_costs = self._compute_ethical_costs(stakeholder_risks)

        return ActionRiskAssessment(
            action=action,
            stakeholder_risks=stakeholder_risks,
            stakeholder_total_risk=round(stakeholder_total_risk, 3),
            ego_vehicle_risk=round(ego_vehicle_risk, 3),
            total_risk=round(total_risk, 3),
            ethical_costs=ethical_costs,
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
        base_likelihood = self._base_collision_likelihood(
            distance_m=obstacle.distance_m,
            time_to_impact_s=obstacle.time_to_impact_s,
            scenario=scenario,
            global_metrics=global_metrics,
        )
        trajectory = self._canonical_trajectory(obstacle.trajectory)
        likelihood_multiplier = profile["trajectory_likelihood_multiplier"].get(trajectory, 0.75)
        collision_likelihood = _clamp(base_likelihood * likelihood_multiplier)

        impact_speed_mps = self._impact_speed_mps(
            scenario.ego_vehicle.speed_kmh,
            profile["speed_factor"],
            self._canonical_weather(scenario.environment.weather),
        )
        impact_angle_deg = profile["impact_angles"].get(trajectory, 35.0)
        protected = self._is_protected(obstacle.vulnerability_class, obstacle.type)

        # Δv for each party — Equation 1, Geisslinger et al. (2023)
        ego_delta_v, obs_delta_v = self._calc_delta_v(
            ego_mass_kg=scenario.ego_vehicle.mass_kg,
            obs_mass_kg=obstacle.mass_kg,
            impact_speed_mps=impact_speed_mps,
            impact_angle_deg=impact_angle_deg,
        )

        # H via logistic regression — Equation 2, Geisslinger et al. (2023)
        harm_estimate = self._harm_estimate(
            delta_v_mps=obs_delta_v,
            protected=protected,
        )
        ego_harm_estimate = self._harm_estimate(
            delta_v_mps=ego_delta_v,
            protected=True,   # ego vehicle always has crash structure
        )

        # R = p × H — Geisslinger et al. (2023), risk definition.
        # In this architecture, p is operationalised as collision_likelihood.
        risk_score = collision_likelihood * harm_estimate
        ego_risk_score = collision_likelihood * ego_harm_estimate

        constraint_flags: list[str] = []
        if (
            not obstacle.responsible_for_risk
            and trajectory in self.PRIORITY_TRAJECTORIES
            and collision_likelihood >= self.RIGHT_OF_WAY_LIKELIHOOD_THRESHOLD
        ):
            constraint_flags.append(f"potential_right_of_way_violation:{obstacle.id}")
        if "swerve" in action and trajectory == "crossing" and collision_likelihood >= 0.60:
            constraint_flags.append(f"high_speed_swerve_toward_crossing_stakeholder:{obstacle.id}")

        sr = StakeholderRisk(
            stakeholder_id=obstacle.id,
            stakeholder_type=obstacle.type,
            label=obstacle.type,
            source="obstacle",
            collision_likelihood=round(collision_likelihood, 3),
            harm_estimate=round(harm_estimate, 3),
            ego_harm_estimate=round(ego_harm_estimate, 3),
            risk_score=round(risk_score, 3),
            ego_risk_score=round(ego_risk_score, 3),
            impact_speed_kmh=round(impact_speed_mps * 3.6, 3),
            impact_angle_deg=round(impact_angle_deg, 3),
            delta_v_mps=round(obs_delta_v, 3),
            responsible_for_risk=obstacle.responsible_for_risk,
            constraint_flags=constraint_flags,
        )
        return sr, ego_risk_score

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
        collision_likelihood = self._occlusion_likelihood(
            exposure=exposure,
            zone_factor=zone_profile["zone_factor"],
            global_metrics=global_metrics,
            collision_unavoidable=scenario.collision_unavoidable,
        )
        impact_speed_mps = self._impact_speed_mps(
            scenario.ego_vehicle.speed_kmh,
            profile["speed_factor"],
            self._canonical_weather(scenario.environment.weather),
        )
        impact_angle_deg = profile["impact_angles"].get(zone_profile["trajectory"], 40.0)
        protected = self._is_protected(zone_profile["vulnerability_class"], zone_profile["stakeholder_type"])

        # Δv — Equation 1, Geisslinger et al. (2023)
        ego_delta_v, obs_delta_v = self._calc_delta_v(
            ego_mass_kg=scenario.ego_vehicle.mass_kg,
            obs_mass_kg=zone_profile["mass_kg"],
            impact_speed_mps=impact_speed_mps,
            impact_angle_deg=impact_angle_deg,
        )

        # H — Equation 2, Geisslinger et al. (2023)
        harm_estimate = self._harm_estimate(delta_v_mps=obs_delta_v, protected=protected)
        ego_harm_estimate = self._harm_estimate(delta_v_mps=ego_delta_v, protected=True)

        risk_score = collision_likelihood * harm_estimate
        ego_risk_score = collision_likelihood * ego_harm_estimate

        constraint_flags: list[str] = []
        if "swerve" in action and exposure >= 0.60:
            constraint_flags.append(f"steers_into_occluded_zone:{canonical_zone.replace(' ', '_')}")

        stakeholder_id = f"occlusion:{canonical_zone.replace(' ', '_')}"
        sr = StakeholderRisk(
            stakeholder_id=stakeholder_id,
            stakeholder_type=zone_profile["stakeholder_type"],
            label=canonical_zone,
            source="occlusion_zone",
            collision_likelihood=round(collision_likelihood, 3),
            harm_estimate=round(harm_estimate, 3),
            ego_harm_estimate=round(ego_harm_estimate, 3),
            risk_score=round(risk_score, 3),
            ego_risk_score=round(ego_risk_score, 3),
            impact_speed_kmh=round(impact_speed_mps * 3.6, 3),
            impact_angle_deg=round(impact_angle_deg, 3),
            delta_v_mps=round(obs_delta_v, 3),
            responsible_for_risk=None,
            constraint_flags=constraint_flags,
        )
        return sr, ego_risk_score

    # ------------------------------------------------------------------
    # Harm model — Geisslinger et al. (2023), Equations 1–2
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_delta_v(
        ego_mass_kg: float,
        obs_mass_kg: float,
        impact_speed_mps: float,
        impact_angle_deg: float,
    ) -> tuple[float, float]:
        """
        Calculate the change in velocity for each collision party.

        Implements Equation 1 from Geisslinger et al. (2023):

            Δv_A = m_B / (m_A + m_B) × √(v_A² + v_B² − 2·v_A·v_B·cos α)

        The obstacle velocity (v_B) is not available from the scenario input
        and is conservatively set to zero (stationary obstacle assumption),
        which gives Δv = v_ego and is consistent with the paper's simplified
        crash-angle mode (crash_angle_simplified=True in the reference
        implementation, risk_assessment/harm_estimation.py).

        Source: Geisslinger et al. (2023), Eq. 1; reference implementation
        risk_assessment/helpers/properties.py → calc_delta_v().
        """
        alpha_rad = radians(impact_angle_deg)
        # v_obs = 0 (conservative stationary assumption; see docstring)
        combined_speed = impact_speed_mps * abs(cos(alpha_rad / 2.0))
        total_mass = ego_mass_kg + max(obs_mass_kg, 1.0)
        ego_delta_v = (obs_mass_kg / total_mass) * combined_speed
        obs_delta_v = (ego_mass_kg / total_mass) * combined_speed
        return max(ego_delta_v, 0.0), max(obs_delta_v, 0.0)

    @staticmethod
    def _harm_estimate(delta_v_mps: float, protected: bool) -> float:
        """
        Logistic regression harm estimate H ∈ [0, 1].

        Implements Equation 2 from Geisslinger et al. (2023):

            H = 1 / (1 + exp(c₀ − c₁ · Δv))

        Two variants are used, consistent with the paper's Methods section:
          - Protected road users (vehicles): NHTSA CRSS logistic regression
            trained on MAIS3+ injury probability, angle ignored (LR1S model).
            Coefficients from harm_parameters.json → log_reg.ignore_angle.
          - Unprotected road users (pedestrians, cyclists): pedestrian-specific
            logistic regression.
            Coefficients from harm_parameters.json → pedestrian.

        Source: Geisslinger et al. (2023), Eq. 2 and Methods;
        reference implementation:
          risk_assessment/utils/logistic_regression_symmetrical.py
            → get_protected_inj_prob_log_reg_ignore_angle()
          risk_assessment/harm_estimation.py
            → get_unprotected_log_reg_harm() (pedestrian branch)
        """
        if protected:
            # H = 1 / (1 + exp(−c0 − c1·Δv))  [Eq. 2, protected branch]
            exponent = -_LR_PROTECTED_CONST - _LR_PROTECTED_SPEED * delta_v_mps
        else:
            # H = 1 / (1 + exp(c0 − c1·Δv))  [Eq. 2, unprotected/pedestrian branch]
            exponent = _LR_PEDESTRIAN_CONST - _LR_PEDESTRIAN_SPEED * delta_v_mps
        return _clamp(1.0 / (1.0 + exp(exponent)))

    # ------------------------------------------------------------------
    # Ethical cost functions — Geisslinger et al. (2023), Equations 8–12
    # ------------------------------------------------------------------

    def _compute_ethical_costs(self, stakeholder_risks: list[StakeholderRisk]) -> EthicalCosts:
        """
        Compute the three ethical cost terms and their weighted combination.

        Sources: Geisslinger et al. (2023)
          Bayes    → Eq. 9  and risk_assessment/risk_costs.py → get_bayesian_costs()
          Equality → Eq. 10 and risk_assessment/risk_costs.py → get_equality_costs()
          Maximin  → Eq. 11 and risk_assessment/risk_costs.py → get_maximin_costs()
          Combined → Eq. 8  J_Risk = w_B·J_B + w_E·J_E + w_M·J_M − w_B·J_R
        Equal weights (w_B = w_E = w_M = 1) are used as the default parameter
        set from the paper's validation runs.
        """
        bayes = self._bayes_cost(stakeholder_risks)
        equality = self._equality_cost(stakeholder_risks)
        maximin = self._maximin_cost(stakeholder_risks)
        responsibility = self._responsibility_cost(stakeholder_risks)

        # Eq. 8: J_Risk = w_B*J_B + w_E*J_E + w_M*J_M − w_B*J_R  (equal weights → 1)
        combined = bayes + equality + maximin - responsibility

        return EthicalCosts(
            bayes_cost=round(bayes, 6),
            equality_cost=round(equality, 6),
            maximin_cost=round(maximin, 6),
            responsibility_cost=round(responsibility, 6),
            combined_cost=round(combined, 6),
        )

    @staticmethod
    def _bayes_cost(stakeholder_risks: list[StakeholderRisk]) -> float:
        """
        Bayesian / utilitarian principle — minimise total risk.

        Equation 9, Geisslinger et al. (2023):
            J_B(u) = Σ R_i(u) / |S_R|

        Sums ego and stakeholder risk for every obstacle pair and normalises
        by the number of road users (×2 for ego + opponent per pair),
        matching the reference implementation's get_bayesian_costs() in
        risk_assessment/risk_costs.py.
        """
        if not stakeholder_risks:
            return 0.0
        total = sum(sr.risk_score + sr.ego_risk_score for sr in stakeholder_risks)
        n_road_users = len(stakeholder_risks) * 2   # ego + stakeholder per pair
        return total / n_road_users

    @staticmethod
    def _equality_cost(stakeholder_risks: list[StakeholderRisk]) -> float:
        """
        Equality principle — equalise risk distribution between parties.

        Equation 10, Geisslinger et al. (2023):
            J_E(u) = Σ|R_ego_i − R_obs_i| / |S_R|

        Penalises large per-pair risk disparities between the ego vehicle and
        each other road user. Reference implementation: get_equality_costs()
        in risk_assessment/risk_costs.py.
        """
        if not stakeholder_risks:
            return 0.0
        total = sum(abs(sr.ego_risk_score - sr.risk_score) for sr in stakeholder_risks)
        return total / len(stakeholder_risks)

    @staticmethod
    def _maximin_cost(
        stakeholder_risks: list[StakeholderRisk],
        scale_factor: int = 10,
        eps: float = 1e-9,
    ) -> float:
        """
        Maximin principle — minimise the greatest possible harm.

        Equation 11, Geisslinger et al. (2023):
            J_M(u) = [max H_i(u)]^γ   (γ = scale_factor)

        Focuses on harm (not risk), ignoring low-probability events when their
        risk contribution is negligible (< eps). This implements the 'priority
        for the worst-off' concept derived from Rawls' theory of justice.
        Reference implementation: get_maximin_costs() in
        risk_assessment/risk_costs.py.
        """
        if not stakeholder_risks:
            return 0.0
        # Include harm only when the corresponding risk is non-negligible
        candidate_harms = []
        for sr in stakeholder_risks:
            if sr.risk_score >= eps:
                candidate_harms.append(sr.harm_estimate)
            if sr.ego_risk_score >= eps:
                candidate_harms.append(sr.ego_harm_estimate)
        if not candidate_harms:
            return 0.0
        return max(candidate_harms) ** scale_factor

    @staticmethod
    def _responsibility_cost(stakeholder_risks: list[StakeholderRisk]) -> float:
        """
        Responsibility principle — discount risk of parties responsible for the hazard.

        Equation 12, Geisslinger et al. (2023):
            J_R(u) = Σ r_i(u) · R_i(u),  r_i ∈ [0, 1)

        When an obstacle is flagged responsible_for_risk=True, a responsibility
        factor r_i = 0.5 is applied (conservative partial discount), consistent
        with the paper's note that r_i must never reach 1 (the algorithm must
        still account for some risk from responsible parties).
        Reference implementation: get_responsibility_cost() in
        risk_assessment/risk_costs.py.
        """
        if not stakeholder_risks:
            return 0.0
        r_i = 0.5   # partial responsibility factor; r_i ∈ [0, 1)  [Eq. 12]
        return sum(
            r_i * sr.risk_score
            for sr in stakeholder_risks
            if sr.responsible_for_risk is True
        )

    # ------------------------------------------------------------------
    # Collision likelihood — deterministic surrogate model
    #
    # Architecture note
    # -----------------
    # The collision probability module in Geisslinger et al. (2023) uses a
    # CommonRoad-coupled LSTM predictor that outputs per-timestep bivariate
    # Gaussian position distributions (ref. 48 in the paper). Porting that
    # module requires CommonRoad scenario objects and per-obstacle state
    # covariance matrices, neither of which exists in this architecture's
    # Scenario representation.
    #
    # We therefore implement a deterministic surrogate that is *inspired by*
    # — but does not claim to replicate — the following works:
    #
    #   Brännström et al. (2010): model-based threat assessment using
    #       kinematic maneuver models. Their core insight is that collision
    #       likelihood is governed by whether the ego vehicle can execute a
    #       physically feasible avoidance maneuver within the available
    #       distance and time. We operationalise this with two dominant terms:
    #       braking_pressure (ratio of braking distance to obstacle distance,
    #       mirroring their braking avoidance condition in Section IV-C) and
    #       time_pressure (proximity of time-to-impact to a critical reaction
    #       window, mirroring their prediction horizon T_max and critical TTC
    #       curves in Section VI-B).
    #
    #   Berthelot et al. (2011): uncertainty-aware criticality assessment.
    #       Their motivation for probabilistic activation conditions is that
    #       sensor noise and scene ambiguity should modulate the criticality
    #       score. We incorporate this with two secondary terms:
    #       scene_uncertainty (derived from sensor fusion confidence, analogous
    #       to their covariance-driven uncertainty) and visibility_pressure
    #       (environmental perception degradation).
    #
    # The surrogate does NOT implement the statistical linearisation (Unscented
    # Transformation) or left-saturated normal distribution used in Berthelot
    # et al. (2011), as those require a covariance matrix Σ from a Kalman
    # filter. Instead, scene_uncertainty is derived deterministically from the
    # sensor fusion confidence score already present in the Scenario model.
    #
    # Weather and traffic density act as minor additive modifiers with no
    # direct literature mapping; they require no specific justification beyond
    # physical plausibility.
    #
    # Term grouping and weight rationale
    # -----------------------------------
    # Weights are calibrated so that physically dominant predictors of
    # imminent collision contribute the majority of the score, while
    # contextual and perceptual factors act as secondary modifiers:
    #
    #   GROUP 1 — Physical collision imminence (0.45 + 0.35 = 0.80)
    #       braking_pressure  (0.45): primary kinematic threat indicator,
    #           grounded in the braking avoidance feasibility condition of
    #           Brännström et al. (2010), Section IV-C.
    #       time_pressure     (0.35): time-to-impact relative to a 2.5 s
    #           reaction window, grounded in the critical TTC analysis of
    #           Brännström et al. (2010), Section VI-B.
    #
    #   GROUP 2 — Perceptual / epistemic uncertainty (0.07 + 0.05 = 0.12)
    #       scene_uncertainty (0.07): sensor fusion uncertainty, conceptually
    #           grounded in the probabilistic uncertainty motivation of
    #           Berthelot et al. (2011), Section I–II.
    #       visibility_pressure (0.05): reduced perception range; environmental
    #           analogue of increased measurement noise in Berthelot et al.
    #
    #   GROUP 3 — Environmental context (0.04 + 0.04 = 0.08)
    #       weather_pressure  (0.04): road condition degradation modifier.
    #       traffic_pressure  (0.04): scene complexity modifier.
    #
    # This weighting ensures the model behaves consistently with the physical
    # reasoning in both reference works while remaining compatible with the
    # deterministic Scenario representation used in this architecture.
    # ------------------------------------------------------------------

    def _base_collision_likelihood(
        self,
        distance_m: float,
        time_to_impact_s: float,
        scenario: Scenario,
        global_metrics: dict[str, Any],
    ) -> float:
        """
        Deterministic collision likelihood estimate for a known obstacle.

        This is a surrogate model inspired by Brännström et al. (2010) and
        Berthelot et al. (2011). It does not replicate either paper's exact
        formulation, but preserves the main physical intuition of model-based
        criticality assessment over the same core variables. See the
        class-level collision likelihood block for the full rationale and
        term-by-term literature mapping.

        Returns a scalar in [0, 1] representing collision likelihood under
        the current scenario conditions, before action-specific multipliers
        are applied in _analyze_obstacle().
        """

        # --- GROUP 1: Physical collision imminence ---

        # Braking pressure: ratio of ego braking distance to obstacle distance.
        # Conceptually mirrors the braking avoidance condition in Brännström
        # et al. (2010), Section IV-C: a collision cannot be avoided by braking
        # when the required deceleration exceeds driver capability, which
        # emerges from this same distance ratio.
        braking_pressure = _clamp(
            scenario.ego_vehicle.braking_distance_m / max(distance_m, 0.5),
            maximum=2.0,
        ) / 2.0

        # Time pressure: how close the time-to-impact is to a 2.5 s critical
        # window. Mirrors the prediction horizon and critical TTC curves in
        # Brännström et al. (2010), Section VI-B (their evaluations show
        # critical TTC in the 0.8–2.0 s range across speeds and scenarios).
        time_pressure = _clamp(
            (self.TIME_PRESSURE_WINDOW_S - time_to_impact_s) / self.TIME_PRESSURE_WINDOW_S,
        )

        # --- GROUP 2: Perceptual / epistemic uncertainty ---

        # Scene uncertainty: complement of sensor fusion confidence.
        # Conceptually grounded in the motivation of Berthelot et al. (2011),
        # Section I: sensor noise and ambiguity should modulate criticality
        # scores. Here it is derived deterministically from the Scenario's
        # sensor confidence fields rather than from a Kalman covariance matrix,
        # as the latter is not available in this architecture.
        visibility_pressure = global_metrics.get("visibility_pressure")
        scene_uncertainty = global_metrics.get("scene_uncertainty")

        # --- GROUP 3: Environmental context modifiers ---

        # Minor additive terms for road condition and scene complexity.
        # No specific literature mapping; included for physical plausibility.
        weather_pressure = (
            self.WEATHER_PRESSURE.get(self._canonical_weather(scenario.environment.weather), 0.15)
            if scenario.environment.weather
            else None
        )
        traffic_pressure = (
            self.TRAFFIC_PRESSURE.get(scenario.environment.traffic_density, 0.45)
            if scenario.environment.traffic_density
            else None
        )

        # Unavoidable-collision flag: small additive bonus when the scenario
        # is marked collision_unavoidable, ensuring the model does not
        # underestimate risk in declared unavoidable situations.
        unavoidable_bonus = 0.10 if scenario.collision_unavoidable else 0.0

        terms = [
            (0.45, braking_pressure),
            (0.35, time_pressure),
            (0.07, scene_uncertainty),
            (0.05, visibility_pressure),
            (0.04, weather_pressure),
            (0.04, traffic_pressure),
        ]
        available_terms = [(weight, value) for weight, value in terms if value is not None]
        weighted_sum = sum(weight * float(value) for weight, value in available_terms)
        weight_total = sum(weight for weight, _value in available_terms)
        if weight_total <= 0.0:
            return _clamp(unavoidable_bonus)
        return _clamp((weighted_sum / weight_total) + unavoidable_bonus)

    def _occlusion_likelihood(
        self,
        exposure: float,
        zone_factor: float,
        global_metrics: dict[str, Any],
        collision_unavoidable: bool,
    ) -> float:
        """
        Collision likelihood for a hidden stakeholder in an occluded zone.

        Exposure (action-specific, from ACTION_PROFILES) represents how much
        the chosen maneuver directs the ego vehicle toward the occluded zone.
        It is scaled by zone_factor (zone-specific hazard weight) and modulated
        by the same uncertainty and visibility terms used in
        _base_collision_likelihood(), consistent with the epistemic uncertainty
        motivation of Berthelot et al. (2011).

        The result is capped at 0.40 to reflect that occluded stakeholders are
        hypothetical: their existence is uncertain, so their contribution to
        total risk should remain bounded relative to confirmed obstacles.
        """
        scene_uncertainty = global_metrics.get("scene_uncertainty")
        visibility_pressure = global_metrics.get("visibility_pressure")
        unavoidable_bonus = 0.02 if collision_unavoidable and exposure >= 0.50 else 0.0
        likelihood_terms = [0.05 + unavoidable_bonus]
        if scene_uncertainty is not None:
            likelihood_terms.append(0.55 * float(scene_uncertainty))
        if visibility_pressure is not None:
            likelihood_terms.append(0.25 * float(visibility_pressure))
        likelihood = exposure * zone_factor * sum(likelihood_terms)
        return _clamp(likelihood, maximum=0.40)

    # ------------------------------------------------------------------
    # Global metrics and rule flags
    # ------------------------------------------------------------------

    def _compute_global_metrics(self, scenario: Scenario) -> dict[str, Any]:
        sensor_terms = [
            (0.40, scenario.sensor_confidence.overall_scene_confidence),
            (0.20, scenario.sensor_confidence.lidar),
            (0.20, scenario.sensor_confidence.camera),
            (0.20, scenario.sensor_confidence.radar),
        ]
        available_sensor_terms = [(weight, value) for weight, value in sensor_terms if value is not None]
        sensor_fusion_confidence = None
        if available_sensor_terms:
            weight_total = sum(weight for weight, _value in available_sensor_terms)
            sensor_fusion_confidence = _clamp(
                sum(weight * value for weight, value in available_sensor_terms) / weight_total,
            )
        scene_uncertainty = (
            1.0 - sensor_fusion_confidence if sensor_fusion_confidence is not None else None
        )

        speed_limit_delta_kmh = (
            scenario.ego_vehicle.speed_kmh - scenario.environment.speed_limit_kmh
            if scenario.environment.speed_limit_kmh is not None
            else None
        )

        visibility_pressure = (
            _clamp(
                (self.VISIBILITY_REFERENCE_M - scenario.environment.visibility_m)
                / self.VISIBILITY_REFERENCE_M,
            )
            if scenario.environment.visibility_m is not None
            else None
        )

        closest_obstacle_distance_m = min(
            (obstacle.distance_m for obstacle in scenario.obstacles if obstacle.distance_m is not None),
            default=float("inf"),
        )
        braking_margin_m = (
            closest_obstacle_distance_m - scenario.ego_vehicle.braking_distance_m
            if closest_obstacle_distance_m != float("inf")
            else float("inf")
        )

        return {
            "sensor_fusion_confidence": (
                round(sensor_fusion_confidence, 3)
                if sensor_fusion_confidence is not None else None
            ),
            "scene_uncertainty": round(scene_uncertainty, 3) if scene_uncertainty is not None else None,
            "speed_limit_delta_kmh": (
                round(speed_limit_delta_kmh, 3) if speed_limit_delta_kmh is not None else None
            ),
            "visibility_pressure": (
                round(visibility_pressure, 3) if visibility_pressure is not None else None
            ),
            "canonical_weather": self._canonical_weather(scenario.environment.weather),
            "canonical_road_type": self._canonical_road_type(scenario.environment.road_type),
            "closest_obstacle_distance_m": (
                round(closest_obstacle_distance_m, 3)
                if closest_obstacle_distance_m != float("inf") else None
            ),
            "braking_margin_m": (
                round(braking_margin_m, 3)
                if braking_margin_m != float("inf") else None
            ),
            # Epistemic uncertainty flag: below 0.85 sensor confidence the
            # scene interpretation is considered unreliable.
            "scene_interpretable": (
                sensor_fusion_confidence >= 0.85
                if sensor_fusion_confidence is not None else False
            ),
        }

    def _compute_rule_flags(self, scenario: Scenario, global_metrics: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        if global_metrics["speed_limit_delta_kmh"] is not None and global_metrics["speed_limit_delta_kmh"] > 0:
            flags.append("speed_limit_exceeded")
        if (
            scenario.environment.visibility_m is not None
            and scenario.ego_vehicle.braking_distance_m > scenario.environment.visibility_m
        ):
            flags.append("cannot_stop_within_visible_distance")
        return flags

    # ------------------------------------------------------------------
    # Speed helpers
    # ------------------------------------------------------------------

    def _impact_speed_mps(self, ego_speed_kmh: float, speed_factor: float, weather: str) -> float:
        degraded_factor = min(
            1.0,
            speed_factor * self.WEATHER_SPEED_DEGRADATION.get(weather, 1.05),
        )
        return max(0.1, self._kmh_to_mps(ego_speed_kmh) * degraded_factor)

    def _action_profile(self, action: str) -> dict[str, Any]:
        return self.ACTION_PROFILES.get(action, self.ACTION_PROFILES["brake_straight"])

    def _canonical_weather(self, weather: str) -> str:
        return canonicalize_weather(weather)

    def _canonical_road_type(self, road_type: str) -> str:
        return canonicalize_road_type(road_type)

    def _canonical_trajectory(self, trajectory: str) -> str:
        return canonicalize_trajectory(trajectory)

    def _is_protected(self, vulnerability_class: str, stakeholder_type: str | None = None) -> bool:
        vulnerability_key = vulnerability_class.strip().lower()
        if self.VULNERABILITY_TO_PROTECTED.get(vulnerability_key) is False:
            return False
        if stakeholder_type:
            type_key = stakeholder_type.strip().lower()
            if self.VULNERABILITY_TO_PROTECTED.get(type_key) is False:
                return False
        if vulnerability_key in self.VULNERABILITY_TO_PROTECTED:
            return self.VULNERABILITY_TO_PROTECTED[vulnerability_key]
        if stakeholder_type:
            type_key = stakeholder_type.strip().lower()
            if type_key in self.VULNERABILITY_TO_PROTECTED:
                return self.VULNERABILITY_TO_PROTECTED[type_key]
        return self.VULNERABILITY_TO_PROTECTED.get(vulnerability_key, True)

    def _canonicalize_zone(self, zone: str) -> str:
        return zone.strip().lower().replace("_", " ")

    @staticmethod
    def _kmh_to_mps(speed_kmh: float) -> float:
        return speed_kmh / 3.6
