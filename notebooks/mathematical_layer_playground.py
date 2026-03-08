# %% [markdown]
# # Mathematical Layer Playground
#
# This notebook isolates the deterministic mathematical layer used in the thesis.
#
# It assumes the scenario is already available as structured JSON and focuses on:
# - global metrics and rule flags
# - per-action risk totals
# - stakeholder-level risk contributions, including occluded zones
#
# The intent is to make the mathematical substrate inspectable before the RAG,
# reasoning, and critic stages are introduced.

# %%
from pathlib import Path
import json
import sys

root = Path.cwd()
if not (root / "thesis").exists():
    root = root.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from thesis import DeterministicMathematicalLayer, DeterministicScenarioParser

parser = DeterministicScenarioParser()
math_layer = DeterministicMathematicalLayer()


def print_json(payload):
    print(json.dumps(payload, indent=2))


def summarize_actions(result):
    rows = []
    for assessment in result.action_assessments:
        rows.append(
            {
                "action": assessment.action,
                "stakeholder_total_risk": assessment.stakeholder_total_risk,
                "ego_vehicle_risk": assessment.ego_vehicle_risk,
                "total_risk": assessment.total_risk,
                "constraint_flags": assessment.constraint_flags,
            }
        )
    rows.sort(key=lambda item: item["total_risk"])
    return rows


print(f"Project root: {root}")

# %% [markdown]
# ## 1. Structured scenario
#
# This mirrors the primary evaluation setup for the mathematical layer. The
# parser still runs here to guarantee the input is normalized before analysis.

# %%
structured_payload = {
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

scenario = parser.parse(structured_payload).scenario
analysis = math_layer.analyze(scenario)

print("Normalized scenario:")
print_json(scenario.to_dict())

# %% [markdown]
# ## 2. Global metrics and rule flags
#
# These quantities are derived once per scenario and then reused across action
# analyses. They encode uncertainty, braking feasibility, visibility pressure,
# and basic legal-rule checks such as speeding.

# %%
print("Global metrics:")
print_json(analysis.global_metrics)

print("\nScenario-level rule flags:")
print_json(analysis.violated_rules)

# %% [markdown]
# ## 3. Action ranking
#
# Each candidate action aggregates stakeholder risk, ego-vehicle risk, and
# action-specific constraint flags. This is the object the later ethical
# reasoning layer can interpret rather than recomputing raw physics.

# %%
action_summary = summarize_actions(analysis)
print_json(action_summary)
print(f"\nBest action by total risk: {analysis.best_action_by_total_risk}")

# %% [markdown]
# ## 4. Stakeholder-level breakdown for one action
#
# Use this section to inspect why an action received its total score and which
# stakeholders dominate the risk profile.

# %%
by_action = {assessment.action: assessment.to_dict() for assessment in analysis.action_assessments}
print_json(by_action["brake_straight"])

# %% [markdown]
# ## 5. Full risk matrix
#
# The matrix includes explicit obstacles, occlusion-derived pseudo-stakeholders,
# and an aggregated `ego_vehicle` entry for each action. This is the main
# machine-readable output that the later reasoning stage should consume.

# %%
print_json(analysis.risk_score_matrix)
