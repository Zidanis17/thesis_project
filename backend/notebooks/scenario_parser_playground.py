# %% [markdown]
# # Scenario Parser Playground
#
# This notebook documents the deterministic parsing layer that feeds the rest of
# the thesis pipeline.
#
# It covers two supported entry modes:
# - direct structured JSON, which is the main evaluation path
# - natural-language descriptions, which are normalized into the JSON schema first
#
# The goal is to make parser behavior inspectable before risk scoring and ethical
# reasoning are added on top.

# %%
from pathlib import Path
import json
import sys

root = Path.cwd()
if not (root / "thesis").exists():
    root = root.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from thesis import DeterministicScenarioParser, ScenarioParseError

parser = DeterministicScenarioParser()


def dump_result(result):
    print(json.dumps(result.to_dict(), indent=2))


print(f"Project root: {root}")

# %% [markdown]
# ## 1. Structured JSON input
#
# This should be the main thesis evaluation path because it avoids ambiguity in
# entity extraction and lets downstream layers operate on a stable schema.

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

structured_result = parser.parse(structured_payload)
dump_result(structured_result)

# %% [markdown]
# ## 2. Natural-language input
#
# This tests the convenience layer. The parser should recover the same semantic
# scenario representation even though the input starts as free text.

# %%
nl_scenario = (
    "An autonomous vehicle weighing 1800 kg is traveling at 60 km/h in the center lane on a "
    "residential road with a 50 km/h speed limit. It is braking at 2.1 m/s2 and has a braking "
    "distance of 42.5 m. Weather is clear, visibility is 120 m, it is daytime, and traffic is low. "
    "A child pedestrian is crossing 10.2 m ahead with time to impact 0.61 s. "
    "A parked vehicle is 6.5 m ahead with time to impact 0.39 s. "
    "Lidar confidence is 0.97, camera 0.91, radar 0.95, and overall scene confidence is 0.93. "
    "The left sidewalk is occluded. Available actions are brake straight, swerve left, swerve right, "
    "and brake and swerve left. Collision is unavoidable."
)

nl_result = parser.parse(nl_scenario)
dump_result(nl_result)

# %% [markdown]
# ## 3. Missing optional fields and warnings
#
# This section is useful for documenting the parser's deterministic defaults.
# Those warnings are important later because the reasoning layer should know
# when a scenario includes inferred rather than explicitly observed values.

# %%
incomplete_payload = {
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

incomplete_result = parser.parse(incomplete_payload)
dump_result(incomplete_result)

print("Warnings:")
for warning in incomplete_result.warnings:
    print("-", warning)

# %% [markdown]
# ## 4. Markdown-wrapped JSON input
#
# This handles scenarios pasted from notes, prompts, or thesis drafts where the
# JSON may already be fenced in markdown.

# %%
markdown_json = """```json
{
  "ego_vehicle": {"speed_kmh": 40, "acceleration_ms2": -4.0},
  "environment": {"road_type": "urban"},
  "obstacles": [{"type": "vehicle", "distance_m": 15}]
}
```"""

markdown_result = parser.parse(markdown_json)
dump_result(markdown_result)

# %% [markdown]
# ## 5. Quick batch experiments
#
# Use this section for fast parser regression checks while you expand the schema
# or tune extraction rules.

# %%
batch_inputs = [
    {
        "name": "simple_structured",
        "payload": {
            "ego_vehicle": {"speed_kmh": 30, "acceleration_ms2": -3.0},
            "environment": {"road_type": "school_zone"},
            "obstacles": [{"type": "child_pedestrian", "distance_m": 8}],
        },
    },
    {
        "name": "simple_text",
        "payload": (
            "An autonomous vehicle is traveling at 35 km/h in a school zone. "
            "A child is crossing 7 m ahead. Available actions are brake straight and swerve right."
        ),
    },
]

for item in batch_inputs:
    print(f"\n=== {item['name']} ===")
    try:
        result = parser.parse(item["payload"])
        print("mode:", result.input_mode)
        print("warnings:", len(result.warnings))
        print("actions:", result.scenario.available_actions)
        print("obstacles:", [obs.type for obs in result.scenario.obstacles])
    except ScenarioParseError as exc:
        print("parse error:", exc)

# %% [markdown]
# ## 6. Notes
#
# The parser now feeds directly into the mathematical layer through the
# `ScenarioPipeline`. Keep this notebook focused on extraction and
# normalization behavior rather than downstream ethical analysis.
