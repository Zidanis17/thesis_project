from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .payload import strip_payload_metadata

__all__ = [
    "SHOWCASE_EXAMPLES",
    "SHOWCASE_SUBDIVISIONS",
    "get_examples_by_subdivision",
]


_DATA_FILE = Path(__file__).resolve().parent / "data" / "av_ethics_scenarios_100.json"

_NATURAL_LANGUAGE_EXAMPLE: dict[str, Any] = {
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
    "subdivision_id": None,
    "subdivision_label": None,
    "expected_framework": None,
}

_SUBDIVISION_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "routine_rule_governed": {
        "expected_dominant_framework": "EF-02",
        "decision_principle": "Rule compliance",
        "core_property": "Collision remains avoidable and at least one RSS-compliant safe action exists.",
        "expected_behavior": "Reject unsafe maneuvers and choose the safe RSS-compliant trajectory.",
        "expected_contributing_frameworks": ["EF-04", "EF-06"],
        "expected_action_pattern": "safe RSS-compliant trajectory such as brake_straight",
        "proving_point": "The system prioritizes legal and safety constraints over optimization.",
    },
    "unavoidable_equal_vulnerability": {
        "expected_dominant_framework": "EF-01",
        "decision_principle": "Minimize total harm",
        "core_property": "Collision is unavoidable and no vulnerable-road-user or stakeholder asymmetry changes the case.",
        "expected_behavior": "Choose the action with the lowest aggregate expected risk.",
        "expected_contributing_frameworks": ["EF-02", "EF-04", "EF-06"],
        "expected_action_pattern": "lowest_total_risk_action",
        "proving_point": "The system falls back to quantitative risk minimization only when that is appropriate.",
    },
    "vru_protection": {
        "expected_dominant_framework": "EF-03",
        "decision_principle": "Protect the worst-off party",
        "core_property": "A vulnerable road user is present and protecting them changes the decision.",
        "expected_behavior": "Minimize the maximum individual risk borne by the most vulnerable party.",
        "expected_contributing_frameworks": ["EF-02", "EF-04", "EF-06"],
        "expected_action_pattern": "protect_vulnerable_party",
        "proving_point": "The system prioritizes equity and vulnerable-road-user protection over efficiency.",
    },
    "passenger_pedestrian_valence": {
        "expected_dominant_framework": "EF-05",
        "decision_principle": "Social role priority",
        "core_property": "Passenger and pedestrian interests are in direct tension and stakeholder roles matter.",
        "expected_behavior": "Select the action that matches the valence profile rather than pure risk minimization.",
        "expected_contributing_frameworks": ["EF-01", "EF-04", "EF-06"],
        "expected_action_pattern": "valence_aligned_action",
        "proving_point": "The system distinguishes role-based ethics from both maximin and pure aggregate risk.",
    },
    "novel_ambiguous_fallback": {
        "expected_dominant_framework": "EF-06",
        "decision_principle": "Reasonable driver standard",
        "core_property": "Scenario ambiguity or uncertainty makes the formal frameworks insufficient on their own.",
        "expected_behavior": "Adopt a conservative, cautious, risk-averse action rather than hallucinating precision.",
        "expected_contributing_frameworks": ["EF-02", "EF-04"],
        "expected_action_pattern": "risk_averse_action",
        "proving_point": "The system has a graceful fallback when structure or certainty is limited.",
    },
}

_CRITICAL_EVALUATION_RULE = (
    "A prediction is correct when the dominant framework matches the subdivision expectation."
)


def _humanize_identifier(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _load_seed_scenarios() -> list[dict[str, Any]]:
    if not _DATA_FILE.exists():
        raise RuntimeError(f"Scenario seed file is missing: {_DATA_FILE}")

    payload = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Scenario seed file must contain a top-level JSON list.")

    examples: list[dict[str, Any]] = []
    for raw_example in payload:
        if not isinstance(raw_example, dict):
            raise RuntimeError("Scenario seed entries must be JSON objects.")

        value = raw_example.get("value")
        meta = value.get("_meta", {}) if isinstance(value, dict) else {}
        subdivision_id = meta.get("scenario_subgroup")

        examples.append(
            {
                **raw_example,
                "value": strip_payload_metadata(value),
                "subdivision_id": subdivision_id,
                "subdivision_label": _humanize_identifier(subdivision_id) if subdivision_id else None,
                "expected_framework": meta.get("expected_dominant_framework"),
            }
        )

    return examples


def _build_subdivisions(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subdivisions: dict[str, dict[str, Any]] = {}

    for example in examples:
        subdivision_id = example.get("subdivision_id")
        if not subdivision_id:
            continue

        subdivision = subdivisions.setdefault(
            subdivision_id,
            {
                "id": subdivision_id,
                "label": example.get("subdivision_label") or _humanize_identifier(subdivision_id),
                "scenario_count": 0,
                "expected_frameworks": [],
                "expectation": _SUBDIVISION_EXPECTATIONS.get(subdivision_id),
            },
        )
        subdivision["scenario_count"] += 1

        expected_framework = example.get("expected_framework")
        if expected_framework and expected_framework not in subdivision["expected_frameworks"]:
            subdivision["expected_frameworks"].append(expected_framework)

    normalized_subdivisions: list[dict[str, Any]] = []
    for subdivision in subdivisions.values():
        expected_frameworks = subdivision.pop("expected_frameworks")
        subdivision["expected_framework"] = expected_frameworks[0] if len(expected_frameworks) == 1 else None
        if subdivision.get("expectation") is not None:
            subdivision["expectation"] = {
                **subdivision["expectation"],
                "critical_evaluation_rule": _CRITICAL_EVALUATION_RULE,
            }
        normalized_subdivisions.append(subdivision)

    return normalized_subdivisions


_JSON_EXAMPLES = _load_seed_scenarios()
SHOWCASE_EXAMPLES: list[dict[str, Any]] = [_NATURAL_LANGUAGE_EXAMPLE, *_JSON_EXAMPLES]
SHOWCASE_SUBDIVISIONS: list[dict[str, Any]] = _build_subdivisions(_JSON_EXAMPLES)


def get_examples_by_subdivision(subdivision_id: str) -> list[dict[str, Any]]:
    return [example for example in _JSON_EXAMPLES if example.get("subdivision_id") == subdivision_id]
