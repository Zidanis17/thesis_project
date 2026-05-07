from __future__ import annotations

from typing import Any

__all__ = [
    "ACTION_NAMES",
    "ROAD_TYPE_CANONICAL_ALIASES",
    "TRAJECTORY_CANONICAL_ALIASES",
    "VRU_TYPES",
    "VRU_VULNERABILITY_CLASSES",
    "VULNERABILITY_TO_PROTECTED",
    "WEATHER_CANONICAL_ALIASES",
    "canonicalize_action",
    "canonicalize_obstacle_type",
    "canonicalize_occlusion_zone",
    "canonicalize_road_type",
    "canonicalize_time_of_day",
    "canonicalize_traffic_density",
    "canonicalize_trajectory",
    "canonicalize_vulnerability",
    "normalize_token",
]


ACTION_NAMES = {
    "brake_straight",
    "swerve_left",
    "swerve_right",
    "brake_swerve_left",
    "brake_swerve_right",
    "maintain_lane",
}

ROAD_TYPE_CANONICAL_ALIASES = {
    "residential road": "residential",
    "residential street": "residential",
    "residential_street": "residential",
    "urban road": "urban",
    "urban street": "urban",
    "urban arterial": "urban",
    "urban_arterial": "urban",
    "urban intersection": "intersection",
    "urban_intersection": "intersection",
    "school zone": "school_zone",
    "parking lot": "parking_lot",
    "ring road": "highway",
    "ring_road": "highway",
    "highway merge": "highway",
    "highway_merge": "highway",
    "hospital access road": "hospital_zone",
    "hospital_access_road": "hospital_zone",
}

WEATHER_CANONICAL_ALIASES = {
    "rainy": "rain",
    "light rain": "rain",
    "light_rain": "rain",
    "foggy": "fog",
    "snowy": "snow",
    "stormy": "storm",
    "overcast": "clear",
}

TRAJECTORY_CANONICAL_ALIASES = {
    "same lane": "same_lane",
    "same_lane_braking": "same_lane",
    "same lane braking": "same_lane",
    "same_lane_edge": "same_lane",
    "same lane edge": "same_lane",
    "same_lane_stationary": "stationary",
    "same lane stationary": "stationary",
    "ahead in lane": "same_lane",
    "lawful_crosswalk": "crossing",
    "lawful crosswalk": "crossing",
    "crossing_jaywalking": "crossing",
    "crossing jaywalking": "crossing",
    "crossing_from_between_parked_cars": "crossing",
    "crossing from between parked cars": "crossing",
    "merge_from_ramp": "merging",
    "merge from ramp": "merging",
    "partial_lane_obstruction": "stationary",
    "partial lane obstruction": "stationary",
    "right_shoulder_stationary": "stationary",
    "right shoulder stationary": "stationary",
    "right_edge_barrier": "stationary",
    "right edge barrier": "stationary",
    "left_edge_barrier": "stationary",
    "left edge barrier": "stationary",
    "left_fixed_barrier": "stationary",
    "left fixed barrier": "stationary",
    "right_side_fixed": "stationary",
    "right side fixed": "stationary",
    "adjacent_left_lane_moving": "same_lane",
    "adjacent left lane moving": "same_lane",
    "parked": "stationary",
    "stopped": "stationary",
}

OBSTACLE_TYPE_ALIASES = {
    "child": "child_pedestrian",
    "child pedestrian": "child_pedestrian",
    "elderly": "elderly_pedestrian",
    "elderly pedestrian": "elderly_pedestrian",
    "adult pedestrian": "adult_pedestrian",
    "pedestrian adult": "adult_pedestrian",
    "pedestrian_adult": "adult_pedestrian",
    "parked car": "parked_vehicle",
    "parked vehicle": "parked_vehicle",
    "motor cyclist": "motorcyclist",
    "car": "vehicle",
    "deer": "animal",
    "dog": "animal",
}

VRU_TYPES = {
    "pedestrian",
    "adult_pedestrian",
    "child_pedestrian",
    "elderly_pedestrian",
    "cyclist",
    "motorcyclist",
    "hidden_pedestrian",
    "hidden_cyclist",
}

VRU_VULNERABILITY_CLASSES = {"high", "child", "elderly", "cyclist", "pedestrian"}

VULNERABILITY_TO_PROTECTED: dict[str, bool] = {
    "high": False,
    "child": False,
    "elderly": False,
    "pedestrian": False,
    "adult_pedestrian": False,
    "child_pedestrian": False,
    "elderly_pedestrian": False,
    "cyclist": False,
    "motorcyclist": False,
    "hidden_pedestrian": False,
    "hidden_cyclist": False,
    "medium": True,
    "low": True,
    "vehicle": True,
    "vehicle_sedan": True,
    "vehicle_suv": True,
    "vehicle_hatchback": True,
    "delivery_van": True,
    "parked_vehicle": True,
    "guardrail": True,
    "traffic_light_pole": True,
}


def normalize_token(value: Any, *, spaces: bool = False) -> str:
    if value is None:
        return ""
    token = str(value).strip().lower().replace("-", " ")
    if spaces:
        return "_".join(token.replace("_", " ").split()).replace("_", " ")
    return "_".join(token.replace("_", " ").split())


def _spaced(value: Any) -> str:
    return normalize_token(value, spaces=True)


def canonicalize_action(value: Any) -> str | None:
    token = _spaced(value)
    aliases = {
        "brake straight": "brake_straight",
        "straight braking": "brake_straight",
        "brake in lane": "brake_straight",
        "hard brake": "brake_straight",
        "full braking": "brake_straight",
        "swerve left": "swerve_left",
        "steer left": "swerve_left",
        "swerve right": "swerve_right",
        "steer right": "swerve_right",
        "brake swerve left": "brake_swerve_left",
        "brake and swerve left": "brake_swerve_left",
        "brake steer left": "brake_swerve_left",
        "brake and steer left": "brake_swerve_left",
        "brake swerve right": "brake_swerve_right",
        "brake and swerve right": "brake_swerve_right",
        "brake steer right": "brake_swerve_right",
        "brake and steer right": "brake_swerve_right",
        "maintain lane": "maintain_lane",
    }
    compact = normalize_token(value)
    if compact in ACTION_NAMES:
        return compact
    return aliases.get(token)


def canonicalize_road_type(value: Any) -> str:
    token = _spaced(value)
    if not token:
        return ""
    if token in ROAD_TYPE_CANONICAL_ALIASES:
        return ROAD_TYPE_CANONICAL_ALIASES[token]
    compact = normalize_token(value)
    if compact in ROAD_TYPE_CANONICAL_ALIASES:
        return ROAD_TYPE_CANONICAL_ALIASES[compact]
    return compact


def canonicalize_weather(value: Any) -> str:
    token = _spaced(value)
    if not token:
        return ""
    if token in WEATHER_CANONICAL_ALIASES:
        return WEATHER_CANONICAL_ALIASES[token]
    compact = normalize_token(value)
    return WEATHER_CANONICAL_ALIASES.get(compact, compact)


def canonicalize_time_of_day(value: Any) -> str:
    token = _spaced(value)
    if not token:
        return ""
    aliases = {
        "day": "daytime",
        "day time": "daytime",
        "daylight": "daytime",
    }
    return aliases.get(token, normalize_token(value))


def canonicalize_traffic_density(value: Any) -> str:
    token = _spaced(value)
    if not token:
        return ""
    if token in {"light", "low", "low traffic", "light traffic"}:
        return "low"
    if token in {"moderate", "medium", "moderate traffic", "medium traffic"}:
        return "medium"
    if token in {"heavy", "dense", "high", "heavy traffic", "dense traffic", "high traffic"}:
        return "high"
    return normalize_token(value)


def canonicalize_obstacle_type(value: Any) -> str:
    token = _spaced(value)
    if not token:
        return ""
    if token in OBSTACLE_TYPE_ALIASES:
        return OBSTACLE_TYPE_ALIASES[token]
    compact = normalize_token(value)
    return OBSTACLE_TYPE_ALIASES.get(compact, compact)


def canonicalize_trajectory(value: Any) -> str:
    token = _spaced(value)
    if not token:
        return ""
    if token in TRAJECTORY_CANONICAL_ALIASES:
        return TRAJECTORY_CANONICAL_ALIASES[token]
    compact = normalize_token(value)
    return TRAJECTORY_CANONICAL_ALIASES.get(compact, compact)


def canonicalize_vulnerability(value: Any) -> str:
    token = _spaced(value)
    if not token:
        return ""
    if "high" in token:
        return "high"
    if "medium" in token or "moderate" in token:
        return "medium"
    if "low" in token:
        return "low"
    return normalize_token(value)


def canonicalize_occlusion_zone(value: Any) -> str:
    return _spaced(value)
