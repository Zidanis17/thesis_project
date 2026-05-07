from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ValidationError(ValueError):
    pass


def _ensure_number(name: str, value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a number") from exc


def _ensure_string(name: str, value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ensure_bool(name: str, value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    else:
        raise ValidationError(f"{name} must be a boolean")
    raise ValidationError(f"{name} must be a boolean")


def _ensure_string_list(name: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list of strings")
    output: list[str] = []
    for item in value:
        text = _ensure_string(name, item)
        if text:
            output.append(text)
    return output


@dataclass(slots=True)
class EgoVehicle:
    speed_kmh: float | None
    acceleration_ms2: float | None
    heading_deg: float | None
    lane_position: str
    braking_distance_m: float | None
    mass_kg: float | None
    passenger_at_risk: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EgoVehicle":
        if not isinstance(data, dict):
            data = {}
        return cls(
            speed_kmh=_ensure_number("ego_vehicle.speed_kmh", data.get("speed_kmh")),
            acceleration_ms2=_ensure_number("ego_vehicle.acceleration_ms2", data.get("acceleration_ms2")),
            heading_deg=_ensure_number("ego_vehicle.heading_deg", data.get("heading_deg")),
            lane_position=_ensure_string("ego_vehicle.lane_position", data.get("lane_position")),
            braking_distance_m=_ensure_number("ego_vehicle.braking_distance_m", data.get("braking_distance_m")),
            mass_kg=_ensure_number("ego_vehicle.mass_kg", data.get("mass_kg")),
            passenger_at_risk=_ensure_bool("ego_vehicle.passenger_at_risk", data.get("passenger_at_risk")),
        )


@dataclass(slots=True)
class Environment:
    road_type: str
    speed_limit_kmh: float | None
    weather: str
    visibility_m: float | None
    time_of_day: str
    traffic_density: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Environment":
        if not isinstance(data, dict):
            data = {}
        return cls(
            road_type=_ensure_string("environment.road_type", data.get("road_type")),
            speed_limit_kmh=_ensure_number("environment.speed_limit_kmh", data.get("speed_limit_kmh")),
            weather=_ensure_string("environment.weather", data.get("weather")),
            visibility_m=_ensure_number("environment.visibility_m", data.get("visibility_m")),
            time_of_day=_ensure_string("environment.time_of_day", data.get("time_of_day")),
            traffic_density=_ensure_string("environment.traffic_density", data.get("traffic_density")),
        )


@dataclass(slots=True)
class Obstacle:
    id: str
    type: str
    distance_m: float | None
    relative_speed_kmh: float | None
    time_to_impact_s: float | None
    trajectory: str
    vulnerability_class: str
    mass_kg: float | None
    responsible_for_risk: bool | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Obstacle":
        if not isinstance(data, dict):
            data = {}
        return cls(
            id=_ensure_string("obstacles[].id", data.get("id")),
            type=_ensure_string("obstacles[].type", data.get("type")),
            distance_m=_ensure_number("obstacles[].distance_m", data.get("distance_m")),
            relative_speed_kmh=_ensure_number("obstacles[].relative_speed_kmh", data.get("relative_speed_kmh")),
            time_to_impact_s=_ensure_number("obstacles[].time_to_impact_s", data.get("time_to_impact_s")),
            trajectory=_ensure_string("obstacles[].trajectory", data.get("trajectory")),
            vulnerability_class=_ensure_string("obstacles[].vulnerability_class", data.get("vulnerability_class")),
            mass_kg=_ensure_number("obstacles[].mass_kg", data.get("mass_kg")),
            responsible_for_risk=_ensure_bool("obstacles[].responsible_for_risk", data.get("responsible_for_risk")),
        )


@dataclass(slots=True)
class SensorConfidence:
    lidar: float | None
    camera: float | None
    radar: float | None
    overall_scene_confidence: float | None
    occluded_zones: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorConfidence":
        if not isinstance(data, dict):
            data = {}
        return cls(
            lidar=_ensure_number("sensor_confidence.lidar", data.get("lidar")),
            camera=_ensure_number("sensor_confidence.camera", data.get("camera")),
            radar=_ensure_number("sensor_confidence.radar", data.get("radar")),
            overall_scene_confidence=_ensure_number(
                "sensor_confidence.overall_scene_confidence",
                data.get("overall_scene_confidence"),
            ),
            occluded_zones=_ensure_string_list("sensor_confidence.occluded_zones", data.get("occluded_zones", [])),
        )


@dataclass(slots=True)
class Scenario:
    ego_vehicle: EgoVehicle
    environment: Environment
    obstacles: list[Obstacle]
    sensor_confidence: SensorConfidence
    available_actions: list[str]
    collision_unavoidable: bool | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        if not isinstance(data, dict):
            raise ValidationError("scenario must be a dictionary")
        obstacles_raw = data.get("obstacles", [])
        if obstacles_raw is None:
            obstacles_raw = []
        if not isinstance(obstacles_raw, list):
            raise ValidationError("obstacles must be a list")
        available_actions = _ensure_string_list("available_actions", data.get("available_actions", []))
        return cls(
            ego_vehicle=EgoVehicle.from_dict(data.get("ego_vehicle", {})),
            environment=Environment.from_dict(data.get("environment", {})),
            obstacles=[Obstacle.from_dict(item) for item in obstacles_raw],
            sensor_confidence=SensorConfidence.from_dict(data.get("sensor_confidence", {})),
            available_actions=available_actions,
            collision_unavoidable=_ensure_bool("collision_unavoidable", data.get("collision_unavoidable")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParserResult:
    scenario: Scenario
    input_mode: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.scenario.to_dict()
