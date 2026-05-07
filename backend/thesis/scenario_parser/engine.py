from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Mapping, Protocol

from ..core.models import ParserResult, Scenario, ValidationError
from ..core.normalization import (
    canonicalize_action,
    canonicalize_obstacle_type,
    canonicalize_occlusion_zone,
    canonicalize_road_type,
    canonicalize_time_of_day,
    canonicalize_traffic_density,
    canonicalize_trajectory,
    canonicalize_vulnerability,
    canonicalize_weather,
)
from .llm_agent import LLMScenarioParserAgent, LLMScenarioParserAgentResult

__all__ = ["DeterministicScenarioParser", "LLMScenarioParserAgent", "ScenarioParseError"]


class ScenarioParseError(ValueError):
    pass


class ScenarioParserAgent(Protocol):
    def extract(self, text: str) -> LLMScenarioParserAgentResult:
        ...


class DeterministicScenarioParser:
    """
    Parse scenarios in exactly two modes:

    - JSON mode: accept a dict or JSON object string and normalize only the
      provided structured fields.
    - natural-language mode: ask the parser LLM agent to translate the text
      into the same JSON schema, then normalize that structured result.

    Missing facts remain blank. The parser may canonicalize names and coerce
    directly provided scalar values, but it does not supply domain defaults or
    merge text heuristics into LLM output.
    """

    def __init__(self, *, llm_agent: ScenarioParserAgent | None = None) -> None:
        self.llm_agent = llm_agent if llm_agent is not None else LLMScenarioParserAgent()

    def parse(self, payload: str | Mapping[str, Any]) -> ParserResult:
        if isinstance(payload, Mapping):
            return self.parse_dict(dict(payload))
        if not isinstance(payload, str):
            raise ScenarioParseError("Scenario input must be a dict or a string")

        stripped = self._strip_json_wrappers(payload.strip())
        if not stripped:
            raise ScenarioParseError("Scenario input cannot be empty")

        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return self.parse_text(stripped)
        if not isinstance(decoded, dict):
            raise ScenarioParseError("Structured scenario input must decode to a JSON object")
        return self.parse_dict(decoded)

    def parse_dict(self, payload: dict[str, Any]) -> ParserResult:
        normalized, warnings = self._normalize_scenario(payload)
        return self._build_result(normalized, "structured_json", warnings)

    def parse_text(self, text: str) -> ParserResult:
        if self.llm_agent is None:
            raise ScenarioParseError("Natural-language parsing requires a scenario parser LLM agent")

        extraction = self.llm_agent.extract(text)
        if not extraction.runtime_available:
            detail = f": {extraction.runtime_error}" if extraction.runtime_error else ""
            raise ScenarioParseError(f"Natural-language parser agent failed{detail}")

        normalized, warnings = self._normalize_scenario(extraction.payload)
        warnings.insert(
            0,
            (
                "scenario_parser_agent translated natural-language input "
                f"with provider={extraction.provider}, model={extraction.model_name}"
            ),
        )
        return self._build_result(normalized, "natural_language", warnings)

    def _build_result(self, normalized: dict[str, Any], input_mode: str, warnings: list[str]) -> ParserResult:
        try:
            scenario = Scenario.from_dict(normalized)
        except ValidationError as exc:
            raise ScenarioParseError(str(exc)) from exc
        return ParserResult(scenario=scenario, input_mode=input_mode, warnings=warnings)

    def _normalize_scenario(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        ego = self._normalize_ego_vehicle(payload.get("ego_vehicle"), warnings)
        environment = self._normalize_environment(payload.get("environment"), warnings)
        sensor_confidence = self._normalize_sensor_confidence(payload.get("sensor_confidence"), warnings)
        obstacles = self._normalize_obstacles(payload.get("obstacles"), warnings)
        actions = self._normalize_actions(payload.get("available_actions"), warnings)
        collision_unavoidable = self._normalize_collision_unavoidable(
            payload.get("collision_unavoidable"),
            warnings,
        )
        return (
            {
                "ego_vehicle": ego,
                "environment": environment,
                "obstacles": obstacles,
                "sensor_confidence": sensor_confidence,
                "available_actions": actions,
                "collision_unavoidable": collision_unavoidable,
            },
            warnings,
        )

    def _normalize_ego_vehicle(self, raw: Any, warnings: list[str]) -> dict[str, Any]:
        data = dict(raw or {}) if isinstance(raw, Mapping) else {}
        speed = self._coerce_number(data.get("speed_kmh"))
        acceleration = self._coerce_number(data.get("acceleration_ms2"))
        heading = self._coerce_number(data.get("heading_deg"))
        lane_position = self._coerce_string(data.get("lane_position")) or ""
        braking_distance = self._coerce_number(data.get("braking_distance_m"))
        mass = self._coerce_number(data.get("mass_kg"))
        passenger_at_risk = self._coerce_bool(data.get("passenger_at_risk"))

        for path, value in (
            ("ego_vehicle.speed_kmh", speed),
            ("ego_vehicle.acceleration_ms2", acceleration),
            ("ego_vehicle.heading_deg", heading),
            ("ego_vehicle.lane_position", lane_position),
            ("ego_vehicle.braking_distance_m", braking_distance),
            ("ego_vehicle.mass_kg", mass),
            ("ego_vehicle.passenger_at_risk", passenger_at_risk),
        ):
            if value is None or value == "":
                warnings.append(f"{path} left blank")

        return {
            "speed_kmh": self._round_or_none(speed),
            "acceleration_ms2": self._round_or_none(acceleration),
            "heading_deg": self._round_or_none(heading),
            "lane_position": lane_position,
            "braking_distance_m": self._round_or_none(braking_distance),
            "mass_kg": self._round_or_none(mass),
            "passenger_at_risk": passenger_at_risk,
        }

    def _normalize_environment(self, raw: Any, warnings: list[str]) -> dict[str, Any]:
        data = dict(raw or {}) if isinstance(raw, Mapping) else {}
        road_type = canonicalize_road_type(self._coerce_string(data.get("road_type")) or "")
        speed_limit = self._coerce_number(data.get("speed_limit_kmh"))
        weather = canonicalize_weather(self._coerce_string(data.get("weather")) or "")
        visibility = self._coerce_number(data.get("visibility_m"))
        time_of_day = canonicalize_time_of_day(self._coerce_string(data.get("time_of_day")) or "")
        traffic_density = canonicalize_traffic_density(
            self._coerce_string(data.get("traffic_density")) or ""
        )

        for path, value in (
            ("environment.road_type", road_type),
            ("environment.speed_limit_kmh", speed_limit),
            ("environment.weather", weather),
            ("environment.visibility_m", visibility),
            ("environment.time_of_day", time_of_day),
            ("environment.traffic_density", traffic_density),
        ):
            if value is None or value == "":
                warnings.append(f"{path} left blank")

        return {
            "road_type": road_type,
            "speed_limit_kmh": self._round_or_none(speed_limit),
            "weather": weather,
            "visibility_m": self._round_or_none(visibility),
            "time_of_day": time_of_day,
            "traffic_density": traffic_density,
        }

    def _normalize_sensor_confidence(self, raw: Any, warnings: list[str]) -> dict[str, Any]:
        data = dict(raw or {}) if isinstance(raw, Mapping) else {}
        lidar = self._coerce_confidence(data.get("lidar"))
        camera = self._coerce_confidence(data.get("camera"))
        radar = self._coerce_confidence(data.get("radar"))
        overall = self._coerce_confidence(data.get("overall_scene_confidence"))

        for path, value in (
            ("sensor_confidence.lidar", lidar),
            ("sensor_confidence.camera", camera),
            ("sensor_confidence.radar", radar),
            ("sensor_confidence.overall_scene_confidence", overall),
        ):
            if value is None:
                warnings.append(f"{path} left blank")

        raw_zones = data.get("occluded_zones") or []
        if not isinstance(raw_zones, list):
            warnings.append("sensor_confidence.occluded_zones was not a list and was cleared")
            raw_zones = []
        occluded_zones = []
        for zone in raw_zones:
            canonical = canonicalize_occlusion_zone(zone)
            if canonical and canonical not in occluded_zones:
                occluded_zones.append(canonical)

        return {
            "lidar": self._round_or_none(lidar),
            "camera": self._round_or_none(camera),
            "radar": self._round_or_none(radar),
            "overall_scene_confidence": self._round_or_none(overall),
            "occluded_zones": occluded_zones,
        }

    def _normalize_obstacles(self, raw: Any, warnings: list[str]) -> list[dict[str, Any]]:
        if raw is None or raw == []:
            warnings.append("obstacles left blank")
            return []
        if not isinstance(raw, list):
            warnings.append("obstacles was not a list and was cleared")
            return []

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ScenarioParseError("Each obstacle must be an object")

            obstacle = dict(item)
            obstacle_type = canonicalize_obstacle_type(self._coerce_string(obstacle.get("type")) or "")
            distance_m = self._coerce_number(obstacle.get("distance_m"))
            relative_speed_kmh = self._coerce_number(obstacle.get("relative_speed_kmh"))
            time_to_impact_s = self._coerce_number(obstacle.get("time_to_impact_s"))
            trajectory = canonicalize_trajectory(self._coerce_string(obstacle.get("trajectory")) or "")
            vulnerability = canonicalize_vulnerability(
                self._coerce_string(obstacle.get("vulnerability_class")) or ""
            )
            mass_kg = self._coerce_number(obstacle.get("mass_kg"))
            responsible_for_risk = self._coerce_bool(obstacle.get("responsible_for_risk"))
            obstacle_id = self._coerce_string(obstacle.get("id")) or f"obj_{index + 1:02d}"

            for field_name, value in (
                ("type", obstacle_type),
                ("distance_m", distance_m),
                ("relative_speed_kmh", relative_speed_kmh),
                ("time_to_impact_s", time_to_impact_s),
                ("trajectory", trajectory),
                ("vulnerability_class", vulnerability),
                ("mass_kg", mass_kg),
                ("responsible_for_risk", responsible_for_risk),
            ):
                if value is None or value == "":
                    warnings.append(f"obstacles[{index}].{field_name} left blank")

            normalized.append(
                {
                    "id": obstacle_id,
                    "type": obstacle_type,
                    "distance_m": self._round_or_none(distance_m),
                    "relative_speed_kmh": self._round_or_none(relative_speed_kmh),
                    "time_to_impact_s": self._round_or_none(time_to_impact_s),
                    "trajectory": trajectory,
                    "vulnerability_class": vulnerability,
                    "mass_kg": self._round_or_none(mass_kg),
                    "responsible_for_risk": responsible_for_risk,
                }
            )
        return normalized

    def _normalize_actions(self, raw: Any, warnings: list[str]) -> list[str]:
        if raw is None or raw == []:
            warnings.append("available_actions left blank")
            return []
        if not isinstance(raw, list):
            warnings.append("available_actions was not a list and was cleared")
            return []

        normalized: list[str] = []
        for index, action in enumerate(raw):
            canonical = canonicalize_action(action)
            if canonical is None:
                if self._coerce_string(action):
                    warnings.append(f"available_actions[{index}] was not recognized and was ignored")
                continue
            if canonical not in normalized:
                normalized.append(canonical)

        if not normalized:
            warnings.append("available_actions left blank because no canonical actions were recognized")
        return normalized

    def _normalize_collision_unavoidable(self, raw: Any, warnings: list[str]) -> bool | None:
        value = self._coerce_bool(raw)
        if value is None:
            warnings.append("collision_unavoidable left blank")
        return value

    def _coerce_number(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_string(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _coerce_bool(self, value: Any) -> bool | None:
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
        return None

    def _coerce_confidence(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            value = value.strip()
            if value.endswith("%"):
                try:
                    return max(0.0, min(1.0, float(value[:-1]) / 100.0))
                except ValueError:
                    return None
        number = self._coerce_number(value)
        if number is None:
            return None
        if number > 1.0:
            number = number / 100.0
        return max(0.0, min(1.0, number))

    def _round_or_none(self, value: float | None) -> float | None:
        return round(value, 3) if value is not None else None

    def to_dict(self, result: ParserResult) -> dict[str, Any]:
        return result.to_dict()

    def to_json(self, result: ParserResult) -> str:
        return json.dumps(asdict(result.scenario), indent=2)

    def _strip_json_wrappers(self, text: str) -> str:
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        lowered = text.lower()
        if lowered.startswith("json"):
            candidate = text[4:].lstrip()
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate
        return text
