from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Iterable, Mapping

from ..models import ParserResult, Scenario, ValidationError

__all__ = ["DeterministicScenarioParser", "ScenarioParseError"]


class ScenarioParseError(ValueError):
    pass


class DeterministicScenarioParser:
    DEFAULT_EGO_MASS_KG = 1800.0
    DEFAULT_BRAKING_DECEL_MS2 = 7.0
    DEFAULT_SENSOR_CONFIDENCE = 0.90
    DEFAULT_ACTIONS = [
        "brake_straight",
        "swerve_left",
        "swerve_right",
        "brake_swerve_left",
    ]
    ROAD_SPEED_LIMITS_KMH = {
        "residential": 50.0,
        "urban": 50.0,
        "intersection": 50.0,
        "school_zone": 30.0,
        "rural": 80.0,
        "highway": 120.0,
        "motorway": 130.0,
        "parking_lot": 20.0,
    }
    WEATHER_DEFAULT_VISIBILITY_M = {
        "clear": 120.0,
        "rain": 80.0,
        "fog": 40.0,
        "snow": 50.0,
        "storm": 35.0,
    }
    TYPE_DEFAULTS = {
        "child_pedestrian": {"mass_kg": 30.0, "vulnerability_class": "high", "trajectory": "crossing"},
        "adult_pedestrian": {"mass_kg": 75.0, "vulnerability_class": "high", "trajectory": "crossing"},
        "pedestrian": {"mass_kg": 75.0, "vulnerability_class": "high", "trajectory": "crossing"},
        "elderly_pedestrian": {"mass_kg": 70.0, "vulnerability_class": "high", "trajectory": "crossing"},
        "cyclist": {"mass_kg": 90.0, "vulnerability_class": "high", "trajectory": "crossing"},
        "motorcyclist": {"mass_kg": 250.0, "vulnerability_class": "high", "trajectory": "same_lane"},
        "parked_vehicle": {"mass_kg": 1500.0, "vulnerability_class": "low", "trajectory": "stationary"},
        "vehicle": {"mass_kg": 1500.0, "vulnerability_class": "medium", "trajectory": "same_lane"},
        "truck": {"mass_kg": 8000.0, "vulnerability_class": "low", "trajectory": "same_lane"},
        "bus": {"mass_kg": 12000.0, "vulnerability_class": "low", "trajectory": "same_lane"},
        "animal": {"mass_kg": 80.0, "vulnerability_class": "medium", "trajectory": "crossing"},
    }
    ACTION_ALIASES = [
        ("brake_swerve_left", r"\bbrake(?:\s+and)?\s+(?:swerve|steer)\s+left\b"),
        ("brake_swerve_right", r"\bbrake(?:\s+and)?\s+(?:swerve|steer)\s+right\b"),
        ("brake_straight", r"\bbrake straight\b"),
        ("brake_straight", r"\bstraight braking\b"),
        ("brake_straight", r"\bbrake in lane\b"),
        ("brake_straight", r"\bhard brake\b"),
        ("brake_straight", r"\bfull braking\b"),
        ("maintain_lane", r"\bmaintain lane\b"),
        ("maintain_lane", r"\bmaintain_lane\b"),
        ("swerve_left", r"\bswerve left\b"),
        ("swerve_left", r"\bsteer left\b"),
        ("swerve_right", r"\bswerve right\b"),
        ("swerve_right", r"\bsteer right\b"),
    ]
    OCCLUSION_ZONES = [
        "left sidewalk",
        "right sidewalk",
        "crosswalk",
        "bike lane",
        "left lane",
        "right lane",
        "intersection",
    ]
    ROAD_TYPE_PATTERNS = [
        ("school_zone", r"\bschool zone\b"),
        ("parking_lot", r"\bparking lot\b"),
        ("residential", r"\bresidential\b"),
        ("intersection", r"\bintersection\b"),
        ("motorway", r"\bmotorway\b"),
        ("highway", r"\bhighway\b"),
        ("urban", r"\burban\b"),
        ("rural", r"\brural\b"),
    ]
    WEATHER_PATTERNS = [
        ("clear", r"\bclear\b"),
        ("rain", r"\brain(?:y)?\b"),
        ("fog", r"\bfog(?:gy)?\b"),
        ("snow", r"\bsnow(?:y)?\b"),
        ("storm", r"\bstorm(?:y)?\b"),
    ]
    TIME_OF_DAY_PATTERNS = [
        ("daytime", r"\bdaytime\b"),
        ("night", r"\bnight\b"),
        ("dawn", r"\bdawn\b"),
        ("dusk", r"\bdusk\b"),
    ]
    TRAFFIC_DENSITY_PATTERNS = [
        ("low", r"\b(?:low|light)\s+traffic\b|\btraffic density\s+low\b|\btraffic\s+is\s+low\b"),
        ("medium", r"\b(?:moderate|medium)\s+traffic\b|\btraffic density\s+(?:moderate|medium)\b|\btraffic\s+is\s+(?:moderate|medium)\b"),
        ("high", r"\b(?:dense|heavy|high)\s+traffic\b|\btraffic density\s+high\b|\btraffic\s+is\s+high\b"),
    ]
    HEADING_CARDINALS = {
        "north": 0.0,
        "east": 90.0,
        "south": 180.0,
        "west": 270.0,
    }
    OBJECT_ALIASES = [
        ("child_pedestrian", r"\bchild pedestrian\b"),
        ("elderly_pedestrian", r"\belderly pedestrian\b"),
        ("adult_pedestrian", r"\badult pedestrian\b"),
        ("parked_vehicle", r"\bparked vehicle\b"),
        ("parked_vehicle", r"\bparked car\b"),
        ("motorcyclist", r"\bmotorcyclist\b"),
        ("cyclist", r"\bcyclist\b"),
        ("pedestrian", r"\bpedestrian\b"),
        ("child_pedestrian", r"\bchild\b"),
        ("elderly_pedestrian", r"\belderly\b"),
        ("truck", r"\btruck\b"),
        ("bus", r"\bbus\b"),
        ("animal", r"\banimal\b"),
        ("animal", r"\bdeer\b"),
        ("animal", r"\bdog\b"),
        ("vehicle", r"\bvehicle\b"),
        ("vehicle", r"\bcar\b"),
    ]
    CLAUSE_BOUNDARY_RE = re.compile(r"(?<!\d)[.!?;](?!\d)")
    NUMBER_RE = r"(?P<value>-?\d+(?:\.\d+)?)"

    def __init__(self, strict: bool = True, default_actions: Iterable[str] | None = None) -> None:
        self.strict = strict
        self.default_actions = list(default_actions or self.DEFAULT_ACTIONS)

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
        partial = {
            "ego_vehicle": self._extract_ego_vehicle(text),
            "environment": self._extract_environment(text),
            "obstacles": self._extract_obstacles(text),
            "sensor_confidence": self._extract_sensor_confidence(text),
            "available_actions": self._extract_available_actions(text),
            "collision_unavoidable": self._extract_collision_unavoidable(text),
        }
        normalized, warnings = self._normalize_scenario(partial)
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
        obstacles = self._normalize_obstacles(payload.get("obstacles"), ego["speed_kmh"], warnings)
        actions = self._normalize_actions(payload.get("available_actions"), warnings)
        collision_unavoidable = self._normalize_collision_unavoidable(
            payload.get("collision_unavoidable"),
            ego,
            obstacles,
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
        data = dict(raw or {})
        speed = self._coerce_number(data.get("speed_kmh"))
        if speed is None:
            raise ScenarioParseError("Missing required field: ego_vehicle.speed_kmh")

        acceleration = self._coerce_number(data.get("acceleration_ms2"))
        if acceleration is None:
            acceleration = 0.0
            warnings.append("ego_vehicle.acceleration_ms2 defaulted to 0.0")

        heading = self._coerce_number(data.get("heading_deg"))
        if heading is None:
            heading = 0.0
            warnings.append("ego_vehicle.heading_deg defaulted to 0.0")

        lane_position = self._coerce_string(data.get("lane_position")) or "center"
        if "lane_position" not in data:
            warnings.append("ego_vehicle.lane_position defaulted to center")

        braking_distance = self._coerce_number(data.get("braking_distance_m"))
        if braking_distance is None:
            braking_distance = self._calculate_braking_distance(speed, acceleration)
            warnings.append("ego_vehicle.braking_distance_m inferred from speed and braking defaults")

        mass = self._coerce_number(data.get("mass_kg"))
        if mass is None:
            mass = self.DEFAULT_EGO_MASS_KG
            warnings.append(f"ego_vehicle.mass_kg defaulted to {self.DEFAULT_EGO_MASS_KG}")

        return {
            "speed_kmh": round(speed, 3),
            "acceleration_ms2": round(acceleration, 3),
            "heading_deg": round(heading, 3),
            "lane_position": lane_position,
            "braking_distance_m": round(braking_distance, 3),
            "mass_kg": round(mass, 3),
        }

    def _normalize_environment(self, raw: Any, warnings: list[str]) -> dict[str, Any]:
        data = dict(raw or {})
        road_type = self._coerce_string(data.get("road_type"))
        if road_type is None:
            raise ScenarioParseError("Missing required field: environment.road_type")

        speed_limit = self._coerce_number(data.get("speed_limit_kmh"))
        if speed_limit is None:
            speed_limit = self.ROAD_SPEED_LIMITS_KMH.get(road_type, 50.0)
            warnings.append(f"environment.speed_limit_kmh defaulted from road_type={road_type}")

        weather = self._coerce_string(data.get("weather")) or "clear"
        if "weather" not in data:
            warnings.append("environment.weather defaulted to clear")

        time_of_day = self._coerce_string(data.get("time_of_day")) or "daytime"
        if "time_of_day" not in data:
            warnings.append("environment.time_of_day defaulted to daytime")

        traffic_density = self._coerce_string(data.get("traffic_density")) or "low"
        if "traffic_density" not in data:
            warnings.append("environment.traffic_density defaulted to low")

        visibility = self._coerce_number(data.get("visibility_m"))
        if visibility is None:
            visibility = self.WEATHER_DEFAULT_VISIBILITY_M.get(weather, 100.0)
            warnings.append(f"environment.visibility_m defaulted from weather={weather}")

        return {
            "road_type": road_type,
            "speed_limit_kmh": round(speed_limit, 3),
            "weather": weather,
            "visibility_m": round(visibility, 3),
            "time_of_day": time_of_day,
            "traffic_density": traffic_density,
        }

    def _normalize_sensor_confidence(self, raw: Any, warnings: list[str]) -> dict[str, Any]:
        data = dict(raw or {})
        overall = self._coerce_confidence(data.get("overall_scene_confidence"))
        lidar = self._coerce_confidence(data.get("lidar"))
        camera = self._coerce_confidence(data.get("camera"))
        radar = self._coerce_confidence(data.get("radar"))

        if lidar is None:
            lidar = overall if overall is not None else self.DEFAULT_SENSOR_CONFIDENCE
            warnings.append("sensor_confidence.lidar defaulted from overall/default confidence")
        if camera is None:
            camera = overall if overall is not None else self.DEFAULT_SENSOR_CONFIDENCE
            warnings.append("sensor_confidence.camera defaulted from overall/default confidence")
        if radar is None:
            radar = overall if overall is not None else self.DEFAULT_SENSOR_CONFIDENCE
            warnings.append("sensor_confidence.radar defaulted from overall/default confidence")
        if overall is None:
            overall = (lidar + camera + radar) / 3.0
            warnings.append("sensor_confidence.overall_scene_confidence inferred from sensor scores")

        occluded_zones = data.get("occluded_zones") or []
        if not isinstance(occluded_zones, list):
            raise ScenarioParseError("sensor_confidence.occluded_zones must be a list")

        return {
            "lidar": round(lidar, 3),
            "camera": round(camera, 3),
            "radar": round(radar, 3),
            "overall_scene_confidence": round(overall, 3),
            "occluded_zones": [str(zone).strip() for zone in occluded_zones if str(zone).strip()],
        }

    def _normalize_obstacles(
        self,
        raw: Any,
        ego_speed_kmh: float,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list) or not raw:
            raise ScenarioParseError("At least one obstacle is required")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, Mapping):
                raise ScenarioParseError("Each obstacle must be an object")

            obstacle = dict(item)
            obstacle_type = self._coerce_string(obstacle.get("type"))
            if obstacle_type is None:
                raise ScenarioParseError(f"Missing required field: obstacles[{index - 1}].type")

            default_spec = self.TYPE_DEFAULTS.get(obstacle_type, self.TYPE_DEFAULTS["vehicle"])

            distance_m = self._coerce_number(obstacle.get("distance_m"))
            relative_speed_kmh = self._coerce_number(obstacle.get("relative_speed_kmh"))
            time_to_impact_s = self._coerce_number(obstacle.get("time_to_impact_s"))
            trajectory = self._coerce_string(obstacle.get("trajectory"))

            if trajectory is None:
                trajectory = default_spec["trajectory"]
                warnings.append(f"obstacles[{index - 1}].trajectory defaulted from type={obstacle_type}")

            if relative_speed_kmh is None:
                relative_speed_kmh = ego_speed_kmh
                warnings.append(f"obstacles[{index - 1}].relative_speed_kmh defaulted from ego speed")

            if time_to_impact_s is None and distance_m is not None and relative_speed_kmh > 0:
                time_to_impact_s = distance_m / self._kmh_to_mps(relative_speed_kmh)
                warnings.append(f"obstacles[{index - 1}].time_to_impact_s inferred from distance and speed")

            if distance_m is None and time_to_impact_s is not None and relative_speed_kmh > 0:
                distance_m = time_to_impact_s * self._kmh_to_mps(relative_speed_kmh)
                warnings.append(f"obstacles[{index - 1}].distance_m inferred from time to impact and speed")

            if distance_m is None:
                raise ScenarioParseError(f"Missing required field: obstacles[{index - 1}].distance_m")
            if time_to_impact_s is None:
                raise ScenarioParseError(f"Missing required field: obstacles[{index - 1}].time_to_impact_s")

            vulnerability = self._coerce_string(obstacle.get("vulnerability_class")) or default_spec["vulnerability_class"]
            if "vulnerability_class" not in obstacle:
                warnings.append(f"obstacles[{index - 1}].vulnerability_class defaulted from type={obstacle_type}")

            mass_kg = self._coerce_number(obstacle.get("mass_kg"))
            if mass_kg is None:
                mass_kg = default_spec["mass_kg"]
                warnings.append(f"obstacles[{index - 1}].mass_kg defaulted from type={obstacle_type}")

            responsible_for_risk = self._coerce_bool(obstacle.get("responsible_for_risk"))
            if responsible_for_risk is None:
                responsible_for_risk = False
                warnings.append(f"obstacles[{index - 1}].responsible_for_risk defaulted to false")

            obstacle_id = self._coerce_string(obstacle.get("id")) or f"obj_{index:02d}"
            normalized.append(
                {
                    "id": obstacle_id,
                    "type": obstacle_type,
                    "distance_m": round(distance_m, 3),
                    "relative_speed_kmh": round(relative_speed_kmh, 3),
                    "time_to_impact_s": round(time_to_impact_s, 3),
                    "trajectory": trajectory,
                    "vulnerability_class": vulnerability,
                    "mass_kg": round(mass_kg, 3),
                    "responsible_for_risk": responsible_for_risk,
                }
            )
        return normalized

    def _normalize_actions(self, raw: Any, warnings: list[str]) -> list[str]:
        if not raw:
            warnings.append("available_actions defaulted to parser defaults")
            return list(self.default_actions)

        if not isinstance(raw, list):
            raise ScenarioParseError("available_actions must be a list")

        normalized: list[str] = []
        for action in raw:
            canonical = self._canonicalize_action(str(action))
            if canonical and canonical not in normalized:
                normalized.append(canonical)

        if not normalized:
            if self.strict:
                raise ScenarioParseError("No valid available_actions were found")
            warnings.append("available_actions defaulted because no canonical actions were recognized")
            return list(self.default_actions)
        return normalized

    def _normalize_collision_unavoidable(
        self,
        raw: Any,
        ego: dict[str, Any],
        obstacles: list[dict[str, Any]],
        warnings: list[str],
    ) -> bool:
        explicit = self._coerce_bool(raw)
        if explicit is not None:
            return explicit

        closest_distance = min(obstacle["distance_m"] for obstacle in obstacles)
        warnings.append("collision_unavoidable inferred from braking distance and nearest obstacle")
        return ego["braking_distance_m"] >= closest_distance

    def _extract_ego_vehicle(self, text: str) -> dict[str, Any]:
        braking_distance = self._extract_distance_value(
            text,
            [
                r"\bbraking distance(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*m(?:eters?)?\b",
                r"\bstopping distance(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*m(?:eters?)?\b",
            ],
        )
        return {
            "speed_kmh": self._extract_speed_kmh(text),
            "acceleration_ms2": self._extract_acceleration_ms2(text),
            "heading_deg": self._extract_heading_deg(text),
            "lane_position": self._extract_lane_position(text),
            "braking_distance_m": braking_distance,
            "mass_kg": self._extract_mass_kg(text, ["ego vehicle", "autonomous vehicle", "vehicle", "car"]),
        }

    def _extract_environment(self, text: str) -> dict[str, Any]:
        road_type = None
        for name, pattern in self.ROAD_TYPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                road_type = name
                break

        weather = None
        for name, pattern in self.WEATHER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                weather = name
                break

        time_of_day = None
        for name, pattern in self.TIME_OF_DAY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                time_of_day = name
                break

        traffic_density = None
        for name, pattern in self.TRAFFIC_DENSITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                traffic_density = name
                break

        return {
            "road_type": road_type,
            "speed_limit_kmh": self._extract_speed_limit_kmh(text),
            "weather": weather,
            "visibility_m": self._extract_distance_value(
                text,
                [r"\bvisibility(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*m(?:eters?)?\b"],
            ),
            "time_of_day": time_of_day,
            "traffic_density": traffic_density,
        }

    def _extract_obstacles(self, text: str) -> list[dict[str, Any]]:
        matches = self._match_obstacle_mentions(text)
        obstacles: list[dict[str, Any]] = []
        for index, match in enumerate(matches, start=1):
            snippet = self._slice_obstacle_region(text, matches, index - 1)
            sentence = self._slice_sentence(text, match["start"])
            obstacles.append(self._build_obstacle_from_text(snippet, sentence, match["type"], index))
        return obstacles

    def _extract_sensor_confidence(self, text: str) -> dict[str, Any]:
        overall = self._extract_confidence(text, "overall scene")
        if overall is None:
            overall = self._extract_confidence(text, "scene")
        return {
            "lidar": self._extract_confidence(text, "lidar"),
            "camera": self._extract_confidence(text, "camera"),
            "radar": self._extract_confidence(text, "radar"),
            "overall_scene_confidence": overall,
            "occluded_zones": self._extract_occluded_zones(text),
        }

    def _extract_available_actions(self, text: str) -> list[str]:
        matches: list[dict[str, Any]] = []
        for canonical, pattern in self.ACTION_ALIASES:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({"start": match.start(), "end": match.end(), "action": canonical})
        matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))

        selected: list[dict[str, Any]] = []
        for match in matches:
            if selected and match["start"] < selected[-1]["end"]:
                continue
            selected.append(match)

        actions: list[str] = []
        for match in selected:
            if match["action"] not in actions:
                actions.append(match["action"])
        return actions

    def _extract_collision_unavoidable(self, text: str) -> bool | None:
        patterns = [
            r"\bcollision is unavoidable\b",
            r"\bunavoidable collision\b",
            r"\bno safe option\b",
            r"\bcannot avoid(?: a)? collision\b",
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return None

    def _match_obstacle_mentions(self, text: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for obstacle_type, pattern in self.OBJECT_ALIASES:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if obstacle_type == "vehicle" and self._looks_like_ego_reference(text, match.start()):
                    continue
                matches.append({"start": match.start(), "end": match.end(), "type": obstacle_type})
        matches.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))

        selected: list[dict[str, Any]] = []
        for match in matches:
            if selected and match["start"] < selected[-1]["end"]:
                continue
            selected.append(match)
        return selected

    def _slice_obstacle_region(self, text: str, matches: list[dict[str, Any]], index: int) -> str:
        start = matches[index]["start"]
        sentence_end = self._next_clause_boundary(text, start)
        region_end = sentence_end
        if index + 1 < len(matches) and matches[index + 1]["start"] < sentence_end:
            region_end = matches[index + 1]["start"]
        return text[start:region_end]

    def _slice_sentence(self, text: str, position: int) -> str:
        previous = max(
            text.rfind(".", 0, position),
            text.rfind("?", 0, position),
            text.rfind("!", 0, position),
            text.rfind(";", 0, position),
        )
        next_boundary = self._next_clause_boundary(text, position)
        return text[previous + 1 : next_boundary]

    def _next_clause_boundary(self, text: str, position: int) -> int:
        match = self.CLAUSE_BOUNDARY_RE.search(text, position)
        return match.start() if match else len(text)

    def _build_obstacle_from_text(
        self,
        snippet: str,
        sentence: str,
        obstacle_type: str,
        index: int,
    ) -> dict[str, Any]:
        distance_m = self._extract_distance_value(snippet, self._distance_patterns())
        if distance_m is None:
            distance_m = self._extract_distance_value(sentence, self._distance_patterns())

        relative_speed_kmh = self._extract_relative_speed_kmh(snippet)
        if relative_speed_kmh is None:
            relative_speed_kmh = self._extract_relative_speed_kmh(sentence)

        time_to_impact_s = self._extract_time_to_impact_s(snippet)
        if time_to_impact_s is None:
            time_to_impact_s = self._extract_time_to_impact_s(sentence)

        trajectory = self._extract_trajectory(snippet, obstacle_type)
        if trajectory is None:
            trajectory = self._extract_trajectory(sentence, obstacle_type)

        vulnerability_class = self._extract_vulnerability_class(snippet)
        if vulnerability_class is None:
            vulnerability_class = self._extract_vulnerability_class(sentence)

        labels = [obstacle_type.replace("_", " ")]
        mass_kg = self._extract_mass_kg(snippet, labels)
        if mass_kg is None:
            mass_kg = self._extract_mass_kg(sentence, labels)

        responsible_for_risk = self._extract_responsibility(snippet)
        if responsible_for_risk is None:
            responsible_for_risk = self._extract_responsibility(sentence)

        return {
            "id": f"obj_{index:02d}",
            "type": obstacle_type,
            "distance_m": distance_m,
            "relative_speed_kmh": relative_speed_kmh,
            "time_to_impact_s": time_to_impact_s,
            "trajectory": trajectory,
            "vulnerability_class": vulnerability_class,
            "mass_kg": mass_kg,
            "responsible_for_risk": responsible_for_risk,
        }

    def _distance_patterns(self) -> list[str]:
        return [
            r"\b" + self.NUMBER_RE + r"\s*m(?:eters?)?\s+(?:ahead|away|in front)\b",
            r"\bdistance(?:\s+of|\s+is)?\s*" + self.NUMBER_RE + r"\s*m(?:eters?)?\b",
            r"\b" + self.NUMBER_RE + r"\s*m(?:eters?)?\b",
        ]

    def _extract_speed_kmh(self, text: str) -> float | None:
        patterns = [
            r"\b(?:ego vehicle|autonomous vehicle|self-driving car|av)\b[^.!?;]{0,60}?\b(?:speed|(?:travelling|traveling|moving|going)(?:\s+at)?|at)\s*"
            + self.NUMBER_RE
            + r"\s*(?P<unit>km/h|kph|mph|m/s)\b",
            r"\b(?:vehicle|car)\b[^.!?;]{0,30}?\b(?:speed|(?:travelling|traveling|moving|going)(?:\s+at)?|at)\s*"
            + self.NUMBER_RE
            + r"\s*(?P<unit>km/h|kph|mph|m/s)\b",
            r"\bspeed(?:\s+is|\s+of)?\s*"
            + self.NUMBER_RE
            + r"\s*(?P<unit>km/h|kph|mph|m/s)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._convert_speed_to_kmh(float(match.group("value")), match.group("unit"))
        return None

    def _extract_speed_limit_kmh(self, text: str) -> float | None:
        patterns = [
            r"\bspeed limit(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*(?P<unit>km/h|kph|mph|m/s)\b",
            r"\bposted limit(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*(?P<unit>km/h|kph|mph|m/s)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._convert_speed_to_kmh(float(match.group("value")), match.group("unit"))
        return None

    def _extract_relative_speed_kmh(self, text: str) -> float | None:
        patterns = [
            r"\brelative speed(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*(?P<unit>km/h|kph|mph|m/s)\b",
            r"\bclosing speed(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*(?P<unit>km/h|kph|mph|m/s)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._convert_speed_to_kmh(float(match.group("value")), match.group("unit"))
        return None

    def _extract_acceleration_ms2(self, text: str) -> float | None:
        patterns = [
            r"\b(?:acceleration|accelerating at)\s*" + self.NUMBER_RE + r"\s*m/s(?:2|\^2)\b",
            r"\b(?:deceleration|decelerating at|braking at)\s*" + self.NUMBER_RE + r"\s*m/s(?:2|\^2)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            value = float(match.group("value"))
            if "deceler" in match.group(0).lower() or "braking" in match.group(0).lower():
                return -abs(value)
            return value
        return None

    def _extract_heading_deg(self, text: str) -> float | None:
        degree_pattern = r"\bheading(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*(?:deg|degrees?)\b"
        match = re.search(degree_pattern, text, re.IGNORECASE)
        if match:
            return float(match.group("value"))

        for direction, degrees in self.HEADING_CARDINALS.items():
            if re.search(rf"\bheading\s+{direction}\b", text, re.IGNORECASE):
                return degrees
            if re.search(
                rf"\b(?:travelling|traveling|moving)\s+{direction}\b",
                text,
                re.IGNORECASE,
            ):
                return degrees
        return None

    def _extract_lane_position(self, text: str) -> str | None:
        patterns = {
            "center": r"\b(?:center|centre|middle)\s+lane\b|\blane(?:\s+position)?\s+center\b",
            "left": r"\bleft\s+lane\b|\blane(?:\s+position)?\s+left\b",
            "right": r"\bright\s+lane\b|\blane(?:\s+position)?\s+right\b",
        }
        for value, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return value
        return None

    def _extract_time_to_impact_s(self, text: str) -> float | None:
        patterns = [
            r"\btime[- ]to[- ]impact(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*(?:s|sec|seconds?)\b",
            r"\btti(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*(?:s|sec|seconds?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group("value"))
        return None

    def _extract_trajectory(self, text: str, obstacle_type: str) -> str | None:
        patterns = [
            ("stationary", r"\bstationary\b|\bparked\b"),
            ("crossing", r"\bcrossing\b|\bentering the lane\b|\brunning into the road\b"),
            ("oncoming", r"\boncoming\b"),
            ("merging", r"\bmerging\b"),
            ("same_lane", r"\bsame lane\b|\bahead in lane\b"),
        ]
        for value, pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return value
        default_spec = self.TYPE_DEFAULTS.get(obstacle_type)
        return default_spec["trajectory"] if default_spec else None

    def _extract_vulnerability_class(self, text: str) -> str | None:
        patterns = {
            "high": r"\bvulnerability(?:\s+class)?\s+high\b|\bhigh vulnerability\b",
            "medium": r"\bvulnerability(?:\s+class)?\s+medium\b|\bmedium vulnerability\b",
            "low": r"\bvulnerability(?:\s+class)?\s+low\b|\blow vulnerability\b",
        }
        for value, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return value
        return None

    def _extract_responsibility(self, text: str) -> bool | None:
        if re.search(
            r"\b(?:jaywalking|ran a red light|ignored right of way|illegal(?:ly)?|responsible for risk)\b",
            text,
            re.IGNORECASE,
        ):
            return True
        if re.search(r"\bnot responsible for risk\b", text, re.IGNORECASE):
            return False
        return None

    def _extract_confidence(self, text: str, label: str) -> float | None:
        label_pattern = re.escape(label)
        patterns = [
            rf"\b{label_pattern}\b[^.!?;]{{0,20}}?\bconfidence(?:\s+is|\s+of)?\s*(?P<value>\d+(?:\.\d+)?%?)",
            rf"\b{label_pattern}\b[^.!?;]{{0,20}}?(?P<value>\d+(?:\.\d+)?%?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._coerce_confidence(match.group("value"))
        return None

    def _extract_occluded_zones(self, text: str) -> list[str]:
        zones: list[str] = []
        for zone in self.OCCLUSION_ZONES:
            pattern = (
                rf"(?:occluded|hidden|blind)[^.!?;]{{0,20}}?\b{re.escape(zone)}\b|"
                rf"\b{re.escape(zone)}\b[^.!?;]{{0,20}}?(?:occluded|hidden|blind)"
            )
            if re.search(pattern, text, re.IGNORECASE) and zone not in zones:
                zones.append(zone)
        return zones

    def _extract_distance_value(self, text: str, patterns: list[str]) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group("value"))
        return None

    def _extract_mass_kg(self, text: str, labels: list[str]) -> float | None:
        label_alternatives = "|".join(re.escape(label) for label in labels if label)
        patterns: list[str] = []
        if label_alternatives:
            patterns.extend(
                [
                    rf"\b(?:{label_alternatives})\b[^.!?;]{{0,30}}?\b(?:weighs|weighing|mass(?:\s+is|\s+of)?)\s*"
                    + self.NUMBER_RE
                    + r"\s*kg\b",
                    rf"\b(?:{label_alternatives})\b[^.!?;]{{0,30}}?\b" + self.NUMBER_RE + r"\s*kg\b",
                ]
            )
        patterns.append(r"\bmass(?:\s+is|\s+of)?\s*" + self.NUMBER_RE + r"\s*kg\b")
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group("value"))
        return None

    def _convert_speed_to_kmh(self, value: float, unit: str) -> float:
        unit = unit.lower()
        if unit in {"km/h", "kph"}:
            return value
        if unit == "mph":
            return value * 1.60934
        if unit == "m/s":
            return value * 3.6
        raise ScenarioParseError(f"Unsupported speed unit: {unit}")

    def _calculate_braking_distance(self, speed_kmh: float, acceleration_ms2: float) -> float:
        speed_mps = self._kmh_to_mps(speed_kmh)
        decel = abs(acceleration_ms2) if acceleration_ms2 < 0 else self.DEFAULT_BRAKING_DECEL_MS2
        decel = max(decel, 0.1)
        return (speed_mps**2) / (2.0 * decel)

    def _kmh_to_mps(self, speed_kmh: float) -> float:
        return speed_kmh / 3.6

    def _looks_like_ego_reference(self, text: str, position: int) -> bool:
        prefix = text[max(0, position - 20) : position].lower()
        return "ego " in prefix or "autonomous " in prefix or "self-driving " in prefix

    def _canonicalize_action(self, action: str) -> str | None:
        canonical = action.strip().lower().replace("-", " ").replace("_", " ")
        if canonical == "brake straight":
            return "brake_straight"
        if canonical in {"swerve left", "steer left"}:
            return "swerve_left"
        if canonical in {"swerve right", "steer right"}:
            return "swerve_right"
        if canonical in {"brake swerve left", "brake and swerve left"}:
            return "brake_swerve_left"
        if canonical in {"brake swerve right", "brake and swerve right"}:
            return "brake_swerve_right"
        if canonical == "maintain lane":
            return "maintain_lane"
        return None

    def _coerce_number(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ScenarioParseError(f"Expected numeric value, got {value!r}") from exc

    def _coerce_string(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _coerce_bool(self, value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes"}:
                return True
            if lowered in {"false", "no"}:
                return False
        raise ScenarioParseError(f"Expected boolean value, got {value!r}")

    def _coerce_confidence(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            value = value.strip()
            if value.endswith("%"):
                return max(0.0, min(1.0, float(value[:-1]) / 100.0))
        number = self._coerce_number(value)
        if number is None:
            return None
        if number > 1.0:
            number = number / 100.0
        return max(0.0, min(1.0, number))

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
