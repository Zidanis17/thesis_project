from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .._env import load_project_env

__all__ = ["LLMScenarioParserAgent", "LLMScenarioParserAgentResult"]


SCENARIO_EXTRACTION_SYSTEM_PROMPT = """
You are the scenario parsing agent for an autonomous-vehicle ethics pipeline.
Extract only facts that are explicitly present or directly unit-convertible from
the user's natural-language scenario. Return exactly one JSON object matching
the schema. Do not invent values, do not use common-sense defaults, and do not
fill a field simply because it is typical.

Use null for unknown numeric and boolean fields. Use "" for unknown string
fields. Use [] for unknown lists. Keep obstacle ids as obj_01, obj_02, ... if
the user does not provide ids.

Canonical action names: brake_straight, swerve_left, swerve_right,
brake_swerve_left, brake_swerve_right, maintain_lane.
Canonical common road types: residential, urban, intersection, school_zone,
rural, highway, motorway, parking_lot.
Canonical common obstacle types: child_pedestrian, adult_pedestrian,
elderly_pedestrian, pedestrian, cyclist, motorcyclist, parked_vehicle,
vehicle, truck, bus, animal.
Canonical trajectory names when stated: crossing, stationary, same_lane,
oncoming, merging.

Output schema:
{
  "ego_vehicle": {
    "speed_kmh": number|null,
    "acceleration_ms2": number|null,
    "heading_deg": number|null,
    "lane_position": "string",
    "braking_distance_m": number|null,
    "mass_kg": number|null,
    "passenger_at_risk": boolean|null
  },
  "environment": {
    "road_type": "string",
    "speed_limit_kmh": number|null,
    "weather": "string",
    "visibility_m": number|null,
    "time_of_day": "string",
    "traffic_density": "string"
  },
  "obstacles": [
    {
      "id": "string",
      "type": "string",
      "distance_m": number|null,
      "relative_speed_kmh": number|null,
      "time_to_impact_s": number|null,
      "trajectory": "string",
      "vulnerability_class": "string",
      "mass_kg": number|null,
      "responsible_for_risk": boolean|null
    }
  ],
  "sensor_confidence": {
    "lidar": number|null,
    "camera": number|null,
    "radar": number|null,
    "overall_scene_confidence": number|null,
    "occluded_zones": ["string"]
  },
  "available_actions": ["string"],
  "collision_unavoidable": boolean|null
}
""".strip()


@dataclass(slots=True)
class LLMScenarioParserAgentResult:
    payload: dict[str, Any]
    model_name: str
    provider: str
    runtime_available: bool
    runtime_error: str | None = None


class LLMScenarioParserAgent:
    DEFAULT_PROVIDER = "openai"
    DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
    DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
    DEFAULT_HUGGINGFACE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    DEFAULT_HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
    DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = SCENARIO_EXTRACTION_SYSTEM_PROMPT,
    ) -> None:
        load_project_env()
        self.provider = (provider or os.getenv("SCENARIO_PARSER_PROVIDER") or self.DEFAULT_PROVIDER).strip().lower()
        self.model_name = model_name or os.getenv("SCENARIO_PARSER_MODEL") or self._default_model_name(self.provider)
        self.temperature = float(os.getenv("SCENARIO_PARSER_TEMPERATURE", temperature))
        self.system_prompt = system_prompt
        self._runtime_error: RuntimeError | None = None
        self.client: Any | None = None

        try:
            self.client = self._build_client()
        except Exception as exc:
            self._runtime_error = RuntimeError(
                "Scenario parser LLM agent could not be initialized. "
                "Set SCENARIO_PARSER_PROVIDER plus the matching API key."
            )
            self._runtime_error.__cause__ = exc

    @property
    def runtime_available(self) -> bool:
        return self._runtime_error is None and self.client is not None

    def extract(self, text: str) -> LLMScenarioParserAgentResult:
        if not self.runtime_available:
            return LLMScenarioParserAgentResult(
                payload={},
                model_name=self.model_name,
                provider=self.provider,
                runtime_available=False,
                runtime_error=str(self._runtime_error or "Scenario parser LLM agent is unavailable"),
            )

        try:
            response_text = self._invoke(text)
            payload = self._parse_json_response(response_text)
            return LLMScenarioParserAgentResult(
                payload=payload,
                model_name=self.model_name,
                provider=self.provider,
                runtime_available=True,
            )
        except Exception as exc:
            return LLMScenarioParserAgentResult(
                payload={},
                model_name=self.model_name,
                provider=self.provider,
                runtime_available=False,
                runtime_error=str(exc),
            )

    def _invoke(self, text: str) -> str:
        response = self.client.invoke(
            [
                ("system", self.system_prompt),
                ("user", self._user_prompt(text)),
            ]
        )
        return self._message_text(response)

    def _user_prompt(self, text: str) -> str:
        return (
            "Extract the scenario into the exact JSON schema. "
            "Return JSON only, with no markdown fences.\n\n"
            f"Scenario text:\n{text}"
        )

    def _message_text(self, response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1:
            stripped = stripped[start : end + 1]
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("Scenario parser agent output must decode to a JSON object")
        return payload

    def _build_client(self) -> Any:
        if self.provider in {"disabled", "none", "off"}:
            raise RuntimeError("Scenario parser LLM agent disabled by configuration")

        if self.provider == "openai":
            return self._build_openai_compatible_client(
                api_key_env="OPENAI_API_KEY",
                base_url=None,
            )

        if self.provider == "groq":
            return self._build_openai_compatible_client(
                api_key_env="GROQ_API_KEY",
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            )

        if self.provider in {"huggingface", "hf"}:
            return self._build_openai_compatible_client(
                api_key_env="HUGGINGFACE_API_KEY",
                fallback_api_key_env="HF_TOKEN",
                base_url=(
                    os.getenv("HUGGINGFACE_BASE_URL")
                    or os.getenv("HF_BASE_URL")
                    or self.DEFAULT_HUGGINGFACE_BASE_URL
                ),
            )

        raise RuntimeError(f"Unsupported scenario parser provider: {self.provider}")

    def _build_openai_compatible_client(
        self,
        *,
        api_key_env: str,
        base_url: str | None,
        fallback_api_key_env: str | None = None,
    ) -> Any:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv(api_key_env)
        if not api_key and fallback_api_key_env:
            api_key = os.getenv(fallback_api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is required for the scenario parser LLM agent")

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    def _default_model_name(self, provider: str) -> str:
        if provider == "groq":
            return self.DEFAULT_GROQ_MODEL
        if provider in {"huggingface", "hf"}:
            return self.DEFAULT_HUGGINGFACE_MODEL
        return self.DEFAULT_OPENAI_MODEL
