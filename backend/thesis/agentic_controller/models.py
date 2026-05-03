from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class RetrievalIntent:
    scenario_class: str
    required_frameworks: list[str]
    retrieval_focus_terms: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgenticAssessment:
    scene_interpretable: bool
    collision_unavoidable: bool | None
    vulnerable_road_users_present: bool
    passenger_vru_tradeoff_possible: bool
    unknown_object_present: bool
    candidate_frameworks: list[str]
    retrieval_intent: RetrievalIntent
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgenticValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    corrected_dominant_framework: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
