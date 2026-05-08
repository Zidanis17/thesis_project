from __future__ import annotations

from typing import Any

__all__ = [
    "VALID_FRAMEWORK_IDS",
    "REQUIRED_STRING_FIELDS",
    "REQUIRED_LIST_FIELDS",
    "validate_framework",
]

VALID_FRAMEWORK_IDS: frozenset[str] = frozenset(
    {"EF-01", "EF-02", "EF-03", "EF-04", "EF-05", "EF-06"}
)

# Fields that must be non-empty strings
REQUIRED_STRING_FIELDS: tuple[str, ...] = (
    "framework_id",
    "name",
    "title",
    "source",
    "alias",
    "category",
    "foundation",
    "decision_logic",
    "tradeoffs",
)

# Fields that must be non-empty lists of non-empty strings
REQUIRED_LIST_FIELDS: tuple[str, ...] = (
    "pros",
    "cons",
    "best_fit_scenarios",
    "poor_fit_scenarios",
    "use_when",
    "avoid_when",
    "dominant_when",
    "key_parameters",
    "scenario_tags",
    "source_papers",
)


def validate_framework(payload: Any, framework_id: str) -> list[str]:
    """Return a list of validation error messages; empty list means valid."""
    errors: list[str] = []

    if not isinstance(payload, dict):
        return [f"{framework_id}: root value must be a JSON object, got {type(payload).__name__}"]

    for field in REQUIRED_STRING_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"{framework_id}: field '{field}' must be a non-empty string, got {value!r}"
            )

    for field in REQUIRED_LIST_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list):
            errors.append(
                f"{framework_id}: field '{field}' must be a list, got {type(value).__name__}"
            )
        elif len(value) == 0:
            errors.append(f"{framework_id}: field '{field}' must not be empty")
        else:
            for i, item in enumerate(value):
                if not isinstance(item, str) or not item.strip():
                    errors.append(
                        f"{framework_id}: field '{field}[{i}]' must be a non-empty string, "
                        f"got {item!r}"
                    )

    fid = payload.get("framework_id", "")
    if fid not in VALID_FRAMEWORK_IDS:
        errors.append(
            f"{framework_id}: 'framework_id' must be one of "
            f"{sorted(VALID_FRAMEWORK_IDS)}, got {fid!r}"
        )

    category = payload.get("category", "")
    if category != "ethical_frameworks":
        errors.append(
            f"{framework_id}: 'category' must be 'ethical_frameworks', got {category!r}"
        )

    title = payload.get("title", "")
    if isinstance(title, str) and title and not title.startswith(framework_id):
        errors.append(
            f"{framework_id}: 'title' must start with '{framework_id}', got {title!r}"
        )

    return errors
