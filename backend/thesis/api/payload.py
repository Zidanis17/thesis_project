from __future__ import annotations

from typing import Any

__all__ = ["strip_payload_metadata"]


def strip_payload_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_payload_metadata(item)
            for key, item in value.items()
            if key != "_meta"
        }
    if isinstance(value, list):
        return [strip_payload_metadata(item) for item in value]
    return value
