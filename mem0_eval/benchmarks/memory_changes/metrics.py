from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def flatten_text(value: Any) -> str:
    """Return all textual fields in a nested Mem0 response."""
    parts: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return "\n".join(parts)


def contains_marker(value: Any, marker: str) -> bool:
    return marker.casefold() in flatten_text(value).casefold()


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def rate(flags: Iterable[bool]) -> float | None:
    values = list(flags)
    return sum(values) / len(values) if values else None

