from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryChangeCase:
    case_id: str
    subject: str
    baseline_statement: str
    updated_statement: str
    query: str
    old_marker: str
    new_marker: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryChangeCase":
        case = cls(**{field: value[field] for field in cls.__dataclass_fields__})
        case.validate()
        return case

    def validate(self) -> None:
        fields = asdict(self)
        for name, value in fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.old_marker.casefold() == self.new_marker.casefold():
            raise ValueError("old_marker and new_marker must differ")
        if self.old_marker.casefold() not in self.baseline_statement.casefold():
            raise ValueError("baseline_statement must contain old_marker")
        if self.new_marker.casefold() not in self.updated_statement.casefold():
            raise ValueError("updated_statement must contain new_marker")
        if self.old_marker.casefold() in self.updated_statement.casefold():
            raise ValueError(
                "updated_statement must not repeat old_marker; that would make leakage scoring ambiguous"
            )
