from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PersonaMemCase:
    row_index: int
    persona_id: int
    user_query: str
    correct_answer: str
    incorrect_answers: tuple[str, ...]
    preference: str
    previous_preference: str | None
    messages: tuple[dict[str, str], ...]
    preference_type: str
    updated: bool
    who: str
    sensitive: bool
    scenario: str

    @property
    def case_id(self) -> str:
        return f"row-{self.row_index}-persona-{self.persona_id}"

    @classmethod
    def from_row(cls, row: dict[str, Any], row_index: int) -> "PersonaMemCase":
        query = ast.literal_eval(row["user_query"])
        messages = json.loads(row["related_conversation_snippet"])
        incorrect = json.loads(row["incorrect_answers"])
        if not isinstance(query, dict) or not isinstance(query.get("content"), str):
            raise ValueError(f"row {row_index}: user_query is not a role/content object")
        if not isinstance(messages, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("content"), str)
            for item in messages
        ):
            raise ValueError(f"row {row_index}: invalid conversation snippet")
        if not isinstance(incorrect, list) or len(incorrect) != 3:
            raise ValueError(f"row {row_index}: expected three incorrect answers")
        return cls(
            row_index=row_index,
            persona_id=int(row["persona_id"]),
            user_query=query["content"],
            correct_answer=str(row["correct_answer"]),
            incorrect_answers=tuple(str(item) for item in incorrect),
            preference=str(row["preference"]),
            previous_preference=(
                str(row["prev_pref"]) if row.get("prev_pref") is not None else None
            ),
            messages=tuple(
                {"role": item["role"], "content": item["content"]}
                for item in messages
            ),
            preference_type=str(row["pref_type"]),
            updated=bool(row["updated"]),
            who=str(row["who"]),
            sensitive=bool(row["sensitive_info"]),
            scenario=str(row["conversation_scenario"]),
        )


def load_cases(path: Path) -> list[PersonaMemCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("PersonaMem dataset must contain a JSON list")
    return [PersonaMemCase.from_row(row, index) for index, row in enumerate(rows)]


def sample_cases(
    cases: list[PersonaMemCase],
    *,
    limit: int,
    seed: int,
    include_sensitive: bool = False,
    include_multimodal: bool = False,
) -> list[PersonaMemCase]:
    """Deterministic round-robin sample across preference types."""
    if limit < 1:
        raise ValueError("limit must be positive")
    eligible = [
        case
        for case in cases
        if (include_sensitive or not case.sensitive)
        and (include_multimodal or case.preference_type != "multimodal")
    ]
    rng = random.Random(seed)
    groups: dict[str, list[PersonaMemCase]] = {}
    for case in eligible:
        groups.setdefault(case.preference_type, []).append(case)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[PersonaMemCase] = []
    labels = sorted(groups)
    while len(selected) < min(limit, len(eligible)):
        made_progress = False
        for label in labels:
            if groups[label] and len(selected) < limit:
                selected.append(groups[label].pop())
                made_progress = True
        if not made_progress:
            break
    return selected
