from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}


@dataclass(frozen=True)
class LoCoMoQuestion:
    conversation_index: int
    question_index: int
    question: str
    answer: str
    evidence: tuple[str, ...]
    category: int

    @property
    def question_id(self) -> str:
        return f"conversation-{self.conversation_index}-question-{self.question_index}"


@dataclass(frozen=True)
class LoCoMoConversation:
    conversation_index: int
    sample_id: str
    raw_conversation: dict[str, Any]
    questions: tuple[LoCoMoQuestion, ...]

    def session_keys(self, limit: int) -> list[str]:
        keys = [
            key
            for key in self.raw_conversation
            if re.fullmatch(r"session_\d+", key)
        ]
        return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))[:limit]

    def session_text(self, key: str) -> str:
        date = self.raw_conversation.get(f"{key}_date_time", "unknown")
        turns = self.raw_conversation[key]
        body = "\n".join(
            f"{turn['speaker']}: {turn['text']}" for turn in turns
        )
        return f"SESSION DATE: {date}\n{body}"


def load_conversations(path: Path) -> list[LoCoMoConversation]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    conversations: list[LoCoMoConversation] = []
    for conversation_index, row in enumerate(rows):
        questions = tuple(
            LoCoMoQuestion(
                conversation_index=conversation_index,
                question_index=question_index,
                question=str(item["question"]),
                answer=str(item.get("answer", "")),
                evidence=tuple(str(value) for value in item.get("evidence", [])),
                category=int(item["category"]),
            )
            for question_index, item in enumerate(row["qa"])
        )
        conversations.append(
            LoCoMoConversation(
                conversation_index=conversation_index,
                sample_id=str(row.get("sample_id", conversation_index)),
                raw_conversation=row["conversation"],
                questions=questions,
            )
        )
    return conversations


def _evidence_session(evidence: str) -> int | None:
    match = re.match(r"D(\d+):", evidence)
    return int(match.group(1)) if match else None


def sample_questions(
    conversation: LoCoMoConversation,
    *,
    session_count: int,
    questions_per_category: int,
    categories: tuple[int, ...],
    seed: int,
) -> list[LoCoMoQuestion]:
    """Sample only questions whose annotated evidence has been ingested."""
    rng = random.Random(f"{seed}:{conversation.conversation_index}")
    selected: list[LoCoMoQuestion] = []
    for category in categories:
        eligible = [
            question
            for question in conversation.questions
            if question.category == category
            and question.evidence
            and all(
                (session := _evidence_session(item)) is not None
                and session <= session_count
                for item in question.evidence
            )
        ]
        rng.shuffle(eligible)
        selected.extend(eligible[:questions_per_category])
    return selected
