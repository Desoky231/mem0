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
class LoCoMoMessage:
    session: str
    session_date: str
    speaker: str
    text: str
    dialogue_id: str

    def formatted(self) -> str:
        return (
            f"[{self.session_date}; {self.dialogue_id}] "
            f"{self.speaker}: {self.text}"
        )


@dataclass(frozen=True)
class LoCoMoExchange:
    session: str
    session_date: str
    exchange_index: int
    messages: tuple[LoCoMoMessage, ...]

    @property
    def exchange_id(self) -> str:
        return f"{self.session}-exchange-{self.exchange_index}"


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

    @property
    def speakers(self) -> tuple[str, str]:
        return (
            str(self.raw_conversation["speaker_a"]),
            str(self.raw_conversation["speaker_b"]),
        )

    def session_messages(self, key: str) -> tuple[LoCoMoMessage, ...]:
        date = str(self.raw_conversation.get(f"{key}_date_time", "unknown"))
        messages = []
        for turn in self.raw_conversation[key]:
            text = str(turn.get("text", "")).strip()
            image_description = str(turn.get("blip_caption", "")).strip()
            image_query = str(turn.get("query", "")).strip()
            if image_description:
                text = (
                    f"{text} [Shared image: {image_description}]"
                    if text
                    else f"[Shared image: {image_description}]"
                )
            elif image_query:
                text = (
                    f"{text} [Shared image related to: {image_query}]"
                    if text
                    else f"[Shared image related to: {image_query}]"
                )
            if not text:
                continue
            messages.append(
                LoCoMoMessage(
                    session=key,
                    session_date=date,
                    speaker=str(turn.get("speaker", "unknown")),
                    text=text,
                    dialogue_id=str(turn.get("dia_id", "unknown")),
                )
            )
        return tuple(messages)

    def incremental_exchanges(self, limit: int) -> tuple[LoCoMoExchange, ...]:
        """Return chronological, non-overlapping message pairs per session."""
        exchanges = []
        for key in self.session_keys(limit):
            messages = self.session_messages(key)
            for start in range(0, len(messages), 2):
                exchanges.append(
                    LoCoMoExchange(
                        session=key,
                        session_date=messages[start].session_date,
                        exchange_index=start // 2,
                        messages=messages[start : start + 2],
                    )
                )
        return tuple(exchanges)


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
