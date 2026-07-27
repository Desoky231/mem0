from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


RESULTS_GENERATION_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:

You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:

1. Carefully analyze all provided memories from both speakers

2. Pay special attention to the timestamps to determine the answer

3. If the question asks about a specific event or fact, look for direct evidence in the memories

4. If the memories contain contradictory information, prioritize the most recent memory

5. If there is a question about time references (like "last year", "two months ago", etc.),
calculate the actual date based on the memory timestamp. For example, if a memory from
4 May 2022 mentions "went to India last year," then the trip occurred in 2021.

6. Always convert relative time references to specific dates, months, or years. For example,
convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
timestamp. Ignore the reference while answering the question.

7. Focus only on the content of the memories from both speakers. Do not confuse character
names mentioned in memories with the actual users who created those memories.

8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):

1. First, examine all memories that contain information related to the question

2. Examine the timestamps and content of these memories carefully

3. Look for explicit mentions of dates, times, locations, or events that answer the question

4. If the answer requires calculation (e.g., converting relative time references), show your work

5. Formulate a precise, concise answer based solely on the evidence in the memories

6. Double-check that your answer directly addresses the question asked

7. Ensure your final answer is specific and avoids vague time references

Memories for user {speaker_1_user_id}:

{speaker_1_memories}

Memories for user {speaker_2_user_id}:

{speaker_2_memories}

Question: {question}

Answer:"""


JUDGE_PROMPT = """Your task is to label an answer to a question as "CORRECT" or "WRONG".
You will be given the following data:
(1) a question (posed by one user to another user),
(2) a ‘gold’ (ground truth) answer,
(3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:

Question: Do you remember what I got the last time I went to Hawaii?

Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like ‘last Tuesday’ or ‘next month’), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., ‘May 7th’ vs ‘7 May’), consider it CORRECT if it’s the same date.

Now it’s time for the real question:

Question: {question}

Gold answer: {gold_answer}

Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def _render_speaker_memories(retrieval: Any) -> tuple[str, str, str, str]:
    speakers = (
        retrieval.get("speakers", [])
        if isinstance(retrieval, dict)
        else []
    )
    padded = list(speakers[:2])
    while len(padded) < 2:
        padded.append(
            {
                "speaker": f"speaker_{len(padded) + 1}",
                "memories": [],
            }
        )
    first, second = padded
    return (
        str(first.get("speaker") or first.get("user_id") or "speaker_1"),
        json.dumps(
            first.get("memories", []),
            ensure_ascii=False,
            default=str,
        ),
        str(second.get("speaker") or second.get("user_id") or "speaker_2"),
        json.dumps(
            second.get("memories", []),
            ensure_ascii=False,
            default=str,
        ),
    )


def _judge_label(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.casefold().startswith("json"):
            cleaned = cleaned[4:].strip()
    parsed = json.loads(cleaned)
    label = str(parsed.get("label", "")).strip().upper()
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError(f"Invalid judge label: {label!r}")
    return label


class AnswerGenerator:
    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def answer(self, *, question: str, category: int, retrieval: Any) -> str:
        speaker_1, memories_1, speaker_2, memories_2 = (
            _render_speaker_memories(retrieval)
        )
        prompt = RESULTS_GENERATION_PROMPT.format(
            speaker_1_user_id=speaker_1,
            speaker_1_memories=memories_1,
            speaker_2_user_id=speaker_2,
            speaker_2_memories=memories_2,
            question=question,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=128,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    def judge(
        self,
        *,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> str:
        prompt = JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=64,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": prompt}],
        )
        return _judge_label(response.choices[0].message.content or "")


class ConversationSummarizer:
    """Maintain compact key knowledge across five-pair conversation batches."""

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def update(
        self,
        *,
        previous_summary: str,
        batch_index: int,
        messages: list[str],
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Maintain a compact key-knowledge summary for an "
                        "ongoing conversation between two people. Organize it "
                        "by person plus a shared events/timeline section. "
                        "Preserve stable personal facts, preferences, "
                        "relationships, important events and absolute dates, "
                        "plans, commitments, and changes that supersede older "
                        "facts. Preserve who said or experienced each fact. "
                        "Discard greetings, filler, repeated facts, and "
                        "turn-by-turn narration. Update the previous summary "
                        "using only the latest messages. Aim for 500-650 tokens "
                        "and never exceed 800 tokens. Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Previous key-knowledge summary:\n"
                        f"{previous_summary or '(none)'}\n\n"
                        f"Latest conversation-pair batch {batch_index}:\n"
                        + "\n".join(messages)
                        + "\n\nReturn the updated key-knowledge summary only."
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()
