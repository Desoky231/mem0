from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


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
        date_instruction = (
            " Give an approximate conversation date when relevant."
            if category == 2
            else ""
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=128,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied retrieved memories. "
                        "Be concise and factual. If the memories do not contain "
                        "the answer, say 'Not mentioned in the retrieved memory'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Retrieved memories:\n"
                        + json.dumps(retrieval, ensure_ascii=False, default=str)
                        + f"\n\nQuestion: {question}{date_instruction}"
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()
