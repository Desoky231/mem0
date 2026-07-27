from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


def parse_choice(raw: str) -> str | None:
    cleaned = raw.strip().upper()
    patterns = (
        r"^([ABCD])[\.\)]?$",
        r"^ANSWER\s*:\s*([ABCD])[\.\)]?$",
        r"^\\BOXED\{([ABCD])\}$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, cleaned)
        if match:
            return match.group(1)
    return None


class MCQSelector:
    """Select an answer using only the query, options, and retrieved memory."""

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def select(
        self,
        *,
        query: str,
        options: list[tuple[str, str]],
        retrieval: Any,
    ) -> dict[str, str | None]:
        rendered_options = "\n".join(f"{letter}. {text}" for letter, text in options)
        retrieval_text = json.dumps(retrieval, ensure_ascii=False, default=str)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=50,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Choose the single response that best answers the user while "
                        "respecting the retrieved long-term memory. Treat an explicit "
                        "request to forget as authoritative. Reply with only A, B, C, or D."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Retrieved memory:\n{retrieval_text}\n\n"
                        f"User query:\n{query}\n\nOptions:\n{rendered_options}"
                    ),
                },
            ],
        )
        message = response.choices[0].message
        raw = message.content or ""
        if not raw:
            raw = str(getattr(message, "reasoning_content", "") or "")
        return {"choice": parse_choice(raw), "raw": raw}
