from __future__ import annotations

from typing import Any


def disable_deepseek_thinking(llm: Any) -> bool:
    """Inject DeepSeek V4 non-thinking mode into an existing OpenAI client."""
    model = str(getattr(getattr(llm, "config", None), "model", "")).lower()
    if not model.startswith("deepseek-v4"):
        return False
    completions = llm.client.chat.completions
    if getattr(completions, "_mem0_non_thinking_wrapped", False):
        return True
    original_create = completions.create

    def create(*args: Any, **kwargs: Any) -> Any:
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body["thinking"] = {"type": "disabled"}
        return original_create(*args, extra_body=extra_body, **kwargs)

    completions.create = create
    completions._mem0_non_thinking_wrapped = True
    return True
