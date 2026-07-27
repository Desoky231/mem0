from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from mem0 import Memory

from mem0_eval.integrations.deepseek import disable_deepseek_thinking


EXPECTED_MEM0_VERSION = "2.0.14"


class TextMemoryAdapter:
    def __init__(self, memory: Memory, *, top_k: int, threshold: float) -> None:
        self.memory = memory
        self.top_k = top_k
        self.threshold = threshold

    def add(self, statement: str, *, user_id: str) -> Any:
        return self.memory.add(statement, user_id=user_id, infer=True)

    def add_exchange(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        speaker: str,
        session: str,
        session_date: str,
        conversation_summary: str,
        recent_messages: list[str],
    ) -> Any:
        context = (
            f"Extract memories only about {speaker}. The new exchange occurred "
            f"on {session_date}; use that as the observation date and preserve "
            "absolute dates in temporal memories. Mem0 supplies its native "
            "previous-10-message window. Use the following key-knowledge "
            "summary only to resolve broader context, references, and pronouns. "
            "Never extract an old fact merely because it appears in the "
            "summary.\n\n"
            f"Key-knowledge summary:\n{conversation_summary or '(none)'}"
        )
        return self.memory.add(
            messages,
            user_id=user_id,
            metadata={
                "speaker": speaker,
                "session": session,
                "conversation_date": session_date,
            },
            infer=True,
            prompt=context,
        )

    def search(self, query: str, *, user_id: str) -> Any:
        return self.memory.search(
            query,
            top_k=self.top_k,
            threshold=self.threshold,
            filters={"user_id": user_id},
        )

    def get_all(self, *, user_id: str) -> Any:
        return self.memory.get_all(
            filters={"user_id": user_id},
            top_k=max(self.top_k * 4, 50),
        )

    def delete_all(self, *, user_id: str) -> Any:
        return self.memory.delete_all(user_id=user_id)


def build_text_memory(
    *,
    state_dir: Path,
    top_k: int,
    threshold: float,
) -> TextMemoryAdapter:
    installed = version("mem0ai")
    if installed != EXPECTED_MEM0_VERSION:
        raise RuntimeError(
            f"Text memory requires mem0ai {EXPECTED_MEM0_VERSION}; found "
            f"{installed}. Run from the root project environment."
        )
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required in the environment or .env")

    state_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": api_key,
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                "openai_base_url": "https://api.deepseek.com",
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {"model": "BAAI/bge-small-en-v1.5"},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_text_384",
                "embedding_model_dims": 384,
                "path": str(state_dir / "qdrant"),
            },
        },
        "history_db_path": str(state_dir / "history.db"),
        "custom_instructions": (
            "Preserve exact alphanumeric identifier codes in extracted facts. "
            "When a new fact supersedes an existing fact about the same subject, "
            "retain only the current fact."
        ),
    }
    memory = Memory.from_config(config)
    disable_deepseek_thinking(memory.llm)
    return TextMemoryAdapter(memory, top_k=top_k, threshold=threshold)
