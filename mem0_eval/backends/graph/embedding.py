from __future__ import annotations

from typing import Any

from fastembed import TextEmbedding


class GraphFastEmbed:
    """Expose FastEmbed through the interface expected by Mem0 v0.1.45."""

    def __init__(self, config: Any) -> None:
        model_name = config.model or "BAAI/bge-small-en-v1.5"
        self.model = TextEmbedding(model_name=model_name)
        config.model = model_name
        config.embedding_dims = config.embedding_dims or 384

    def embed(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()
