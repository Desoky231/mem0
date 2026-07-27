from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mem0_eval/backends/graph/adapter.py"
)


def _load_identifier_function():
    source = MODULE_PATH.read_text(encoding="utf-8")
    start = source.index("def _safe_cypher_identifier")
    end = source.index("\n\ndef _harden_generated_graph_identifiers", start)
    namespace = {"Any": object, "re": __import__("re")}
    exec(source[start:end], namespace)
    return namespace["_safe_cypher_identifier"]


def test_generated_cypher_identifier_is_sanitized() -> None:
    sanitize = _load_identifier_function()
    assert sanitize("platform/social_media", fallback="unknown") == "platform_social_media"
    assert sanitize("123 place", fallback="unknown") == "n_123_place"
    assert sanitize("/", fallback="unknown") == "unknown"
