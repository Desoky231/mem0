from __future__ import annotations

import os
import logging
import re
from typing import Any
from types import MethodType

from importlib.metadata import version
from mem0.configs.base import MemoryConfig
from mem0.memory.graph_memory import MemoryGraph
from mem0.utils.factory import EmbedderFactory
from neo4j import GraphDatabase
from mem0_eval.integrations.deepseek import disable_deepseek_thinking


EXPECTED_MEM0_VERSION = "0.1.45"
logger = logging.getLogger(__name__)


def _safe_cypher_identifier(value: Any, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", str(value)).strip("_").lower()
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        return fallback
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    return normalized


def _harden_generated_graph_identifiers(graph: MemoryGraph) -> None:
    """Guard v0.1.45 against malformed LLM JSON and unsafe Cypher labels."""
    original_retrieve = graph._retrieve_nodes_from_data
    original_establish = graph._establish_nodes_relations_from_data
    original_delete_candidates = graph._get_delete_entities_from_search_output

    def safe_retrieve(self, data, filters):
        try:
            entity_map = original_retrieve(data, filters)
        except Exception as exc:
            logger.warning("Graph entity extraction failed; using no entities: %s", exc)
            return {}
        return {
            str(entity): _safe_cypher_identifier(entity_type, fallback="unknown")
            for entity, entity_type in entity_map.items()
        }

    def sanitize_relations(items):
        sanitized = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            candidate["relationship"] = _safe_cypher_identifier(
                candidate.get("relationship"), fallback="related_to"
            )
            sanitized.append(candidate)
        return sanitized

    def safe_establish(self, data, filters, entity_type_map):
        try:
            return sanitize_relations(
                original_establish(data, filters, entity_type_map)
            )
        except Exception as exc:
            logger.warning("Graph relation extraction failed; adding no relations: %s", exc)
            return []

    def safe_delete_candidates(self, search_output, data, filters):
        try:
            return sanitize_relations(
                original_delete_candidates(search_output, data, filters)
            )
        except Exception as exc:
            logger.warning("Graph deletion extraction failed; deleting no relations: %s", exc)
            return []

    graph._retrieve_nodes_from_data = MethodType(safe_retrieve, graph)
    graph._establish_nodes_relations_from_data = MethodType(safe_establish, graph)
    graph._get_delete_entities_from_search_output = MethodType(
        safe_delete_candidates, graph
    )


class GraphMemoryAdapter:
    def __init__(self, graph: MemoryGraph, *, top_k: int = 5) -> None:
        if top_k != 5:
            raise ValueError(
                "Mem0 v0.1.45 MemoryGraph hardcodes BM25 retrieval to 5 "
                "results; use top_k=5 for an honest baseline."
            )
        self.graph = graph
        self.top_k = top_k

    def add(self, statement: str, *, user_id: str) -> Any:
        return self.graph.add(statement, {"user_id": user_id})

    def search(self, query: str, *, user_id: str) -> Any:
        return self.graph.search(
            query, {"user_id": user_id}, limit=self.top_k
        )

    def get_all(self, *, user_id: str) -> Any:
        return self.graph.get_all({"user_id": user_id}, limit=100)

    def delete_all(self, *, user_id: str) -> Any:
        return self.graph.delete_all({"user_id": user_id})


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _neo4j_password() -> str:
    return os.getenv("NEO4J_LEGACY_PASSWORD") or _required_env(
        "NEO4J_PASSWORD"
    )


def verify_neo4j() -> dict[str, str]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = _neo4j_password()
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            record = session.run(
                "CALL dbms.components() YIELD versions "
                "RETURN versions[0] AS version LIMIT 1"
            ).single(strict=True)
        return {"uri": uri, "version": str(record["version"])}
    finally:
        driver.close()


def build_graph_adapter(*, top_k: int = 5) -> GraphMemoryAdapter:
    installed = version("mem0ai")
    if installed != EXPECTED_MEM0_VERSION:
        raise RuntimeError(
            f"Expected mem0ai {EXPECTED_MEM0_VERSION}, found {installed}. "
            "Run through `uv run --project mem0_eval/backends/graph`."
        )
    if "graph_store" not in MemoryConfig.model_fields:
        raise RuntimeError("Historical MemoryConfig does not expose graph_store")

    verify_neo4j()
    EmbedderFactory.provider_to_class["huggingface"] = (
        "mem0_eval.backends.graph.embedding.GraphFastEmbed"
    )
    api_key = _required_env("DEEPSEEK_API_KEY")
    config = MemoryConfig(
        **{
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": api_key,
                    "model": os.getenv(
                        "DEEPSEEK_MODEL", "deepseek-chat"
                    ),
                    "openai_base_url": os.getenv(
                        "DEEPSEEK_API_BASE", "https://api.deepseek.com"
                    ),
                    "temperature": 0,
                    "max_tokens": 8192,
                    "top_p": 1,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5",
                    "embedding_dims": 384,
                },
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": os.getenv(
                        "NEO4J_URI", "bolt://localhost:7687"
                    ),
                    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                    "password": _neo4j_password(),
                },
                "custom_prompt": (
                    "Preserve exact identifier codes as entities and preserve "
                    "their exact spelling. New statements are more recent than "
                    "existing statements."
                ),
            },
            "version": "v1.1",
        }
    )
    graph = MemoryGraph(config)
    disable_deepseek_thinking(graph.llm)
    _harden_generated_graph_identifiers(graph)
    return GraphMemoryAdapter(graph, top_k=top_k)
