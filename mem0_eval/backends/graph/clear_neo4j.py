#!/usr/bin/env python3
"""Delete all benchmark-created nodes and relationships from Neo4j."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[3]


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required in .env or the environment")
    return value


def main() -> int:
    load_dotenv(ROOT / ".env")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_LEGACY_PASSWORD") or _required(
        "NEO4J_PASSWORD"
    )

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            before = session.run(
                "MATCH (n) "
                "OPTIONAL MATCH (n)-[r]-() "
                "RETURN count(DISTINCT n) AS nodes, "
                "count(DISTINCT r) AS relationships"
            ).single(strict=True)
            session.run("MATCH (n) DETACH DELETE n").consume()
            after = session.run(
                "MATCH (n) "
                "OPTIONAL MATCH (n)-[r]-() "
                "RETURN count(DISTINCT n) AS nodes, "
                "count(DISTINCT r) AS relationships"
            ).single(strict=True)
    finally:
        driver.close()

    print(
        json.dumps(
            {
                "uri": uri,
                "deleted": {
                    "nodes": int(before["nodes"]),
                    "relationships": int(before["relationships"]),
                },
                "remaining": {
                    "nodes": int(after["nodes"]),
                    "relationships": int(after["relationships"]),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
