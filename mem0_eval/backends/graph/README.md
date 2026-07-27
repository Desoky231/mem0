# Mem0g composite memory

This backend combines the repository's standard Mem0 text memory with a direct
Neo4j implementation of the graph pipeline described in the Mem0 paper.

Each chronological message pair is sent concurrently to text extraction and a
two-stage entity/relation extractor. Graph entities and relations carry source
dialogue IDs, observation/session metadata, creation and validity timestamps,
and an active/obsolete status. Conflicting edges are closed rather than
deleted. Retrieval combines normal text search, entity-centric graph traversal,
and semantic triplet ranking before using the paper's Mem0g answer prompt.

The old `mem0ai==0.1.45` graph implementation is no longer used.

## Setup and run

```bash
docker compose up -d
uv sync
uv run python -m mem0_eval.run check-graph
```

Available experiments:

```bash
uv run python -m mem0_eval.run locomo --backend graph
uv run python -m mem0_eval.run personamem --backend graph
uv run python -m mem0_eval.run memory-changes --backend graph
```
