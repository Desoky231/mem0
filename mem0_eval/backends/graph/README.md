# Graph-memory baseline

This folder contains Mem0's older graph-memory implementation from
`mem0ai==0.1.45`.

It is separate because the current package used by the rest of this repository
does not expose the same graph API. Keeping a second environment prevents the
two incompatible Mem0 versions from being mixed accidentally.

The baseline uses Neo4j for graph storage, DeepSeek for extracting entities and
relationships, and the same local BGE embedding model as the text experiment.

## Setup and run

```bash
docker compose up -d
uv sync --project mem0_eval/backends/graph
uv run --project mem0_eval/backends/graph \
  python -m mem0_eval.run check-graph
```

Available experiments:

```bash
uv run --project mem0_eval/backends/graph \
  python -m mem0_eval.run locomo --backend graph
uv run --project mem0_eval/backends/graph \
  python -m mem0_eval.run personamem --backend graph
uv run --project mem0_eval/backends/graph \
  python -m mem0_eval.run memory-changes --backend graph
```

This is a historical baseline, so comparisons with the current text-memory
implementation include both an architecture difference and a version
difference.
